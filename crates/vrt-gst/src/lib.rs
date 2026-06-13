//! GStreamer RTSP source with Jetson NVMM zero-copy output.
//!
//! Always decodes via `nvv4l2decoder` and converts to RGBA in NVMM-backed
//! memory.  No CPU frame path — every frame is a DMA-BUF that can be imported
//! directly into CUDA without any host copies.
//!
//! ## Typical usage
//! ```no_run
//! use vrt_gst::{RtspSource, NvmmPreprocessStage};
//! use vrt::{Pipeline, Stream};
//!
//! let source  = RtspSource::connect("rtsp://camera/stream").unwrap();
//! let stream  = Stream::new_standalone().unwrap().cuda_stream().clone();
//! let preproc = NvmmPreprocessStage::new(
//!     stream.clone(), source.width(), source.height(),
//!     source.width(), source.height(),
//! ).unwrap();
//! ```

pub use vrt_preproc::Preprocessor;


use std::ffi::c_void;
use std::sync::{Arc, Mutex, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};

use gstreamer::prelude::*;
use vrt::{Source, Operator, ExecCtx, BoxError, VrtTensor, VrtImage, Format, MemKind};
use vrt_preproc::PreprocError;

/// Errors from the GStreamer NVMM source and preprocessing stage.
#[derive(Debug, thiserror::Error)]
pub enum GstSourceError {
    #[error("GStreamer: {0}")]
    Glib(#[from] gstreamer::glib::Error),
    #[error("GStreamer state change: {0}")]
    StateChange(#[from] gstreamer::StateChangeError),
    #[error("pipeline setup: {0}")]
    Setup(&'static str),
    #[error("stream ended before first frame — check RTSP URL and H.264 codec")]
    NoFirstFrame,
    #[error("cudaImportExternalMemory failed (fd={fd}, size={size}, err={code})")]
    NvmmImport { fd: i32, size: u64, code: i32 },
    #[error(transparent)]
    Preproc(#[from] PreprocError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}

// ── CudaMemory ────────────────────────────────────────────────────────────────

/// RAII wrapper for a CUDA external-memory import of an NVMM DMA-BUF.
///
/// Calls `nvbuf_cuda_release` on drop.  Drop only after syncing any CUDA
/// stream that has used `dev_ptr`.
pub struct CudaMemory {
    pub dev_ptr: *mut c_void,
    ext_mem:     *mut c_void,
}

unsafe impl Send for CudaMemory {}

impl Drop for CudaMemory {
    fn drop(&mut self) {
        unsafe { nvbuf_sys::nvbuf_cuda_release(self.ext_mem, self.dev_ptr); }
    }
}

// ── NvmmFrame ─────────────────────────────────────────────────────────────────

/// A single decoded NVMM RGBA frame from an RTSP stream.
///
/// `_keep_alive` holds whatever value must live for as long as `fd` is in use
/// (typically a GStreamer `Sample`) — erased to avoid a gstreamer dep on callers.
pub struct NvmmFrame {
    _keep_alive: Box<dyn Send + Sync + 'static>,
    /// DMA-BUF file descriptor — valid for the lifetime of this frame.
    pub fd:    i32,
    /// Row pitch in bytes.
    pub pitch: u32,
    /// Total NVMM allocation size in bytes, required for `cudaImportExternalMemory`.
    pub size:  u64,
}

impl NvmmFrame {
    pub fn new(keep_alive: impl Send + Sync + 'static, fd: i32, pitch: u32, size: u64) -> Self {
        Self { _keep_alive: Box::new(keep_alive), fd, pitch, size }
    }

    /// Import this NVMM buffer into CUDA device memory.
    ///
    /// # Safety
    /// Calls `nvbuf_cuda_import`.  `self` must remain alive for the duration
    /// of any CUDA work using the returned `dev_ptr`.
    pub unsafe fn cuda_import(&self) -> Result<CudaMemory, GstSourceError> {
        let mut ext_mem: *mut c_void = std::ptr::null_mut();
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let rc = nvbuf_sys::nvbuf_cuda_import(self.fd, self.size, &mut ext_mem, &mut dev_ptr);
        if rc != 0 {
            return Err(GstSourceError::NvmmImport { fd: self.fd, size: self.size, code: rc });
        }
        Ok(CudaMemory { dev_ptr, ext_mem })
    }
}

/// A CPU RGBA snapshot for visualization: `(rgba_bytes, width, height)`.
pub type CpuFrame = (Vec<u8>, u32, u32);

// ── RtspSource ────────────────────────────────────────────────────────────────

/// RTSP source that delivers frames as NVMM RGBA using Jetson hardware decode.
///
/// # Pipeline
/// ```text
/// rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvidconv
///         → video/x-raw(memory:NVMM),format=RGBA → tee
///              ├→ appsink(NVMM)   [main inference path]
///              └→ nvvidconv → video/x-raw,format=RGBA → appsink(CPU)  [viz snapshot]
/// ```
pub struct RtspSource {
    pipeline:   gstreamer::Pipeline,
    rx:         mpsc::Receiver<NvmmFrame>,
    width:      u32,
    height:     u32,
    /// Latest CPU RGBA frame for visualization.  Updated asynchronously by GStreamer;
    /// take with `latest_cpu_frame()` and lock to read.
    cpu_frame:  Arc<Mutex<Option<CpuFrame>>>,
}

impl RtspSource {
    /// Open an RTSP stream and block until the first frame arrives.
    ///
    /// Frames are delivered at the camera's native resolution.
    /// Use [`connect_resized`] to have the VIC scaler downsize before CUDA.
    ///
    /// [`connect_resized`]: RtspSource::connect_resized
    pub fn connect(url: &str) -> Result<Self, GstSourceError> {
        Self::connect_internal(url, None)
    }

    /// Open an RTSP stream and resize every frame to `(width, height)` in
    /// the GStreamer pipeline before it reaches CUDA.
    ///
    /// The resize is done by the Jetson VIC hardware scaler inside `nvvidconv`,
    /// so it costs no CUDA or CPU cycles.  `source.width()` / `source.height()`
    /// return the resized dimensions.
    pub fn connect_resized(url: &str, width: u32, height: u32) -> Result<Self, GstSourceError> {
        Self::connect_internal(url, Some((width, height)))
    }

    fn connect_internal(url: &str, resize: Option<(u32, u32)>) -> Result<Self, GstSourceError> {
        gstreamer::init()?;

        // Optional VIC resize: add width/height to nvvidconv output caps.
        // Two appsinks via tee: NVMM path for inference, CPU path for visualization.
        // leaky=upstream on both queues: if a branch falls behind it drops new frames
        // rather than blocking the decoder.
        let nvmm_caps = match resize {
            None         => "video/x-raw(memory:NVMM),format=RGBA".to_string(),
            Some((w, h)) => format!("video/x-raw(memory:NVMM),format=RGBA,width={w},height={h}"),
        };
        let pipeline_str = format!(
            "rtspsrc location={url} latency=100 ! rtph264depay ! h264parse ! \
             nvv4l2decoder enable-max-performance=1 disable-dpb=true ! \
             nvvidconv ! {nvmm_caps} ! tee name=t \
             t. ! queue max-size-buffers=2 leaky=upstream ! \
                  appsink name=sink max-buffers=1 drop=true sync=false \
             t. ! queue max-size-buffers=2 leaky=upstream ! \
                  nvvidconv ! video/x-raw,format=RGBA ! \
                  appsink name=sink_cpu max-buffers=1 drop=true sync=false"
        );

        let pipeline = gstreamer::parse_launch(&pipeline_str)?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(|_| GstSourceError::Setup("pipeline cast failed"))?;

        let appsink = pipeline
            .by_name("sink").ok_or(GstSourceError::Setup("no appsink"))?
            .dynamic_cast::<gstreamer_app::AppSink>()
            .map_err(|_| GstSourceError::Setup("element is not AppSink"))?;

        let appsink_cpu = pipeline
            .by_name("sink_cpu").ok_or(GstSourceError::Setup("no sink_cpu"))?
            .dynamic_cast::<gstreamer_app::AppSink>()
            .map_err(|_| GstSourceError::Setup("sink_cpu is not AppSink"))?;

        let (frame_tx, frame_rx) = mpsc::sync_channel::<NvmmFrame>(2);
        let (dim_tx, dim_rx)     = mpsc::sync_channel::<(u32, u32)>(1);
        let dims_sent = Arc::new(AtomicBool::new(false));

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let buffer = sample.buffer()
                        .ok_or(gstreamer::FlowError::Error)?;
                    let caps = sample.caps()
                        .ok_or(gstreamer::FlowError::Error)?;
                    let st = caps.structure(0)
                        .ok_or(gstreamer::FlowError::Error)?;
                    let width = st.get::<i32>("width")
                        .map_err(|_| gstreamer::FlowError::Error)? as u32;
                    let height = st.get::<i32>("height")
                        .map_err(|_| gstreamer::FlowError::Error)? as u32;

                    let map  = buffer.map_readable()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let surf = map.as_slice().as_ptr() as *const c_void;

                    let fd     = unsafe { nvbuf_sys::nvbuf_dmabuf_fd(surf) };
                    let pitch  = unsafe { nvbuf_sys::nvbuf_pitch(surf) };
                    let size   = unsafe { nvbuf_sys::nvbuf_data_size(surf) };
                    let layout = unsafe { nvbuf_sys::nvbuf_layout(surf) };
                    drop(map);

                    if fd < 0 || pitch == 0 || size == 0 || layout != 0 {
                        return Err(gstreamer::FlowError::Error);
                    }

                    if !dims_sent.swap(true, Ordering::SeqCst) {
                        let _ = dim_tx.try_send((width, height));
                    }

                    let _ = frame_tx.try_send(NvmmFrame::new(sample, fd, pitch, size));
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        let cpu_frame: Arc<Mutex<Option<CpuFrame>>> = Arc::new(Mutex::new(None));
        let cpu_frame_cb = Arc::clone(&cpu_frame);

        appsink_cpu.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let buffer = sample.buffer()
                        .ok_or(gstreamer::FlowError::Error)?;
                    let caps = sample.caps()
                        .ok_or(gstreamer::FlowError::Error)?;
                    let st = caps.structure(0)
                        .ok_or(gstreamer::FlowError::Error)?;
                    let w = st.get::<i32>("width")
                        .map_err(|_| gstreamer::FlowError::Error)? as u32;
                    let h = st.get::<i32>("height")
                        .map_err(|_| gstreamer::FlowError::Error)? as u32;
                    let map = buffer.map_readable()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let data = map.as_slice().to_vec();
                    drop(map);
                    if let Ok(mut g) = cpu_frame_cb.lock() {
                        *g = Some((data, w, h));
                    }
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        pipeline.set_state(gstreamer::State::Playing)?;

        let (width, height) = dim_rx.recv()
            .map_err(|_| GstSourceError::NoFirstFrame)?;

        Ok(Self { pipeline, rx: frame_rx, width, height, cpu_frame })
    }

    pub fn width(&self)  -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }

    /// Returns a shared handle to the latest CPU RGBA snapshot.
    ///
    /// Clone the `Arc` before moving the source into a `Pipeline`.  The GStreamer
    /// thread updates this every frame (overwriting old data); lock and `take()` to
    /// consume without holding the lock during PNG save.
    pub fn latest_cpu_frame(&self) -> Arc<Mutex<Option<CpuFrame>>> {
        Arc::clone(&self.cpu_frame)
    }
}

impl Iterator for RtspSource {
    type Item = NvmmFrame;
    fn next(&mut self) -> Option<Self::Item> { self.rx.recv().ok() }
}

impl Source for RtspSource {
    type Frame = NvmmFrame;
    fn next_frame(&mut self) -> Option<NvmmFrame> { self.rx.recv().ok() }
}

impl Drop for RtspSource {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}

// ── NvmmPreprocessStage ───────────────────────────────────────────────────────

/// Pipeline stage: [`NvmmFrame`] → [`VrtTensor`] (CHW FP32).
///
/// Bridges the NVMM import lifecycle with the GPU letterbox kernel in
/// [`Preprocessor`].  The two are kept separate so `Preprocessor` has no
/// NVMM/GStreamer dependency.
///
/// ## Drop ordering across enqueue → sync → finalize
/// 1. `finalize` calls `preproc.finalize()` → drops `TextureGuard` (GPU texture object).
/// 2. Then drops `_pending` → drops `CudaMemory` (NVMM external memory handles).
///
/// This order is mandatory: the texture must be released before the device
/// memory it references is unmapped.
pub struct NvmmPreprocessStage {
    preproc:  Preprocessor,
    src_w:    u32,
    src_h:    u32,
    _pending: Option<CudaMemory>,
}

impl NvmmPreprocessStage {
    pub fn new(
        stream: Arc<vrt::CudaStream>,
        src_w: u32, src_h: u32,
        dst_w: u32, dst_h: u32,
    ) -> Result<Self, GstSourceError> {
        let preproc = Preprocessor::new(stream, src_w, src_h, dst_w, dst_h)?;
        Ok(Self { preproc, src_w, src_h, _pending: None })
    }
}

impl Operator for NvmmPreprocessStage {
    type Input   = NvmmFrame;
    type Pending = VrtTensor;   // borrowed view of the preprocessor's output
    type Output  = ();

    fn enqueue(&mut self, frame: &NvmmFrame, ctx: &ExecCtx) -> Result<VrtTensor, BoxError> {
        // A still-pending import means the previous frame never reached
        // finalize (error path).  Recover in the mandatory release order:
        // drain the GPU, drop the texture object, then the NVMM import.
        if self._pending.is_some() {
            self.preproc.stream().synchronize()?;
            self.preproc.release_pending();  // drops TextureGuard first ✓
            self._pending = None;             // then CudaMemory ✓
        }
        let mem = unsafe { frame.cuda_import()? };
        // SAFETY: the import's dev_ptr stays mapped while `mem` is held in
        // `_pending` (released only in finalize, after the stream sync).
        let image = unsafe {
            VrtImage::borrowed(
                mem.dev_ptr, self.src_w, self.src_h, frame.pitch,
                Format::Rgba8, MemKind::Imported,
            )
        };
        self._pending = Some(mem);
        self.preproc.enqueue(&image, ctx)
    }

    fn finalize(&mut self, pending: VrtTensor, ctx: &ExecCtx) -> Result<(), BoxError> {
        self.preproc.finalize(pending, ctx)?;  // drops TextureGuard first ✓
        self._pending = None;                    // then drops CudaMemory ✓
        Ok(())
    }
}
