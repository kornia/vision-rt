//! RF-DETR object detection: GPU stretch-resize → TRT backbone → **GPU decode**.
//!
//! [`RfDetr`] is a single `Image<u8,3> → Vec<Detection>` detector. RF-DETR is a
//! transformer set-predictor: a fixed set of query boxes + class logits, and it
//! is **NMS-free** (no duplicate suppression). The pipeline is zero-copy and
//! all-GPU: a stretch-resize kernel (kornia `Preprocessor` in `Stretch` mode +
//! ImageNet mean/std — this export has no baked-in normalization), the TRT
//! backbone, and a decode kernel; only the surviving detections reach the host.
//!
//! Model: the fixed-resolution official export (`rfdetr-small`) — input
//! `[1,3,512,512]`, outputs `pred_boxes [1,300,4]` (cxcywh, normalized) +
//! `pred_logits [1,300,91]` (logits; class 0 = background).
//!
//! Like the rest of the workspace the API is **fully async / caller-owned**
//! (VPI-style): allocate a [`DetectResult`], `submit` enqueues all GPU work with
//! **no sync**, the caller syncs the shared stream once, then reads.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kornia_image::Image;
use kornia_imgproc::preprocess::{Normalize, Preprocessor, PreprocessorBuilder, ResizeMode};
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::cuda::cfg_1d;
use vrt::{BoxError, Engine, ModelSession};

/// Errors from RF-DETR inference / decode.
#[derive(Debug, thiserror::Error)]
pub enum RfDetrError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("kornia CUDA: {0}")]
    Cuda(#[from] kornia_tensor::CudaError),
    #[error(transparent)]
    Preproc(#[from] kornia_imgproc::preprocess::PreprocessError),
    #[error("engine output '{0}' missing")]
    MissingOutput(String),
}

/// A detected object in original-image coordinate space.
#[derive(Debug, Clone)]
pub struct Detection {
    /// COCO category id (1–90); 0 (background) is never emitted.
    pub class_id: u32,
    pub score: f32,
    /// Bounding box `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
}

// One thread per query: argmax the class logits (sigmoid is monotonic, so argmax
// on the raw logit), skip background (class 0), threshold the sigmoid score, and
// — for survivors — atomically append the box (cxcywh normalized → xyxy in source
// pixels, since the stretch maps a normalized coord straight to the source).
const KERNEL_SRC: &str = r#"
extern "C" __global__ void rfdetr_decode(
    const float* __restrict__ boxes,    // [Q*4] cxcywh, normalized [0,1]
    const float* __restrict__ logits,   // [Q*C] raw logits
    int Q, int C, float conf,
    float src_w, float src_h,
    float* __restrict__ dets,           // [Q*6] out: x1,y1,x2,y2,class,score
    int*   __restrict__ count
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q) return;

    const float* lo = logits + (long)q * C;
    int   best_c = 0;
    float best_l = -1e30f;
    for (int c = 1; c < C; ++c) {        // class 0 = background, skip
        float l = lo[c];
        if (l > best_l) { best_l = l; best_c = c; }
    }
    float score = 1.0f / (1.0f + __expf(-best_l));
    if (best_c == 0 || score < conf) return;

    int slot = atomicAdd(count, 1);
    if (slot >= Q) return;

    const float* b = boxes + (long)q * 4;
    float cx = b[0] * src_w, cy = b[1] * src_h;
    float bw = b[2] * src_w, bh = b[3] * src_h;
    float* o = dets + (long)slot * 6;
    o[0] = cx - bw * 0.5f;
    o[1] = cy - bh * 0.5f;
    o[2] = cx + bw * 0.5f;
    o[3] = cy + bh * 0.5f;
    o[4] = (float)best_c;
    o[5] = score;
}
"#;

/// Caller-owned RF-DETR output (VPI-style), allocated once and reused.
///
/// [`RfDetr::submit`] writes the decoded detections here (async); after the
/// stream sync, [`detections`](Self::detections) downloads only the survivors.
pub struct DetectResult {
    dets: CudaSlice<f32>, // [q*6] x1,y1,x2,y2,class,score
    count_pin: vrt::PinnedBuffer<i32>,
    stream: Arc<CudaStream>, // the stream these buffers live on (for the readout D2H)
    q: usize,
}

impl DetectResult {
    /// Pre-allocate for `q` queries (use [`RfDetr::num_queries`]).
    pub fn alloc(stream: &Arc<CudaStream>, q: usize) -> Result<Self, RfDetrError> {
        Ok(Self {
            dets: unsafe { stream.alloc::<f32>(q * 6)? },
            count_pin: vrt::PinnedBuffer::<i32>::alloc(1)?,
            stream: stream.clone(),
            q,
        })
    }

    /// Number of surviving detections (reads the pinned scalar — call after sync).
    pub fn count(&self) -> usize {
        (self.count_pin.as_slice()[0].max(0) as usize).min(self.q)
    }

    /// Download the surviving detections to host (original-image pixels). Call
    /// after the stream sync that follows [`RfDetr::submit`].
    pub fn detections(&self) -> Result<Vec<Detection>, cudarc::driver::DriverError> {
        let n = self.count();
        if n == 0 {
            return Ok(Vec::new());
        }
        let flat = self.stream.clone_dtoh(&self.dets.slice(0..n * 6))?;
        Ok(flat
            .chunks_exact(6)
            .map(|d| Detection {
                class_id: d[4] as u32,
                score: d[5],
                bbox: [d[0], d[1], d[2], d[3]],
            })
            .collect())
    }
}

/// RF-DETR detector (payload): owns the backbone session, stretch preprocessor,
/// decode kernel, and the shared stream. Construct once, reuse every frame.
pub struct RfDetr {
    model: ModelSession,
    preproc: Preprocessor,
    stream: Arc<CudaStream>,
    input: Tensor<f32, 4>, // [1,3,mh,mw] CHW f32 device, reused
    decode: CudaKernel,
    conf: f32,
    q: usize, // query slots (fixed by engine)
    c: usize, // class-logit channels (fixed by engine)
    box_name: String,
    logit_name: String,
}

impl RfDetr {
    /// Build a detector sharing `stream` with the rest of the application.
    ///
    /// The input size is read from the engine's static `[1,3,H,W]` input, so any
    /// fixed-resolution RF-DETR export works. The two outputs are identified by
    /// **shape** (naming-agnostic): `[1,Q,4]` is boxes (cxcywh), `[1,Q,C]` the
    /// class logits — validated here once, so `submit` needs no per-frame checks.
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>, conf: f32) -> Result<Self, BoxError> {
        let inp = engine
            .inputs()
            .next()
            .ok_or("rfdetr: engine has no input")?;
        let d = &inp.dims;
        if d.len() != 4 || d.iter().any(|&x| x <= 0) {
            return Err(format!("rfdetr: input must be static [1,3,H,W], got {d:?}").into());
        }
        let (mh, mw) = (d[2] as usize, d[3] as usize);

        // Classify the two outputs by shape (a name-substring match would misfire
        // on an export whose class output happens to contain "box").
        let (mut box_name, mut logit_name, mut q, mut c) = (None, None, 0usize, 0usize);
        for s in engine.outputs() {
            match s.dims.as_slice() {
                [1, _, 4] => box_name = Some(s.name.clone()),
                [1, nq, nc] if *nc > 4 => {
                    logit_name = Some(s.name.clone());
                    q = *nq as usize;
                    c = *nc as usize;
                }
                _ => {}
            }
        }
        let box_name = box_name.ok_or("rfdetr: no boxes output [1,Q,4]")?;
        let logit_name = logit_name.ok_or("rfdetr: no logits output [1,Q,C]")?;
        if q == 0 {
            return Err("rfdetr: dynamic/unknown query count unsupported".into());
        }

        // RF-DETR: anisotropic stretch to the model size + ImageNet normalization
        // (this export has no baked-in mean/std — it starts straight with Conv).
        let preproc = PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build_cuda(stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, mh, mw], &stream)?;
        let decode = CudaKernel::compile(stream.context(), KERNEL_SRC, "rfdetr_decode")?;
        let model = ModelSession::new(engine, stream.clone())?;

        Ok(Self {
            model,
            preproc,
            stream,
            input,
            decode,
            conf,
            q,
            c,
            box_name,
            logit_name,
        })
    }

    /// Construct from a prebuilt `.engine` file (creates its own Logger/Runtime).
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        conf: f32,
    ) -> Result<Self, BoxError> {
        let logger = vrt::Logger::new(vrt::logger::Severity::Warning)?;
        let runtime = vrt::Runtime::new(logger)?;
        let engine = vrt::Engine::from_file(runtime, engine_path.as_ref())?;
        Self::new(engine, stream, conf)
    }

    /// Build (and cache) an engine from an ONNX file, then construct. First call
    /// builds on-device (cache hit thereafter, keyed by ONNX + TRT version + GPU
    /// arch). Requires feature `hub` (trtexec build) or `builder` (in-process).
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        conf: f32,
    ) -> Result<Self, BoxError> {
        // RF-DETR export is fixed-resolution (static shapes) → no shape profile.
        let profile = vrt_hub::EngineProfile {
            input: None,
            fp16: true,
            workspace_mb: 2048,
        };
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or("rfdetr: onnx path is not valid UTF-8")?;
        let engine_path =
            vrt_hub::EngineCache::default().resolve("rfdetr", model_path, &profile)?;
        Self::from_engine_file(engine_path, stream, conf)
    }

    /// Pull the pinned RF-DETR ONNX from Hugging Face (`kornia/rfdetr`), build/
    /// cache the engine on-device, and construct. Network is needed only on the
    /// first run; for a private/gated HF repo set `HF_TOKEN`. Requires feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>, conf: f32) -> Result<Self, BoxError> {
        // Prefer a prebuilt engine matching this box's TRT+SM (skips the on-device
        // build, which is slow for the transformer); otherwise pull the ONNX.
        if let Some(engine) = vrt_hub::ModelHub::get_engine("rfdetr")? {
            return Self::from_engine_file(engine, stream, conf);
        }
        let onnx = vrt_hub::ModelHub::get("rfdetr")?;
        Self::from_onnx(onnx, stream, conf)
    }

    /// Number of query slots (fixed by the engine) — the [`DetectResult`] capacity.
    pub fn num_queries(&self) -> usize {
        self.q
    }

    /// Allocate a reusable output for this detector.
    pub fn alloc_result(&self) -> Result<DetectResult, RfDetrError> {
        DetectResult::alloc(&self.stream, self.q)
    }

    /// Submit one frame's async GPU work — stretch-resize → backbone → decode —
    /// into the caller-owned `out`, with **no sync**. Sync the stream, then read
    /// `out.detections(...)`. `img` is a device image of any resolution; boxes
    /// come back in its pixel space.
    pub fn submit(
        &mut self,
        img: &Image<u8, 3>,
        out: &mut DetectResult,
    ) -> Result<(), RfDetrError> {
        self.preproc.run(img, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;

        // Outputs are identified + shape-validated in `new`; fetch by name and
        // trust the (static) shapes. f32_ptr() still guards against a non-F32
        // (e.g. fp16-output) binding.
        let b_raw = tmap
            .get(&self.box_name)
            .ok_or_else(|| RfDetrError::MissingOutput(self.box_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;
        let l_raw = tmap
            .get(&self.logit_name)
            .ok_or_else(|| RfDetrError::MissingOutput(self.logit_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;

        // Per-frame device count (atomic, zeroed); dets go into the caller's buffer.
        let count_dev: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let dets_raw = out.dets.device_ptr(self.stream.as_ref()).0;
        let cnt_raw = count_dev.device_ptr(self.stream.as_ref()).0;
        let (qi, ci) = (self.q as i32, self.c as i32);
        let conf = self.conf;
        let (sw, sh) = (img.width() as f32, img.height() as f32);
        self.decode
            .launch_builder(&self.stream)
            .arg(&b_raw)
            .arg(&l_raw)
            .arg(&qi)
            .arg(&ci)
            .arg(&conf)
            .arg(&sw)
            .arg(&sh)
            .arg(&dets_raw)
            .arg(&cnt_raw)
            .launch_cfg(cfg_1d(self.q, 256))?;

        // Async D2H of the count into the caller's pinned buffer (no sync).
        let cnt_pin = out.count_pin.as_mut_ptr();
        let vstream = vrt::Stream::from_cuda_stream(self.stream.clone());
        unsafe {
            vstream.memcpy_d2h_raw(
                cnt_pin as *mut u8,
                cnt_raw as usize as *const _,
                std::mem::size_of::<i32>(),
            )?;
        }
        Ok(())
    }
}
