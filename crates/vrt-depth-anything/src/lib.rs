//! Depth Anything V2 **metric** monocular depth: GPU stretch+ImageNet-normalize →
//! TRT → dense metric depth map, plus GPU **depth-at-box / depth-at-mask** fusion.
//!
//! [`DepthAnything`] is an `Image<u8,3> → DepthImage` (dense metric depth in meters)
//! estimator on the Depth Anything V2 Metric-Small export. Everything stays on the
//! **GPU** and the pipeline is fully **async / caller-owned** (VPI-style), mirroring
//! the detector crates: `submit` enqueues stretch+normalize → TRT → a copy of the
//! depth map into the caller-owned [`DepthResult`], with **no sync and no host
//! copy**; the caller syncs the shared stream once, then pulls what it needs.
//!
//! # Composing with a detector (the parallel detect+depth pattern)
//!
//! Build the detector and `DepthAnything` on **one shared stream**; per frame,
//! `submit` both from the **same** device image (each only enqueues), sample
//! per-detection depth, then a **single** `stream.synchronize()` drains everything:
//!
//! ```ignore
//! det.submit(&img, &mut d)?;                                   // enqueue, no sync
//! depth.submit(&img, &mut z)?;                                 // same stream, no sync
//! let zs = z.depth_image()
//!     .sample_masks(d.masks_slice(), d.mask_size(), d.count_slice(), &stream)?; // enqueue fusion
//! stream.synchronize()?;                                        // ONE sync drains all
//! let zs = stream.clone_dtoh(&zs)?;                             // per-instance metric z
//! ```
//!
//! Sampling from the instance **mask** (not the box) isolates the object — the box
//! bleeds background depth. Feed the sampled `z` to a tracker's `Detection::depth`.
//!
//! Engine I/O: input `[1,3,H,W]` (Stretch + ImageNet norm); output `depth [1,1,H,W]`
//! (or `[1,H,W]`) f32 **metric meters**.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::preprocess::{Normalize, Preprocessor, PreprocessorBuilder, ResizeMode};
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::cuda::cfg_1d;
use vrt::{BoxError, Engine, ModelSession};
use vrt_types::DepthImage;

/// Errors from depth inference / fusion.
#[derive(Debug, thiserror::Error)]
pub enum DepthError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("kornia CUDA: {0}")]
    Cuda(#[from] kornia_tensor::CudaError),
    #[error(transparent)]
    Preproc(#[from] kornia_imgproc::preprocess::PreprocessError),
    #[error(transparent)]
    Image(#[from] kornia_image::ImageError),
    #[error("engine output '{0}' missing")]
    MissingOutput(String),
}

// Copy the TRT depth output into the caller-owned buffer (the output view aliases
// session memory reused on the next run — we must own a stable copy).
const COPY_SRC: &str = r#"
extern "C" __global__ void depth_copy(const float* __restrict__ src, int n, float* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
"#;

/// Caller-owned depth output (VPI-style): a GPU-resident metric depth map, filled
/// async by [`DepthAnything::submit`]. Copy to host on request ([`depth_host`]) or
/// hand the device slice to the fusion kernels / downstream GPU work.
///
/// [`depth_host`]: DepthResult::depth_host
pub struct DepthResult {
    depth: DepthImage, // device-resident [mh*mw] metric meters, row-major
    stream: Arc<CudaStream>,
    src: (f32, f32), // original-image (w, h), stamped by submit
}

impl DepthResult {
    fn alloc(stream: &Arc<CudaStream>, mh: usize, mw: usize) -> Result<Self, DepthError> {
        Ok(Self {
            depth: DepthImage::zeros_cuda(
                ImageSize {
                    width: mw,
                    height: mh,
                },
                stream,
            )?,
            stream: stream.clone(),
            src: (0.0, 0.0),
        })
    }

    /// Depth-map grid `(width, height)`.
    pub fn map_size(&self) -> (usize, usize) {
        let s = self.depth.size();
        (s.width, s.height)
    }

    /// Original-image `(width, height)` stamped by the last [`DepthAnything::submit`]
    /// — pass as `src_wh` to [`DepthImage::sample_boxes`], whose box coords are in
    /// source pixels.
    pub fn src_wh(&self) -> (f32, f32) {
        self.src
    }

    /// GPU-resident dense metric depth map — pass to the [`DepthImage`] sampling
    /// builtins (`sample_masks` / `sample_boxes`) or other downstream GPU work.
    /// Valid after the stream sync that follows [`DepthAnything::submit`].
    pub fn depth_image(&self) -> &DepthImage {
        &self.depth
    }

    /// GPU-resident depth map `[mh*mw]` (metric meters) as a raw device slice.
    /// Valid after the stream sync.
    pub fn depth_slice(&self) -> &CudaSlice<f32> {
        self.depth
            .as_cudaslice()
            .expect("DepthResult depth map is device-resident")
    }

    /// Download the dense metric depth map to a host [`DepthImage`] (meters). Call
    /// after the stream sync that follows [`DepthAnything::submit`].
    pub fn depth_host(&self) -> Result<DepthImage, DepthError> {
        Ok(self.depth.to_host(&self.stream)?)
    }
}

/// Depth Anything V2 metric depth estimator (payload): backbone session +
/// stretch/ImageNet preprocessor + fusion kernels + shared stream. Build once,
/// reuse every frame.
pub struct DepthAnything {
    model: ModelSession,
    preproc: Preprocessor,
    stream: Arc<CudaStream>,
    input: Tensor<f32, 4>, // [1,3,ih,iw] CHW f32 device, reused
    mh: usize,
    mw: usize,
    out_name: String,
    copy_k: CudaKernel,
}

impl DepthAnything {
    /// Build a depth estimator sharing `stream`. Input size read from the engine's
    /// static `[1,3,H,W]`; the single dense depth output is identified by shape
    /// (`[1,1,H,W]` or `[1,H,W]`).
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let inp = engine.inputs().next().ok_or("depth: engine has no input")?;
        let d = &inp.dims;
        if d.len() != 4 || d.iter().any(|&x| x <= 0) {
            return Err(format!("depth: input must be static [1,3,H,W], got {d:?}").into());
        }
        let (ih, iw) = (d[2] as usize, d[3] as usize);

        // The dense depth map: rank-4 [1,1,H,W] or rank-3 [1,H,W], positive dims.
        // Reject ambiguity: a permissive rank-3 match would also bind an auxiliary
        // output (e.g. a [1,Q,4] boxes head), so require *exactly one* dense-map
        // output and error out rather than silently pick the last one.
        let mut matched: Option<(String, usize, usize)> = None;
        for s in engine.outputs() {
            if let [1, 1, nh, nw] | [1, nh, nw] = s.dims.as_slice() {
                if *nh > 0 && *nw > 0 {
                    if matched.is_some() {
                        return Err("depth: engine exposes multiple dense-map outputs; \
                             cannot identify the depth output by shape"
                            .into());
                    }
                    matched = Some((s.name.clone(), *nh as usize, *nw as usize));
                }
            }
        }
        let (out_name, mh, mw) = matched.ok_or("depth: no dense depth output [1,1,H,W]/[1,H,W]")?;

        // Stretch + ImageNet normalization (DA2 expects ImageNet mean/std).
        let preproc = PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build_cuda(stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, ih, iw], &stream)?;
        let copy_k = CudaKernel::compile(stream.context(), COPY_SRC, "depth_copy")?;
        let model = ModelSession::new(engine, stream.clone())?;

        Ok(Self {
            model,
            preproc,
            stream,
            input,
            mh,
            mw,
            out_name,
            copy_k,
        })
    }

    /// Construct from a prebuilt `.engine` file.
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        Self::new(Engine::load(engine_path)?, stream)
    }

    /// The engine build profile — fixed-resolution (static shapes).
    #[cfg(any(feature = "hub", feature = "builder"))]
    fn engine_profile() -> vrt_hub::EngineProfile {
        vrt_hub::EngineProfile {
            input: None,
            fp16: true,
            workspace_mb: 2048,
        }
    }

    /// Build (and cache) an engine from an ONNX file, then construct. Requires
    /// feature `hub` (trtexec build) or `builder` (in-process).
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or("depth: onnx path is not valid UTF-8")?;
        let engine_path = vrt_hub::EngineCache::default().resolve(
            "depth-anything-v2-metric-small",
            model_path,
            &Self::engine_profile(),
        )?;
        Self::from_engine_file(engine_path, stream)
    }

    /// Pull from Hugging Face (`kornia/depth-anything`) and construct. Requires
    /// feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let engine =
            vrt_hub::resolve_engine("depth-anything-v2-metric-small", &Self::engine_profile())?;
        Self::from_engine_file(engine, stream)
    }

    /// Depth-map grid `(width, height)` the engine emits.
    pub fn map_size(&self) -> (usize, usize) {
        (self.mw, self.mh)
    }

    /// Allocate a reusable output for this estimator.
    pub fn alloc_result(&self) -> Result<DepthResult, DepthError> {
        DepthResult::alloc(&self.stream, self.mh, self.mw)
    }

    /// Submit one frame — stretch+normalize → backbone → copy the dense metric
    /// depth map into `out`, with **no sync and no host copy**. Sync the stream,
    /// then read `out.depth_host()` or use `out.depth_slice()` / the fusion kernels.
    pub fn submit(&mut self, img: &Image<u8, 3>, out: &mut DepthResult) -> Result<(), DepthError> {
        self.preproc.run(img, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;
        let src_raw = tmap
            .get(&self.out_name)
            .ok_or_else(|| DepthError::MissingOutput(self.out_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;
        let dst_raw = out.depth_slice().device_ptr(self.stream.as_ref()).0;
        let n = (self.mh * self.mw) as i32;
        self.copy_k
            .launch_builder(&self.stream)
            .arg(&src_raw)
            .arg(&n)
            .arg(&dst_raw)
            .launch_cfg(cfg_1d(self.mh * self.mw, 256))?;
        out.src = (img.width() as f32, img.height() as f32);
        Ok(())
    }
}
