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
//! let zs = depth.sample_masks(&z, d.masks_slice(), d.mask_size(), d.count())?; // enqueue fusion
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
use vrt::cuda::{cfg_1d, cfg_per_item};
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

// One block per box: average the depth over the box's inner-50% central patch
// (avoids DPT depth bleeding across object edges). Box xyxy is in source pixels;
// the Stretch preprocess is full-frame, so source→map is a plain linear scale.
const BOX_SRC: &str = r#"
extern "C" __global__ void depth_box(
    const float* __restrict__ depth, int dmh, int dmw,
    const float* __restrict__ boxes, int stride, int n,
    float src_w, float src_h,
    float* __restrict__ out_z
) {
    int b = blockIdx.x;
    if (b >= n) return;
    const float* box = boxes + (long)b * stride;
    float sx = (float)dmw / src_w, sy = (float)dmh / src_h;
    float x1 = box[0] * sx, y1 = box[1] * sy, x2 = box[2] * sx, y2 = box[3] * sy;
    float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
    float hw = (x2 - x1) * 0.25f, hh = (y2 - y1) * 0.25f;  // inner 50%
    int ix0 = max(0, (int)floorf(cx - hw)), ix1 = min(dmw - 1, (int)ceilf(cx + hw));
    int iy0 = max(0, (int)floorf(cy - hh)), iy1 = min(dmh - 1, (int)ceilf(cy + hh));
    int W = ix1 - ix0 + 1, H = iy1 - iy0 + 1, N = (W > 0 && H > 0) ? W * H : 0;

    float sum = 0.0f; int cnt = 0;
    for (int p = threadIdx.x; p < N; p += blockDim.x) {
        int x = ix0 + p % W, y = iy0 + p / W;
        float d = depth[(long)y * dmw + x];
        if (d > 0.0f) { sum += d; ++cnt; }
    }
    __shared__ float ssum[256]; __shared__ int scnt[256];
    int t = threadIdx.x;
    ssum[t] = sum; scnt[t] = cnt; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { ssum[t] += ssum[t + s]; scnt[t] += scnt[t + s]; }
        __syncthreads();
    }
    if (t == 0) out_z[b] = scnt[0] > 0 ? ssum[0] / scnt[0] : 0.0f;
}
"#;

// One block per instance: average the depth over the instance mask's foreground
// pixels. Mask grid (mmh,mmw) and depth grid (dmh,dmw) both span the full frame,
// so a mask pixel maps to a depth pixel by a plain scale. `masks` is packed per
// surviving slot ([count*mmh*mmw], 1 = foreground).
const MASK_SRC: &str = r#"
extern "C" __global__ void depth_mask(
    const float* __restrict__ depth, int dmh, int dmw,
    const unsigned char* __restrict__ masks, int mmh, int mmw,
    int count,
    float* __restrict__ out_z
) {
    int m = blockIdx.x;
    if (m >= count) return;
    const unsigned char* mask = masks + (long)m * mmh * mmw;
    int L = mmh * mmw;
    float sx = (float)dmw / mmw, sy = (float)dmh / mmh;

    float sum = 0.0f; int cnt = 0;
    for (int p = threadIdx.x; p < L; p += blockDim.x) {
        if (mask[p] != 0) {
            int mx = p % mmw, my = p / mmw;
            int dx = min(dmw - 1, (int)(mx * sx)), dy = min(dmh - 1, (int)(my * sy));
            float d = depth[(long)dy * dmw + dx];
            if (d > 0.0f) { sum += d; ++cnt; }
        }
    }
    __shared__ float ssum[256]; __shared__ int scnt[256];
    int t = threadIdx.x;
    ssum[t] = sum; scnt[t] = cnt; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { ssum[t] += ssum[t + s]; scnt[t] += scnt[t + s]; }
        __syncthreads();
    }
    if (t == 0) out_z[m] = scnt[0] > 0 ? ssum[0] / scnt[0] : 0.0f;
}
"#;

/// Caller-owned depth output (VPI-style): a GPU-resident metric depth map, filled
/// async by [`DepthAnything::submit`]. Copy to host on request ([`depth_host`]) or
/// hand the device slice to the fusion kernels / downstream GPU work.
///
/// [`depth_host`]: DepthResult::depth_host
pub struct DepthResult {
    depth: CudaSlice<f32>, // [mh*mw] metric meters, row-major
    stream: Arc<CudaStream>,
    mh: usize,
    mw: usize,
    src: (f32, f32), // original-image (w, h), stamped by submit
}

impl DepthResult {
    fn alloc(stream: &Arc<CudaStream>, mh: usize, mw: usize) -> Result<Self, DepthError> {
        Ok(Self {
            depth: unsafe { stream.alloc::<f32>(mh * mw)? },
            stream: stream.clone(),
            mh,
            mw,
            src: (0.0, 0.0),
        })
    }

    /// Depth-map grid `(width, height)`.
    pub fn map_size(&self) -> (usize, usize) {
        (self.mw, self.mh)
    }

    /// GPU-resident depth map `[mh*mw]` (metric meters) — for the fusion kernels /
    /// downstream on-device work. Valid after the stream sync.
    pub fn depth_slice(&self) -> &CudaSlice<f32> {
        &self.depth
    }

    /// Download the dense metric depth map to a host [`DepthImage`] (meters). Call
    /// after the stream sync that follows [`DepthAnything::submit`].
    pub fn depth_host(&self) -> Result<DepthImage, DepthError> {
        let v = self.stream.clone_dtoh(&self.depth)?;
        Ok(DepthImage::from_size_vec(
            ImageSize {
                width: self.mw,
                height: self.mh,
            },
            v,
        )?)
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
    box_k: CudaKernel,
    mask_k: CudaKernel,
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
        let (mut out_name, mut mh, mut mw) = (None, 0usize, 0usize);
        for s in engine.outputs() {
            match s.dims.as_slice() {
                [1, 1, nh, nw] | [1, nh, nw] if *nh > 0 && *nw > 0 => {
                    out_name = Some(s.name.clone());
                    (mh, mw) = (*nh as usize, *nw as usize);
                }
                _ => {}
            }
        }
        let out_name = out_name.ok_or("depth: no dense depth output [1,1,H,W]/[1,H,W]")?;
        if mh == 0 || mw == 0 {
            return Err("depth: dynamic/unknown output dims unsupported".into());
        }

        // Stretch + ImageNet normalization (DA2 expects ImageNet mean/std).
        let preproc = PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build_cuda(stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, ih, iw], &stream)?;
        let copy_k = CudaKernel::compile(stream.context(), COPY_SRC, "depth_copy")?;
        let box_k = CudaKernel::compile(stream.context(), BOX_SRC, "depth_box")?;
        let mask_k = CudaKernel::compile(stream.context(), MASK_SRC, "depth_mask")?;
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
            box_k,
            mask_k,
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
        let dst_raw = out.depth.device_ptr(self.stream.as_ref()).0;
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

    /// Sample per-**box** metric depth (mean of the box's inner-50% central patch).
    /// `boxes` is a GPU `[n*stride]` buffer with `x1,y1,x2,y2` (source pixels) in the
    /// first 4 lanes — e.g. a detector's `dets_slice()` (`stride=6`). Enqueues the
    /// kernel on the shared stream; returns a GPU `[n]` z buffer (meters), valid
    /// after the single sync — read it with `stream.clone_dtoh`.
    pub fn sample_boxes(
        &self,
        out: &DepthResult,
        boxes: &CudaSlice<f32>,
        stride: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, DepthError> {
        let z: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(n)?;
        if n > 0 {
            let depth_raw = out.depth.device_ptr(self.stream.as_ref()).0;
            let boxes_raw = boxes.device_ptr(self.stream.as_ref()).0;
            let z_raw = z.device_ptr(self.stream.as_ref()).0;
            let (dmh, dmw) = (out.mh as i32, out.mw as i32);
            let (st, ni) = (stride as i32, n as i32);
            let (sw, sh) = out.src;
            self.box_k
                .launch_builder(&self.stream)
                .arg(&depth_raw)
                .arg(&dmh)
                .arg(&dmw)
                .arg(&boxes_raw)
                .arg(&st)
                .arg(&ni)
                .arg(&sw)
                .arg(&sh)
                .arg(&z_raw)
                .launch_cfg(cfg_per_item(n, 256))?;
        }
        Ok(z)
    }

    /// Sample per-**instance** metric depth over each instance mask's foreground
    /// pixels (isolates the object — no background bleed). `masks` is a GPU
    /// `[q*mmh*mmw]` u8 buffer packed per slot (a detector's `masks_slice()`),
    /// `mask_wh = (mmw, mmh)`, `count` valid instances. Enqueues on the shared
    /// stream; returns a GPU `[count]` z buffer (meters), valid after the single sync.
    pub fn sample_masks(
        &self,
        out: &DepthResult,
        masks: &CudaSlice<u8>,
        mask_wh: (usize, usize),
        count: usize,
    ) -> Result<CudaSlice<f32>, DepthError> {
        let z: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(count)?;
        if count > 0 {
            let (mmw, mmh) = mask_wh;
            let depth_raw = out.depth.device_ptr(self.stream.as_ref()).0;
            let masks_raw = masks.device_ptr(self.stream.as_ref()).0;
            let z_raw = z.device_ptr(self.stream.as_ref()).0;
            let (dmh, dmw) = (out.mh as i32, out.mw as i32);
            let (mmhi, mmwi, cnt) = (mmh as i32, mmw as i32, count as i32);
            self.mask_k
                .launch_builder(&self.stream)
                .arg(&depth_raw)
                .arg(&dmh)
                .arg(&dmw)
                .arg(&masks_raw)
                .arg(&mmhi)
                .arg(&mmwi)
                .arg(&cnt)
                .arg(&z_raw)
                .launch_cfg(cfg_per_item(count, 256))?;
        }
        Ok(z)
    }
}
