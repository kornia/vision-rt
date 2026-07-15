//! OSNet **person re-identification** embeddings: GPU crop+resize+normalize → TRT →
//! L2-normed identity vectors.
//!
//! [`OsNetReid`] turns person detections into **metric-learned** appearance embeddings
//! for a tracker's identity decisions (re-id after occlusion, gallery resurrection
//! after leaving the scene). Unlike detector-backbone tokens — which are fine as a
//! soft, IoU-gated *tie-breaker* but collapse toward each other under occlusion —
//! OSNet is trained with a re-id metric objective (MSMT17 pedestrians), so
//! same-person-across-viewpoints stays closer than different-person: the property an
//! identity *decision* needs.
//!
//! Engine I/O (batch export): input `[B,3,256,128]` (ImageNet-normalized CHW crops),
//! output `[B,D]` embeddings (e.g. `B=16, D=512` for `osnet_x0_25_msmt17`).
//!
//! Caller-owned async (VPI-style), like the other model crates: [`submit`] enqueues
//! crop → TRT → GPU L2-normalize into a caller-owned [`ReidResult`] and returns
//! **without syncing** — the embeddings stay **on device**. The caller drains it with
//! its own `stream.synchronize()` (folding OSNet into whatever else is on the stream),
//! then reads the device slice ([`ReidResult::embeddings_slice`], cosine-ready) or
//! copies to host ([`ReidResult::embeddings_host`]). Typically run *after* the frame's
//! main readout, only when person boxes exist. ~1 ms for a few crops on an Orin Nano.
//!
//! [`submit`]: OsNetReid::submit

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaStream, DevicePtr};
use kornia_image::Image;
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::{BoxError, Engine, ModelSession};

/// Errors from re-id embedding extraction.
#[derive(Debug, thiserror::Error)]
pub enum ReidError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("kornia CUDA: {0}")]
    Cuda(#[from] kornia_tensor::CudaError),
    #[error("expected a device-resident image")]
    NotOnDevice,
    #[error("engine output '{0}' missing")]
    MissingOutput(String),
}

// One output pixel per thread, one crop per grid-z: bilinear-sample the box region of
// the HWC u8 source frame at 256×128, then ImageNet-normalize into the CHW f32 batch
// slot. The stretch (no aspect preservation) matches torchreid's Resize((256,128)).
const CROP_SRC: &str = r#"
extern "C" __global__ void crop_resize_norm(
    const unsigned char* __restrict__ img, int iw, int ih,  // HWC RGB source
    const float* __restrict__ boxes, int n,                 // [n*4] xyxy source px
    int cw, int chh,                                        // crop grid (128, 256)
    float* __restrict__ out                                 // [B,3,chh,cw] CHW
) {
    int b  = blockIdx.z;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= n || ox >= cw || oy >= chh) return;

    const float* bx = boxes + (long)b * 4;
    float x1 = bx[0], y1 = bx[1], x2 = bx[2], y2 = bx[3];
    float u = x1 + (x2 - x1) * ((ox + 0.5f) / cw)  - 0.5f;
    float v = y1 + (y2 - y1) * ((oy + 0.5f) / chh) - 0.5f;

    int u0 = (int)floorf(u), v0 = (int)floorf(v);
    float fu = u - u0, fv = v - v0;
    const float MEAN[3] = {0.485f, 0.456f, 0.406f};
    const float STD[3]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (int dy = 0; dy < 2; ++dy) {
            int yy = min(max(v0 + dy, 0), ih - 1);
            float wy = dy ? fv : 1.0f - fv;
            for (int dx = 0; dx < 2; ++dx) {
                int xx = min(max(u0 + dx, 0), iw - 1);
                float wx = dx ? fu : 1.0f - fu;
                acc += wy * wx * (float)img[((long)yy * iw + xx) * 3 + c];
            }
        }
        float val = (acc / 255.0f - MEAN[c]) / STD[c];
        out[(((long)b * 3 + c) * chh + oy) * cw + ox] = val;
    }
}
"#;

// One block per row: L2-normalize the TRT embedding output `[n, dim]` into the caller's
// device buffer, so the returned embeddings are cosine-ready on device (no host round
// trip). A zero row stays ~0 (rsqrtf of the eps floor → tiny scale on zero data).
const NORM_SRC: &str = r#"
extern "C" __global__ void l2norm_rows(
    const float* __restrict__ in, float* __restrict__ out, int n, int dim
) {
    int r = blockIdx.x;
    if (r >= n) return;
    const float* src = in + (long)r * dim;
    float* dst = out + (long)r * dim;
    extern __shared__ float ss[];
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = src[i]; local += v * v; }
    ss[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ss[threadIdx.x] += ss[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(ss[0] + 1e-24f); // (1e-12)^2 floor → matches the old host norm
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i] * inv;
}
"#;

/// OSNet person re-id embedder: crop kernel + TRT session + shared stream. Build once,
/// call [`submit`](Self::submit) per frame (only when person boxes exist).
pub struct OsNetReid {
    model: ModelSession,
    crop_k: CudaKernel,
    norm_k: CudaKernel,
    stream: Arc<CudaStream>,
    input: Tensor<f32, 4>, // [B,3,ch,cw] CHW f32 device, reused
    out_name: String,
    batch: usize,
    dim: usize,
    ch: usize,
    cw: usize,
}

/// Caller-owned re-id output (VPI-style): a device buffer of L2-normed embeddings,
/// allocated once via [`OsNetReid::alloc_result`] and reused every frame. [`OsNetReid::submit`]
/// fills it **async**; after the caller's `stream.synchronize()` read the rows via
/// [`embeddings_slice`](Self::embeddings_slice) (device) or [`embeddings_host`](Self::embeddings_host).
pub struct ReidResult {
    embeddings: cudarc::driver::CudaSlice<f32>, // [batch*dim] device, L2-normed, rows 0..n valid
    // Crop-kernel input boxes, kept alive until the caller syncs (the GPU reads this
    // device pointer during the caller's later sync — a local would free it too soon).
    boxes_dev: Option<cudarc::driver::CudaSlice<f32>>,
    n: usize,
    dim: usize,
    stream: Arc<CudaStream>,
}

impl ReidResult {
    /// Live embedding count this frame (`boxes.len().min(batch)` from the last submit).
    pub fn count(&self) -> usize {
        self.n
    }

    /// Embedding dimensionality (`D`).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// GPU-resident embeddings `[batch*dim]`, L2-normed, rows `0..count()` valid and
    /// aligned with the submitted box order. Stays on device (cosine-ready). Read after
    /// the caller's stream sync.
    pub fn embeddings_slice(&self) -> &cudarc::driver::CudaSlice<f32> {
        &self.embeddings
    }

    /// Copy the live embeddings to host, one L2-normed `D`-vector per box (submit order).
    /// Empty when no boxes were submitted. Call **after** the stream sync that follows
    /// [`OsNetReid::submit`].
    pub fn embeddings_host(&self) -> Result<Vec<Vec<f32>>, ReidError> {
        if self.n == 0 {
            return Ok(Vec::new());
        }
        let flat = self
            .stream
            .clone_dtoh(&self.embeddings.slice(0..self.n * self.dim))?;
        Ok(flat.chunks_exact(self.dim).map(<[f32]>::to_vec).collect())
    }
}

impl OsNetReid {
    /// Build from an engine whose input is `[B,3,H,W]` and output `[B,D]`.
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let inp = engine.inputs().next().ok_or("reid: engine has no input")?;
        let d = &inp.dims;
        if d.len() != 4 || d.iter().any(|&x| x <= 0) {
            return Err(format!("reid: input must be static [B,3,H,W], got {d:?}").into());
        }
        let (batch, ch, cw) = (d[0] as usize, d[2] as usize, d[3] as usize);
        let out = engine
            .outputs()
            .next()
            .ok_or("reid: engine has no output")?;
        let od = &out.dims;
        if od.len() != 2 || od[0] as usize != batch || od[1] <= 0 {
            return Err(format!("reid: output must be [B,D], got {od:?}").into());
        }
        let dim = od[1] as usize;
        let out_name = out.name.clone();

        let input = zeros_cuda::<f32, 4>([batch, 3, ch, cw], &stream)?;
        let crop_k = CudaKernel::compile(stream.context(), CROP_SRC, "crop_resize_norm")?;
        let norm_k = CudaKernel::compile(stream.context(), NORM_SRC, "l2norm_rows")?;
        let model = ModelSession::new(engine, stream.clone())?;
        Ok(Self {
            model,
            crop_k,
            norm_k,
            stream,
            input,
            out_name,
            batch,
            dim,
            ch,
            cw,
        })
    }

    /// Construct from a prebuilt `.engine` file.
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        Self::new(Engine::load(engine_path)?, stream)
    }

    /// The engine build profile — fixed batch/resolution (static shapes from the ONNX,
    /// `[16,3,256,128]`). FP16 off: the re-id embedding's cosine margin is small enough
    /// that FP16 rounding erodes it, so build at full precision (still ~1 ms for a few
    /// crops).
    #[cfg(any(feature = "hub", feature = "builder"))]
    fn engine_profile() -> vrt_hub::EngineProfile {
        vrt_hub::EngineProfile {
            input: None,
            fp16: false,
            workspace_mb: 1024,
        }
    }

    /// Build (and cache) an engine from an ONNX file, then construct. Requires feature
    /// `hub` (trtexec build) or `builder` (in-process).
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or("reid: onnx path is not valid UTF-8")?;
        let engine_path = vrt_hub::EngineCache::default().resolve(
            "osnet-reid",
            model_path,
            &Self::engine_profile(),
        )?;
        Self::from_engine_file(engine_path, stream)
    }

    /// Pull from Hugging Face (`kornia/osnet`) and construct — a matching prebuilt engine
    /// if the registry has one for this box, else the pinned ONNX built on-device.
    /// Requires feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let engine = vrt_hub::resolve_engine("osnet-reid", &Self::engine_profile())?;
        Self::from_engine_file(engine, stream)
    }

    /// Embedding dimensionality (`D`, e.g. 512).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Max crops per call (the engine's batch, e.g. 16); excess boxes are ignored.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Allocate a reusable [`ReidResult`] (device buffer for `batch` embeddings). Build
    /// once, pass to every [`submit`](Self::submit).
    pub fn alloc_result(&self) -> Result<ReidResult, ReidError> {
        Ok(ReidResult {
            embeddings: self.stream.alloc_zeros::<f32>(self.batch * self.dim)?,
            boxes_dev: None,
            n: 0,
            dim: self.dim,
            stream: self.stream.clone(),
        })
    }

    /// Enqueue one frame's re-id work — crop+normalize → TRT → GPU L2-normalize — for up
    /// to `batch` person boxes (`xyxy`, source pixels) on a device RGB frame, writing the
    /// L2-normed embeddings into `out` on device. **Async: does not sync** — the caller
    /// drains it with its own `stream.synchronize()`, then reads via
    /// [`ReidResult::embeddings_slice`] (device) or [`ReidResult::embeddings_host`].
    /// Rows are aligned with the submitted box order; excess boxes beyond `batch` are
    /// ignored. Keep `img` and `out` alive until the sync (the GPU reads their pointers).
    pub fn submit(
        &mut self,
        img: &Image<u8, 3>,
        boxes: &[[f32; 4]],
        out: &mut ReidResult,
    ) -> Result<(), ReidError> {
        let n = boxes.len().min(self.batch);
        out.n = n;
        if n == 0 {
            out.boxes_dev = None;
            return Ok(());
        }
        let src = img.as_cudaslice().ok_or(ReidError::NotOnDevice)?;
        let (iw, ih) = (img.width() as i32, img.height() as i32);
        let flat: Vec<f32> = boxes[..n].iter().flatten().copied().collect();
        let boxes_dev = self.stream.clone_htod(&flat)?;

        let (ni, cwi, chi) = (n as i32, self.cw as i32, self.ch as i32);
        let src_raw = src.device_ptr(self.stream.as_ref()).0;
        let in_raw = self
            .input
            .as_cudaslice()
            .expect("input tensor is device-resident (zeros_cuda)")
            .device_ptr(self.stream.as_ref())
            .0;
        let boxes_raw = boxes_dev.device_ptr(self.stream.as_ref()).0;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                (self.cw as u32).div_ceil(16),
                (self.ch as u32).div_ceil(16),
                n as u32,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        self.crop_k
            .launch_builder(&self.stream)
            .arg(&src_raw)
            .arg(&iw)
            .arg(&ih)
            .arg(&boxes_raw)
            .arg(&ni)
            .arg(&cwi)
            .arg(&chi)
            .arg(&in_raw)
            .launch_cfg(cfg)?;

        let tmap = self.model.run(&self.input)?;
        let emb_ptr = tmap
            .get(&self.out_name)
            .ok_or_else(|| ReidError::MissingOutput(self.out_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;

        // L2-normalize the TRT output rows into the caller's device buffer (async, one
        // block per row). Enqueued after `run`, so FIFO order guarantees it sees the
        // finished embeddings; the caller's later sync completes the whole chain.
        let out_raw = out.embeddings.device_ptr(self.stream.as_ref()).0;
        let (dimi, block) = (self.dim as i32, 256u32);
        let norm_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * std::mem::size_of::<f32>() as u32,
        };
        self.norm_k
            .launch_builder(&self.stream)
            .arg(&emb_ptr)
            .arg(&out_raw)
            .arg(&ni)
            .arg(&dimi)
            .launch_cfg(norm_cfg)?;

        // Keep the crop-input boxes alive until the caller syncs (freeing here would
        // pull the pointer out from under the still-pending crop kernel).
        out.boxes_dev = Some(boxes_dev);
        Ok(())
    }
}
