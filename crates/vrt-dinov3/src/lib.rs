//! DINOv3 **global image descriptors**: GPU stretch+ImageNet-normalize → TRT →
//! one L2-normed vector per frame, plus a GPU cosine [`DescriptorBank`] for retrieval.
//!
//! [`DinoV3`] is an `Image<u8,3> → [f32; D]` whole-frame embedder on a DINOv3 ViT-S/16
//! export. The descriptor is the model's own **CLS token** (HF `pooler_output`) — the
//! representation DINOv3 pools into by default — L2-normalized on device so every
//! downstream cosine similarity is a plain dot product. Useful for retrieval, visual
//! place recognition, relocalization candidates and scene-change detection.
//!
//! Everything stays on the **GPU** and the pipeline is fully **async / caller-owned**
//! (VPI-style), mirroring the other model crates: `submit` enqueues stretch+normalize →
//! TRT → L2-norm into the caller-owned [`DinoV3Result`], with **no sync and no host
//! copy**; the caller syncs the shared stream once, then pulls what it needs.
//!
//! # Retrieval against a keyframe bank
//!
//! ```no_run
//! use std::sync::Arc;
//! use cudarc::driver::CudaStream;
//! use kornia_image::Image;
//! use vrt_dinov3::{DescriptorBank, DinoV3};
//!
//! fn frame(
//!     dino: &mut DinoV3,
//!     bank: &DescriptorBank,
//!     img: &Image<u8, 3>,
//!     stream: &Arc<CudaStream>,
//! ) -> Result<Vec<f32>, vrt::BoxError> {
//!     // In a real loop both of these are allocated ONCE, outside it.
//!     let mut r = dino.alloc_result()?;
//!     let mut scores = stream.alloc_zeros::<f32>(bank.capacity())?;
//!
//!     dino.submit(img, &mut r)?;                            // enqueue, no sync
//!     bank.match_into(r.descriptor_slice(), &mut scores)?;   // enqueue, no sync
//!     stream.synchronize()?;                                // ONE sync drains both
//!     Ok(stream.clone_dtoh(&scores.slice(0..bank.len()))?)   // tiny D2H, post-sync
//! }
//! ```
//!
//! Both operands are unit-norm, so the bank's "cosine" kernel is literally a dot
//! product — no division, no normalization on the query side.
//!
//! # Engine I/O
//!
//! Input `[1,3,S,S]` with `S % 16 == 0` (Stretch + ImageNet norm). Outputs, bound **by
//! shape**: `[1,D]` → the descriptor (required); `[1,N,D]` → the full token sequence
//! (optional). When the token output is present the crate copies out the **patch grid
//! only** — the leading CLS + register tokens are skipped, so `patch_tokens_slice()`
//! never contains DINOv3's 4 register artifacts. The prefix length is derived from
//! `N - (S/16)^2` rather than hardcoded, so register-free variants work unchanged.
//!
//! Binding the token output costs an output buffer and **zero FLOPs** — the ViT computes
//! every patch token regardless; only the binding differs.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig};
use kornia_image::Image;
use kornia_imgproc::preprocess::{Normalize, Preprocessor, PreprocessorBuilder, ResizeMode};
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::cuda::cfg_1d;
use vrt::{BoxError, Engine, ModelSession};

/// ViT patch size for the DINOv3 `/16` family — the input side must be a multiple.
const PATCH: usize = 16;

/// Reduction block width for the row kernels. Must stay a power of two: the shared-mem
/// tree reduction in `l2norm_rows` / `cosine_bank` halves the stride each step.
const RED_BLOCK: u32 = 256;

/// Errors from descriptor extraction / retrieval.
#[derive(Debug, thiserror::Error)]
pub enum DinoError {
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
    #[error("descriptor dim {got} does not match the bank's {want}")]
    DimMismatch { got: usize, want: usize },
    #[error("scores buffer holds {got} slots, need {want}")]
    ScoresTooSmall { got: usize, want: usize },
    #[error("bank is full ({capacity} descriptors)")]
    BankFull { capacity: usize },
}

// One block per row: L2-normalize `[n, dim]` into the caller's device buffer, so the
// descriptor is cosine-ready on device (no host round trip). This is *also* the
// mandatory copy-out — the TRT output view aliases session memory that the next `run`
// overwrites. A zero row stays ~0 (rsqrtf of the eps floor → tiny scale on zero data).
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
    float inv = rsqrtf(ss[0] + 1e-24f); // (1e-12)^2 floor → matches a host-side norm
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i] * inv;
}
"#;

// Copy the patch-token block out of the TRT view into the caller-owned buffer (same
// aliasing reason as above). The caller offsets `src` past the CLS + register prefix.
const COPY_SRC: &str = r#"
extern "C" __global__ void token_copy(const float* __restrict__ src, int n, float* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
"#;

// One block per bank row: dot the query against every stored descriptor. Both operands
// are unit-norm, so the dot IS the cosine — no division, no re-normalization.
const MATCH_SRC: &str = r#"
extern "C" __global__ void cosine_bank(
    const float* __restrict__ bank, const float* __restrict__ q,
    float* __restrict__ scores, int n, int dim
) {
    int r = blockIdx.x;
    if (r >= n) return;
    const float* row = bank + (long)r * dim;
    extern __shared__ float ss[];
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) local += row[i] * q[i];
    ss[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ss[threadIdx.x] += ss[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[r] = ss[0];
}
"#;

/// Engine I/O resolved by shape — the pure part of construction, so it is unit-testable
/// without a GPU or a real engine (see the tests at the bottom of this file).
#[derive(Debug, PartialEq, Eq)]
struct EngineIo {
    ih: usize,
    iw: usize,
    gh: usize,
    gw: usize,
    dim: usize,
    desc_name: String,
    /// `None` when the export omits the token sequence (descriptor-only engine).
    tok_name: Option<String>,
    /// Leading non-patch tokens (CLS + registers); 5 for stock DINOv3.
    prefix: usize,
}

/// Classify a DINOv3 engine's tensors **by shape, not name** — export tools rename
/// tensors freely, but the ranks here are unambiguous: `[1,D]` is the pooled descriptor
/// and `[1,N,D]` is the token sequence.
///
/// Positive-dim guards matter: a dynamic axis reports `-1`, which would wrap to a huge
/// `usize` if cast blindly.
fn resolve_io(inp: &[i64], outs: &[(String, Vec<i64>)]) -> Result<EngineIo, String> {
    if inp.len() != 4 || inp.iter().any(|&x| x <= 0) {
        return Err(format!("dinov3: input must be static [1,3,S,S], got {inp:?}"));
    }
    let (ih, iw) = (inp[2] as usize, inp[3] as usize);
    if !ih.is_multiple_of(PATCH) || !iw.is_multiple_of(PATCH) {
        return Err(format!(
            "dinov3: input dims must be multiples of {PATCH} (ViT/16 patch grid), got {ih}x{iw}"
        ));
    }
    let (gh, gw) = (ih / PATCH, iw / PATCH);

    // Reject ambiguity rather than silently picking the last match, the way the sibling
    // crates do — a second output of either rank means the export is not what we expect.
    let mut desc: Option<(String, usize)> = None;
    let mut toks: Option<(String, usize, usize)> = None;
    for (name, dims) in outs {
        match dims.as_slice() {
            [1, d] if *d > 0 => {
                if desc.is_some() {
                    return Err("dinov3: engine exposes multiple [1,D] outputs; \
                         cannot identify the descriptor by shape"
                        .into());
                }
                desc = Some((name.clone(), *d as usize));
            }
            [1, n, d] if *n > 0 && *d > 0 => {
                if toks.is_some() {
                    return Err("dinov3: engine exposes multiple [1,N,D] outputs; \
                         cannot identify the token sequence by shape"
                        .into());
                }
                toks = Some((name.clone(), *n as usize, *d as usize));
            }
            _ => {}
        }
    }
    let (desc_name, dim) = desc.ok_or(
        "dinov3: no [1,D] descriptor output — export `pooler_output` (the CLS token)",
    )?;

    // The prefix is DERIVED (N - patches), not hardcoded to DINOv3's 5, so a variant
    // with a different register count needs no code change.
    let (tok_name, prefix) = match toks {
        None => (None, 0),
        Some((name, ntok, tdim)) => {
            if tdim != dim {
                return Err(format!(
                    "dinov3: token output dim {tdim} != descriptor dim {dim}"
                ));
            }
            let n_patch = gh * gw;
            if ntok < n_patch {
                return Err(format!(
                    "dinov3: token output has {ntok} tokens, fewer than the {n_patch}-patch \
                     grid implied by the {ih}x{iw} input"
                ));
            }
            (Some(name), ntok - n_patch)
        }
    };

    Ok(EngineIo {
        ih,
        iw,
        gh,
        gw,
        dim,
        desc_name,
        tok_name,
        prefix,
    })
}

/// Caller-owned descriptor output (VPI-style): a GPU-resident L2-normed embedding,
/// filled async by [`DinoV3::submit`]. Allocate once via [`DinoV3::alloc_result`] and
/// reuse every frame.
pub struct DinoV3Result {
    descriptor: CudaSlice<f32>, // [dim] device, L2-normed — the headline output
    patches: Option<CudaSlice<f32>>, // [gh*gw*dim] device, raw (not normed), registers stripped
    stream: Arc<CudaStream>,
    dim: usize,
    gh: usize,
    gw: usize,
}

impl DinoV3Result {
    /// Descriptor dimensionality (`D`, 384 for ViT-S/16).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Patch grid `(width, height)` — `(S/16, S/16)`.
    pub fn grid(&self) -> (usize, usize) {
        (self.gw, self.gh)
    }

    /// GPU-resident L2-normed global descriptor `[dim]`. Cosine-ready: dot it against
    /// another unit descriptor. Valid after the caller's stream sync.
    pub fn descriptor_slice(&self) -> &CudaSlice<f32> {
        &self.descriptor
    }

    /// GPU-resident dense patch tokens `[gh*gw, dim]`, row-major, **registers stripped**
    /// — `None` when the engine does not expose the token output. Raw (not normalized);
    /// normalize downstream if you need cosine over patches. Valid after the sync.
    pub fn patch_tokens_slice(&self) -> Option<&CudaSlice<f32>> {
        self.patches.as_ref()
    }

    /// Copy the global descriptor to host. Call **after** the stream sync that follows
    /// [`DinoV3::submit`].
    pub fn descriptor_host(&self) -> Result<Vec<f32>, DinoError> {
        Ok(self.stream.clone_dtoh(&self.descriptor)?)
    }

    /// Copy the patch tokens to host, one `dim`-vector per patch in row-major grid order.
    /// `None` when the engine does not expose them. Call **after** the stream sync.
    pub fn patch_tokens_host(&self) -> Option<Result<Vec<Vec<f32>>, DinoError>> {
        let p = self.patches.as_ref()?;
        Some(
            self.stream
                .clone_dtoh(p)
                .map(|flat| flat.chunks_exact(self.dim).map(<[f32]>::to_vec).collect())
                .map_err(DinoError::from),
        )
    }
}

/// DINOv3 global descriptor extractor (payload): backbone session + stretch/ImageNet
/// preprocessor + post-process kernels + shared stream. Build once, reuse every frame.
pub struct DinoV3 {
    model: ModelSession,
    preproc: Preprocessor,
    stream: Arc<CudaStream>,
    input: Tensor<f32, 4>, // [1,3,ih,iw] CHW f32 device, reused
    desc_name: String,
    tok_name: Option<String>,
    dim: usize,
    gh: usize,
    gw: usize,
    prefix: usize,
    norm_k: CudaKernel,
    copy_k: CudaKernel,
}

impl DinoV3 {
    /// Build a descriptor extractor sharing `stream`. Input size is read from the
    /// engine's static `[1,3,S,S]`; outputs are identified by shape (see [`resolve_io`]).
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let inp = engine.inputs().next().ok_or("dinov3: engine has no input")?;
        let outs: Vec<(String, Vec<i64>)> = engine
            .outputs()
            .map(|s| (s.name.clone(), s.dims.clone()))
            .collect();
        let io = resolve_io(&inp.dims, &outs)?;

        // Stretch + ImageNet normalization — DINOv3 uses ImageNet mean/std, and the
        // full-frame stretch keeps cross-model coordinate mapping a plain grid/src ratio.
        let preproc = PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build_cuda(stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, io.ih, io.iw], &stream)?;
        let norm_k = CudaKernel::compile(stream.context(), NORM_SRC, "l2norm_rows")?;
        let copy_k = CudaKernel::compile(stream.context(), COPY_SRC, "token_copy")?;
        let model = ModelSession::new(engine, stream.clone())?;

        Ok(Self {
            model,
            preproc,
            stream,
            input,
            desc_name: io.desc_name,
            tok_name: io.tok_name,
            dim: io.dim,
            gh: io.gh,
            gw: io.gw,
            prefix: io.prefix,
            norm_k,
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
    ///
    /// **BF16, and FP16 explicitly off** — not a stylistic choice. FP16 produces
    /// all-NaN for this model: DINOv3's attention logits reach 2.1e6 while fp16 tops
    /// out at 65504, so they overflow to `inf` in the first block and
    /// `softmax(inf - inf)` poisons every output. It cannot be pinned around either —
    /// TensorRT fuses the attention block, so `--layerPrecisions` matches no layer
    /// names and is silently ignored. See the crate README for the measurements.
    ///
    /// BF16 keeps fp32's exponent range, which is what this model actually needs:
    /// **7.64 ms vs fp32's 13.39 ms**, cosine 0.999568 against the PyTorch reference.
    ///
    /// Never set both flags — TensorRT then picks per layer on speed alone, chooses
    /// fp16 for attention, and the NaN returns.
    #[cfg(any(feature = "hub", feature = "builder"))]
    fn engine_profile() -> vrt_hub::EngineProfile {
        vrt_hub::EngineProfile {
            input: None,
            fp16: false,
            bf16: true,
            workspace_mb: 2048,
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
            .ok_or("dinov3: onnx path is not valid UTF-8")?;
        let engine_path = vrt_hub::EngineCache::default().resolve(
            "dinov3-vits16-336",
            model_path,
            &Self::engine_profile(),
        )?;
        Self::from_engine_file(engine_path, stream)
    }

    /// Pull from Hugging Face (`kornia/dinov3`) and construct — a matching
    /// prebuilt engine if the registry has one for this box, else the pinned ONNX built
    /// on-device. Requires feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let engine = vrt_hub::resolve_engine("dinov3-vits16-336", &Self::engine_profile())?;
        Self::from_engine_file(engine, stream)
    }

    /// Descriptor dimensionality (`D`, 384 for ViT-S/16).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Patch grid `(width, height)` the engine emits — `(S/16, S/16)`.
    pub fn grid(&self) -> (usize, usize) {
        (self.gw, self.gh)
    }

    /// Leading non-patch tokens in the engine's token output (CLS + registers; 5 for
    /// stock DINOv3). `0` when the engine exposes no token output.
    pub fn prefix_tokens(&self) -> usize {
        self.prefix
    }

    /// Allocate a reusable output for this extractor.
    pub fn alloc_result(&self) -> Result<DinoV3Result, DinoError> {
        let patches = match self.tok_name {
            Some(_) => Some(self.stream.alloc_zeros::<f32>(self.gh * self.gw * self.dim)?),
            None => None,
        };
        Ok(DinoV3Result {
            descriptor: self.stream.alloc_zeros::<f32>(self.dim)?,
            patches,
            stream: self.stream.clone(),
            dim: self.dim,
            gh: self.gh,
            gw: self.gw,
        })
    }

    /// Submit one frame — stretch+normalize → backbone → L2-normalize the CLS descriptor
    /// (and copy out the patch grid, when bound) into `out`, with **no sync and no host
    /// copy**. Sync the stream, then read `out.descriptor_slice()` (device, cosine-ready)
    /// or `out.descriptor_host()`.
    ///
    /// Keep `img` and `out` alive until the sync — the GPU reads their device pointers
    /// during it.
    pub fn submit(&mut self, img: &Image<u8, 3>, out: &mut DinoV3Result) -> Result<(), DinoError> {
        self.preproc.run(img, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;

        // Descriptor: one block, L2-norm straight into the caller's buffer. This doubles
        // as the mandatory copy-out (the TRT view aliases session memory the next `run`
        // overwrites). Enqueued after `run`, so stream FIFO order guarantees it sees the
        // finished output.
        let desc_ptr = tmap
            .get(&self.desc_name)
            .ok_or_else(|| DinoError::MissingOutput(self.desc_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;
        let dst_raw = out.descriptor.device_ptr(self.stream.as_ref()).0;
        let (one, dimi) = (1i32, self.dim as i32);
        self.norm_k
            .launch_builder(&self.stream)
            .arg(&desc_ptr)
            .arg(&dst_raw)
            .arg(&one)
            .arg(&dimi)
            .launch_cfg(red_cfg(1))?;

        // Patch tokens, when bound: copy the grid out, skipping the CLS + register
        // prefix so the caller's buffer holds patches only.
        if let (Some(name), Some(dst)) = (self.tok_name.as_ref(), out.patches.as_ref()) {
            let base = tmap
                .get(name)
                .ok_or_else(|| DinoError::MissingOutput(name.clone()))?
                .f32_ptr()? as usize as CUdeviceptr;
            let src = base + (self.prefix * self.dim * std::mem::size_of::<f32>()) as CUdeviceptr;
            let n = self.gh * self.gw * self.dim;
            let tok_dst = dst.device_ptr(self.stream.as_ref()).0;
            let ni = n as i32;
            self.copy_k
                .launch_builder(&self.stream)
                .arg(&src)
                .arg(&ni)
                .arg(&tok_dst)
                .launch_cfg(cfg_1d(n, 256))?;
        }
        Ok(())
    }
}

/// Launch config for the one-block-per-row reduction kernels.
fn red_cfg(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (RED_BLOCK, 1, 1),
        shared_mem_bytes: RED_BLOCK * std::mem::size_of::<f32>() as u32,
    }
}

/// A GPU-resident gallery of L2-normed descriptors with a cosine match kernel — the
/// retrieval primitive behind keyframe banks, place recognition and relocalization
/// candidate search.
///
/// Both the stored rows and the query are unit-norm, so matching is one dot product per
/// row: at `capacity = 512, dim = 384` that is ~196k MACs, free next to the backbone.
pub struct DescriptorBank {
    data: CudaSlice<f32>, // [capacity*dim] device; rows 0..len live
    len: usize,
    capacity: usize,
    dim: usize,
    stream: Arc<CudaStream>,
    match_k: CudaKernel,
}

impl DescriptorBank {
    /// Allocate an empty bank on the shared stream.
    pub fn new(
        capacity: usize,
        dim: usize,
        stream: Arc<CudaStream>,
    ) -> Result<Self, DinoError> {
        let match_k = CudaKernel::compile(stream.context(), MATCH_SRC, "cosine_bank")?;
        Ok(Self {
            data: stream.alloc_zeros::<f32>(capacity * dim)?,
            len: 0,
            capacity,
            dim,
            stream,
            match_k,
        })
    }

    /// Live descriptor count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when nothing has been enrolled yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum descriptors this bank can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Descriptor dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// GPU-resident descriptor rows `[capacity*dim]`, rows `0..len()` live.
    pub fn data_slice(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// Enqueue the cosine of `q` against every live row into `scores[0..len()]`.
    /// **Async: does not sync.** A no-op on an empty bank.
    ///
    /// `scores` must be caller-owned and stay alive until the caller's sync — the GPU
    /// writes it during that sync.
    pub fn match_into(
        &self,
        q: &CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
    ) -> Result<(), DinoError> {
        if self.len == 0 {
            return Ok(());
        }
        if q.len() != self.dim {
            return Err(DinoError::DimMismatch {
                got: q.len(),
                want: self.dim,
            });
        }
        if scores.len() < self.len {
            return Err(DinoError::ScoresTooSmall {
                got: scores.len(),
                want: self.len,
            });
        }
        let bank_raw = self.data.device_ptr(self.stream.as_ref()).0;
        let q_raw = q.device_ptr(self.stream.as_ref()).0;
        let sc_raw = scores.device_ptr_mut(self.stream.as_ref()).0;
        let (ni, dimi) = (self.len as i32, self.dim as i32);
        self.match_k
            .launch_builder(&self.stream)
            .arg(&bank_raw)
            .arg(&q_raw)
            .arg(&sc_raw)
            .arg(&ni)
            .arg(&dimi)
            .launch_cfg(red_cfg(self.len))?;
        Ok(())
    }

    /// Copy `q` into the next free slot (device-to-device, no host round trip) and
    /// return its id. Enqueued on the shared stream like everything else.
    pub fn enroll(&mut self, q: &CudaSlice<f32>) -> Result<usize, DinoError> {
        if q.len() != self.dim {
            return Err(DinoError::DimMismatch {
                got: q.len(),
                want: self.dim,
            });
        }
        if self.len == self.capacity {
            return Err(DinoError::BankFull {
                capacity: self.capacity,
            });
        }
        let id = self.len;
        let stream = self.stream.clone();
        let mut dst = self.data.slice_mut(id * self.dim..(id + 1) * self.dim);
        stream.memcpy_dtod(q, &mut dst)?;
        self.len += 1;
        Ok(id)
    }

    /// Drop every enrolled descriptor. The allocation is kept for reuse.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outs(v: &[(&str, &[i64])]) -> Vec<(String, Vec<i64>)> {
        v.iter().map(|(n, d)| (n.to_string(), d.to_vec())).collect()
    }

    #[test]
    fn binds_descriptor_and_tokens_by_shape() {
        // Stock DINOv3 ViT-S/16 @ 336: 21x21 = 441 patches + 1 CLS + 4 registers = 446.
        let io = resolve_io(
            &[1, 3, 336, 336],
            &outs(&[("tokens", &[1, 446, 384]), ("descriptor", &[1, 384])]),
        )
        .expect("stock DINOv3 shapes should resolve");
        assert_eq!(io.desc_name, "descriptor");
        assert_eq!(io.tok_name.as_deref(), Some("tokens"));
        assert_eq!((io.gh, io.gw, io.dim), (21, 21, 384));
        assert_eq!(io.prefix, 5, "1 CLS + 4 register tokens");
    }

    #[test]
    fn descriptor_only_engine_is_valid() {
        let io = resolve_io(&[1, 3, 224, 224], &outs(&[("pooler_output", &[1, 384])]))
            .expect("a descriptor-only export is supported");
        assert_eq!(io.tok_name, None);
        assert_eq!(io.prefix, 0);
    }

    #[test]
    fn derives_prefix_rather_than_assuming_registers() {
        // A register-free variant (CLS only) must work without a code change.
        let io = resolve_io(
            &[1, 3, 224, 224],
            &outs(&[("t", &[1, 197, 384]), ("d", &[1, 384])]),
        )
        .expect("register-free ViT should resolve");
        assert_eq!(io.prefix, 1, "CLS only");
    }

    #[test]
    fn rejects_dynamic_input_axis() {
        // A dynamic axis reports -1; casting it blindly would wrap to a huge usize.
        let err = resolve_io(&[1, 3, -1, -1], &outs(&[("d", &[1, 384])])).unwrap_err();
        assert!(err.contains("static"), "got: {err}");
    }

    #[test]
    fn rejects_non_multiple_of_patch() {
        let err = resolve_io(&[1, 3, 330, 330], &outs(&[("d", &[1, 384])])).unwrap_err();
        assert!(err.contains("multiples of 16"), "got: {err}");
    }

    #[test]
    fn rejects_ambiguous_descriptor_outputs() {
        let err = resolve_io(
            &[1, 3, 336, 336],
            &outs(&[("a", &[1, 384]), ("b", &[1, 384])]),
        )
        .unwrap_err();
        assert!(err.contains("multiple [1,D]"), "got: {err}");
    }

    #[test]
    fn rejects_missing_descriptor() {
        let err = resolve_io(&[1, 3, 336, 336], &outs(&[("tokens", &[1, 446, 384])])).unwrap_err();
        assert!(err.contains("no [1,D] descriptor"), "got: {err}");
    }

    #[test]
    fn rejects_token_dim_mismatch() {
        let err = resolve_io(
            &[1, 3, 336, 336],
            &outs(&[("t", &[1, 446, 768]), ("d", &[1, 384])]),
        )
        .unwrap_err();
        assert!(err.contains("!= descriptor dim"), "got: {err}");
    }

    #[test]
    fn rejects_token_count_below_patch_grid() {
        // Fewer tokens than patches means the input size and the export disagree.
        let err = resolve_io(
            &[1, 3, 336, 336],
            &outs(&[("t", &[1, 197, 384]), ("d", &[1, 384])]),
        )
        .unwrap_err();
        assert!(err.contains("fewer than"), "got: {err}");
    }
}
