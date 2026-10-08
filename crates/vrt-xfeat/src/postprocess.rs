//! XFeat post-processing: NMS → TopK → descriptor sampling → L2-norm.
//!
//! Works with the TRT backbone engine that outputs three tensors:
//!   `descriptors`  (B, 64, H/8, W/8)  — dense feature maps (FP32 on device)
//!   `heatmap`      (B,  1,   H,   W)  — keypoint confidence (FP32 on device)
//!   `reliability`  (B,  1,   H,   W)  — channel reliability  (FP32 on device)
//!
//! B is 1 or 2 (a stereo pair); each stage is one launch over the batch.
//!
//! Stages (entirely on the GPU — no device→host→device round trip):
//!   GPU  xfeat_score_nms      → score_map (H×W), masked to local-max pixels above threshold
//!   GPU  xfeat_topk_histogram → bin survivor scores into NBINS buckets
//!   GPU  xfeat_topk_cutoff    → score threshold for ~K survivors (one thread)
//!   GPU  xfeat_topk_select    → atomically gather survivors ≥ cutoff, capped K
//!   GPU  xfeat_sample_descs   → K×64 descriptor vectors (bilinear sample from desc_map)
//!   GPU  xfeat_l2_norm        → in-place L2 normalise
//!   (only the keypoint count is read to host; kpts/descs/scores stay on device)
//!
//! [`XFeatResult`] holds the device buffers + `count`. Descriptor matching is a
//! separate concern — see the [`matching`](crate::matching) module. Output
//! keypoints are in GPU-select (atomic-append) order, not score-sorted; `kpts`,
//! `descs`, and `scores` share that order. Kernels are JIT-compiled via kornia's
//! `CudaKernel::compile_many` (arch auto-detected) and launched with explicit
//! configs through `CudaLaunchBuilder::launch_cfg`.

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;
use std::sync::Arc;

use vrt::cuda::{cfg_1d, cfg_2d, cfg_per_item};

/// The descriptor kernels with the width substituted in, so `XFEAT_DESC_DIM` is the one
/// place it is written rather than a constant that happens to agree with three literals.
fn kernels_src() -> String {
    KERNELS_SRC.replace("{XFEAT_D}", &XFEAT_DESC_DIM.to_string())
}

/// XFeat's descriptor width, owned by the post-processing that produces the descriptors.
///
/// It lives here, not on `Matcher`, because it describes the *data*: the sampling and
/// L2-norm kernels below emit exactly this many floats per keypoint. Sourcing it from the
/// matcher would make `check_descriptors` compare the matcher's constant against itself,
/// which is the tautology the width check exists to avoid.
pub const XFEAT_DESC_DIM: usize = 64;

/// Errors from XFeat post-processing and matching.
#[derive(Debug, thiserror::Error)]
pub enum XFeatError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("kornia CUDA: {0}")]
    Cuda(#[from] kornia_tensor::CudaError),
    #[error("backbone output '{0}' missing from engine")]
    MissingOutput(&'static str),
    #[error(transparent)]
    Preproc(#[from] kornia_imgproc::preprocess::PreprocessError),
    #[error("input image {0}x{1} too small — each side must be ≥ 32px")]
    InputTooSmall(usize, usize),
    #[error("batch of {0} images; XFeat post-processing takes 1 or 2")]
    BatchSize(usize),
    #[error("stereo pair sizes differ: left {0}x{1}, right {2}x{3}")]
    StereoSizeMismatch(usize, usize, usize, usize),
    #[error("batched results must share one capacity: got {0}, expected {1}")]
    BatchCapacity(usize, usize),
    #[error(
        "result was allocated on a different Arc<CudaStream> than the one filling it; \
         allocate it with XFeat::alloc_result() or XFeatResult::alloc() on that same Arc"
    )]
    StreamMismatch,
    #[error("score map holds {got} floats but {need} are needed (batch × h × w)")]
    ScoreMapTooSmall { got: usize, need: usize },
    #[error(
        "descriptor width {0} is not supported — it must be a non-zero multiple of 32 and \
         at most 128 (the query array is held in registers, and 256 floats exceeds CUDA's \
         255-register limit per thread)"
    )]
    UnsupportedDim(usize),
    #[error(
        "{which} holds {got} floats but {expected} are needed for {dim}-D descriptors; \
         matching them with a {dim}-D kernel would stride the buffer wrongly"
    )]
    DescriptorDim {
        which: &'static str,
        expected: usize,
        got: usize,
        dim: usize,
    },
    #[error(
        "{which} holds {buf_dim}-D descriptors but this matcher was compiled for \
         {kernel_dim}-D; the kernel would stride the buffer wrongly and return \
         plausible-looking nonsense"
    )]
    DescriptorWidth {
        which: &'static str,
        buf_dim: usize,
        kernel_dim: usize,
    },
    #[error("{which} holds {count} descriptors but the match output has capacity {cap}")]
    MatchCapacity {
        which: &'static str,
        count: usize,
        cap: usize,
    },
}

// ── Kernel source ─────────────────────────────────────────────────────────────

const KERNELS_SRC: &str = r#"
#define XFEAT_D {XFEAT_D}
/* xfeat_score_nms — fused NMS + score map.
   For each pixel (x,y): if heatmap[y,x] > threshold AND no neighbour in the
   5×5 window has a strictly greater value, write heatmap[y,x]*reliability[y,x]
   to score_out; otherwise write 0.

   Uses __ldg (read-only cache via texture path) for the 25 neighbour reads so
   overlapping 5×5 windows reuse L1 instead of re-fetching from DRAM. */
extern "C" __global__ void xfeat_score_nms(
    const float* __restrict__ heatmap,
    const float* __restrict__ reliability,
    float* __restrict__ score_out,
    int H, int W,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    size_t plane = (size_t)blockIdx.z * H * W;   /* batch image */
    heatmap += plane; reliability += plane; score_out += plane;

    int idx = y * W + x;
    float h = __ldg(&heatmap[idx]);

    if (h <= threshold) { score_out[idx] = 0.0f; return; }

    for (int dy = -2; dy <= 2; dy++) {
        int ny = y + dy;
        if (ny < 0 || ny >= H) continue;
        for (int dx = -2; dx <= 2; dx++) {
            int nx = x + dx;
            if (nx < 0 || nx >= W) continue;
            if (__ldg(&heatmap[ny * W + nx]) > h) {
                score_out[idx] = 0.0f;
                return;
            }
        }
    }

    score_out[idx] = h * __ldg(&reliability[idx]);
}

/* xfeat_sample_descs — bilinear descriptor sampling.
   For each of K keypoints (pixel-space x, y), sample the 64-channel descriptor
   map (stored CHW: [64, Hd, Wd]) using align_corners=False bilinear interpolation.
   Also publishes, per image, the clamped count and the keypoints in original-image
   pixels (x*sx, y*sy — the same single multiply kpts_to_host does on the host).
   Launch config: grid=(K,1,B), block=(64,1,1); slots past count[b] exit. */
extern "C" __global__ void xfeat_sample_descs(
    const float* __restrict__ desc_map,
    const float* __restrict__ kpts0, const float* __restrict__ kpts1,
    float* __restrict__ descs0, float* __restrict__ descs1,
    const int* __restrict__ count,
    int* __restrict__ count_out0, int* __restrict__ count_out1,
    float* __restrict__ kpx0, float* __restrict__ kpx1,
    float sx, float sy,
    int Hd, int Wd,
    int H,  int W, int K
) {
    int k = blockIdx.x;
    int c = threadIdx.x;
    int b = blockIdx.z;
    int n = min(count[b], K);
    if (k == 0 && c == 0) *(b ? count_out1 : count_out0) = n;
    if (k >= n) return;
    desc_map += (size_t)b * XFEAT_D * Hd * Wd;
    const float* kpts = b ? kpts1 : kpts0;
    float* descs_out  = b ? descs1 : descs0;

    float px = __ldg(&kpts[k * 2 + 0]);
    float py = __ldg(&kpts[k * 2 + 1]);
    if (c == 0) {
        float* kpx = b ? kpx1 : kpx0;
        kpx[k * 2 + 0] = px * sx;
        kpx[k * 2 + 1] = py * sy;
    }
    float dx = (px + 0.5f) / (float)W * (float)Wd - 0.5f;
    float dy = (py + 0.5f) / (float)H * (float)Hd - 0.5f;

    int x0 = (int)floorf(dx);
    int y0 = (int)floorf(dy);
    float wx = dx - (float)x0;
    float wy = dy - (float)y0;

    // Clamp all four sample indices to [0, dim-1] (border replicate), for BOTH
    // bounds — a coordinate <= -1 would otherwise give x0+1 <= 0, i.e. a negative
    // x1/y1 index and an out-of-bounds read. wx/wy keep the true fractional offset.
    int x1 = min(max(x0 + 1, 0), Wd - 1);
    int y1 = min(max(y0 + 1, 0), Hd - 1);
    x0 = min(max(x0, 0), Wd - 1);
    y0 = min(max(y0, 0), Hd - 1);

    int base = c * Hd * Wd;
    float val = (1.0f - wx) * (1.0f - wy) * __ldg(&desc_map[base + y0 * Wd + x0])
              +           wx * (1.0f - wy) * __ldg(&desc_map[base + y0 * Wd + x1])
              + (1.0f - wx) *           wy * __ldg(&desc_map[base + y1 * Wd + x0])
              +           wx *           wy * __ldg(&desc_map[base + y1 * Wd + x1]);

    descs_out[k * XFEAT_D + c] = val;
}

/* xfeat_l2_norm — in-place L2-normalise each 64-D descriptor row.
   block_dim=64 = exactly 2 warps; grid=(K,1,B). */
extern "C" __global__ void xfeat_l2_norm(
    float* __restrict__ descs0, float* __restrict__ descs1,
    const int* __restrict__ count,
    int K
) {
    int k = blockIdx.x;
    int c = threadIdx.x;
    int b = blockIdx.z;
    if (k >= min(count[b], K)) return;   /* whole block exits: no split __syncthreads */
    float* descs = b ? descs1 : descs0;

    /* Sized and looped from XFEAT_D rather than hardcoded to two warps: the previous
       form's shmem[2] / `if (c == 32)` silently divided by a half-norm at any other
       width, which is the failure the width constant is supposed to make impossible. */
    __shared__ float shmem[XFEAT_D / 32];

    float v = descs[k * XFEAT_D + c];
    float s = v * v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    if ((c & 31) == 0) shmem[c >> 5] = s;
    __syncthreads();
    float total = 0.0f;
    #pragma unroll
    for (int w = 0; w < XFEAT_D / 32; w++) total += shmem[w];
    float norm = sqrtf(total);
    if (norm < 1e-8f) norm = 1e-8f;
    descs[k * XFEAT_D + c] = v / norm;
}

/* xfeat_compact_scores — stream-compact NMS survivors.
   GPU top-K by histogram cutoff — keeps the whole select on the device so
   the postproc is a pure async tail (no mid-frame D2H→CPU-sort→H2D round trip).

   1. xfeat_topk_histogram: bin every survivor score (>0) into NBINS buckets.
   2. xfeat_topk_cutoff:    one thread scans buckets high→low, finds the score
      threshold below which fewer than K survivors remain.
   3. xfeat_topk_select:    atomically gather survivors >= threshold, capped at
      K, writing (x,y) and score.  Approximate only at the boundary bucket
      (NBINS=1024 → indistinguishable from exact for keypoint selection); the
      boundary ties are as arbitrary as a CPU unstable sort's were. */
#define TOPK_NBINS 1024
extern "C" __global__ void xfeat_topk_histogram(
    const float* __restrict__ score_map,
    int*   __restrict__ hist,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    score_map += (size_t)blockIdx.z * total;
    hist      += blockIdx.z * TOPK_NBINS;
    float s = __ldg(&score_map[i]);
    if (s <= 0.0f) return;
    int b = (int)(s * (float)TOPK_NBINS);
    if (b < 0) b = 0;
    if (b >= TOPK_NBINS) b = TOPK_NBINS - 1;
    atomicAdd(&hist[b], 1);
}

extern "C" __global__ void xfeat_topk_cutoff(
    const int* __restrict__ hist,
    int K,
    float* __restrict__ cutoff_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    hist       += blockIdx.z * TOPK_NBINS;
    cutoff_out += blockIdx.z;
    float cut = 0.0f;          // default: take every survivor (total < K)
    int cum = 0;
    for (int i = TOPK_NBINS - 1; i >= 0; --i) {
        cum += hist[i];
        if (cum >= K) { cut = (float)i / (float)TOPK_NBINS; break; }
    }
    *cutoff_out = cut;
}

extern "C" __global__ void xfeat_topk_select(
    const float* __restrict__ score_map,
    const float* __restrict__ cutoff,
    float* __restrict__ kpts0, float* __restrict__ kpts1,      /* [K*2] (x,y) */
    float* __restrict__ scores0, float* __restrict__ scores1,  /* [K] */
    int*   __restrict__ count,
    int H, int W, int K
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W) return;
    int b = blockIdx.z;
    score_map += (size_t)b * H * W;
    cutoff += b; count += b;
    float* kpts_xy    = b ? kpts1 : kpts0;
    float* scores_out = b ? scores1 : scores0;
    float s = __ldg(&score_map[i]);
    float cut = *cutoff;
    if (s <= 0.0f || s < cut) return;
    int slot = atomicAdd(count, 1);     // counts all >= cut; may exceed K
    if (slot >= K) return;              // cap: extras dropped (boundary-bucket ties)
    kpts_xy[slot * 2 + 0] = (float)(i % W);
    kpts_xy[slot * 2 + 1] = (float)(i / W);
    scores_out[slot]      = s;
}
"#;

// ── Public types ──────────────────────────────────────────────────────────────

/// Output of one XFeat extraction — **caller-owned, pre-allocated, reusable**
/// (VPI-style output buffer).
///
/// Allocate once with [`alloc`](Self::alloc) (capacity `top_k`), pass `&mut` to
/// [`XFeat::submit`], sync the stream, then read. All device buffers stay on the
/// GPU — descriptor matching (`matching::Matcher`) runs on them without a
/// download; [`count`](Self::count) reads the pinned scalar (valid **after** the
/// sync). Reuse across frames; hold several to keep multiple frames outstanding.
pub struct XFeatResult {
    /// Device (x, y) in **model space** (floor-32 backbone input), capacity `top_k×2`.
    /// [`kpts_to_host`](Self::kpts_to_host) applies [`scale`](Self::scale) → original px.
    pub kpts: CudaSlice<f32>,
    /// L2-normalised 64-D descriptors on device, capacity `top_k×64`.
    /// Ask [`desc_dim`](Self::desc_dim) for the width rather than assuming it.
    pub descs: CudaSlice<f32>,
    /// Combined NMS scores on device, capacity `top_k`.
    pub scores: CudaSlice<f32>,
    /// Keypoint count on device, clamped to `top_k`; see [`count_device`](Self::count_device).
    count_dev: CudaSlice<i32>,
    /// Device (x, y) in original-image pixels; see [`kpts_px`](Self::kpts_px).
    kpts_px: CudaSlice<f32>,
    /// Pinned host target for the count scalar (the only D2H), written by `submit`.
    count_pin: vrt::PinnedBuffer<i32>,
    /// The stream these buffers live on (used for the readout D2H).
    stream: Arc<CudaStream>,
    top_k: usize,
    /// Model→original scale `(rw, rh)`, stamped by [`XFeat::submit`].
    scale: (f32, f32),
}

impl XFeatResult {
    /// Pre-allocate an extraction output of capacity `top_k` on `stream`.
    pub fn alloc(stream: &Arc<CudaStream>, top_k: usize) -> Result<Self, XFeatError> {
        Ok(Self {
            kpts: stream.alloc_zeros::<f32>(top_k * 2)?,
            descs: unsafe { stream.alloc::<f32>(top_k * XFEAT_DESC_DIM)? },
            scores: stream.alloc_zeros::<f32>(top_k)?,
            count_dev: stream.alloc_zeros::<i32>(1)?,
            kpts_px: stream.alloc_zeros::<f32>(top_k * 2)?,
            count_pin: vrt::PinnedBuffer::<i32>::alloc(1)?,
            stream: stream.clone(),
            top_k,
            scale: (1.0, 1.0),
        })
    }

    /// Capacity (max keypoints) this result was allocated for.
    pub fn capacity(&self) -> usize {
        self.top_k
    }

    /// Descriptor width of [`descs`](Self::descs).
    ///
    /// Sourced from the extractor that produced them, so a `Descriptors` built from this
    /// carries the *data's* width. Passing a matcher's own `dim()` instead compares a
    /// value against itself and passes unconditionally.
    pub fn desc_dim(&self) -> usize {
        XFEAT_DESC_DIM
    }

    /// Valid keypoint count — reads the pinned scalar, so call **after** the
    /// stream sync following [`XFeat::submit`].
    pub fn count(&self) -> usize {
        (self.count_pin.as_slice()[0].max(0) as usize).min(self.top_k)
    }
    pub fn len(&self) -> usize {
        self.count()
    }
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Download the valid keypoints to host: interleaved `[x0,y0,x1,y1,…]`,
    /// length `count × 2`, in **original image pixels** ([`scale`](Self::scale)
    /// applied). Call after the stream sync.
    pub fn kpts_to_host(&self) -> Result<Vec<f32>, cudarc::driver::DriverError> {
        let n = self.count();
        let mut xy = self.stream.clone_dtoh(&self.kpts.slice(0..n * 2))?;
        let (sx, sy) = self.scale;
        if (sx, sy) != (1.0, 1.0) {
            for p in xy.chunks_exact_mut(2) {
                p[0] *= sx;
                p[1] *= sy;
            }
        }
        Ok(xy)
    }

    /// Download the valid scores to host (length `count`). Call after the sync.
    pub fn scores_to_host(&self) -> Result<Vec<f32>, cudarc::driver::DriverError> {
        let n = self.count();
        self.stream.clone_dtoh(&self.scores.slice(0..n))
    }

    /// Download the valid descriptors to host, row-major `count × `[`desc_dim`](Self::desc_dim).
    /// Call after the sync.
    ///
    /// The counterpart to the other two `*_to_host` accessors, added because it was the only one
    /// missing: a caller wanting descriptors had to slice the public [`descs`](Self::descs) buffer
    /// itself, and doing that means writing `n * 64` at the call site — the one place the width is
    /// *not* sourced from the data that produced it.
    pub fn descs_to_host(&self) -> Result<Vec<f32>, cudarc::driver::DriverError> {
        let n = self.count();
        self.stream
            .clone_dtoh(&self.descs.slice(0..n * XFEAT_DESC_DIM))
    }

    /// Valid keypoint count on the device, already clamped to
    /// [`capacity`](Self::capacity). Valid in stream order after
    /// [`XFeat::submit`](crate::XFeat::submit) — a consumer on the same stream (e.g.
    /// kornia-3d's stereo matcher via `KeypointCount::Device`) reads it with no host
    /// sync; [`count`](Self::count) is the host copy, valid only after a sync.
    pub fn count_device(&self) -> &CudaSlice<i32> {
        &self.count_dev
    }

    /// Device keypoints in **original-image pixels**, interleaved `[x0, y0, …]`,
    /// capacity `top_k × 2`; the first `count` are valid. Bit-identical to
    /// [`kpts_to_host`](Self::kpts_to_host). Use these, not [`kpts`](Self::kpts) (model
    /// space), wherever coordinates must match the image, e.g. stereo matching against
    /// the rectified frames when the width is not a multiple of 32.
    pub fn kpts_px(&self) -> &CudaSlice<f32> {
        &self.kpts_px
    }

    /// Mutable pinned-count pointer (for the async count D2H in `launch_topk_batch`).
    pub(crate) fn count_pin_mut(&mut self) -> *mut i32 {
        self.count_pin.as_mut_ptr()
    }

    /// Stamp the model→original keypoint scale. Set by [`XFeat::submit`] BEFORE the
    /// post-processing launch, which bakes it into [`kpts_px`](Self::kpts_px).
    pub(crate) fn set_scale(&mut self, scale: (f32, f32)) {
        self.scale = scale;
    }
}

// ── XFeatPostproc ─────────────────────────────────────────────────────────────

pub struct XFeatPostproc {
    fn_score_nms: CudaKernel,
    fn_sample_descs: CudaKernel,
    fn_l2_norm: CudaKernel,
    fn_histogram: CudaKernel,
    fn_cutoff: CudaKernel,
    fn_select: CudaKernel,
    stream: Arc<CudaStream>,
    threshold: f32,
    /// `[MAX_BATCH × NBINS histogram | MAX_BATCH counts]`, zeroed by one memset per call.
    scratch: CudaSlice<i32>,
    /// Per-image top-K cutoff, written by `xfeat_topk_cutoff` (no zeroing needed).
    cutoff: CudaSlice<f32>,
}

const TOPK_NBINS: usize = 1024;

/// Images one post-processing pass handles: the kernels pick per-image output
/// pointers by `blockIdx.z`, so this is a stereo pair at most.
pub const MAX_BATCH: usize = 2;

impl XFeatPostproc {
    /// Compile all CUDA kernels and return a ready post-processor. The keypoint
    /// cap comes from the output [`XFeatResult`]'s capacity, not from here.
    pub fn new(stream: Arc<CudaStream>, threshold: f32) -> Result<Self, XFeatError> {
        // Compile the kernel suite once; load all six functions from the module.
        let names = [
            "xfeat_score_nms",
            "xfeat_sample_descs",
            "xfeat_l2_norm",
            "xfeat_topk_histogram",
            "xfeat_topk_cutoff",
            "xfeat_topk_select",
        ];
        let [fn_score_nms, fn_sample_descs, fn_l2_norm, fn_histogram, fn_cutoff, fn_select]: [CudaKernel; 6] =
            CudaKernel::compile_many(stream.context(), &kernels_src(), &names)?
                .try_into().unwrap_or_else(|_| unreachable!("compile_many returns names.len() kernels"));

        let scratch = stream.alloc_zeros::<i32>(MAX_BATCH * (TOPK_NBINS + 1))?;
        let cutoff = stream.alloc_zeros::<f32>(MAX_BATCH)?;
        Ok(Self {
            fn_score_nms,
            fn_sample_descs,
            fn_l2_norm,
            fn_histogram,
            fn_cutoff,
            fn_select,
            stream,
            threshold,
            scratch,
            cutoff,
        })
    }

    /// Enqueue the NMS score kernel for `batch` stacked `h × w` maps into `score_dev`
    /// (async — caller must sync before reading).
    ///
    /// `heat_ptr`/`rel_ptr` point at `[batch,1,h,w]` tensors; `score_dev` must hold
    /// at least `batch * h * w` f32 elements (checked).
    pub fn launch_score_nms(
        &self,
        heat_ptr: *const f32,
        rel_ptr: *const f32,
        score_dev: &CudaSlice<f32>,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<(), XFeatError> {
        use cudarc::driver::DevicePtr;
        check_batch(batch)?;
        check_score_len(score_dev, batch, h, w)?;
        let heat_raw: CUdeviceptr = heat_ptr as usize as CUdeviceptr;
        let rel_raw: CUdeviceptr = rel_ptr as usize as CUdeviceptr;
        let score_raw: CUdeviceptr = score_dev.device_ptr(self.stream.as_ref()).0;

        let mut cfg = cfg_2d(w, h);
        cfg.grid_dim.2 = batch as u32;
        let h_i = h as i32;
        let w_i = w as i32;
        let thr = self.threshold;
        self.fn_score_nms
            .launch_builder(&self.stream)
            .arg(&heat_raw)
            .arg(&rel_raw)
            .arg(&score_raw)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&thr)
            .launch_cfg(cfg)?;
        Ok(())
    }

    /// Single-image [`launch_topk_batch`](Self::launch_topk_batch).
    pub fn launch_topk(
        &mut self,
        desc_ptr: *const f32,
        score_dev: &CudaSlice<f32>,
        h: usize,
        w: usize,
        out: &mut XFeatResult,
    ) -> Result<(), XFeatError> {
        self.launch_topk_batch(desc_ptr, score_dev, h, w, &mut [out])
    }

    /// All `outs` must share one capacity and this stream (a foreign stream races silently).
    pub(crate) fn check_outs(&self, outs: &[&mut XFeatResult]) -> Result<(), XFeatError> {
        check_batch(outs.len())?;
        let k = outs[0].top_k;
        for o in outs {
            if o.top_k != k {
                return Err(XFeatError::BatchCapacity(o.top_k, k));
            }
            if !Arc::ptr_eq(&o.stream, &self.stream) {
                return Err(XFeatError::StreamMismatch);
            }
        }
        Ok(())
    }

    /// Launch the entire top-K + descriptor postproc **asynchronously** for
    /// `outs.len()` stacked images — GPU histogram-cutoff top-K, descriptor
    /// sampling, L2-norm, and the async D2H of each keypoint count into its
    /// `out`'s pinned buffer — with **no `stream.synchronize()`**. Every stage is
    /// one launch covering the whole batch (`blockIdx.z` = image).
    ///
    /// `desc_ptr` is the `[B,64,h/8,w/8]` backbone output and `score_dev` the
    /// `[B,h,w]` NMS map (see [`launch_score_nms`](Self::launch_score_nms)); `outs[b]`
    /// receives image `b`. All `outs` must share one capacity and this stream.
    /// Sync once, then read each `out`.
    pub fn launch_topk_batch(
        &mut self,
        desc_ptr: *const f32,
        score_dev: &CudaSlice<f32>,
        h: usize,
        w: usize,
        outs: &mut [&mut XFeatResult],
    ) -> Result<(), XFeatError> {
        use cudarc::driver::DevicePtr;
        self.check_outs(outs)?;
        let batch = outs.len();
        let k = outs[0].top_k;
        check_score_len(score_dev, batch, h, w)?;
        let (hd, wd) = (h / 8, w / 8);
        let n_pixels = h * w;

        // Zero the histograms + counts in one memset (replaces three per-frame
        // allocs). The `out` tails past `count` keep stale values but are never read.
        self.stream.memset_zeros(&mut self.scratch)?;

        let st = self.stream.as_ref();
        let raw = |s: &CudaSlice<f32>| -> CUdeviceptr { s.device_ptr(st).0 };
        let score_raw = raw(score_dev);
        let hist_raw = self.scratch.device_ptr(st).0;
        let cnt_raw =
            hist_raw + (MAX_BATCH * TOPK_NBINS * std::mem::size_of::<i32>()) as CUdeviceptr;
        let cut_raw = raw(&self.cutoff);
        // Image 1's pointers alias image 0's for a single image; never read then.
        let o1 = batch - 1;
        let (kxy0, kxy1) = (raw(&outs[0].kpts), raw(&outs[o1].kpts));
        let (sco0, sco1) = (raw(&outs[0].scores), raw(&outs[o1].scores));
        let (dsc0, dsc1) = (raw(&outs[0].descs), raw(&outs[o1].descs));
        let (kpx0, kpx1) = (raw(&outs[0].kpts_px), raw(&outs[o1].kpts_px));
        let raw_i = |s: &CudaSlice<i32>| -> CUdeviceptr { s.device_ptr(self.stream.as_ref()).0 };
        let (cnt0, cnt1) = (raw_i(&outs[0].count_dev), raw_i(&outs[o1].count_dev));
        // Same-size images (submit_pair enforces it), so one scale serves the batch.
        let (sx, sy) = outs[0].scale;
        let desc_raw = desc_ptr as usize as CUdeviceptr;
        let total = n_pixels as i32;
        let (k_i, h_i, w_i) = (k as i32, h as i32, w as i32);
        let (hd_i, wd_i) = (hd as i32, wd as i32);
        let z = |mut c: cudarc::driver::LaunchConfig| {
            c.grid_dim.2 = batch as u32;
            c
        };
        let cfg64 = z(cfg_per_item(k, XFEAT_DESC_DIM as u32));

        // 1. histogram → 2. cutoff → 3. select survivors into out.kpts/out.scores
        self.fn_histogram
            .launch_builder(&self.stream)
            .arg(&score_raw)
            .arg(&hist_raw)
            .arg(&total)
            .launch_cfg(z(cfg_1d(n_pixels, 256)))?;
        self.fn_cutoff
            .launch_builder(&self.stream)
            .arg(&hist_raw)
            .arg(&k_i)
            .arg(&cut_raw)
            .launch_cfg(z(cfg_1d(1, 1)))?;
        self.fn_select
            .launch_builder(&self.stream)
            .arg(&score_raw)
            .arg(&cut_raw)
            .arg(&kxy0)
            .arg(&kxy1)
            .arg(&sco0)
            .arg(&sco1)
            .arg(&cnt_raw)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&k_i)
            .launch_cfg(z(cfg_1d(n_pixels, 256)))?;
        // 4. sample 64-D descriptors into out.descs → 5. L2-normalise in place
        self.fn_sample_descs
            .launch_builder(&self.stream)
            .arg(&desc_raw)
            .arg(&kxy0)
            .arg(&kxy1)
            .arg(&dsc0)
            .arg(&dsc1)
            .arg(&cnt_raw)
            .arg(&cnt0)
            .arg(&cnt1)
            .arg(&kpx0)
            .arg(&kpx1)
            .arg(&sx)
            .arg(&sy)
            .arg(&hd_i)
            .arg(&wd_i)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&k_i)
            .launch_cfg(cfg64)?;
        self.fn_l2_norm
            .launch_builder(&self.stream)
            .arg(&dsc0)
            .arg(&dsc1)
            .arg(&cnt_raw)
            .arg(&k_i)
            .launch_cfg(cfg64)?;

        // 6. async D2H of each count scalar (the ONLY host transfers) into the
        //    callers' pinned buffers — pinned makes cudaMemcpyAsync truly async.
        let vstream = vrt::Stream::from_cuda_stream(self.stream.clone());
        for (b, out) in outs.iter_mut().enumerate() {
            let src = cnt_raw + (b * std::mem::size_of::<i32>()) as CUdeviceptr;
            unsafe {
                vstream.memcpy_d2h_raw(
                    out.count_pin_mut() as *mut u8,
                    src as usize as *const _,
                    std::mem::size_of::<i32>(),
                )?;
            }
        }
        Ok(())
    }
}

fn check_batch(batch: usize) -> Result<(), XFeatError> {
    if batch == 0 || batch > MAX_BATCH {
        return Err(XFeatError::BatchSize(batch));
    }
    Ok(())
}

fn check_score_len(
    score_dev: &CudaSlice<f32>,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<(), XFeatError> {
    let need = batch * h * w;
    if score_dev.len() < need {
        return Err(XFeatError::ScoreMapTooSmall {
            got: score_dev.len(),
            need,
        });
    }
    Ok(())
}

#[cfg(test)]
mod gpu_compact_tests {
    use super::*;

    /// GPU top-K must select the right keypoints from a synthetic score map and
    /// produce L2-normalized descriptors.  The GPU `select` gathers via atomic
    /// append, so the output order is unspecified — assertions are order-free.
    /// The device count is clamped to capacity even when more survivors pass the
    /// cutoff, and `kpts_px` is bit-identical to the host-scaled `kpts_to_host`.
    #[test]
    #[ignore]
    fn gpu_device_count_and_pixel_kpts() {
        use cudarc::driver::DevicePtr;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut pp = XFeatPostproc::new(stream.clone(), 0.05).unwrap();
        let mut res = XFeatResult::alloc(&stream, 2).unwrap();
        res.set_scale((1.0217391, 0.75)); // 752/736 and an arbitrary y scale

        let (h, w) = (32usize, 32usize);
        // Three equal-score survivors land in one histogram bin, so all three pass the
        // cutoff and the raw atomic count (3) exceeds top_k (2).
        let mut scores = vec![0.0f32; h * w];
        for i in [100, 517, 999] {
            scores[i] = 0.5;
        }
        let score_dev = stream.clone_htod(&scores).unwrap();
        let desc_dev = stream
            .clone_htod(&vec![1.0f32; 64 * (h / 8) * (w / 8)])
            .unwrap();
        let desc_ptr = desc_dev.device_ptr(stream.as_ref()).0 as *const f32;
        pp.launch_topk(desc_ptr, &score_dev, h, w, &mut res)
            .unwrap();
        stream.synchronize().unwrap();

        assert_eq!(res.count(), 2);
        assert_eq!(stream.clone_dtoh(res.count_device()).unwrap(), vec![2]);
        let host: Vec<u32> = res
            .kpts_to_host()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let dev: Vec<u32> = stream
            .clone_dtoh(&res.kpts_px().slice(0..4))
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(dev, host);
    }

    #[test]
    #[ignore]
    fn gpu_topk_selects_correct_keypoints() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut pp = XFeatPostproc::new(stream.clone(), 0.05).unwrap();
        let mut res = XFeatResult::alloc(&stream, 2).unwrap(); // top_k = 2

        let (h, w) = (32usize, 32usize);
        let (hd, wd) = (h / 8, w / 8);

        // Three survivors; top-2 by score are at flat idx 100 (x=4,y=3) and 999 (x=7,y=31).
        let mut scores = vec![0.0f32; h * w];
        scores[100] = 0.9;
        scores[999] = 0.7;
        scores[500] = 0.1;
        let score_dev = stream.clone_htod(&scores).unwrap();

        // Constant-per-channel descriptor map: sampled vector = (c+1) before norm.
        let mut desc_map = vec![0.0f32; 64 * hd * wd];
        for c in 0..64 {
            for i in 0..hd * wd {
                desc_map[c * hd * wd + i] = (c + 1) as f32;
            }
        }
        let desc_dev = stream.clone_htod(&desc_map).unwrap();
        let desc_ptr = {
            use cudarc::driver::DevicePtr;
            desc_dev.device_ptr(stream.as_ref()).0 as *const f32
        };

        pp.launch_topk(desc_ptr, &score_dev, h, w, &mut res)
            .unwrap();
        stream.synchronize().unwrap();

        // Exactly the top-2 keypoints, in any order: pair (score, x, y) and sort.
        assert_eq!(res.count(), 2);
        let scores = res.scores_to_host().unwrap();
        let kpts = res.kpts_to_host().unwrap();
        let mut got: Vec<(i32, u32, u32)> = scores
            .iter()
            .zip(kpts.chunks_exact(2))
            .map(|(s, xy)| ((s * 1000.0) as i32, xy[0] as u32, xy[1] as u32))
            .collect();
        got.sort_by(|a, b| b.0.cmp(&a.0));
        assert_eq!(
            got,
            vec![
                (900, (100 % w) as u32, (100 / w) as u32),
                (700, (999 % w) as u32, (999 / w) as u32),
            ]
        );

        // Descriptors (count rows of the capacity-K buffer) must be L2-normalized
        // samples of the constant map.
        let descs: Vec<f32> = stream.clone_dtoh(&res.descs).unwrap();
        for row in descs.chunks_exact(64).take(res.count()) {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "descriptor not normalized: {norm}"
            );
            // direction must follow (1, 2, ..., 64) / |(1,...,64)|
            let expect0 = 1.0 / (1..=64).map(|c| (c * c) as f32).sum::<f32>().sqrt();
            assert!(
                (row[0] - expect0).abs() < 1e-3,
                "row[0]={} expect {}",
                row[0],
                expect0
            );
        }
    }

    /// A score map shorter than `batch × h × w` must be rejected before any launch.
    #[test]
    #[ignore]
    fn gpu_short_score_map_is_rejected() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut pp = XFeatPostproc::new(stream.clone(), 0.05).unwrap();
        let (h, w) = (32usize, 32usize);
        let score_dev = stream.alloc_zeros::<f32>(h * w).unwrap();
        let (mut r0, mut r1) = (
            XFeatResult::alloc(&stream, 2).unwrap(),
            XFeatResult::alloc(&stream, 2).unwrap(),
        );
        let null = std::ptr::null::<f32>();
        let err = pp
            .launch_score_nms(null, null, &score_dev, 2, h, w)
            .unwrap_err();
        assert!(matches!(err, XFeatError::ScoreMapTooSmall { .. }), "{err}");
        let err = pp
            .launch_topk_batch(null, &score_dev, h, w, &mut [&mut r0, &mut r1])
            .unwrap_err();
        assert!(matches!(err, XFeatError::ScoreMapTooSmall { .. }), "{err}");
    }

    /// Batch 2 through NMS -> top-K: each result must get exactly its own image's
    /// keypoints, scores and descriptors (catches plane-offset / slot-routing bugs).
    #[test]
    #[ignore]
    fn gpu_batch2_routes_each_image_to_its_slot() {
        use cudarc::driver::DevicePtr;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut pp = XFeatPostproc::new(stream.clone(), 0.05).unwrap();
        let (h, w) = (32usize, 32usize);
        let (hd, wd, n) = (h / 8, w / 8, h * w);
        let d = XFEAT_DESC_DIM;

        // Isolated peaks (>2 px apart, so all survive NMS); top-2 per image is the first two.
        let peaks: [[(usize, f32); 3]; 2] = [
            [(100, 0.9), (999, 0.7), (500, 0.1)],
            [(37, 0.8), (700, 0.6), (300, 0.2)],
        ];
        let rel_val = [1.0f32, 0.5]; // per-image reliability: proves the rel plane offset
        let mut heat = vec![0.0f32; 2 * n];
        let mut rel = vec![0.0f32; 2 * n];
        for b in 0..2 {
            rel[b * n..(b + 1) * n].fill(rel_val[b]);
            for &(i, s) in &peaks[b] {
                heat[b * n + i] = s;
            }
        }
        // Per-image constant-per-channel desc maps with different directions.
        let chan = |b: usize, c: usize| {
            if b == 0 {
                (c + 1) as f32
            } else {
                (d - c) as f32
            }
        };
        let mut desc = vec![0.0f32; 2 * d * hd * wd];
        for b in 0..2 {
            for c in 0..d {
                let o = (b * d + c) * hd * wd;
                desc[o..o + hd * wd].fill(chan(b, c));
            }
        }
        let (heat_dev, rel_dev, desc_dev) = (
            stream.clone_htod(&heat).unwrap(),
            stream.clone_htod(&rel).unwrap(),
            stream.clone_htod(&desc).unwrap(),
        );
        let score_dev = stream.alloc_zeros::<f32>(2 * n).unwrap();
        let ptr = |s: &CudaSlice<f32>| s.device_ptr(stream.as_ref()).0 as *const f32;

        let (mut r0, mut r1) = (
            XFeatResult::alloc(&stream, 2).unwrap(),
            XFeatResult::alloc(&stream, 2).unwrap(),
        );
        pp.launch_score_nms(ptr(&heat_dev), ptr(&rel_dev), &score_dev, 2, h, w)
            .unwrap();
        pp.launch_topk_batch(ptr(&desc_dev), &score_dev, h, w, &mut [&mut r0, &mut r1])
            .unwrap();
        stream.synchronize().unwrap();

        for (b, res) in [&r0, &r1].into_iter().enumerate() {
            assert_eq!(res.count(), 2, "image {b}");
            // Each slot publishes its own device count.
            assert_eq!(
                stream.clone_dtoh(res.count_device()).unwrap(),
                vec![2],
                "image {b}"
            );
            let (kpts, scores) = (res.kpts_to_host().unwrap(), res.scores_to_host().unwrap());
            let mut got: Vec<(u32, u32, f32)> = kpts
                .chunks_exact(2)
                .zip(&scores)
                .map(|(xy, &s)| (xy[0] as u32, xy[1] as u32, s))
                .collect();
            got.sort_by(|a, b| b.2.total_cmp(&a.2));
            let want: Vec<(u32, u32, f32)> = peaks[b][..2]
                .iter()
                .map(|&(i, s)| ((i % w) as u32, (i / w) as u32, s * rel_val[b]))
                .collect();
            assert_eq!(got, want, "image {b}");

            let norm = (0..d).map(|c| chan(b, c).powi(2)).sum::<f32>().sqrt();
            for row in res.descs_to_host().unwrap().chunks_exact(d) {
                for (c, v) in row.iter().enumerate() {
                    let e = chan(b, c) / norm;
                    assert!((v - e).abs() < 1e-5, "image {b} ch {c}: {v} vs {e}");
                }
            }
        }
    }
}
