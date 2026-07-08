//! XFeat post-processing: NMS → TopK → descriptor sampling → L2-norm + GPU matching.
//!
//! Works with the TRT backbone engine that outputs three tensors:
//!   `descriptors`  (1, 64, H/8, W/8)  — dense feature maps (FP32 on device)
//!   `heatmap`      (1,  1,   H,   W)  — keypoint confidence (FP32 on device)
//!   `reliability`  (1,  1,   H,   W)  — channel reliability  (FP32 on device)
//!
//! Stages (entirely on the GPU — no device→host→device round trip):
//!   GPU  xfeat_score_nms      → score_map (H×W), masked to local-max pixels above threshold
//!   GPU  xfeat_topk_histogram → bin survivor scores into NBINS buckets
//!   GPU  xfeat_topk_cutoff    → score threshold for ~K survivors (one thread)
//!   GPU  xfeat_topk_select    → atomically gather survivors ≥ cutoff, capped K
//!   GPU  xfeat_sample_descs   → K×64 descriptor vectors (bilinear sample from desc_map)
//!   GPU  xfeat_l2_norm        → in-place L2 normalise
//!   (only the keypoint count is read to host; kpts/descs/scores stay on device)
//!   GPU  xfeat_match_argmax   → tiled mutual-NN matching (two calls, swapped args)
//!
//! [`XFeatResult`] holds the device buffers + `count`; matching runs on them
//! without a download. Output keypoints are in GPU-select (atomic-append) order,
//! not score-sorted; `kpts`, `descs`, and `scores` share that order. Kernels are
//! JIT-compiled via kornia's `CudaKernel::compile_many` (arch auto-detected) and
//! launched with explicit configs through `CudaLaunchBuilder::launch_cfg`.

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;
use std::sync::Arc;

use vrt::cuda::{cfg_1d, cfg_2d, cfg_per_item};

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
}

// ── Kernel source ─────────────────────────────────────────────────────────────

const KERNELS_SRC: &str = r#"
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
   Launch config: grid=(K,1,1), block=(64,1,1). */
extern "C" __global__ void xfeat_sample_descs(
    const float* __restrict__ desc_map,
    const float* __restrict__ kpts,
    float* __restrict__ descs_out,
    int Hd, int Wd,
    int H,  int W
) {
    int k = blockIdx.x;
    int c = threadIdx.x;

    float px = __ldg(&kpts[k * 2 + 0]);
    float py = __ldg(&kpts[k * 2 + 1]);
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

    descs_out[k * 64 + c] = val;
}

/* xfeat_l2_norm — in-place L2-normalise each 64-D descriptor row.
   block_dim=64 = exactly 2 warps; uses 2-element shared memory for cross-warp sum. */
extern "C" __global__ void xfeat_l2_norm(
    float* __restrict__ descs,
    int K
) {
    int k = blockIdx.x;
    int c = threadIdx.x;
    if (k >= K) return;

    __shared__ float shmem[2];

    float v = descs[k * 64 + c];
    float s = v * v;
    s += __shfl_down_sync(0xFFFFFFFF, s, 16);
    s += __shfl_down_sync(0xFFFFFFFF, s,  8);
    s += __shfl_down_sync(0xFFFFFFFF, s,  4);
    s += __shfl_down_sync(0xFFFFFFFF, s,  2);
    s += __shfl_down_sync(0xFFFFFFFF, s,  1);
    if (c ==  0) shmem[0] = s;
    if (c == 32) shmem[1] = s;
    __syncthreads();
    float norm = sqrtf(shmem[0] + shmem[1]);
    if (norm < 1e-8f) norm = 1e-8f;
    descs[k * 64 + c] = v / norm;
}

/* xfeat_match_argmax — argmax dot-product search: Q[t] → nearest in R.
   Direction-agnostic: call once with (D0, D1) and once with (D1, D0).

   One THREAD per query (not one block): the 64-D query lives in registers
   and reference descriptors stream through a shared-memory tile, so the
   inner loop is a pure unrolled MAC chain with no per-candidate barrier.
   (The previous one-block-per-query version spent ~10ms at K=4096 on two
   __syncthreads per candidate; this shape is compute/bandwidth bound.)

   Launch: grid = ceil(Nq/128), block = 128. Shared: 64×64 floats (16 KB).
   sim_out may be NULL (the reverse direction doesn't need similarities). */
#define MATCH_BLOCK 128
#define MATCH_TILE   64
extern "C" __global__ void xfeat_match_argmax(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    int*   __restrict__ match_out,
    float* __restrict__ sim_out,
    int Nq, int Nr
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;

    float q[64];
    if (qi < Nq) {
        #pragma unroll
        for (int c = 0; c < 64; c++) q[c] = __ldg(&Q[qi * 64 + c]);
    }

    __shared__ float tile[MATCH_TILE][64];

    int   best_j = 0;
    float best_s = -1e30f;

    for (int j0 = 0; j0 < Nr; j0 += MATCH_TILE) {
        int jt = min(MATCH_TILE, Nr - j0);

        /* Cooperative, coalesced tile load (rows of R are contiguous). */
        for (int idx = threadIdx.x; idx < jt * 64; idx += MATCH_BLOCK) {
            tile[idx >> 6][idx & 63] = __ldg(&R[j0 * 64 + idx]);
        }
        __syncthreads();

        if (qi < Nq) {
            for (int j = 0; j < jt; j++) {
                float s = 0.0f;
                /* All threads read the same tile row in lockstep → broadcast. */
                #pragma unroll
                for (int c = 0; c < 64; c++) s += q[c] * tile[j][c];
                if (s > best_s) { best_s = s; best_j = j0 + j; }
            }
        }
        __syncthreads();
    }

    if (qi < Nq) {
        match_out[qi] = best_j;
        if (sim_out) sim_out[qi] = best_s;
    }
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
    float* __restrict__ kpts_xy,         /* [K*2] (x,y) */
    float* __restrict__ scores_out,      /* [K] */
    int*   __restrict__ count,
    int H, int W, int K
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W) return;
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

/// Output of one XFeat extraction — **entirely on the GPU**.
///
/// All three buffers have capacity `top_k`; [`count`](Self::count) is the valid
/// keypoint count (the only host-side scalar). They share the same (GPU-select,
/// atomic-append) order. Nothing is downloaded — descriptor matching stays on
/// device ([`XFeatPostproc::match_mutual_nn_gpu`]); a consumer that needs pixel
/// coordinates or scores on the host downloads them explicitly with
/// [`kpts_to_host`](Self::kpts_to_host) / [`scores_to_host`](Self::scores_to_host).
pub struct XFeatResult {
    /// Pixel-space (x, y) coordinates on device, capacity [top_k × 2].
    pub kpts: CudaSlice<f32>,
    /// L2-normalised 64-D descriptors on device, capacity [top_k × 64].
    pub descs: CudaSlice<f32>,
    /// Combined NMS scores on device, capacity [top_k].
    pub scores: CudaSlice<f32>,
    /// Number of valid keypoints (≤ top_k) — bounds any access to the buffers.
    pub count: usize,
}

impl XFeatResult {
    /// Valid keypoint count.
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Download the valid keypoints to host: interleaved `[x0,y0,x1,y1,…]`,
    /// length `count × 2`. Call only when you actually need pixel coordinates on
    /// the CPU (e.g. drawing) — descriptor matching stays on device.
    pub fn kpts_to_host(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>, cudarc::driver::DriverError> {
        stream.clone_dtoh(&self.kpts.slice(0..self.count * 2))
    }

    /// Download the valid scores to host (length `count`).
    pub fn scores_to_host(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>, cudarc::driver::DriverError> {
        stream.clone_dtoh(&self.scores.slice(0..self.count))
    }
}

// ── XFeatPostproc ─────────────────────────────────────────────────────────────

/// Device buffers carrying one frame's GPU-selected keypoints from the async
/// launch (`launch_topk`) to the post-sync read (`finish_topk`).
///
/// All buffers have **capacity `top_k`**; the valid keypoint count is read from
/// the post-processor's reused pinned count buffer (the only D2H) in `finish_topk`.
pub struct TopkBufs {
    kpts_dev: CudaSlice<f32>,   // [top_k * 2]
    descs_dev: CudaSlice<f32>,  // [top_k * 64]
    scores_dev: CudaSlice<f32>, // [top_k]
    top_k: usize,
}

pub struct XFeatPostproc {
    fn_score_nms: CudaKernel,
    fn_sample_descs: CudaKernel,
    fn_l2_norm: CudaKernel,
    fn_match_argmax: CudaKernel,
    fn_histogram: CudaKernel,
    fn_cutoff: CudaKernel,
    fn_select: CudaKernel,
    stream: Arc<CudaStream>,
    top_k: usize,
    threshold: f32,
    // Pinned host buffer for the count scalar (the only D2H), reused each frame.
    count_pin: vrt::PinnedBuffer<i32>, // [1]
}

const TOPK_NBINS: usize = 1024;

impl XFeatPostproc {
    /// The CUDA stream used for all GPU work.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Compile all CUDA kernels and return a ready post-processor.
    pub fn new(stream: Arc<CudaStream>, top_k: usize, threshold: f32) -> Result<Self, XFeatError> {
        // Compile the kernel suite once; load all seven functions from the module.
        let names = [
            "xfeat_score_nms",
            "xfeat_sample_descs",
            "xfeat_l2_norm",
            "xfeat_match_argmax",
            "xfeat_topk_histogram",
            "xfeat_topk_cutoff",
            "xfeat_topk_select",
        ];
        let [fn_score_nms, fn_sample_descs, fn_l2_norm, fn_match_argmax,
             fn_histogram, fn_cutoff, fn_select]: [CudaKernel; 7] =
            CudaKernel::compile_many(stream.context(), KERNELS_SRC, &names)?
                .try_into().unwrap_or_else(|_| unreachable!("compile_many returns names.len() kernels"));

        let count_pin = vrt::PinnedBuffer::<i32>::alloc(1)?;

        Ok(Self {
            fn_score_nms,
            fn_sample_descs,
            fn_l2_norm,
            fn_match_argmax,
            fn_histogram,
            fn_cutoff,
            fn_select,
            stream,
            top_k,
            threshold,
            count_pin,
        })
    }

    /// Enqueue the NMS score kernel into `score_dev` (async — caller must sync before reading).
    ///
    /// `score_dev` must be pre-allocated with `h * w` f32 elements.
    pub fn launch_score_nms(
        &self,
        heat_ptr: *const f32,
        rel_ptr: *const f32,
        score_dev: &CudaSlice<f32>,
        h: usize,
        w: usize,
    ) -> Result<(), XFeatError> {
        use cudarc::driver::DevicePtr;
        let heat_raw: CUdeviceptr = heat_ptr as usize as CUdeviceptr;
        let rel_raw: CUdeviceptr = rel_ptr as usize as CUdeviceptr;
        let score_raw: CUdeviceptr = score_dev.device_ptr(self.stream.as_ref()).0;

        let cfg = cfg_2d(w, h);
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

    /// Launch the entire top-K + descriptor postproc **asynchronously** — GPU
    /// histogram-cutoff top-K, descriptor sampling, L2-norm, and async D2H of
    /// the host-side results — with **no `stream.synchronize()`**.
    ///
    /// The NMS score map must already be in `score_dev` (see [`launch_score_nms`]).
    /// The returned [`TopkBufs`] owns the device buffers; the caller syncs the
    /// stream once and then calls [`finish_topk`] to read the count.
    ///
    /// **Only one frame may be outstanding at a time.** The keypoint count is
    /// staged through a single reused pinned buffer, so a second `launch_topk`
    /// before the first's `finish_topk` overwrites the first frame's count.
    /// `process_topk_sample` (and `XFeat::run`) are serial and safe; if you drive
    /// the split API yourself, sync + `finish_topk` one frame before launching
    /// the next.
    ///
    /// [`launch_score_nms`]: XFeatPostproc::launch_score_nms
    /// [`finish_topk`]: XFeatPostproc::finish_topk
    pub fn launch_topk(
        &mut self,
        desc_ptr: *const f32,
        score_dev: &CudaSlice<f32>,
        h: usize,
        w: usize,
    ) -> Result<TopkBufs, XFeatError> {
        use cudarc::driver::DevicePtr;
        let (hd, wd) = (h / 8, w / 8);
        let n_pixels = h * w;
        let k = self.top_k;

        // Per-frame scratch + outputs (counts/cutoff zeroed; kpts zeroed so the
        // unused [count..k) tail samples at (0,0) instead of garbage coords).
        let hist_dev: CudaSlice<i32> = self.stream.alloc_zeros(TOPK_NBINS)?;
        let cutoff_dev: CudaSlice<f32> = self.stream.alloc_zeros(1)?;
        let count_dev: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let kpts_dev: CudaSlice<f32> = self.stream.alloc_zeros(k * 2)?;
        let scores_dev: CudaSlice<f32> = self.stream.alloc_zeros(k)?;
        let descs_dev: CudaSlice<f32> = unsafe { self.stream.alloc(k * 64)? };

        let raw = |s: &CudaSlice<f32>| -> CUdeviceptr { s.device_ptr(self.stream.as_ref()).0 };
        let raw_i = |s: &CudaSlice<i32>| -> CUdeviceptr { s.device_ptr(self.stream.as_ref()).0 };

        let score_raw = score_dev.device_ptr(self.stream.as_ref()).0;
        let hist_raw = raw_i(&hist_dev);
        let cut_raw = raw(&cutoff_dev);
        let cnt_raw = raw_i(&count_dev);
        let kxy_raw = raw(&kpts_dev);
        let sco_raw = raw(&scores_dev);
        let total = n_pixels as i32;
        let k_i = k as i32;
        let h_i = h as i32;
        let w_i = w as i32;

        // 1. histogram of survivor scores
        self.fn_histogram
            .launch_builder(&self.stream)
            .arg(&score_raw)
            .arg(&hist_raw)
            .arg(&total)
            .launch_cfg(cfg_1d(n_pixels, 256))?;
        // 2. find the score cutoff for ~K survivors (one block, one thread)
        self.fn_cutoff
            .launch_builder(&self.stream)
            .arg(&hist_raw)
            .arg(&k_i)
            .arg(&cut_raw)
            .launch_cfg(cfg_1d(1, 1))?;
        // 3. gather survivors >= cutoff, capped at K
        self.fn_select
            .launch_builder(&self.stream)
            .arg(&score_raw)
            .arg(&cut_raw)
            .arg(&kxy_raw)
            .arg(&sco_raw)
            .arg(&cnt_raw)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&k_i)
            .launch_cfg(cfg_1d(n_pixels, 256))?;
        // 4. sample 64-D descriptors at the selected keypoints (cap K; the
        //    unused tail samples at (0,0) and is ignored by `finish_topk`).
        let desc_raw = desc_ptr as usize as CUdeviceptr;
        let descs_raw = raw(&descs_dev);
        let hd_i = hd as i32;
        let wd_i = wd as i32;
        let cfg64 = cfg_per_item(k, 64);
        self.fn_sample_descs
            .launch_builder(&self.stream)
            .arg(&desc_raw)
            .arg(&kxy_raw)
            .arg(&descs_raw)
            .arg(&hd_i)
            .arg(&wd_i)
            .arg(&h_i)
            .arg(&w_i)
            .launch_cfg(cfg64)?;
        // 5. L2-normalise each descriptor row in place
        self.fn_l2_norm
            .launch_builder(&self.stream)
            .arg(&descs_raw)
            .arg(&k_i)
            .launch_cfg(cfg64)?;

        // 6. async D2H of the count scalar (the ONLY host transfer) into the
        //    reused pinned buffer — pinned host memory makes cudaMemcpyAsync
        //    truly asynchronous, so the host thread is free until the sync.
        //    Keypoints/descs/scores stay on device (downloaded only on demand).
        let vstream = vrt::Stream::from_cuda_stream(self.stream.clone());
        unsafe {
            vstream.memcpy_d2h_raw(
                self.count_pin.as_mut_ptr() as *mut u8,
                cnt_raw as usize as *const _,
                std::mem::size_of::<i32>(),
            )?;
        }

        Ok(TopkBufs {
            kpts_dev,
            descs_dev,
            scores_dev,
            top_k: k,
        })
    }

    /// Assemble the final [`XFeatResult`] from [`TopkBufs`] **after the stream
    /// has been synced** (the async count D2H is then done).
    ///
    /// All buffers stay on device; only the keypoint `count` is read here (from
    /// the pinned scalar). Consumers use `result.count` (or `result.len()`).
    pub fn finish_topk(&self, bufs: TopkBufs) -> XFeatResult {
        let count = (self.count_pin.as_slice()[0].max(0) as usize).min(bufs.top_k);
        XFeatResult {
            kpts: bufs.kpts_dev,
            descs: bufs.descs_dev,
            scores: bufs.scores_dev,
            count,
        }
    }

    /// Synchronous one-shot: [`launch_topk`] + a single sync + [`finish_topk`].
    ///
    /// The convenience used by `XFeat::run`; callers that overlap work can instead
    /// drive `launch_topk` / sync / `finish_topk` themselves.
    ///
    /// [`launch_topk`]: XFeatPostproc::launch_topk
    /// [`finish_topk`]: XFeatPostproc::finish_topk
    pub fn process_topk_sample(
        &mut self,
        desc_ptr: *const f32,
        score_dev: &CudaSlice<f32>,
        h: usize,
        w: usize,
    ) -> Result<XFeatResult, XFeatError> {
        let bufs = self.launch_topk(desc_ptr, score_dev, h, w)?;
        self.stream.synchronize()?;
        Ok(self.finish_topk(bufs))
    }

    /// Run the full post-processing pipeline (NMS → top-K → sample → L2-norm).
    ///
    /// * `desc_ptr` — device pointer, shape `(1, 64, H/8, W/8)` CHW FP32
    /// * `heat_ptr` — device pointer, shape `(1, 1, H, W)` FP32
    /// * `rel_ptr`  — device pointer, shape `(1, 1, H, W)` FP32
    /// * `h`, `w`   — backbone input dimensions (multiples of 32)
    pub fn process(
        &mut self,
        desc_ptr: *const f32,
        heat_ptr: *const f32,
        rel_ptr: *const f32,
        h: usize,
        w: usize,
    ) -> Result<XFeatResult, XFeatError> {
        let score_dev: CudaSlice<f32> = unsafe { self.stream.alloc(h * w)? };
        self.launch_score_nms(heat_ptr, rel_ptr, &score_dev, h, w)?;
        // launch_score_nms and process_topk_sample's kernels share the stream
        // (ordered) — no intermediate sync needed; the single sync is inside.
        self.process_topk_sample(desc_ptr, &score_dev, h, w)
    }

    /// GPU mutual nearest-neighbour matching between two `XFeatResult`s.
    ///
    /// Descriptors already live on device — no re-upload.
    /// Returns pairs `(i, j)` where keypoint `i` from `res0` matches `j` from `res1`.
    pub fn match_mutual_nn_gpu(
        &self,
        res0: &XFeatResult,
        res1: &XFeatResult,
        min_cossim: f32,
    ) -> Result<Vec<(usize, usize)>, XFeatError> {
        let n0 = res0.count;
        let n1 = res1.count;
        if n0 == 0 || n1 == 0 {
            return Ok(Vec::new());
        }

        let match12_dev: CudaSlice<i32> = unsafe { self.stream.alloc(n0)? };
        let match21_dev: CudaSlice<i32> = unsafe { self.stream.alloc(n1)? };
        let sim12_dev: CudaSlice<f32> = unsafe { self.stream.alloc(n0)? };

        let n0_i = n0 as i32;
        let n1_i = n1 as i32;

        let d0_raw: CUdeviceptr = {
            use cudarc::driver::DevicePtr;
            res0.descs.device_ptr(self.stream.as_ref()).0
        };
        let d1_raw: CUdeviceptr = {
            use cudarc::driver::DevicePtr;
            res1.descs.device_ptr(self.stream.as_ref()).0
        };
        let m12_raw: CUdeviceptr = {
            use cudarc::driver::DevicePtr;
            match12_dev.device_ptr(self.stream.as_ref()).0
        };
        let m21_raw: CUdeviceptr = {
            use cudarc::driver::DevicePtr;
            match21_dev.device_ptr(self.stream.as_ref()).0
        };
        let s12_raw: CUdeviceptr = {
            use cudarc::driver::DevicePtr;
            sim12_dev.device_ptr(self.stream.as_ref()).0
        };
        let null_sim: CUdeviceptr = 0;

        // One tiled argmax kernel, both directions (sim only needed for 1→2).
        // Block size must match MATCH_BLOCK in the kernel source.
        self.fn_match_argmax
            .launch_builder(&self.stream)
            .arg(&d0_raw)
            .arg(&d1_raw)
            .arg(&m12_raw)
            .arg(&s12_raw)
            .arg(&n0_i)
            .arg(&n1_i)
            .launch_cfg(cfg_1d(n0, 128))?;
        self.fn_match_argmax
            .launch_builder(&self.stream)
            .arg(&d1_raw)
            .arg(&d0_raw)
            .arg(&m21_raw)
            .arg(&null_sim)
            .arg(&n1_i)
            .arg(&n0_i)
            .launch_cfg(cfg_1d(n1, 128))?;

        self.stream.synchronize()?;
        let match12: Vec<i32> = self.stream.clone_dtoh(&match12_dev)?;
        let match21: Vec<i32> = self.stream.clone_dtoh(&match21_dev)?;
        let sim12: Vec<f32> = self.stream.clone_dtoh(&sim12_dev)?;

        let pairs = (0..n0)
            .filter(|&i| {
                let j = match12[i] as usize;
                match21[j] as usize == i && sim12[i] >= min_cossim
            })
            .map(|i| (i, match12[i] as usize))
            .collect();

        Ok(pairs)
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    /// Independent CPU mutual nearest-neighbour reference (O(n²×64)) — the oracle
    /// the GPU tiled-argmax kernel is validated against. Test-only; the shipped
    /// crate does matching on the GPU (`XFeatPostproc::match_mutual_nn_gpu`).
    /// Descriptors must be L2-normalised; cosim = dot product.
    fn cpu_match_reference(descs0: &[f32], descs1: &[f32], min_cossim: f32) -> Vec<(usize, usize)> {
        const D: usize = 64;
        let n0 = descs0.len() / D;
        let n1 = descs1.len() / D;
        if n0 == 0 || n1 == 0 {
            return Vec::new();
        }

        let mut d1t = vec![0.0f32; D * n1];
        for j in 0..n1 {
            for d in 0..D {
                d1t[d * n1 + j] = descs1[j * D + d];
            }
        }

        let mut match12 = vec![0usize; n0];
        let mut sim12 = vec![f32::NEG_INFINITY; n0];
        for i in 0..n0 {
            let d0 = &descs0[i * D..(i + 1) * D];
            for j in 0..n1 {
                let mut s = 0.0f32;
                for d in 0..D {
                    s += d0[d] * d1t[d * n1 + j];
                }
                if s > sim12[i] {
                    sim12[i] = s;
                    match12[i] = j;
                }
            }
        }

        let mut match21 = vec![0usize; n1];
        let mut sim21 = vec![f32::NEG_INFINITY; n1];
        for j in 0..n1 {
            let d1 = &descs1[j * D..(j + 1) * D];
            for i in 0..n0 {
                let d0 = &descs0[i * D..(i + 1) * D];
                let s: f32 = d0.iter().zip(d1).map(|(a, b)| a * b).sum();
                if s > sim21[j] {
                    sim21[j] = s;
                    match21[j] = i;
                }
            }
        }

        (0..n0)
            .filter(|&i| match21[match12[i]] == i && sim12[i] >= min_cossim)
            .map(|i| (i, match12[i]))
            .collect()
    }

    /// Deterministic pseudo-random L2-normalized descriptors (LCG, no deps).
    fn random_descs(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mut v: Vec<f32> = (0..n * 64).map(|_| next()).collect();
        for row in v.chunks_exact_mut(64) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            row.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    /// GPU tiled-argmax matching must agree with the CPU reference.
    /// Needs the Jetson GPU; run explicitly:
    ///   cargo test -p trt-xfeat -- --ignored
    #[test]
    #[ignore]
    fn gpu_match_agrees_with_cpu_reference() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let pp = XFeatPostproc::new(stream.clone(), 4096, 0.05).unwrap();

        for (n0, n1) in [(4096usize, 4096usize), (1000, 3000), (1, 4096), (130, 1)] {
            let h0 = random_descs(n0, 42);
            let h1 = random_descs(n1, 7);

            let r0 = XFeatResult {
                kpts: stream.clone_htod(&vec![0.0f32; n0 * 2]).unwrap(),
                descs: stream.clone_htod(&h0).unwrap(),
                scores: stream.clone_htod(&vec![1.0f32; n0]).unwrap(),
                count: n0,
            };
            let r1 = XFeatResult {
                kpts: stream.clone_htod(&vec![0.0f32; n1 * 2]).unwrap(),
                descs: stream.clone_htod(&h1).unwrap(),
                scores: stream.clone_htod(&vec![1.0f32; n1]).unwrap(),
                count: n1,
            };

            // Warm-up (first launch pays module/alloc setup), then timed run.
            let _ = pp.match_mutual_nn_gpu(&r0, &r1, -1.0).unwrap();
            let t0 = std::time::Instant::now();
            let gpu = pp.match_mutual_nn_gpu(&r0, &r1, -1.0).unwrap();
            let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let cpu = cpu_match_reference(&h0, &h1, -1.0);

            let gset: std::collections::HashSet<_> = gpu.iter().copied().collect();
            let cset: std::collections::HashSet<_> = cpu.iter().copied().collect();
            assert_eq!(gset, cset, "GPU/CPU match mismatch at n0={n0} n1={n1}");
            eprintln!(
                "match n0={n0:5} n1={n1:5}: {} pairs, GPU wall {gpu_ms:.2} ms",
                gpu.len()
            );
        }
    }

    /// Kernel-only timing: pre-allocated buffers, CUDA-event bracketed,
    /// averaged over 20 launches.  Run: cargo test -p trt-xfeat --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_match_kernel_only_timing() {
        use cudarc::driver::DevicePtr;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let pp = XFeatPostproc::new(stream.clone(), 4096, 0.05).unwrap();

        let n = 4096usize;
        let d0 = stream.clone_htod(&random_descs(n, 42)).unwrap();
        let d1 = stream.clone_htod(&random_descs(n, 7)).unwrap();
        let m12: CudaSlice<i32> = unsafe { stream.alloc(n).unwrap() };
        let s12: CudaSlice<f32> = unsafe { stream.alloc(n).unwrap() };

        let d0r: CUdeviceptr = d0.device_ptr(stream.as_ref()).0;
        let d1r: CUdeviceptr = d1.device_ptr(stream.as_ref()).0;
        let mr: CUdeviceptr = m12.device_ptr(stream.as_ref()).0;
        let sr: CUdeviceptr = s12.device_ptr(stream.as_ref()).0;
        let n_i = n as i32;

        let launch = || {
            pp.fn_match_argmax
                .launch_builder(&stream)
                .arg(&d0r)
                .arg(&d1r)
                .arg(&mr)
                .arg(&sr)
                .arg(&n_i)
                .arg(&n_i)
                .launch_cfg(vrt::cuda::cfg_1d(n, 128))
                .unwrap();
        };

        launch();
        stream.synchronize().unwrap(); // warm-up

        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = stream.record_event(flags).unwrap();
        for _ in 0..20 {
            launch();
        }
        let stop = stream.record_event(flags).unwrap();
        stream.synchronize().unwrap();
        let ms = start.elapsed_ms(&stop).unwrap() / 20.0;
        eprintln!("match_argmax kernel-only @ {n}x{n}: {ms:.3} ms/direction");
    }
}

#[cfg(test)]
mod gpu_compact_tests {
    use super::*;

    /// GPU top-K must select the right keypoints from a synthetic score map and
    /// produce L2-normalized descriptors.  The GPU `select` gathers via atomic
    /// append, so the output order is unspecified — assertions are order-free.
    #[test]
    #[ignore]
    fn gpu_topk_selects_correct_keypoints() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut pp = XFeatPostproc::new(stream.clone(), 2, 0.05).unwrap(); // top_k = 2

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

        let res = pp.process_topk_sample(desc_ptr, &score_dev, h, w).unwrap();

        // Exactly the top-2 keypoints, in any order: pair (score, x, y) and sort.
        assert_eq!(res.count, 2);
        let scores = res.scores_to_host(&stream).unwrap();
        let kpts = res.kpts_to_host(&stream).unwrap();
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
        for row in descs.chunks_exact(64).take(res.count) {
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
}
