//! XFeat post-processing: NMS → TopK → descriptor sampling → L2-norm.
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
/// device (see the `matching::Matcher`); a consumer that needs pixel
/// coordinates or scores on the host downloads them explicitly with
/// [`kpts_to_host`](Self::kpts_to_host) / [`scores_to_host`](Self::scores_to_host).
pub struct XFeatResult {
    /// Device (x, y) coordinates in **model space** (the floor-32 backbone input),
    /// capacity [top_k × 2]. Multiply by [`scale`](Self::scale) for original pixels.
    pub kpts: CudaSlice<f32>,
    /// L2-normalised 64-D descriptors on device, capacity [top_k × 64].
    pub descs: CudaSlice<f32>,
    /// Combined NMS scores on device, capacity [top_k].
    pub scores: CudaSlice<f32>,
    /// Number of valid keypoints (≤ top_k) — bounds any access to the buffers.
    pub count: usize,
    /// Per-axis model→original scale `(rw, rh)` = `(src_w/model_w, src_h/model_h)`
    /// (matches upstream XFeat's `mkpts * [rw, rh]`). `(1.0, 1.0)` if unset.
    /// Applied to `kpts` on host readout; descriptors/matching are unaffected.
    pub scale: (f32, f32),
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
    /// length `count × 2`, in **original image pixels** ([`scale`](Self::scale)
    /// applied). Call only when you actually need coordinates on the CPU (e.g.
    /// drawing) — descriptor matching stays on device.
    pub fn kpts_to_host(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>, cudarc::driver::DriverError> {
        let mut xy = stream.clone_dtoh(&self.kpts.slice(0..self.count * 2))?;
        let (sx, sy) = self.scale;
        if (sx, sy) != (1.0, 1.0) {
            for p in xy.chunks_exact_mut(2) {
                p[0] *= sx;
                p[1] *= sy;
            }
        }
        Ok(xy)
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
/// All device buffers have **capacity `top_k`**; the valid keypoint count is
/// staged into this frame's **own** pinned buffer (the only D2H) and read in
/// `finish_topk`. Because the count buffer is per-frame (not shared), multiple
/// frames may be outstanding on the stream before a single sync.
pub struct TopkBufs {
    kpts_dev: CudaSlice<f32>,          // [top_k * 2]
    descs_dev: CudaSlice<f32>,         // [top_k * 64]
    scores_dev: CudaSlice<f32>,        // [top_k]
    count_pin: vrt::PinnedBuffer<i32>, // [1] — this frame's async count D2H target
    top_k: usize,
}

pub struct XFeatPostproc {
    fn_score_nms: CudaKernel,
    fn_sample_descs: CudaKernel,
    fn_l2_norm: CudaKernel,
    fn_histogram: CudaKernel,
    fn_cutoff: CudaKernel,
    fn_select: CudaKernel,
    stream: Arc<CudaStream>,
    top_k: usize,
    threshold: f32,
}

const TOPK_NBINS: usize = 1024;

impl XFeatPostproc {
    /// The CUDA stream used for all GPU work.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Compile all CUDA kernels and return a ready post-processor.
    pub fn new(stream: Arc<CudaStream>, top_k: usize, threshold: f32) -> Result<Self, XFeatError> {
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
            CudaKernel::compile_many(stream.context(), KERNELS_SRC, &names)?
                .try_into().unwrap_or_else(|_| unreachable!("compile_many returns names.len() kernels"));

        Ok(Self {
            fn_score_nms,
            fn_sample_descs,
            fn_l2_norm,
            fn_histogram,
            fn_cutoff,
            fn_select,
            stream,
            top_k,
            threshold,
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
    /// The returned [`TopkBufs`] own **per-frame** buffers (including the pinned
    /// count target), so several frames may be outstanding on the stream before a
    /// single sync — submit multiple, sync once, then `finish_topk` each.
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

        // 6. async D2H of the count scalar (the ONLY host transfer) into THIS
        //    frame's pinned buffer — pinned host memory makes cudaMemcpyAsync
        //    truly asynchronous, so the host thread is free until the sync. A
        //    per-frame buffer (not shared) lets several frames stay outstanding.
        //    Keypoints/descs/scores stay on device (downloaded only on demand).
        let mut count_pin = vrt::PinnedBuffer::<i32>::alloc(1)?;
        let vstream = vrt::Stream::from_cuda_stream(self.stream.clone());
        unsafe {
            vstream.memcpy_d2h_raw(
                count_pin.as_mut_ptr() as *mut u8,
                cnt_raw as usize as *const _,
                std::mem::size_of::<i32>(),
            )?;
        }

        Ok(TopkBufs {
            kpts_dev,
            descs_dev,
            scores_dev,
            count_pin,
            top_k: k,
        })
    }

    /// Assemble the final [`XFeatResult`] from [`TopkBufs`] **after the stream
    /// has been synced** (the async count D2H is then done).
    ///
    /// All buffers stay on device; only the keypoint `count` is read here (from
    /// the pinned scalar). Consumers use `result.count` (or `result.len()`).
    pub fn finish_topk(&self, bufs: TopkBufs) -> XFeatResult {
        let count = (bufs.count_pin.as_slice()[0].max(0) as usize).min(bufs.top_k);
        XFeatResult {
            kpts: bufs.kpts_dev,
            descs: bufs.descs_dev,
            scores: bufs.scores_dev,
            count,
            scale: (1.0, 1.0), // caller (XFeat::run) stamps the real model→src scale
        }
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

        let bufs = pp.launch_topk(desc_ptr, &score_dev, h, w).unwrap();
        stream.synchronize().unwrap();
        let res = pp.finish_topk(bufs);

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
