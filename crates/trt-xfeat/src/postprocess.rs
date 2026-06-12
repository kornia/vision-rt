//! XFeat post-processing: NMS → TopK → descriptor sampling → L2-norm + GPU matching.
//!
//! Works with the TRT backbone engine that outputs three tensors:
//!   `descriptors`  (1, 64, H/8, W/8)  — dense feature maps (FP32 on device)
//!   `heatmap`      (1,  1,   H,   W)  — keypoint confidence (FP32 on device)
//!   `reliability`  (1,  1,   H,   W)  — channel reliability  (FP32 on device)
//!
//! Pipeline:
//!   GPU  xfeat_score_nms      → score_map (H×W), masked to local-max pixels above threshold
//!   GPU  xfeat_compact_scores → stream-compact survivors (D2H ∝ survivors, not H×W)
//!   CPU  TopK select          → top-K flat indices (sorted by score descending)
//!   GPU  xfeat_sample_descs   → K×64 descriptor vectors (bilinear sample from desc_map)
//!   GPU  xfeat_l2_norm        → in-place L2 normalise
//!   GPU  xfeat_match_argmax   → tiled mutual-NN matching (two calls, swapped args)
//!
//! Kernels are JIT-compiled via trt::cuda::Kernels (arch auto-detected).

use std::sync::Arc;
use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use cudarc::driver::sys::CUdeviceptr;

use trt::cuda::{Kernels, cfg_2d, cfg_per_item};

/// Errors from XFeat post-processing and matching.
#[derive(Debug, thiserror::Error)]
pub enum XFeatError {
    #[error(transparent)]
    Trt(#[from] trt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("backbone output '{0}' missing from engine")]
    MissingOutput(&'static str),
    #[error("XFeat: finalize called before enqueue")]
    FinalizeBeforeEnqueue,
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

    int x1 = min(x0 + 1, Wd - 1);
    int y1 = min(y0 + 1, Hd - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

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
   Appends (score, flat_index) of every score > 0 via an atomic counter, so
   the host copies only survivors (tens of KB) instead of the full H×W map
   (3.7 MB at 1280×736). Output order is nondeterministic — the host top-K
   sorts anyway. Capacity equals the map size, so no overflow is possible. */
extern "C" __global__ void xfeat_compact_scores(
    const float* __restrict__ score_map,
    float* __restrict__ out_scores,
    int*   __restrict__ out_idx,
    int*   __restrict__ count,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float s = __ldg(&score_map[i]);
    if (s <= 0.0f) return;
    int slot = atomicAdd(count, 1);
    out_scores[slot] = s;
    out_idx[slot]    = i;
}
"#;

// ── Public types ──────────────────────────────────────────────────────────────

/// Output of one XFeat extraction.
pub struct XFeatResult {
    /// Pixel-space (x, y) coordinates on device, shape [count × 2].
    pub kpts:     CudaSlice<f32>,
    /// L2-normalised 64-D descriptors on device, shape [count × 64].
    pub descs:    CudaSlice<f32>,
    /// Combined NMS scores on host, length [count].
    pub scores:   Vec<f32>,
    /// Pixel-space (x, y) coordinates on host — flat interleaved `[x0,y0,x1,y1,…]`.
    /// Same ordering as `scores`. Zero-cost: kept from the top-K selection step.
    pub kpts_cpu: Vec<f32>,
}

// ── XFeatPostproc ─────────────────────────────────────────────────────────────

pub struct XFeatPostproc {
    fn_score_nms:      cudarc::driver::CudaFunction,
    fn_sample_descs:   cudarc::driver::CudaFunction,
    fn_l2_norm:        cudarc::driver::CudaFunction,
    fn_match_argmax:   cudarc::driver::CudaFunction,
    fn_compact_scores: cudarc::driver::CudaFunction,
    stream:            Arc<CudaStream>,
    top_k:             usize,
    threshold:         f32,
}

impl XFeatPostproc {
    /// The CUDA stream used for all GPU work.
    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }

    /// Compile all CUDA kernels and return a ready post-processor.
    pub fn new(
        stream:    Arc<CudaStream>,
        top_k:     usize,
        threshold: f32,
    ) -> Result<Self, XFeatError> {
        let kernels = Kernels::compile(stream.clone(), KERNELS_SRC)?;

        let fn_score_nms      = kernels.function("xfeat_score_nms")?;
        let fn_sample_descs   = kernels.function("xfeat_sample_descs")?;
        let fn_l2_norm        = kernels.function("xfeat_l2_norm")?;
        let fn_match_argmax   = kernels.function("xfeat_match_argmax")?;
        let fn_compact_scores = kernels.function("xfeat_compact_scores")?;

        Ok(Self { fn_score_nms, fn_sample_descs, fn_l2_norm, fn_match_argmax,
                  fn_compact_scores, stream, top_k, threshold })
    }

    /// Enqueue the NMS score kernel into `score_dev` (async — caller must sync before reading).
    ///
    /// `score_dev` must be pre-allocated with `h * w` f32 elements.
    pub fn launch_score_nms(
        &self,
        heat_ptr:  *const f32,
        rel_ptr:   *const f32,
        score_dev: &CudaSlice<f32>,
        h:         usize,
        w:         usize,
    ) -> Result<(), XFeatError> {
        use cudarc::driver::DevicePtr;
        let heat_raw:  CUdeviceptr = heat_ptr  as usize as CUdeviceptr;
        let rel_raw:   CUdeviceptr = rel_ptr   as usize as CUdeviceptr;
        let score_raw: CUdeviceptr = score_dev.device_ptr(self.stream.as_ref()).0;

        let cfg = cfg_2d(w, h);
        let h_i = h as i32;
        let w_i = w as i32;
        let thr = self.threshold;
        unsafe {
            self.stream.launch_builder(&self.fn_score_nms)
                .arg(&heat_raw).arg(&rel_raw).arg(&score_raw)
                .arg(&h_i).arg(&w_i).arg(&thr)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Complete post-processing after the stream has been synced.
    ///
    /// GPU stream-compaction collects the NMS survivors, so the D2H copy is
    /// proportional to the survivor count (tens of KB) instead of the full
    /// H×W score map.  Then: top-K select → descriptor sampling → L2-norm.
    /// Performs internal `stream.synchronize()` calls (compaction + kernels).
    pub fn process_topk_sample(
        &self,
        desc_ptr:  *const f32,
        score_dev: &CudaSlice<f32>,
        h:         usize,
        w:         usize,
    ) -> Result<XFeatResult, XFeatError> {
        use cudarc::driver::DevicePtr;
        let hd = h / 8;
        let wd = w / 8;
        let n_pixels = h * w;

        // ── GPU: compact survivors (score > 0) ────────────────────────────────
        let count_dev: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let cs_dev:    CudaSlice<f32> = unsafe { self.stream.alloc(n_pixels)? };
        let ci_dev:    CudaSlice<i32> = unsafe { self.stream.alloc(n_pixels)? };

        {
            let score_raw: CUdeviceptr = score_dev.device_ptr(self.stream.as_ref()).0;
            let cs_raw:    CUdeviceptr = cs_dev.device_ptr(self.stream.as_ref()).0;
            let ci_raw:    CUdeviceptr = ci_dev.device_ptr(self.stream.as_ref()).0;
            let cnt_raw:   CUdeviceptr = count_dev.device_ptr(self.stream.as_ref()).0;
            let total = n_pixels as i32;
            unsafe {
                self.stream.launch_builder(&self.fn_compact_scores)
                    .arg(&score_raw).arg(&cs_raw).arg(&ci_raw).arg(&cnt_raw)
                    .arg(&total)
                    .launch(trt::cuda::cfg_1d(n_pixels, 256))?;
            }
        }
        self.stream.synchronize()?;

        let n_survivors = self.stream.memcpy_dtov(&count_dev)?[0] as usize;
        let mut candidates: Vec<(f32, u32)> = if n_survivors == 0 {
            Vec::new()
        } else {
            let cs: Vec<f32> = self.stream.memcpy_dtov(&cs_dev.slice(0..n_survivors))?;
            let ci: Vec<i32> = self.stream.memcpy_dtov(&ci_dev.slice(0..n_survivors))?;
            cs.into_iter().zip(ci).map(|(s, i)| (s, i as u32)).collect()
        };

        let k = self.top_k.min(candidates.len());
        if k == 0 {
            let empty:  CudaSlice<f32> = unsafe { self.stream.alloc(0)? };
            let empty2: CudaSlice<f32> = unsafe { self.stream.alloc(0)? };
            return Ok(XFeatResult { kpts: empty, descs: empty2, scores: Vec::new(), kpts_cpu: Vec::new() });
        }

        candidates.select_nth_unstable_by(k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
        candidates.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        let scores_out: Vec<f32> = candidates.iter().map(|(s, _)| *s).collect();
        let kpts_host: Vec<f32>  = candidates.iter()
            .flat_map(|(_, idx)| {
                let i = *idx as usize;
                [(i % w) as f32, (i / w) as f32]
            })
            .collect();

        let kpts_dev:  CudaSlice<f32> = self.stream.memcpy_stod(&kpts_host)?;
        let descs_dev: CudaSlice<f32> = unsafe { self.stream.alloc(k * 64)? };

        let desc_raw:  CUdeviceptr = desc_ptr  as usize as CUdeviceptr;
        let kpts_raw:  CUdeviceptr = kpts_dev.device_ptr(self.stream.as_ref()).0;
        let descs_raw: CUdeviceptr = descs_dev.device_ptr(self.stream.as_ref()).0;

        let hd_i = hd as i32;  let wd_i = wd as i32;
        let h_i  = h  as i32;  let w_i  = w  as i32;
        let k_i  = k as i32;
        let cfg64 = cfg_per_item(k, 64);

        unsafe {
            self.stream.launch_builder(&self.fn_sample_descs)
                .arg(&desc_raw).arg(&kpts_raw).arg(&descs_raw)
                .arg(&hd_i).arg(&wd_i).arg(&h_i).arg(&w_i)
                .launch(cfg64)?;
        }
        let descs_raw: CUdeviceptr = descs_dev.device_ptr(self.stream.as_ref()).0;
        unsafe {
            self.stream.launch_builder(&self.fn_l2_norm)
                .arg(&descs_raw).arg(&k_i)
                .launch(cfg64)?;
        }
        self.stream.synchronize()?;

        Ok(XFeatResult { kpts: kpts_dev, descs: descs_dev, scores: scores_out, kpts_cpu: kpts_host })
    }

    /// Run the full post-processing pipeline (NMS → top-K → sample → L2-norm).
    ///
    /// * `desc_ptr` — device pointer, shape `(1, 64, H/8, W/8)` CHW FP32
    /// * `heat_ptr` — device pointer, shape `(1, 1, H, W)` FP32
    /// * `rel_ptr`  — device pointer, shape `(1, 1, H, W)` FP32
    /// * `h`, `w`   — backbone input dimensions (multiples of 32)
    pub fn process(
        &self,
        desc_ptr: *const f32,
        heat_ptr: *const f32,
        rel_ptr:  *const f32,
        h: usize,
        w: usize,
    ) -> Result<XFeatResult, XFeatError> {
        let score_dev: CudaSlice<f32> = unsafe { self.stream.alloc(h * w)? };
        self.launch_score_nms(heat_ptr, rel_ptr, &score_dev, h, w)?;
        self.stream.synchronize()?;
        self.process_topk_sample(desc_ptr, &score_dev, h, w)
    }

    /// GPU mutual nearest-neighbour matching between two `XFeatResult`s.
    ///
    /// Descriptors already live on device — no re-upload.
    /// Returns pairs `(i, j)` where keypoint `i` from `res0` matches `j` from `res1`.
    pub fn match_mutual_nn_gpu(
        &self,
        res0:       &XFeatResult,
        res1:       &XFeatResult,
        min_cossim: f32,
    ) -> Result<Vec<(usize, usize)>, XFeatError> {
        let n0 = res0.scores.len();
        let n1 = res1.scores.len();
        if n0 == 0 || n1 == 0 { return Ok(Vec::new()); }

        let match12_dev: CudaSlice<i32> = unsafe { self.stream.alloc(n0)? };
        let match21_dev: CudaSlice<i32> = unsafe { self.stream.alloc(n1)? };
        let sim12_dev:   CudaSlice<f32> = unsafe { self.stream.alloc(n0)? };

        let n0_i = n0 as i32;
        let n1_i = n1 as i32;

        let d0_raw:  CUdeviceptr = { use cudarc::driver::DevicePtr; res0.descs.device_ptr(self.stream.as_ref()).0 };
        let d1_raw:  CUdeviceptr = { use cudarc::driver::DevicePtr; res1.descs.device_ptr(self.stream.as_ref()).0 };
        let m12_raw: CUdeviceptr = { use cudarc::driver::DevicePtr; match12_dev.device_ptr(self.stream.as_ref()).0 };
        let m21_raw: CUdeviceptr = { use cudarc::driver::DevicePtr; match21_dev.device_ptr(self.stream.as_ref()).0 };
        let s12_raw: CUdeviceptr = { use cudarc::driver::DevicePtr; sim12_dev.device_ptr(self.stream.as_ref()).0 };
        let null_sim: CUdeviceptr = 0;

        // One tiled argmax kernel, both directions (sim only needed for 1→2).
        // Block size must match MATCH_BLOCK in the kernel source.
        unsafe {
            self.stream.launch_builder(&self.fn_match_argmax)
                .arg(&d0_raw).arg(&d1_raw)
                .arg(&m12_raw).arg(&s12_raw)
                .arg(&n0_i).arg(&n1_i)
                .launch(trt::cuda::cfg_1d(n0, 128))?;
        }
        unsafe {
            self.stream.launch_builder(&self.fn_match_argmax)
                .arg(&d1_raw).arg(&d0_raw)
                .arg(&m21_raw).arg(&null_sim)
                .arg(&n1_i).arg(&n0_i)
                .launch(trt::cuda::cfg_1d(n1, 128))?;
        }

        self.stream.synchronize()?;
        let match12: Vec<i32> = self.stream.memcpy_dtov(&match12_dev)?;
        let match21: Vec<i32> = self.stream.memcpy_dtov(&match21_dev)?;
        let sim12:   Vec<f32> = self.stream.memcpy_dtov(&sim12_dev)?;

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

// ── CPU fallback matching ─────────────────────────────────────────────────────

/// CPU mutual nearest-neighbour matching (fallback, O(n²×64)).
///
/// Descriptors must be L2-normalised; cosim = dot product.
pub fn match_mutual_nn(
    descs0:     &[f32],
    descs1:     &[f32],
    min_cossim: f32,
) -> Vec<(usize, usize)> {
    const D: usize = 64;
    let n0 = descs0.len() / D;
    let n1 = descs1.len() / D;
    if n0 == 0 || n1 == 0 { return Vec::new(); }

    let mut d1t = vec![0.0f32; D * n1];
    for j in 0..n1 {
        for d in 0..D { d1t[d * n1 + j] = descs1[j * D + d]; }
    }

    let mut match12 = vec![0usize; n0];
    let mut sim12   = vec![f32::NEG_INFINITY; n0];
    for i in 0..n0 {
        let d0 = &descs0[i * D..(i + 1) * D];
        for j in 0..n1 {
            let mut s = 0.0f32;
            for d in 0..D { s += d0[d] * d1t[d * n1 + j]; }
            if s > sim12[i] { sim12[i] = s; match12[i] = j; }
        }
    }

    let mut match21 = vec![0usize; n1];
    let mut sim21   = vec![f32::NEG_INFINITY; n1];
    for j in 0..n1 {
        let d1 = &descs1[j * D..(j + 1) * D];
        for i in 0..n0 {
            let d0 = &descs0[i * D..(i + 1) * D];
            let s: f32 = d0.iter().zip(d1).map(|(a, b)| a * b).sum();
            if s > sim21[j] { sim21[j] = s; match21[j] = i; }
        }
    }

    (0..n0)
        .filter(|&i| match21[match12[i]] == i && sim12[i] >= min_cossim)
        .map(|i| (i, match12[i]))
        .collect()
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    /// Deterministic pseudo-random L2-normalized descriptors (LCG, no deps).
    fn random_descs(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        let ctx    = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let pp     = XFeatPostproc::new(stream.clone(), 4096, 0.05).unwrap();

        for (n0, n1) in [(4096usize, 4096usize), (1000, 3000), (1, 4096), (130, 1)] {
            let h0 = random_descs(n0, 42);
            let h1 = random_descs(n1, 7);

            let r0 = XFeatResult {
                kpts:     stream.memcpy_stod(&vec![0.0f32; n0 * 2]).unwrap(),
                descs:    stream.memcpy_stod(&h0).unwrap(),
                scores:   vec![1.0; n0],
                kpts_cpu: Vec::new(),
            };
            let r1 = XFeatResult {
                kpts:     stream.memcpy_stod(&vec![0.0f32; n1 * 2]).unwrap(),
                descs:    stream.memcpy_stod(&h1).unwrap(),
                scores:   vec![1.0; n1],
                kpts_cpu: Vec::new(),
            };

            // Warm-up (first launch pays module/alloc setup), then timed run.
            let _ = pp.match_mutual_nn_gpu(&r0, &r1, -1.0).unwrap();
            let t0 = std::time::Instant::now();
            let gpu = pp.match_mutual_nn_gpu(&r0, &r1, -1.0).unwrap();
            let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let cpu = match_mutual_nn(&h0, &h1, -1.0);

            let gset: std::collections::HashSet<_> = gpu.iter().copied().collect();
            let cset: std::collections::HashSet<_> = cpu.iter().copied().collect();
            assert_eq!(gset, cset, "GPU/CPU match mismatch at n0={n0} n1={n1}");
            eprintln!("match n0={n0:5} n1={n1:5}: {} pairs, GPU wall {gpu_ms:.2} ms", gpu.len());
        }
    }

    /// Kernel-only timing: pre-allocated buffers, CUDA-event bracketed,
    /// averaged over 20 launches.  Run: cargo test -p trt-xfeat --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_match_kernel_only_timing() {
        use cudarc::driver::DevicePtr;
        let ctx    = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let pp     = XFeatPostproc::new(stream.clone(), 4096, 0.05).unwrap();

        let n = 4096usize;
        let d0 = stream.memcpy_stod(&random_descs(n, 42)).unwrap();
        let d1 = stream.memcpy_stod(&random_descs(n, 7)).unwrap();
        let m12: CudaSlice<i32> = unsafe { stream.alloc(n).unwrap() };
        let s12: CudaSlice<f32> = unsafe { stream.alloc(n).unwrap() };

        let d0r: CUdeviceptr = d0.device_ptr(stream.as_ref()).0;
        let d1r: CUdeviceptr = d1.device_ptr(stream.as_ref()).0;
        let mr:  CUdeviceptr = m12.device_ptr(stream.as_ref()).0;
        let sr:  CUdeviceptr = s12.device_ptr(stream.as_ref()).0;
        let n_i = n as i32;

        let launch = || unsafe {
            stream.launch_builder(&pp.fn_match_argmax)
                .arg(&d0r).arg(&d1r).arg(&mr).arg(&sr).arg(&n_i).arg(&n_i)
                .launch(trt::cuda::cfg_1d(n, 128)).unwrap();
        };

        launch(); stream.synchronize().unwrap();  // warm-up

        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = stream.record_event(flags).unwrap();
        for _ in 0..20 { launch(); }
        let stop = stream.record_event(flags).unwrap();
        stream.synchronize().unwrap();
        let ms = start.elapsed_ms(&stop).unwrap() / 20.0;
        eprintln!("match_argmax kernel-only @ {n}x{n}: {ms:.3} ms/direction");
    }
}

#[cfg(test)]
mod gpu_compact_tests {
    use super::*;

    /// Compaction top-K must select the right keypoints from a synthetic
    /// score map and produce L2-normalized descriptors.
    #[test]
    #[ignore]
    fn compact_topk_selects_correct_keypoints() {
        let ctx    = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let pp     = XFeatPostproc::new(stream.clone(), 2, 0.05).unwrap();  // top_k = 2

        let (h, w) = (32usize, 32usize);
        let (hd, wd) = (h / 8, w / 8);

        // Three survivors; top-2 by score are at flat idx 100 (x=4,y=3) and 999 (x=7,y=31).
        let mut scores = vec![0.0f32; h * w];
        scores[100] = 0.9;
        scores[999] = 0.7;
        scores[500] = 0.1;
        let score_dev = stream.memcpy_stod(&scores).unwrap();

        // Constant-per-channel descriptor map: sampled vector = (c+1) before norm.
        let mut desc_map = vec![0.0f32; 64 * hd * wd];
        for c in 0..64 {
            for i in 0..hd * wd { desc_map[c * hd * wd + i] = (c + 1) as f32; }
        }
        let desc_dev = stream.memcpy_stod(&desc_map).unwrap();
        let desc_ptr = {
            use cudarc::driver::DevicePtr;
            desc_dev.device_ptr(stream.as_ref()).0 as *const f32
        };

        let res = pp.process_topk_sample(desc_ptr, &score_dev, h, w).unwrap();

        assert_eq!(res.scores, vec![0.9, 0.7]);
        assert_eq!(res.kpts_cpu, vec![
            (100 % w) as f32, (100 / w) as f32,
            (999 % w) as f32, (999 / w) as f32,
        ]);

        // Descriptors must be L2-normalized samples of the constant map.
        let descs: Vec<f32> = stream.memcpy_dtov(&res.descs).unwrap();
        for row in descs.chunks_exact(64) {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "descriptor not normalized: {norm}");
            // direction must follow (1, 2, ..., 64) / |(1,...,64)|
            let expect0 = 1.0 / (1..=64).map(|c| (c * c) as f32).sum::<f32>().sqrt();
            assert!((row[0] - expect0).abs() < 1e-3, "row[0]={} expect {}", row[0], expect0);
        }
    }
}
