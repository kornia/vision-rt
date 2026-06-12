//! XFeat post-processing: NMS → TopK → descriptor sampling → L2-norm + GPU matching.
//!
//! Works with the TRT backbone engine that outputs three tensors:
//!   `descriptors`  (1, 64, H/8, W/8)  — dense feature maps (FP32 on device)
//!   `heatmap`      (1,  1,   H,   W)  — keypoint confidence (FP32 on device)
//!   `reliability`  (1,  1,   H,   W)  — channel reliability  (FP32 on device)
//!
//! Pipeline:
//!   GPU  xfeat_score_nms    → score_map (H×W), masked to local-max pixels above threshold
//!   CPU  D2H + TopK select  → top-K flat indices (sorted by score descending)
//!   GPU  xfeat_sample_descs → K×64 descriptor vectors (bilinear sample from desc_map)
//!   GPU  xfeat_l2_norm      → in-place L2 normalise
//!
//! Both GPU kernels are JIT-compiled via cudarc nvrtc targeting sm_87 (Jetson Orin).

use std::sync::Arc;
use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use cudarc::driver::sys::CUdeviceptr;
use trt::BoxError;
use trt::cuda::{Kernels, cfg_2d, cfg_per_item};

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

/* xfeat_match_rows — argmax dot-product search: D0[i] → nearest in D1.
   One block per query; block_dim=64 = 2 warps; warp-shuffle + shared-mem reduction. */
extern "C" __global__ void xfeat_match_rows(
    const float* __restrict__ D0,
    const float* __restrict__ D1,
    int*   __restrict__ match_out,
    float* __restrict__ sim_out,
    int N0, int N1
) {
    int qi = blockIdx.x;
    int c  = threadIdx.x;
    if (qi >= N0) return;

    __shared__ float shmem[2];

    float qi_c = __ldg(&D0[qi * 64 + c]);
    int   best_j = 0;
    float best_s = -1e30f;

    for (int j = 0; j < N1; j++) {
        float s = qi_c * __ldg(&D1[j * 64 + c]);
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s,  8);
        s += __shfl_down_sync(0xFFFFFFFF, s,  4);
        s += __shfl_down_sync(0xFFFFFFFF, s,  2);
        s += __shfl_down_sync(0xFFFFFFFF, s,  1);
        if (c ==  0) shmem[0] = s;
        if (c == 32) shmem[1] = s;
        __syncthreads();
        if (c == 0) {
            float total = shmem[0] + shmem[1];
            if (total > best_s) { best_s = total; best_j = j; }
        }
        __syncthreads();
    }

    if (c == 0) { match_out[qi] = best_j; sim_out[qi] = best_s; }
}

/* xfeat_match_cols — argmax dot-product search: D1[j] → nearest in D0. */
extern "C" __global__ void xfeat_match_cols(
    const float* __restrict__ D0,
    const float* __restrict__ D1,
    int*   __restrict__ match_out,
    int N0, int N1
) {
    int qj = blockIdx.x;
    int c  = threadIdx.x;
    if (qj >= N1) return;

    __shared__ float shmem[2];

    float qj_c = __ldg(&D1[qj * 64 + c]);
    int   best_i = 0;
    float best_s = -1e30f;

    for (int i = 0; i < N0; i++) {
        float s = qj_c * __ldg(&D0[i * 64 + c]);
        s += __shfl_down_sync(0xFFFFFFFF, s, 16);
        s += __shfl_down_sync(0xFFFFFFFF, s,  8);
        s += __shfl_down_sync(0xFFFFFFFF, s,  4);
        s += __shfl_down_sync(0xFFFFFFFF, s,  2);
        s += __shfl_down_sync(0xFFFFFFFF, s,  1);
        if (c ==  0) shmem[0] = s;
        if (c == 32) shmem[1] = s;
        __syncthreads();
        if (c == 0) {
            float total = shmem[0] + shmem[1];
            if (total > best_s) { best_s = total; best_i = i; }
        }
        __syncthreads();
    }

    if (c == 0) { match_out[qj] = best_i; }
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
    fn_score_nms:    cudarc::driver::CudaFunction,
    fn_sample_descs: cudarc::driver::CudaFunction,
    fn_l2_norm:      cudarc::driver::CudaFunction,
    fn_match_rows:   cudarc::driver::CudaFunction,
    fn_match_cols:   cudarc::driver::CudaFunction,
    stream:          Arc<CudaStream>,
    top_k:           usize,
    threshold:       f32,
}

impl XFeatPostproc {
    /// The CUDA stream used for all GPU work.
    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }

    /// Compile all CUDA kernels and return a ready post-processor.
    pub fn new(
        stream:    Arc<CudaStream>,
        top_k:     usize,
        threshold: f32,
    ) -> Result<Self, BoxError> {
        let kernels = Kernels::compile(stream.clone(), KERNELS_SRC)?;

        let fn_score_nms    = kernels.function("xfeat_score_nms")?;
        let fn_sample_descs = kernels.function("xfeat_sample_descs")?;
        let fn_l2_norm      = kernels.function("xfeat_l2_norm")?;
        let fn_match_rows   = kernels.function("xfeat_match_rows")?;
        let fn_match_cols   = kernels.function("xfeat_match_cols")?;

        Ok(Self { fn_score_nms, fn_sample_descs, fn_l2_norm, fn_match_rows, fn_match_cols,
                  stream, top_k, threshold })
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
    ) -> Result<(), BoxError> {
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
    /// Reads the NMS scores from `score_dev` (D2H), selects top-K keypoints,
    /// samples descriptors, L2-normalises, and returns the [`XFeatResult`].
    /// Performs one internal `stream.synchronize()` for the descriptor kernels.
    pub fn process_topk_sample(
        &self,
        desc_ptr:  *const f32,
        score_dev: &CudaSlice<f32>,
        h:         usize,
        w:         usize,
    ) -> Result<XFeatResult, BoxError> {
        use cudarc::driver::DevicePtr;
        let hd = h / 8;
        let wd = w / 8;

        // D2H — stream is already synced by the pipeline before finalize.
        let scores_host: Vec<f32> = self.stream.memcpy_dtov(score_dev)?;

        let mut candidates: Vec<(f32, u32)> = scores_host.iter().enumerate()
            .filter(|(_, &s)| s > 0.0)
            .map(|(i, &s)| (s, i as u32))
            .collect();

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

    /// Run the full post-processing pipeline.
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
    ) -> Result<XFeatResult, BoxError> {
        let hd       = h / 8;
        let wd       = w / 8;
        let n_pixels = h * w;

        // ── GPU: NMS score map ────────────────────────────────────────────────
        let score_dev: CudaSlice<f32> = unsafe { self.stream.alloc(n_pixels)? };

        let cfg_nms = cfg_2d(w, h);

        let heat_raw:  CUdeviceptr = heat_ptr as usize as CUdeviceptr;
        let rel_raw:   CUdeviceptr = rel_ptr  as usize as CUdeviceptr;
        let score_raw: CUdeviceptr = { use cudarc::driver::DevicePtr; score_dev.device_ptr(self.stream.as_ref()).0 };

        let h_i  = h as i32;
        let w_i  = w as i32;
        let thr  = self.threshold;

        unsafe {
            self.stream.launch_builder(&self.fn_score_nms)
                .arg(&heat_raw).arg(&rel_raw).arg(&score_raw)
                .arg(&h_i).arg(&w_i).arg(&thr)
                .launch(cfg_nms)?;
        }

        // ── D2H: score map → TopK ─────────────────────────────────────────────
        self.stream.synchronize()?;
        let scores_host: Vec<f32> = self.stream.memcpy_dtov(&score_dev)?;
        drop(score_dev);

        let mut candidates: Vec<(f32, u32)> = scores_host.iter().enumerate()
            .filter(|(_, &s)| s > 0.0)
            .map(|(i, &s)| (s, i as u32))
            .collect();

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

        let kpts_host: Vec<f32> = candidates.iter()
            .flat_map(|(_, idx)| {
                let i = *idx as usize;
                [(i % w) as f32, (i / w) as f32]
            })
            .collect();

        // ── GPU: descriptor sampling ──────────────────────────────────────────
        let kpts_dev:  CudaSlice<f32> = self.stream.memcpy_stod(&kpts_host)?;
        let descs_dev: CudaSlice<f32> = unsafe { self.stream.alloc(k * 64)? };

        let desc_raw:  CUdeviceptr = desc_ptr as usize as CUdeviceptr;
        let kpts_raw:  CUdeviceptr = { use cudarc::driver::DevicePtr; kpts_dev.device_ptr(self.stream.as_ref()).0 };
        let descs_raw: CUdeviceptr = { use cudarc::driver::DevicePtr; descs_dev.device_ptr(self.stream.as_ref()).0 };

        let hd_i  = hd as i32;
        let wd_i  = wd as i32;

        let k_i   = k as i32;

        let cfg64 = cfg_per_item(k, 64);

        unsafe {
            self.stream.launch_builder(&self.fn_sample_descs)
                .arg(&desc_raw).arg(&kpts_raw).arg(&descs_raw)
                .arg(&hd_i).arg(&wd_i).arg(&h_i).arg(&w_i)
                .launch(cfg64)?;
        }

        // ── GPU: L2 normalise in-place ────────────────────────────────────────
        // Re-acquire descs_raw after the sample launch (same address, safe).
        let descs_raw: CUdeviceptr = { use cudarc::driver::DevicePtr; descs_dev.device_ptr(self.stream.as_ref()).0 };

        unsafe {
            self.stream.launch_builder(&self.fn_l2_norm)
                .arg(&descs_raw).arg(&k_i)
                .launch(cfg64)?;
        }

        self.stream.synchronize()?;

        Ok(XFeatResult { kpts: kpts_dev, descs: descs_dev, scores: scores_out, kpts_cpu: kpts_host })
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
    ) -> Result<Vec<(usize, usize)>, BoxError> {
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

        let cfg_rows = cfg_per_item(n0, 64);
        unsafe {
            self.stream.launch_builder(&self.fn_match_rows)
                .arg(&d0_raw).arg(&d1_raw)
                .arg(&m12_raw).arg(&s12_raw)
                .arg(&n0_i).arg(&n1_i)
                .launch(cfg_rows)?;
        }

        let cfg_cols = cfg_per_item(n1, 64);
        unsafe {
            self.stream.launch_builder(&self.fn_match_cols)
                .arg(&d0_raw).arg(&d1_raw)
                .arg(&m21_raw)
                .arg(&n0_i).arg(&n1_i)
                .launch(cfg_cols)?;
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
