//! GPU mutual nearest-neighbour descriptor matching — decoupled from the XFeat
//! extractor's post-processing.
//!
//! [`Matcher`] owns the single `xfeat_match_argmax` kernel and matches two sets of
//! L2-normalised descriptors already on device (no re-upload). Cosine similarity =
//! dot product (unit-norm descriptors). VPI-style, caller-owned output: allocate a
//! [`MatchResult`] once, `submit_match` writes into it (**async, no sync**), sync
//! the shared stream, then read [`MatchResult::pairs`].

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kornia_tensor::CudaKernel;
use std::sync::Arc;

use vrt::cuda::cfg_1d;

use crate::postprocess::XFeatError;

const MATCH_SRC: &str = r#"
/* xfeat_match_argmax — argmax dot-product search: Q[t] → nearest in R.
   Direction-agnostic: call once with (D0, D1) and once with (D1, D0).

   One THREAD per query (not one block): the 64-D query lives in registers
   and reference descriptors stream through a shared-memory tile, so the
   inner loop is a pure unrolled MAC chain with no per-candidate barrier.

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
"#;

/// Caller-owned mutual-NN match output (VPI-style), allocated once and reused.
///
/// Capacity `cap` must be ≥ both descriptor counts of any pair matched into it
/// (allocate with the extractor's `top_k`). [`Matcher::submit_match`] writes the
/// argmax arrays here (async); after the stream sync, [`pairs`](Self::pairs)
/// builds the `(i, j)` list on the host.
pub struct MatchResult {
    // Device match arrays (capacity `cap`) — written by the kernels, copied to pinned.
    m12_dev: CudaSlice<i32>,
    m21_dev: CudaSlice<i32>,
    s12_dev: CudaSlice<f32>,
    // Pinned host targets read in `pairs()` (post-sync).
    m12: vrt::PinnedBuffer<i32>,
    m21: vrt::PinnedBuffer<i32>,
    s12: vrt::PinnedBuffer<f32>,
    cap: usize,
    n0: usize,
    min_cossim: f32,
}

impl MatchResult {
    /// Pre-allocate a match output of capacity `cap` (use the extractor's `top_k`).
    pub fn alloc(stream: &Arc<CudaStream>, cap: usize) -> Result<Self, XFeatError> {
        Ok(Self {
            m12_dev: unsafe { stream.alloc::<i32>(cap)? },
            m21_dev: unsafe { stream.alloc::<i32>(cap)? },
            s12_dev: unsafe { stream.alloc::<f32>(cap)? },
            m12: vrt::PinnedBuffer::<i32>::alloc(cap)?,
            m21: vrt::PinnedBuffer::<i32>::alloc(cap)?,
            s12: vrt::PinnedBuffer::<f32>::alloc(cap)?,
            cap,
            n0: 0,
            min_cossim: 0.0,
        })
    }

    /// Build mutual-NN pairs `(i, j)` from the last [`Matcher::submit_match`],
    /// **after** the stream sync. `i` indexes set 0, `j` set 1; both are mutual
    /// nearest neighbours with cosine ≥ the submitted `min_cossim`.
    pub fn pairs(&self) -> Vec<(usize, usize)> {
        if self.n0 == 0 {
            return Vec::new();
        }
        let (m12, m21, s12) = (
            self.m12.as_slice(),
            self.m21.as_slice(),
            self.s12.as_slice(),
        );
        (0..self.n0)
            .filter(|&i| {
                let j = m12[i] as usize;
                m21[j] as usize == i && s12[i] >= self.min_cossim
            })
            .map(|i| (i, m12[i] as usize))
            .collect()
    }
}

/// GPU mutual nearest-neighbour matcher: owns the argmax kernel + shared stream.
///
/// Construct once (share the extractor's CUDA stream for one end-to-end sync),
/// then match any two device descriptor sets.
pub struct Matcher {
    fn_match_argmax: CudaKernel,
    stream: Arc<CudaStream>,
}

impl Matcher {
    /// Compile the match kernel on `stream`'s context. Share `stream` with the
    /// XFeat extractor so extraction + matching run on one continuous stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, XFeatError> {
        let fn_match_argmax =
            CudaKernel::compile(stream.context(), MATCH_SRC, "xfeat_match_argmax")?;
        Ok(Self {
            fn_match_argmax,
            stream,
        })
    }

    /// The shared CUDA stream this matcher enqueues on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Allocate a reusable match output for this matcher's stream.
    pub fn alloc_result(&self, cap: usize) -> Result<MatchResult, XFeatError> {
        MatchResult::alloc(&self.stream, cap)
    }

    /// Enqueue mutual-NN matching of two device descriptor sets into `out` —
    /// **async, no sync**. `descs*` are `[n* × 64]` L2-normalised device buffers
    /// (e.g. `XFeatResult::descs` with `count`); `out.cap` must be ≥ `n0` and
    /// `n1`. Sync the stream, then read [`MatchResult::pairs`].
    pub fn submit_match(
        &self,
        descs0: &CudaSlice<f32>,
        n0: usize,
        descs1: &CudaSlice<f32>,
        n1: usize,
        min_cossim: f32,
        out: &mut MatchResult,
    ) -> Result<(), XFeatError> {
        debug_assert!(
            n0 <= out.cap && n1 <= out.cap,
            "match output capacity too small"
        );
        out.n0 = n0;
        out.min_cossim = min_cossim;
        if n0 == 0 || n1 == 0 {
            return Ok(());
        }

        let d0_raw = descs0.device_ptr(self.stream.as_ref()).0;
        let d1_raw = descs1.device_ptr(self.stream.as_ref()).0;
        let m12_raw = out.m12_dev.device_ptr(self.stream.as_ref()).0;
        let m21_raw = out.m21_dev.device_ptr(self.stream.as_ref()).0;
        let s12_raw = out.s12_dev.device_ptr(self.stream.as_ref()).0;
        let null_sim: CUdeviceptr = 0;
        let (n0_i, n1_i) = (n0 as i32, n1 as i32);

        // One tiled argmax kernel, both directions (sim only needed for 1→2).
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

        // Async D2H of the match arrays into the caller's pinned buffers (no sync).
        let (m12_pin, m21_pin, s12_pin) = (
            out.m12.as_mut_ptr(),
            out.m21.as_mut_ptr(),
            out.s12.as_mut_ptr(),
        );
        let vstream = vrt::Stream::from_cuda_stream(self.stream.clone());
        unsafe {
            vstream.memcpy_d2h_raw(
                m12_pin as *mut u8,
                m12_raw as usize as *const _,
                n0 * std::mem::size_of::<i32>(),
            )?;
            vstream.memcpy_d2h_raw(
                m21_pin as *mut u8,
                m21_raw as usize as *const _,
                n1 * std::mem::size_of::<i32>(),
            )?;
            vstream.memcpy_d2h_raw(
                s12_pin as *mut u8,
                s12_raw as usize as *const _,
                n0 * std::mem::size_of::<f32>(),
            )?;
        }
        Ok(())
    }

    /// Synchronous one-shot: alloc + [`submit_match`] + one sync + `pairs`.
    /// Convenience; drive the split with a reused [`MatchResult`] for a hot loop.
    ///
    /// [`submit_match`]: Matcher::submit_match
    pub fn match_mutual_nn_gpu(
        &self,
        descs0: &CudaSlice<f32>,
        n0: usize,
        descs1: &CudaSlice<f32>,
        n1: usize,
        min_cossim: f32,
    ) -> Result<Vec<(usize, usize)>, XFeatError> {
        let mut out = MatchResult::alloc(&self.stream, n0.max(n1).max(1))?;
        self.submit_match(descs0, n0, descs1, n1, min_cossim, &mut out)?;
        self.stream.synchronize()?;
        Ok(out.pairs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent CPU mutual nearest-neighbour reference (O(n²×64)) — the oracle
    /// the GPU tiled-argmax kernel is validated against. Test-only.
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
    ///   cargo test -p vrt-xfeat -- --ignored
    #[test]
    #[ignore]
    fn gpu_match_agrees_with_cpu_reference() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let matcher = Matcher::new(stream.clone()).unwrap();

        for (n0, n1) in [(4096usize, 4096usize), (1000, 3000), (1, 4096), (130, 1)] {
            let h0 = random_descs(n0, 42);
            let h1 = random_descs(n1, 7);
            let d0 = stream.clone_htod(&h0).unwrap();
            let d1 = stream.clone_htod(&h1).unwrap();

            let _ = matcher.match_mutual_nn_gpu(&d0, n0, &d1, n1, -1.0).unwrap(); // warm-up
            let t0 = std::time::Instant::now();
            let gpu = matcher.match_mutual_nn_gpu(&d0, n0, &d1, n1, -1.0).unwrap();
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

    /// Kernel-only timing: pre-allocated buffers, CUDA-event bracketed, averaged
    /// over 20 launches. Run: cargo test -p vrt-xfeat --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_match_kernel_only_timing() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let matcher = Matcher::new(stream.clone()).unwrap();

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
            matcher
                .fn_match_argmax
                .launch_builder(&stream)
                .arg(&d0r)
                .arg(&d1r)
                .arg(&mr)
                .arg(&sr)
                .arg(&n_i)
                .arg(&n_i)
                .launch_cfg(cfg_1d(n, 128))
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
