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

   One THREAD per query (not one block): the query lives in registers and
   reference descriptors stream through a shared-memory tile, so the inner
   loop is a pure unrolled MAC chain with no per-candidate barrier.

   DESC_D and MATCH_TILE are substituted at NVRTC compile time so the register
   array and shared tile stay statically sized. MATCH_TILE is chosen per width to
   hold shared at ~16 KB (64 rows at D=64, 32 rows at D=128).

   Launch: grid = ceil(Nq/128), block = 128.
   sim_out may be NULL (the reverse direction doesn't need similarities).

   MEASURED, 128-D, Nq=Nr=3072, Orin Nano @ locked 1020 MHz (ncu --set basic):
     registers/thread 161  ->  Block Limit Registers 3  ->  occupancy 25% (12 warps/SM)
     static shared 16.38 KB/block, grid 24 blocks, SM throughput 64%, L1/LSU 56%
   ncu attributes the occupancy cap to registers and estimates ~75% local headroom.

   The register array `float q[DESC_D]` is the cause: at 128-D it alone is half the
   161. Occupancy has repeatedly beaten instruction-level parallelism on this part, so
   the likely win is to stop holding the whole query in registers — tile the reduction
   over DESC_D in 64-wide chunks, or cap with __launch_bounds__ and let the compiler
   spill. Both are untested: they must be A/B'd with sustained timing on an IDLE box
   (this capture was taken under heavy unrelated GPU load, so no timing conclusion is
   drawn from it — ncu says WHERE the time goes, never how fast it is).

   At 64-D the same kernel is not register-bound, so this only concerns 128-D. */
#define MATCH_BLOCK 128
#define MATCH_TILE   {TILE}
#define DESC_D       {DESC_D}
extern "C" __global__ void xfeat_match_argmax(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    int*   __restrict__ match_out,
    float* __restrict__ sim_out,
    int Nq, int Nr
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;

    float q[DESC_D];
    if (qi < Nq) {
        #pragma unroll
        for (int c = 0; c < DESC_D; c++) q[c] = __ldg(&Q[qi * DESC_D + c]);
    }

    __shared__ float tile[MATCH_TILE][DESC_D];

    int   best_j = 0;
    float best_s = -1e30f;

    for (int j0 = 0; j0 < Nr; j0 += MATCH_TILE) {
        int jt = min(MATCH_TILE, Nr - j0);

        /* Cooperative, coalesced tile load (rows of R are contiguous). */
        for (int idx = threadIdx.x; idx < jt * DESC_D; idx += MATCH_BLOCK) {
            tile[idx / DESC_D][idx % DESC_D] = __ldg(&R[j0 * DESC_D + idx]);
        }
        __syncthreads();

        if (qi < Nq) {
            for (int j = 0; j < jt; j++) {
                float s = 0.0f;
                /* All threads read the same tile row in lockstep → broadcast. */
                #pragma unroll
                for (int c = 0; c < DESC_D; c++) s += q[c] * tile[j][c];
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
    /// Descriptor width this matcher was compiled for; `submit_match` rejects
    /// buffers that do not divide by it.
    dim: usize,
    stream: Arc<CudaStream>,
}

impl Matcher {
    /// XFeat's descriptor width, and the default for [`new`](Self::new).
    pub const XFEAT_DIM: usize = 64;

    /// Compile the match kernel for 64-D descriptors (XFeat). Share `stream` with the
    /// extractor so extraction + matching run on one continuous stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, XFeatError> {
        Self::with_dim(stream, Self::XFEAT_DIM)
    }

    /// Compile for an arbitrary descriptor width — 128 for ALIKED
    /// (`vrt-raco-aliked`), 64 for XFeat.
    ///
    /// The width is a compile-time constant in the kernel, not a runtime argument, so
    /// the per-thread query array and the shared tile stay statically sized and the
    /// inner product stays a fully unrolled MAC chain. The tile height is halved as the
    /// width doubles, holding shared memory at ~16 KB either way.
    ///
    /// Cost is not linear in the width. The query lives in registers, so 128-D doubles
    /// the register array; on this part occupancy has repeatedly beaten
    /// instruction-level parallelism, so measure before assuming 128-D costs only 2x.
    pub fn with_dim(stream: Arc<CudaStream>, dim: usize) -> Result<Self, XFeatError> {
        // Shared tile is MATCH_TILE * dim floats; keep it near 16 KB.
        let tile = match dim {
            64 => 64,
            128 => 32,
            _ => return Err(XFeatError::UnsupportedDim(dim)),
        };
        let src = MATCH_SRC
            .replace("{TILE}", &tile.to_string())
            .replace("{DESC_D}", &dim.to_string());
        let fn_match_argmax = CudaKernel::compile(stream.context(), &src, "xfeat_match_argmax")?;
        Ok(Self {
            fn_match_argmax,
            dim,
            stream,
        })
    }

    /// Descriptor width this matcher was compiled for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Allocate a reusable match output for this matcher's stream.
    pub fn alloc_result(&self, cap: usize) -> Result<MatchResult, XFeatError> {
        MatchResult::alloc(&self.stream, cap)
    }

    /// Enqueue mutual-NN matching of two device descriptor sets into `out` —
    /// **async, no sync**. `descs*` are `[n* × dim]` L2-normalised device buffers
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
        // A width mismatch is not a crash — the kernel would stride the buffers wrongly
        // and return plausible-looking nonsense — so check it rather than trust it.
        for (label, buf, n) in [("descs0", descs0, n0), ("descs1", descs1, n1)] {
            if buf.len() < n * self.dim {
                return Err(XFeatError::DescriptorDim {
                    which: label,
                    expected: n * self.dim,
                    got: buf.len(),
                    dim: self.dim,
                });
            }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent CPU mutual nearest-neighbour reference (O(n²×64)) — the oracle
    /// the GPU tiled-argmax kernel is validated against. Test-only.
    /// Descriptors must be L2-normalised; cosim = dot product.
    fn cpu_match_reference(
        descs0: &[f32],
        descs1: &[f32],
        min_cossim: f32,
        dim: usize,
    ) -> Vec<(usize, usize)> {
        let n0 = descs0.len() / dim;
        let n1 = descs1.len() / dim;
        if n0 == 0 || n1 == 0 {
            return Vec::new();
        }

        let mut d1t = vec![0.0f32; dim * n1];
        for j in 0..n1 {
            for c in 0..dim {
                d1t[c * n1 + j] = descs1[j * dim + c];
            }
        }

        let mut match12 = vec![0usize; n0];
        let mut sim12 = vec![f32::NEG_INFINITY; n0];
        for i in 0..n0 {
            let d0 = &descs0[i * dim..(i + 1) * dim];
            for j in 0..n1 {
                let mut s = 0.0f32;
                for c in 0..dim {
                    s += d0[c] * d1t[c * n1 + j];
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
            let d1 = &descs1[j * dim..(j + 1) * dim];
            for i in 0..n0 {
                let d0 = &descs0[i * dim..(i + 1) * dim];
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
        random_descs_dim(n, seed, 64)
    }

    /// L2-normalised random descriptors of arbitrary width.
    fn random_descs_dim(n: usize, seed: u64, d: usize) -> Vec<f32> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mut v: Vec<f32> = (0..n * d).map(|_| next()).collect();
        for row in v.chunks_exact_mut(d) {
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
            let mut out = matcher.alloc_result(n0.max(n1)).unwrap();

            let run = |out: &mut MatchResult| {
                matcher.submit_match(&d0, n0, &d1, n1, -1.0, out).unwrap();
                stream.synchronize().unwrap();
                out.pairs()
            };
            let _ = run(&mut out); // warm-up
            let t0 = std::time::Instant::now();
            let gpu = run(&mut out);
            let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let cpu = cpu_match_reference(&h0, &h1, -1.0, Matcher::XFEAT_DIM);

            let gset: std::collections::HashSet<_> = gpu.iter().copied().collect();
            let cset: std::collections::HashSet<_> = cpu.iter().copied().collect();
            assert_eq!(gset, cset, "GPU/CPU match mismatch at n0={n0} n1={n1}");
            eprintln!(
                "match n0={n0:5} n1={n1:5}: {} pairs, GPU wall {gpu_ms:.2} ms",
                gpu.len()
            );
        }
    }

    /// The 128-D kernel must agree with the same CPU oracle.
    ///
    /// This is the test that matters for the width change: a wrong stride does not
    /// crash, it reads neighbouring descriptors and returns plausible-looking matches.
    /// Only an independent reference catches it. Also checks that a buffer too short
    /// for its claimed count is rejected rather than silently strided.
    ///
    /// Run: cargo test -p vrt-xfeat --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_match_128d_agrees_with_cpu_reference() {
        const D: usize = 128;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let matcher = Matcher::with_dim(stream.clone(), D).unwrap();
        assert_eq!(matcher.dim(), D);

        for (n0, n1) in [(3072usize, 3072usize), (1024, 3072), (1, 2048), (77, 1)] {
            let h0 = random_descs_dim(n0, 42, D);
            let h1 = random_descs_dim(n1, 7, D);
            let d0 = stream.clone_htod(&h0).unwrap();
            let d1 = stream.clone_htod(&h1).unwrap();
            let mut out = matcher.alloc_result(n0.max(n1)).unwrap();

            matcher
                .submit_match(&d0, n0, &d1, n1, -1.0, &mut out)
                .unwrap();
            stream.synchronize().unwrap();
            let gpu: std::collections::HashSet<_> = out.pairs().into_iter().collect();
            let cpu: std::collections::HashSet<_> =
                cpu_match_reference(&h0, &h1, -1.0, D).into_iter().collect();
            assert_eq!(gpu, cpu, "128-D GPU/CPU mismatch at n0={n0} n1={n1}");
            eprintln!("128-D match n0={n0:5} n1={n1:5}: {} pairs", gpu.len());
        }

        // A 64-D buffer handed to the 128-D kernel must be refused, not strided.
        let short = stream.clone_htod(&random_descs_dim(64, 1, 64)).unwrap();
        let mut out = matcher.alloc_result(64).unwrap();
        assert!(
            matcher
                .submit_match(&short, 64, &short, 64, -1.0, &mut out)
                .is_err(),
            "a buffer too short for its claimed count must be rejected"
        );
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
