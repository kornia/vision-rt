//! GPU mutual nearest-neighbour descriptor matching — decoupled from the XFeat
//! extractor's post-processing.
//!
//! [`Matcher`] owns the single `xfeat_match_argmax` kernel and matches two sets of
//! L2-normalised descriptors already on device (no re-upload). Cosine similarity =
//! dot product (unit-norm descriptors). VPI-style, caller-owned output: allocate a
//! [`MatchResult`] once, `submit` writes into it (**async, no sync**), sync
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
        /* `idx` is unsigned deliberately. Signed division by a non-power-of-two must
           emit the round-toward-zero correction (8x SHF.R.S32.HI in the SASS vs 1),
           costing ~18 extra instructions per 4 elements in this loop; unsigned also
           lands at 90 registers against 93-95 for the signed forms. */
        for (unsigned idx = threadIdx.x; idx < (unsigned)(jt * DESC_D); idx += MATCH_BLOCK) {
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
/// (allocate with the extractor's `top_k`). [`Matcher::submit`] writes the
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

    /// Build mutual-NN pairs `(i, j)` from the last [`Matcher::submit`],
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
    /// Descriptor width this matcher was compiled for; `submit` rejects
    /// [`Descriptors`] declaring any other width.
    dim: usize,
    stream: Arc<CudaStream>,
}

/// A device descriptor set handed to [`Matcher::submit`]: the buffer, how many
/// descriptors are live in it, and how wide they are.
///
/// The width is carried explicitly because it cannot be recovered from the buffer —
/// `XFeatResult::descs` is allocated for `top_k` descriptors and `RaCoAlikedResult::descs`
/// for `k`, so both are longer than `count * dim` in normal use and no length arithmetic
/// distinguishes a 64-D set from a 128-D one.
#[derive(Clone, Copy)]
pub struct Descriptors<'a> {
    buf: &'a CudaSlice<f32>,
    count: usize,
    dim: usize,
}

/// Shared-tile height for a descriptor width, and the validation of that width.
///
/// Pure, so the supported-width rule can be tested without compiling a kernel — the
/// previous version asserted `SHARED_FLOATS / dim` in a test that never called
/// `with_dim`, so it would have passed against a hardcoded lookup table.
fn tile_for_dim(dim: usize) -> Result<usize, XFeatError> {
    // A multiple of the warp width keeps the tile load coalesced; the upper bound is the
    // register file, not the shared budget.
    if dim == 0 || !dim.is_multiple_of(32) || dim > Matcher::MAX_DIM {
        return Err(XFeatError::UnsupportedDim(dim));
    }
    Ok(Matcher::SHARED_FLOATS / dim)
}

/// The `submit` preconditions, as arithmetic over lengths.
///
/// Split out from [`Matcher::submit`] because it needs no GPU: keeping it inline would
/// leave the only coverage of these guards behind an `#[ignore]`d device test, which is
/// how the previous version of the width check shipped broken.
fn check_descriptors(
    which: &'static str,
    buf_len: usize,
    count: usize,
    dim: usize,
    kernel_dim: usize,
    cap: usize,
) -> Result<(), XFeatError> {
    if dim != kernel_dim {
        return Err(XFeatError::DescriptorWidth {
            which,
            buf_dim: dim,
            kernel_dim,
        });
    }
    if buf_len < count * dim {
        return Err(XFeatError::DescriptorDim {
            which,
            expected: count * dim,
            got: buf_len,
            dim,
        });
    }
    if count > cap {
        return Err(XFeatError::MatchCapacity { which, count, cap });
    }
    Ok(())
}

impl<'a> Descriptors<'a> {
    /// `buf` holds at least `count * dim` L2-normalised floats, row-major per descriptor.
    pub fn new(buf: &'a CudaSlice<f32>, count: usize, dim: usize) -> Self {
        Self { buf, count, dim }
    }
}

impl Matcher {
    /// XFeat's descriptor width, and the default for [`new`](Self::new).
    ///
    /// Re-exported from the post-processing that actually emits the descriptors, so this
    /// cannot drift from the data it describes.
    pub const XFEAT_DIM: usize = crate::postprocess::XFEAT_DESC_DIM;

    /// Largest width the kernel can hold. The query lives in registers as
    /// `float q[DESC_D]`, and CUDA caps a thread at 255 registers, so 256-D would spill
    /// the whole query to local memory. Raising this needs the reduction tiled over the
    /// width first.
    pub const MAX_DIM: usize = 128;

    /// Floats held in the shared reference tile — 16 KB of `f32`, the budget the tile
    /// height is derived from.
    const SHARED_FLOATS: usize = 4096;

    /// Compile the match kernel for 64-D descriptors (XFeat). Share `stream` with the
    /// extractor so extraction + matching run on one continuous stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, XFeatError> {
        Self::with_dim(stream, Self::XFEAT_DIM)
    }

    /// Compile for a descriptor width — 128 for ALIKED (`vrt-raco-aliked`), 64 for XFeat.
    ///
    /// Any non-zero multiple of 32 up to [`MAX_DIM`](Self::MAX_DIM) works; the tile height
    /// is derived from the shared budget rather than tabulated. The ceiling is the
    /// register file: the query array is `float q[DESC_D]`, so 256-D would exceed CUDA's
    /// 255 registers per thread and spill.
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
        let tile = tile_for_dim(dim)?;
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
    /// **async, no sync**. Sync the stream, then read [`MatchResult::pairs`].
    ///
    /// Both sets must carry the width this matcher was compiled for. The width travels
    /// with the buffer in [`Descriptors`] rather than being inferred from its length,
    /// because every real descriptor buffer is over-allocated to a capacity: a length
    /// check cannot tell `k x 128` floats handed to a 64-D kernel from `2k x 64`, and
    /// would accept the mismatch that silently matches half-descriptors.
    pub fn submit(
        &self,
        descs0: Descriptors<'_>,
        descs1: Descriptors<'_>,
        min_cossim: f32,
        out: &mut MatchResult,
    ) -> Result<(), XFeatError> {
        let (n0, n1) = (descs0.count, descs1.count);

        // Invalidate first, validate second. `MatchResult` is documented as reusable, so
        // returning an error with `n0` still set from the last successful submit would
        // have `pairs()` hand the caller the *previous* pair's matches; leaving `n0` set
        // to this pair's count would have it read buffers the kernels never wrote. Only
        // zero is safe in both directions.
        out.n0 = 0;
        for (which, d) in [("descs0", &descs0), ("descs1", &descs1)] {
            check_descriptors(which, d.buf.len(), d.count, d.dim, self.dim, out.cap)?;
        }

        out.min_cossim = min_cossim;
        // Either side empty means no match is possible. Leaving `n0` at the requested
        // count would have `pairs()` read buffers the kernels never wrote.
        out.n0 = if n0 == 0 || n1 == 0 { 0 } else { n0 };
        if out.n0 == 0 {
            return Ok(());
        }

        let d0_raw = descs0.buf.device_ptr(self.stream.as_ref()).0;
        let d1_raw = descs1.buf.device_ptr(self.stream.as_ref()).0;
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
mod guard_tests {
    use super::*;

    const D: usize = 128;

    /// The case a length check cannot catch, and the whole reason the width is declared:
    /// a 64-D set long enough to look like a valid 128-D one.
    #[test]
    fn rejects_a_width_that_passes_the_length_check() {
        // 2048 x 64 floats offered as 1024 x 128 — the length check passes.
        assert!(check_descriptors("descs0", 2048 * 64, 1024, 64, D, 4096).is_err());
        // The rejection must come from the declared width, not the length: the same
        // buffer and count with the correct width is accepted, so the length check
        // was never what failed.
        assert!(check_descriptors("descs0", 2048 * 64, 1024, D, D, 4096).is_ok());
    }

    #[test]
    fn rejects_a_buffer_too_short_for_its_count() {
        assert!(check_descriptors("descs0", 4 * D, 64, D, D, 4096).is_err());
    }

    #[test]
    fn rejects_a_count_over_capacity() {
        assert!(check_descriptors("descs0", 64 * D, 64, D, D, 8).is_err());
    }

    #[test]
    fn accepts_an_over_allocated_buffer() {
        // Every real caller over-allocates to a capacity; that must stay legal.
        assert!(check_descriptors("descs0", 4096 * D, 2000, D, D, 4096).is_ok());
    }

    /// The tile derivation and the width rule, through the function `with_dim` calls.
    ///
    /// An earlier version asserted `SHARED_FLOATS / dim` directly, which would have passed
    /// against a hardcoded lookup table — i.e. it could not fail for the thing it named.
    #[test]
    fn tile_is_derived_and_the_width_rule_is_enforced() {
        assert_eq!(tile_for_dim(64).unwrap(), 64);
        assert_eq!(tile_for_dim(128).unwrap(), 32);
        assert_eq!(
            tile_for_dim(96).unwrap(),
            42,
            "not a power of two, still fine"
        );

        assert!(tile_for_dim(0).is_err());
        assert!(
            tile_for_dim(48).is_err(),
            "not a multiple of the warp width"
        );
        // 256 floats of query would exceed CUDA's 255 registers per thread. The docs
        // previously advertised it as working on the strength of a division.
        assert!(tile_for_dim(256).is_err(), "above MAX_DIM");
        assert_eq!(Matcher::MAX_DIM, 128);
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
            ((state >> 32) as f32 / (1u64 << 31) as f32) - 1.0
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
        // The width the *fixtures* are built at, not `Matcher::XFEAT_DIM`: sourcing it
        // from the matcher would compare the matcher's constant against itself, which is
        // the tautology `Descriptors` exists to prevent. Mirrors the 128-D test below.
        const D: usize = 64;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let matcher = Matcher::new(stream.clone()).unwrap();
        assert_eq!(matcher.dim(), D);

        for (n0, n1) in [(4096usize, 4096usize), (1000, 3000), (1, 4096), (130, 1)] {
            let h0 = random_descs(n0, 42);
            let h1 = random_descs(n1, 7);
            let d0 = stream.clone_htod(&h0).unwrap();
            let d1 = stream.clone_htod(&h1).unwrap();
            let mut out = matcher.alloc_result(n0.max(n1)).unwrap();

            let run = |out: &mut MatchResult| {
                matcher
                    .submit(
                        Descriptors::new(&d0, n0, D),
                        Descriptors::new(&d1, n1, D),
                        -1.0,
                        out,
                    )
                    .unwrap();
                stream.synchronize().unwrap();
                out.pairs()
            };
            let _ = run(&mut out); // warm-up
            let t0 = std::time::Instant::now();
            let gpu = run(&mut out);
            let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let cpu = cpu_match_reference(&h0, &h1, -1.0, D);

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
                .submit(
                    Descriptors::new(&d0, n0, D),
                    Descriptors::new(&d1, n1, D),
                    -1.0,
                    &mut out,
                )
                .unwrap();
            stream.synchronize().unwrap();
            let gpu: std::collections::HashSet<_> = out.pairs().into_iter().collect();
            let cpu: std::collections::HashSet<_> =
                cpu_match_reference(&h0, &h1, -1.0, D).into_iter().collect();
            assert_eq!(gpu, cpu, "128-D GPU/CPU mismatch at n0={n0} n1={n1}");
            eprintln!("128-D match n0={n0:5} n1={n1:5}: {} pairs", gpu.len());
        }

        // Guard rejections. Each of these is a silent-wrong-answer case, not a crash:
        // the kernel would stride the buffer and return plausible matches.
        let mut out = matcher.alloc_result(4096).unwrap();

        // (a) The case a length check CANNOT catch, and the reason `Descriptors` carries
        // the width: a 64-D set long enough to look like a valid 128-D set.
        let wide = stream.clone_htod(&random_descs_dim(2048, 1, 64)).unwrap();
        assert!(
            matcher
                .submit(
                    Descriptors::new(&wide, 2048, 64),
                    Descriptors::new(&wide, 2048, 64),
                    -1.0,
                    &mut out,
                )
                .is_err(),
            "64-D descriptors must be refused by a 128-D matcher even when the buffer is \
             long enough for the length check to pass"
        );

        // (b) Genuinely too short for its claimed count.
        let short = stream.clone_htod(&random_descs_dim(4, 1, D)).unwrap();
        assert!(
            matcher
                .submit(
                    Descriptors::new(&short, 64, D),
                    Descriptors::new(&short, 64, D),
                    -1.0,
                    &mut out,
                )
                .is_err(),
            "a buffer too short for its claimed count must be rejected"
        );

        // (c) Over capacity is an error, not a debug-only assert compiled out in release.
        let big = stream.clone_htod(&random_descs_dim(64, 1, D)).unwrap();
        let mut tiny = matcher.alloc_result(8).unwrap();
        assert!(
            matcher
                .submit(
                    Descriptors::new(&big, 64, D),
                    Descriptors::new(&big, 64, D),
                    -1.0,
                    &mut tiny,
                )
                .is_err(),
            "a count exceeding the output capacity must be rejected"
        );

        // (d) An empty side must report no matches, not read buffers the kernels never
        // wrote. `pairs()` on zero-filled pinned memory otherwise fabricates (0, 0).
        let some = stream.clone_htod(&random_descs_dim(32, 3, D)).unwrap();
        let empty = stream.clone_htod(&random_descs_dim(1, 4, D)).unwrap();
        matcher
            .submit(
                Descriptors::new(&some, 32, D),
                Descriptors::new(&empty, 0, D),
                -1.0,
                &mut out,
            )
            .unwrap();
        stream.synchronize().unwrap();
        assert!(
            out.pairs().is_empty(),
            "matching against an empty set must yield no pairs, not a fabricated (0, 0)"
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
