//! **LightGlue+** transformer feature matching on TensorRT: two [`RaCoAlikedResult`]s
//! in, per-keypoint correspondences out — entirely on the GPU.
//!
//! [`LightGlue`] wraps the *matcher half* of fabio-sim's RaCo-ALIKED-LightGlue+ export
//! (see `vrt-raco-aliked/scripts/split_raco_pipeline.py`). Unlike the fused upstream
//! graph — which can only ever match the two images handed to it in one forward pass —
//! this takes descriptors that were extracted *whenever*, so a live frame can be matched
//! against descriptors stored in a map or relocalization database.
//!
//! Same async contract as the rest of vrt: `submit_match` enqueues pack → TRT with
//! **no sync**; the caller syncs the shared stream once, then reads.
//!
//! # Engine I/O
//!
//! ```text
//! normalized_keypoints (2P,1,K,2)   f32   long-edge normalised
//! descriptors          (2P,1,K,128) f32   L2-normalised
//!   -> matches0 (P,K) i32   index into image 1, or -1 if unmatched
//!   -> mscores0 (P,K) f32   match confidence in [0,1]
//! ```
//!
//! Two things about that signature are deliberate:
//!
//! * **Rank-4 padding.** The graph natively takes rank-3 `(2P,K,2)` / `(2P,K,128)`, but
//!   [`ModelSession::run_inputs`] binds `&Tensor<f32,4>`. The split script splices
//!   `Reshape` nodes at the graph boundary so the engine takes rank-4, rather than
//!   making vrt rank-generic for one model.
//! * **`matches0` is int32, with the validity mask folded in as `-1`.** Upstream emits a
//!   compacted `(M,3)` int64 list built by a `NonZero`; that is data-dependent (vrt has
//!   no `IOutputAllocator`) *and* int64 (rejected at engine load). Cutting upstream of
//!   the compaction gives the static per-query form instead, which is also the classic
//!   LightGlue `matches0` convention.
//!
//! # Image pairs are interleaved
//!
//! The leading dimension is `2P` — images stacked `[L0, R0, L1, R1, ...]` — and the
//! pair split happens *inside* the graph. One pair is therefore leading dim 2, not two
//! separate inputs. [`LightGlue::submit_match`] packs a left/right
//! [`RaCoAlikedResult`] into that layout on-device, with no host round-trip.
//!
//! Feed it `normalized_keypoints`, never `keypoints` — see [`vrt_raco_aliked`] on the
//! two coordinate spaces. Mixing them up degrades matching silently.
//!
//! # Model credit
//!
//! LightGlue (Apache-2.0, `cvg/LightGlue`) — Lindenberger, Sarlin, Pollefeys,
//! "LightGlue: Local Feature Matching at Light Speed", ICCV 2023. The `raco_aliked`
//! matcher weights and the ONNX export are from `fabio-sim/LightGlue-ONNX`
//! (Apache-2.0). See README.md.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::cuda::cfg_1d;
use vrt::{BoxError, Engine, ModelSession};
use vrt_raco_aliked::{RaCoAlikedResult, DESC_DIM};

/// Errors from LightGlue matching.
#[derive(Debug, thiserror::Error)]
pub enum LightGlueError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("kornia CUDA: {0}")]
    Cuda(#[from] kornia_tensor::CudaError),
    #[error("engine output '{0}' missing")]
    MissingOutput(&'static str),
    #[error("engine expects K={0} keypoints but the {1} result holds K={2}")]
    KeypointMismatch(usize, &'static str, usize),
}

// Pack one image's keypoints/descriptors into its slot of the interleaved pair buffer,
// and copy the TRT outputs into caller-owned memory (output views alias session memory
// that the next run reuses).
const KERNEL_SRC: &str = r#"
extern "C" __global__ void lg_pack(const float* __restrict__ src, int n,
                                   float* __restrict__ dst, int dst_off) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[dst_off + i] = src[i];
}
extern "C" __global__ void lg_copy_i32(const int* __restrict__ src, int n,
                                       int* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
extern "C" __global__ void lg_copy_f32(const float* __restrict__ src, int n,
                                       float* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
"#;

/// Caller-owned match output (VPI-style): GPU-resident per-keypoint correspondences,
/// filled async by [`LightGlue::submit_match`].
pub struct LightGlueResult {
    matches: CudaSlice<i32>, // [k] index into image 1, -1 = unmatched
    scores: CudaSlice<f32>,  // [k] confidence in [0,1]
    stream: Arc<CudaStream>,
    k: usize,
}

impl LightGlueResult {
    fn alloc(stream: &Arc<CudaStream>, k: usize) -> Result<Self, LightGlueError> {
        Ok(Self {
            matches: stream.alloc_zeros::<i32>(k)?,
            scores: stream.alloc_zeros::<f32>(k)?,
            stream: stream.clone(),
            k,
        })
    }

    /// Keypoint capacity (the engine's `K`).
    pub fn len(&self) -> usize {
        self.k
    }

    /// Always false — the buffer is sized to the engine's fixed `K`.
    pub fn is_empty(&self) -> bool {
        self.k == 0
    }

    /// GPU-resident match indices `[K]` (`-1` = unmatched). Valid after the stream sync.
    pub fn matches_slice(&self) -> &CudaSlice<i32> {
        &self.matches
    }

    /// GPU-resident match confidences `[K]`. Valid after the stream sync.
    pub fn scores_slice(&self) -> &CudaSlice<f32> {
        &self.scores
    }

    /// Download correspondences as `(i0, i1)` index pairs, keeping only matched
    /// keypoints whose confidence is at least `min_score`. Call after the stream sync.
    ///
    /// LightGlue has already applied its own mutual-nearest-neighbour check and
    /// filter threshold inside the graph — everything with `matches0 >= 0` passed both.
    /// `min_score` is an *additional* tightening, so `0.0` keeps all of them.
    pub fn pairs(&self, min_score: f32) -> Result<Vec<(usize, usize)>, LightGlueError> {
        let m = self.stream.clone_dtoh(&self.matches)?;
        let s = self.stream.clone_dtoh(&self.scores)?;
        Ok(m.iter()
            .zip(s.iter())
            .enumerate()
            .filter(|(_, (&j, &sc))| j >= 0 && sc >= min_score)
            .map(|(i, (&j, _))| (i, j as usize))
            .collect())
    }

    /// Download the raw per-keypoint match indices `[K]`. Call after the stream sync.
    pub fn matches_host(&self) -> Result<Vec<i32>, LightGlueError> {
        Ok(self.stream.clone_dtoh(&self.matches)?)
    }

    /// Download the raw per-keypoint confidences `[K]`. Call after the stream sync.
    pub fn scores_host(&self) -> Result<Vec<f32>, LightGlueError> {
        Ok(self.stream.clone_dtoh(&self.scores)?)
    }
}

/// LightGlue+ matcher (payload): TRT session + interleaved pair buffers + pack/copy
/// kernels + shared stream. Build once, reuse for every pair.
pub struct LightGlue {
    model: ModelSession,
    stream: Arc<CudaStream>,
    /// Interleaved pair inputs `[2,1,K,2]` and `[2,1,K,128]`, reused every call.
    kpts_in: Tensor<f32, 4>,
    descs_in: Tensor<f32, 4>,
    k: usize,
    pack_k: CudaKernel,
    copy_i32: CudaKernel,
    copy_f32: CudaKernel,
}

impl LightGlue {
    /// Build a matcher sharing `stream` with the extractor, so one sync per frame covers
    /// extraction and matching together.
    ///
    /// `K` is read from the engine's `descriptors` input and must equal the `K` of the
    /// extractor whose results are fed in.
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let descs = engine
            .inputs()
            .find(|s| s.name == "descriptors")
            .ok_or("lightglue: engine has no 'descriptors' input")?;
        let k = match descs.dims.as_slice() {
            [_, 1, nk, d] if *nk > 0 && *d == DESC_DIM as i64 => *nk as usize,
            dims => {
                return Err(format!(
                    "lightglue: 'descriptors' must be (2P,1,K,{DESC_DIM}), got {dims:?}"
                )
                .into())
            }
        };
        let kpts = engine
            .inputs()
            .find(|s| s.name == "normalized_keypoints")
            .ok_or("lightglue: engine has no 'normalized_keypoints' input")?;
        match kpts.dims.as_slice() {
            [_, 1, nk, 2] if *nk as usize == k => {}
            dims => {
                return Err(
                    format!("lightglue: 'normalized_keypoints' must be (2P,1,{k},2), got {dims:?}")
                        .into(),
                )
            }
        }

        let kpts_in = zeros_cuda::<f32, 4>([2, 1, k, 2], &stream)?;
        let descs_in = zeros_cuda::<f32, 4>([2, 1, k, DESC_DIM], &stream)?;
        let ctx = stream.context();
        let pack_k = CudaKernel::compile(ctx, KERNEL_SRC, "lg_pack")?;
        let copy_i32 = CudaKernel::compile(ctx, KERNEL_SRC, "lg_copy_i32")?;
        let copy_f32 = CudaKernel::compile(ctx, KERNEL_SRC, "lg_copy_f32")?;
        let model = ModelSession::new(engine, Arc::clone(&stream))?;

        Ok(Self {
            model,
            stream,
            kpts_in,
            descs_in,
            k,
            pack_k,
            copy_i32,
            copy_f32,
        })
    }

    /// Construct from a prebuilt TensorRT `.engine` file.
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        Self::new(Engine::load(engine_path)?, stream)
    }

    /// The engine build profile — both inputs are dynamic in the leading (pair) dim,
    /// fp16. `K` must match the extractor's, so it is a parameter rather than a
    /// constant: the k512…k3584 assets each bake in their own.
    ///
    /// This is the workspace's only multi-input shape profile, so it only builds
    /// through vrt-hub's default trtexec path — the in-process `builder` feature binds
    /// a single profile and will refuse it.
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn engine_profile(k: usize) -> vrt_hub::EngineProfile {
        let (k, d) = (k as i64, DESC_DIM as i64);
        vrt_hub::EngineProfile {
            inputs: vec![
                (
                    "normalized_keypoints".into(),
                    vec![2, 1, k, 2],
                    vec![2, 1, k, 2],
                    vec![2, 1, k, 2],
                ),
                (
                    "descriptors".into(),
                    vec![2, 1, k, d],
                    vec![2, 1, k, d],
                    vec![2, 1, k, d],
                ),
            ],
            fp16: true,
            bf16: false,
            workspace_mb: 2048,
        }
    }

    /// Build (and cache) an engine from an ONNX file, then construct. `k` must match
    /// the `K` the ONNX was split at. Requires feature `hub` or `builder`.
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        k: usize,
    ) -> Result<Self, BoxError> {
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or("lightglue: onnx path is not valid UTF-8")?;
        let engine_path = vrt_hub::EngineCache::default().resolve(
            "lightglue-matcher",
            model_path,
            &Self::engine_profile(k),
        )?;
        Self::from_engine_file(engine_path, stream)
    }

    /// Pull from Hugging Face and construct. Requires feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>, k: usize) -> Result<Self, BoxError> {
        let engine = vrt_hub::resolve_engine("lightglue-matcher", &Self::engine_profile(k))?;
        Self::from_engine_file(engine, stream)
    }

    /// Keypoints per image the engine was built for.
    pub fn num_keypoints(&self) -> usize {
        self.k
    }

    /// Allocate a reusable output sized for this matcher's `K`.
    pub fn alloc_result(&self) -> Result<LightGlueResult, BoxError> {
        Ok(LightGlueResult::alloc(&self.stream, self.k)?)
    }

    /// Submit one image pair's async GPU work — pack the two extractor results into the
    /// interleaved layout, run the matcher, copy into the caller-owned `out` — all
    /// enqueued on the shared stream with **no sync**.
    ///
    /// Indices in `out` are into `left`'s keypoints; values are indices into `right`'s.
    pub fn submit_match(
        &mut self,
        left: &RaCoAlikedResult,
        right: &RaCoAlikedResult,
        out: &mut LightGlueResult,
    ) -> Result<(), LightGlueError> {
        for (label, r) in [("left", left), ("right", right)] {
            if r.count() != self.k {
                return Err(LightGlueError::KeypointMismatch(self.k, label, r.count()));
            }
        }

        // Interleaved [L, R]: image 0 occupies the first slot of each pair buffer.
        // Both were allocated by `zeros_cuda` on this stream, so they are device-resident
        // by construction.
        let kpts_dst = self
            .kpts_in
            .as_cudaslice()
            .expect("LightGlue pair keypoint buffer is device-resident")
            .device_ptr(self.stream.as_ref())
            .0;
        let descs_dst = self
            .descs_in
            .as_cudaslice()
            .expect("LightGlue pair descriptor buffer is device-resident")
            .device_ptr(self.stream.as_ref())
            .0;
        for (slot, r) in [(0usize, left), (1, right)] {
            self.pack(
                r.normalized_kpts_slice(),
                self.k * 2,
                kpts_dst,
                slot * self.k * 2,
            )?;
            self.pack(
                r.descs_slice(),
                self.k * DESC_DIM,
                descs_dst,
                slot * self.k * DESC_DIM,
            )?;
        }

        let tmap = self.model.run_inputs(&[
            ("normalized_keypoints", &self.kpts_in),
            ("descriptors", &self.descs_in),
        ])?;

        let m_src = tmap
            .get("matches0")
            .ok_or(LightGlueError::MissingOutput("matches0"))?
            .i32_ptr()? as usize as CUdeviceptr;
        let s_src = tmap
            .get("mscores0")
            .ok_or(LightGlueError::MissingOutput("mscores0"))?
            .f32_ptr()? as usize as CUdeviceptr;

        let m_dst = out.matches.device_ptr(self.stream.as_ref()).0;
        self.copy_i32
            .launch_builder(&self.stream)
            .arg(&m_src)
            .arg(&(self.k as i32))
            .arg(&m_dst)
            .launch_cfg(cfg_1d(self.k, 256))?;

        let s_dst = out.scores.device_ptr(self.stream.as_ref()).0;
        self.copy_f32
            .launch_builder(&self.stream)
            .arg(&s_src)
            .arg(&(self.k as i32))
            .arg(&s_dst)
            .launch_cfg(cfg_1d(self.k, 256))?;

        Ok(())
    }

    fn pack(
        &self,
        src: &CudaSlice<f32>,
        n: usize,
        dst: CUdeviceptr,
        dst_off: usize,
    ) -> Result<(), LightGlueError> {
        let src_raw = src.device_ptr(self.stream.as_ref()).0;
        self.pack_k
            .launch_builder(&self.stream)
            .arg(&src_raw)
            .arg(&(n as i32))
            .arg(&dst)
            .arg(&(dst_off as i32))
            .launch_cfg(cfg_1d(n, 256))?;
        Ok(())
    }
}
