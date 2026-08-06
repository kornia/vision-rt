//! **RaCo** keypoint detection + **ALIKED** 128-D descriptors: GPU stretch resize → TRT
//! → caller-owned keypoints/descriptors.
//!
//! [`RaCoAliked`] is an `Image<u8,3> → RaCoAlikedResult` feature extractor on the
//! extractor half of fabio-sim's RaCo-ALIKED-LightGlue+ export (see
//! `scripts/split_raco_pipeline.py`). RaCo decides *where* to look — it is a
//! rotation-robust detector with a learned ranker — and ALIKED describes *what is
//! there*. That separation is the main thing it buys over [`vrt_xfeat`], along with
//! 128-D descriptors instead of 64-D.
//!
//! Everything stays on the **GPU** and the pipeline is fully **async / caller-owned**
//! (VPI-style), mirroring the sibling crates: `submit` enqueues resize → TRT → copies
//! into the caller-owned [`RaCoAlikedResult`] with **no sync and no host copy**; the
//! caller syncs the shared stream once, then pulls what it needs.
//!
//! # Engine I/O
//!
//! ```text
//! images (B,3,H,W) f32   H,W multiples of 32, RGB in [0,1]
//!   -> keypoints            (B,K,2)   f32   model-resolution pixels (x,y)
//!   -> normalized_keypoints (B,K,2)   f32   long-edge normalised, matcher input
//!   -> descriptors          (B,K,128) f32   already L2-normalised
//! ```
//!
//! `K` is baked into the engine at export time (the k512..k3584 release assets), so
//! unlike XFeat there is no top-K count to read back — [`RaCoAlikedResult::count`] is
//! always `K`.
//!
//! # Preprocessing
//!
//! Feed **RGB in [0,1]** and nothing else: the ImageNet mean/std live *inside* the
//! graph as `extractor.raco.image_mean` / `image_std` buffers, so this crate uses
//! [`Preprocessor::stretch`] (resize + `/255`) and must **not** apply
//! `Normalize::imagenet()`. Double-normalising degrades matching silently rather than
//! erroring. kornia's `Image<u8,3>` is already RGB, so no channel swap is needed
//! (upstream's Python preprocessor swaps only because OpenCV hands it BGR).
//!
//! Each frame is resized to its own floor-of-32 dimensions (RaCo's
//! `input_dim_divisor`), matching what [`vrt_xfeat`] does at floor-of-32 as well;
//! `keypoints` are scaled back to original pixels by [`RaCoAlikedResult::keypoints_host`].
//!
//! # Two coordinate spaces
//!
//! `keypoints` and `normalized_keypoints` are **not** interchangeable. The former is in
//! model pixels (rescale to source pixels for geometry); the latter is RaCo's
//! **long-edge** normalisation `(kpts - size/2) / (size.max()/2)` — note that differs
//! from the per-axis `2*kpts/size - 1` that SuperPoint and DISK use — and is what the
//! LightGlue matcher consumes verbatim. Feeding the wrong one is silent.
//!
//! # Model credit
//!
//! RaCo (Apache-2.0, `cvg/RaCo`) — Shenoi, Lindenberger, Sarlin, Pollefeys, "RaCo:
//! Ranking and Covariance for Practical Learned Keypoints", 3DV 2026,
//! arXiv:2602.15755. ALIKED descriptors (**BSD-3-Clause**, `Shiaoming/ALIKED`).
//! ONNX export tooling: `fabio-sim/LightGlue-ONNX` (Apache-2.0). See README.md.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kornia_image::Image;
use kornia_imgproc::preprocess::Preprocessor;
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::cuda::cfg_1d;
use vrt::{BoxError, Engine, ModelSession};

/// ALIKED descriptor dimensionality. Fixed by the `aliked-n16` weights the export
/// wraps; the engine's `descriptors` output is validated against it at construction.
pub const DESC_DIM: usize = 128;

/// RaCo's `input_dim_divisor`: model H and W must be multiples of this.
const DIM_DIVISOR: usize = 32;

/// Minimum model dimension the reused buffers are seeded with in [`RaCoAliked::new`];
/// the first frame reallocates them to its real floor-32 size.
const SEED_DIM: usize = DIM_DIVISOR;

/// Errors from RaCo-ALIKED extraction.
#[derive(Debug, thiserror::Error)]
pub enum RaCoAlikedError {
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
    MissingOutput(&'static str),
    #[error("input {0}x{1} is smaller than one {DIM_DIVISOR}px cell in some dimension")]
    InputTooSmall(usize, usize),
    #[error("result was allocated for K={0} but the engine emits K={1}")]
    CapacityMismatch(usize, usize),
}

// TRT output views alias session memory that the next `run` reuses, so a result that
// must outlive the next frame needs a copy into caller-owned buffers.
const COPY_SRC: &str = r#"
extern "C" __global__ void raco_copy(const float* __restrict__ src, int n, float* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
"#;

/// Caller-owned extraction output (VPI-style): GPU-resident keypoints and descriptors,
/// filled async by [`RaCoAliked::submit`]. Allocate once with
/// [`RaCoAliked::alloc_result`] and reuse it every frame, or hold several to keep
/// multiple frames outstanding.
pub struct RaCoAlikedResult {
    kpts: CudaSlice<f32>,      // [k*2] model-resolution pixels (x,y)
    norm_kpts: CudaSlice<f32>, // [k*2] long-edge normalised
    descs: CudaSlice<f32>,     // [k*DESC_DIM] L2-normalised
    stream: Arc<CudaStream>,
    k: usize,
    /// Source/model size ratio `(rw, rh)`, stamped by `submit`.
    scale: (f32, f32),
}

impl RaCoAlikedResult {
    fn alloc(stream: &Arc<CudaStream>, k: usize) -> Result<Self, RaCoAlikedError> {
        Ok(Self {
            kpts: stream.alloc_zeros::<f32>(k * 2)?,
            norm_kpts: stream.alloc_zeros::<f32>(k * 2)?,
            descs: stream.alloc_zeros::<f32>(k * DESC_DIM)?,
            stream: stream.clone(),
            k,
            scale: (1.0, 1.0),
        })
    }

    /// Number of keypoints — always `K`, fixed by the engine at export time. Unlike
    /// XFeat there is no threshold-dependent survivor count to read back.
    pub fn count(&self) -> usize {
        self.k
    }

    /// Source/model size ratio `(rw, rh)` from the last [`RaCoAliked::submit`].
    pub fn scale(&self) -> (f32, f32) {
        self.scale
    }

    /// GPU-resident keypoints `[K*2]` in **model** pixels. Valid after the stream sync.
    pub fn kpts_slice(&self) -> &CudaSlice<f32> {
        &self.kpts
    }

    /// GPU-resident **long-edge normalised** keypoints `[K*2]` — the tensor the
    /// LightGlue matcher consumes. Do not use these for geometry; see the module docs.
    pub fn normalized_kpts_slice(&self) -> &CudaSlice<f32> {
        &self.norm_kpts
    }

    /// GPU-resident L2-normalised descriptors `[K*128]`. Valid after the stream sync.
    pub fn descs_slice(&self) -> &CudaSlice<f32> {
        &self.descs
    }

    /// Download keypoints as `(x, y)` in **original source pixels** (model coordinates
    /// rescaled by the stretch ratio). Call after the stream sync.
    pub fn keypoints_host(&self) -> Result<Vec<(f32, f32)>, RaCoAlikedError> {
        let raw = self.stream.clone_dtoh(&self.kpts)?;
        let (rw, rh) = self.scale;
        Ok(raw.chunks_exact(2).map(|p| (p[0] * rw, p[1] * rh)).collect())
    }

    /// Download the raw normalised keypoints `[K*2]`. Call after the stream sync.
    pub fn normalized_keypoints_host(&self) -> Result<Vec<f32>, RaCoAlikedError> {
        Ok(self.stream.clone_dtoh(&self.norm_kpts)?)
    }

    /// Download descriptors as `[K*128]` row-major. Call after the stream sync.
    pub fn descriptors_host(&self) -> Result<Vec<f32>, RaCoAlikedError> {
        Ok(self.stream.clone_dtoh(&self.descs)?)
    }
}

/// RaCo + ALIKED feature extractor (payload): TRT session + stretch preprocessor +
/// copy kernel + shared stream. Build once, reuse every frame.
pub struct RaCoAliked {
    model: ModelSession,
    preproc: Preprocessor,
    /// The one shared stream (== the session's stream); used to (re)alloc the per-frame
    /// buffers so they are stream-ordered with all other GPU work.
    stream: Arc<CudaStream>,
    /// Model input tensor `[1,3,mh,mw]` CHW f32 device, written by `preproc.run`.
    input: Tensor<f32, 4>,
    /// Model dims `(mh, mw)` the input is currently sized for.
    cur: (usize, usize),
    k: usize,
    copy_k: CudaKernel,
}

impl RaCoAliked {
    /// Build an extractor sharing `stream` with the rest of the application (one CUDA
    /// stream so a single sync per frame covers all its GPU work, including a
    /// downstream matcher on the same stream).
    ///
    /// `K` is read from the engine's `descriptors` output.
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        // Outputs are bound by NAME here, deliberately departing from the shape-based
        // binding the sibling crates use: `keypoints` and `normalized_keypoints` are
        // both (B,K,2) and are indistinguishable by shape, while carrying different
        // coordinate spaces. split_raco_pipeline.py emits these exact names, and the
        // dims are asserted below so a renamed or reshaped export fails loudly.
        let mut k = None;
        for s in engine.outputs() {
            if s.name == "descriptors" {
                match s.dims.as_slice() {
                    [_, nk, d] if *nk > 0 && *d == DESC_DIM as i64 => k = Some(*nk as usize),
                    dims => {
                        return Err(format!(
                            "raco-aliked: 'descriptors' must be (B,K,{DESC_DIM}), got {dims:?}"
                        )
                        .into())
                    }
                }
            }
        }
        let k = k.ok_or("raco-aliked: engine has no 'descriptors' output")?;

        for name in ["keypoints", "normalized_keypoints"] {
            let s = engine
                .outputs()
                .find(|s| s.name == name)
                .ok_or_else(|| format!("raco-aliked: engine has no '{name}' output"))?;
            match s.dims.as_slice() {
                [_, nk, 2] if *nk as usize == k => {}
                dims => {
                    return Err(
                        format!("raco-aliked: '{name}' must be (B,{k},2), got {dims:?}").into(),
                    )
                }
            }
        }

        let inp = engine.inputs().next().ok_or("raco-aliked: engine has no input")?;
        if inp.dims.len() != 4 {
            return Err(format!("raco-aliked: input must be rank-4, got {:?}", inp.dims).into());
        }

        // Stretch + /255 only — the ImageNet normalisation is baked into the graph.
        let preproc = Preprocessor::stretch(stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, SEED_DIM, SEED_DIM], &stream)?;
        let copy_k = CudaKernel::compile(stream.context(), COPY_SRC, "raco_copy")?;
        let model = ModelSession::new(engine, Arc::clone(&stream))?;

        Ok(Self {
            model,
            preproc,
            stream,
            input,
            cur: (SEED_DIM, SEED_DIM),
            k,
            copy_k,
        })
    }

    /// Construct from a prebuilt TensorRT `.engine` file (machine-locked to this TRT
    /// version + GPU arch). No `hub`/`builder` feature required.
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        Self::new(Engine::load(engine_path)?, stream)
    }

    /// The engine build profile — dynamic `images` H×W (multiples of 32), fp16.
    ///
    /// fp16 is what upstream recommends, and the released graph already contains the
    /// fix for the one known fp16 hazard: RaCo's flattened image indices exceed fp16's
    /// 65504 limit, so the export lowers that division to integer floor-division.
    ///
    /// min/opt/max mirror `scripts/build_engine.sh`'s defaults, and are deliberately
    /// tight: TensorRT sizes tactic workspaces from the MAX shape, so an over-generous
    /// max makes the builder skip its fastest tactics for lack of memory on a 7.4 GB
    /// Orin Nano — yielding a slower engine after a much longer build.
    #[cfg(any(feature = "hub", feature = "builder"))]
    fn engine_profile() -> vrt_hub::EngineProfile {
        vrt_hub::EngineProfile {
            inputs: vec![(
                "images".into(),
                vec![1, 3, 256, 256],
                vec![2, 3, 512, 512],
                vec![2, 3, 640, 640],
            )],
            fp16: true,
            bf16: false,
            workspace_mb: 2048,
        }
    }

    /// Build (and cache) an engine from an ONNX file, then construct. First call builds
    /// on-device; later calls are cache hits keyed by ONNX content + TRT version + GPU
    /// arch. Requires feature `hub` (trtexec build) or `builder` (in-process).
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or("raco-aliked: onnx path is not valid UTF-8")?;
        let engine_path = vrt_hub::EngineCache::default().resolve(
            "raco-aliked-extractor",
            model_path,
            &Self::engine_profile(),
        )?;
        Self::from_engine_file(engine_path, stream)
    }

    /// Pull from Hugging Face and construct. Requires feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let engine = vrt_hub::resolve_engine("raco-aliked-extractor", &Self::engine_profile())?;
        Self::from_engine_file(engine, stream)
    }

    /// Keypoints per frame, fixed by the engine.
    pub fn num_keypoints(&self) -> usize {
        self.k
    }

    /// Allocate a reusable output sized for this extractor's `K`.
    pub fn alloc_result(&self) -> Result<RaCoAlikedResult, BoxError> {
        Ok(RaCoAlikedResult::alloc(&self.stream, self.k)?)
    }

    /// Submit one frame's async GPU work — resize/normalise → backbone → copy into the
    /// caller-owned `out` — all enqueued on the shared stream with **no sync**. Sync the
    /// stream once (covering any other work on it), then read `out`.
    pub fn submit(
        &mut self,
        img: &Image<u8, 3>,
        out: &mut RaCoAlikedResult,
    ) -> Result<(), RaCoAlikedError> {
        if out.k != self.k {
            return Err(RaCoAlikedError::CapacityMismatch(out.k, self.k));
        }

        let (sw, sh) = (img.width(), img.height());
        let (mw, mh) = ((sw / DIM_DIVISOR) * DIM_DIVISOR, (sh / DIM_DIVISOR) * DIM_DIVISOR);
        if mw == 0 || mh == 0 {
            return Err(RaCoAlikedError::InputTooSmall(sw, sh));
        }
        let (rw, rh) = (sw as f32 / mw as f32, sh as f32 / mh as f32);

        // (Re)allocate the reused input on the shared stream when the frame's model size
        // changes — stream-ordered so it is valid in submit order.
        if self.cur != (mh, mw) {
            self.input = zeros_cuda::<f32, 4>([1, 3, mh, mw], &self.stream)?;
            self.cur = (mh, mw);
        }

        self.preproc.run(img, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;

        for (name, dst, n) in [
            ("keypoints", &out.kpts, self.k * 2),
            ("normalized_keypoints", &out.norm_kpts, self.k * 2),
            ("descriptors", &out.descs, self.k * DESC_DIM),
        ] {
            let src = tmap
                .get(name)
                .ok_or(RaCoAlikedError::MissingOutput(name))?
                .f32_ptr()? as usize as CUdeviceptr;
            let dst_raw = dst.device_ptr(self.stream.as_ref()).0;
            self.copy_k
                .launch_builder(&self.stream)
                .arg(&src)
                .arg(&(n as i32))
                .arg(&dst_raw)
                .launch_cfg(cfg_1d(n, 256))?;
        }

        out.scale = (rw, rh);
        Ok(())
    }
}
