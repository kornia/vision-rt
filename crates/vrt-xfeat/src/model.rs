//! `XFeat` — GPU preprocessing + TRT backbone + GPU post-processing.

use crate::postprocess::{XFeatError, XFeatPostproc, XFeatResult};
use cudarc::driver::CudaSlice;
use kornia_image::Image;
use kornia_imgproc::preprocess::Preprocessor;
use kornia_tensor::{zeros_cuda, Tensor};
use std::sync::Arc;
use vrt::{BoxError, CudaStream, Engine, ModelSession};

// ── Params ────────────────────────────────────────────────────────────────────

/// Configuration for the XFeat feature extractor.
///
/// The backbone input size is NOT configured here — matching upstream XFeat, each
/// frame is resized to its own floor-of-32 dimensions (see [`XFeat::run`]).
#[derive(Debug, Clone)]
pub struct XFeatParams {
    /// Maximum keypoints returned per frame.
    pub top_k: usize,
    /// Minimum NMS score for a keypoint candidate to be kept.
    pub threshold: f32,
}

impl XFeatParams {
    pub fn new(top_k: usize, threshold: f32) -> Self {
        Self { top_k, threshold }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// XFeat feature extractor: GPU resize/normalize + TRT backbone + GPU post-processing.
///
/// A single `Image<u8, 3> → XFeatResult` algorithm: it owns a kornia
/// [`Preprocessor`] in **stretch** mode and, matching upstream XFeat, resizes each
/// frame to its own floor-of-32 dimensions (`(H/32)*32 × (W/32)*32`), then rescales
/// keypoints back to original pixels. Callers hand it an image of any resolution.
///
/// Preprocess, TRT backbone, and post-processing (incl. matching) all run on the
/// one CUDA `stream` shared at construction — a single async submit per frame with
/// one `synchronize()` in [`run`](Self::run).
pub struct XFeat {
    model: ModelSession,
    preproc: Preprocessor,
    postproc: XFeatPostproc,
    /// The one shared stream (== the backbone session's stream); used to (re)alloc
    /// the per-frame buffers so they are stream-ordered with all other GPU work.
    stream: Arc<CudaStream>,
    /// NMS score buffer, sized to the current model dims; reallocated on size change.
    score_dev: CudaSlice<f32>,
    /// Model input tensor `[1,3,mh,mw]` CHW FP32 device, written by `preproc.run`.
    input: Tensor<f32, 4>,
    /// Model dims `(mh, mw)` the buffers are currently sized for.
    cur: (usize, usize),
    /// Keypoint capacity for results allocated by [`run`](Self::run).
    top_k: usize,
}

/// Minimum model dimension (a multiple of 32) the reused buffers are seeded with
/// in [`XFeat::new`]; the first frame reallocates them to its real floor-32 size.
const SEED_DIM: usize = 32;

impl XFeat {
    /// Build an extractor sharing `stream` with the rest of the application
    /// (one CUDA stream so a single sync per frame covers all its GPU work,
    /// including matching via a `matching::Matcher` on the same stream).
    pub fn new(
        engine: Arc<Engine>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        let model = ModelSession::new(Arc::clone(&engine), Arc::clone(&stream))?;
        let preproc = Preprocessor::stretch(stream.clone())?;
        let postproc = XFeatPostproc::new(stream.clone(), params.threshold)?;
        // Seed the reused buffers at a minimal valid size so they're always live
        // (no Option/unwrap); the first frame reallocates them to its floor-32 size.
        let input = zeros_cuda::<f32, 4>([1, 3, SEED_DIM, SEED_DIM], &stream)?;
        let score_dev = stream.alloc_zeros::<f32>(SEED_DIM * SEED_DIM)?;
        Ok(XFeat {
            model,
            preproc,
            postproc,
            stream,
            score_dev,
            input,
            cur: (SEED_DIM, SEED_DIM),
            top_k: params.top_k,
        })
    }

    /// Allocate an output buffer sized for this extractor's `top_k`, to reuse
    /// across [`submit`](Self::submit) calls (VPI-style caller-owned output).
    pub fn alloc_result(&self) -> Result<XFeatResult, BoxError> {
        Ok(XFeatResult::alloc(&self.stream, self.top_k)?)
    }

    /// Construct from a prebuilt TensorRT `.engine` file (machine-locked to this
    /// TRT version + GPU arch). Creates its own `Logger`/`Runtime`; use
    /// [`new`](Self::new) when the application already owns an [`Engine`]. No
    /// `hub`/`builder` feature required.
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        let logger = vrt::Logger::new(vrt::logger::Severity::Warning)?;
        let runtime = vrt::Runtime::new(logger)?;
        let engine = vrt::Engine::from_file(runtime, engine_path.as_ref())?;
        Self::new(engine, stream, params)
    }

    /// Build (and cache) an engine from an ONNX file, then construct. First call
    /// builds on-device (~1–5 min); later calls are cache hits keyed by ONNX
    /// content + TRT version + GPU arch. The build profile matches the XFeat
    /// backbone (dynamic input `image`, fp16). Requires feature `hub` or
    /// `builder`; with only `hub` the build runs via the `trtexec` subprocess.
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        let profile = vrt_hub::EngineProfile {
            // XFeat backbone: dynamic H×W input, downsampled ×8. min/opt/max
            // mirror examples/xfeat_match; opt = 640×640 (the common query size).
            input: Some((
                "image".into(),
                vec![1, 3, 240, 320],
                vec![1, 3, 640, 640],
                vec![1, 3, 1088, 1920],
            )),
            fp16: true,
            workspace_mb: 2048,
        };
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or_else(|| BoxError::from("onnx path is not valid UTF-8"))?;
        let engine_path =
            vrt_hub::EngineCache::default().resolve("xfeat-backbone", model_path, &profile)?;
        Self::from_engine_file(engine_path, stream, params)
    }

    /// Construct from Hugging Face (`kornia/xfeat`). Requires feature `hub`.
    ///
    /// Prefers a **prebuilt engine** when the registry lists one matching this
    /// box's TensorRT version + GPU arch (skips the on-device build entirely);
    /// otherwise pulls the pinned ONNX and builds/caches the engine locally.
    /// Network is needed only on the first run (artifacts are then cached). For a
    /// private/gated HF repo set `HF_TOKEN` in the environment.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>, params: XFeatParams) -> Result<Self, BoxError> {
        if let Some(engine) = vrt_hub::ModelHub::get_engine("xfeat-backbone")? {
            return Self::from_engine_file(engine, stream, params);
        }
        let onnx = vrt_hub::ModelHub::get("xfeat-backbone")?;
        Self::from_onnx(onnx, stream, params)
    }

    /// Submit one frame's async GPU work — resize/normalize → backbone → NMS →
    /// top-K — into the caller-owned `out`, all enqueued on the shared stream with
    /// **no sync** (VPI-style). Sync the stream once (covering any other work on
    /// it), then read `out` (its `count()`/`kpts_to_host` are valid after the
    /// sync). Reuse one `out` per frame, or hold several to keep multiple frames
    /// outstanding. [`run`](Self::run) wraps alloc + submit + sync.
    pub fn submit(&mut self, img: &Image<u8, 3>, out: &mut XFeatResult) -> Result<(), XFeatError> {
        // Upstream XFeat: resize to floor-of-32 dims, keypoints scaled back by (rw,rh).
        let (sw, sh) = (img.width(), img.height());
        let (mw, mh) = ((sw / 32) * 32, (sh / 32) * 32);
        if mw == 0 || mh == 0 {
            return Err(XFeatError::InputTooSmall(sw, sh));
        }
        let (rw, rh) = (sw as f32 / mw as f32, sh as f32 / mh as f32);

        // (Re)allocate the reused buffers on the shared stream when the frame's
        // model size changes — stream-ordered so they're valid in submit order.
        if self.cur != (mh, mw) {
            self.input = zeros_cuda::<f32, 4>([1, 3, mh, mw], &self.stream)?;
            self.score_dev = self.stream.alloc_zeros::<f32>(mh * mw)?;
            self.cur = (mh, mw);
        }

        // Preprocess (stretch resize + /255) into the reused input, then backbone.
        self.preproc.run(img, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;
        let desc_ptr = tmap
            .get("descriptors")
            .ok_or(XFeatError::MissingOutput("descriptors"))?
            .f32_ptr()?;
        let heat_ptr = tmap
            .get("heatmap")
            .ok_or(XFeatError::MissingOutput("heatmap"))?
            .f32_ptr()?;
        let rel_ptr = tmap
            .get("reliability")
            .ok_or(XFeatError::MissingOutput("reliability"))?
            .f32_ptr()?;
        self.postproc
            .launch_score_nms(heat_ptr, rel_ptr, &self.score_dev, mh, mw)?;
        self.postproc
            .launch_topk(desc_ptr, &self.score_dev, mh, mw, out)?;
        out.set_scale((rw, rh));
        Ok(())
    }

    /// Synchronous one-shot inference on a device image of any resolution.
    ///
    /// Allocates a fresh [`XFeatResult`] + [`submit`](Self::submit) + one
    /// `stream.synchronize()`. For a hot loop, reuse a result via `alloc_result` +
    /// `submit` + your own sync. Keypoints carry the model→original `scale`, so
    /// `XFeatResult::kpts_to_host` yields original-image pixels.
    pub fn run(&mut self, img: &Image<u8, 3>) -> Result<XFeatResult, XFeatError> {
        let mut res = XFeatResult::alloc(&self.stream, self.top_k)?;
        self.submit(img, &mut res)?;
        self.stream.synchronize()?;
        Ok(res)
    }
}
