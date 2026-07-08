//! `XFeat` — GPU preprocessing + TRT backbone + GPU post-processing.

use crate::postprocess::{TopkBufs, XFeatError, XFeatPostproc, XFeatResult};
use cudarc::driver::CudaSlice;
use kornia_image::Image;
use kornia_imgproc::preprocess::Preprocessor;
use kornia_tensor::{zeros_cuda, Tensor};
use std::sync::Arc;
use vrt::{BoxError, CudaStream, Engine, ModelSession};

// ── Params ────────────────────────────────────────────────────────────────────

/// Configuration for the XFeat feature extractor.
#[derive(Debug, Clone)]
pub struct XFeatParams {
    /// Maximum keypoints returned per frame.
    pub top_k: usize,
    /// Minimum NMS score for a keypoint candidate to be kept.
    pub threshold: f32,
    /// Model input height — must be a multiple of 32.
    pub h: usize,
    /// Model input width  — must be a multiple of 32.
    pub w: usize,
}

impl XFeatParams {
    pub fn new(top_k: usize, threshold: f32, h: usize, w: usize) -> Self {
        Self {
            top_k,
            threshold,
            h,
            w,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// XFeat feature extractor: GPU letterbox + TRT backbone + GPU post-processing.
///
/// A single `Image<u8, 3> → XFeatResult` algorithm: it owns its [`Preprocessor`]
/// (letterbox/normalize to the model's input size), so callers hand it a camera
/// or image surface of any resolution directly. Run it with [`run`](Self::run).
pub struct XFeat {
    model: ModelSession,
    preproc: Preprocessor,
    postproc: XFeatPostproc,
    score_dev: CudaSlice<f32>, // pre-allocated h×w NMS score buffer
    /// Model input tensor (`[1,3,h,w]` CHW FP32 device), written by `preproc.run`, reused.
    input: Tensor<f32, 4>,
    h: usize,
    w: usize,
}

impl XFeat {
    /// Build an extractor sharing `stream` with the rest of the application
    /// (one CUDA stream so a single sync per frame covers all its GPU work).
    pub fn new(
        engine: Arc<Engine>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        let model = ModelSession::new(Arc::clone(&engine), Arc::clone(&stream))?;
        let (h, w) = (params.h, params.w);
        let preproc = Preprocessor::letterbox(stream.clone())?;
        let postproc = XFeatPostproc::new(stream.clone(), params.top_k, params.threshold)?;
        let score_dev: CudaSlice<f32> = unsafe { stream.alloc(h * w)? };
        // The preprocessor writes the letterboxed frame here; the backbone reads it.
        let input = zeros_cuda::<f32, 4>([1, 3, h, w], &stream)?;

        Ok(XFeat {
            model,
            preproc,
            postproc,
            score_dev,
            input,
            h,
            w,
        })
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

    /// Access the postproc (e.g. to call `match_mutual_nn_gpu` between two results).
    pub fn postproc(&self) -> &XFeatPostproc {
        &self.postproc
    }

    /// Submit one frame's async GPU work — preprocess → backbone → NMS → top-K —
    /// and return the device [`TopkBufs`]. The texture is held in `self` until the
    /// caller syncs and reads with `XFeatPostproc::finish_topk`. [`run`](Self::run)
    /// wraps this with the sync + read.
    fn submit(&mut self, img: &Image<u8, 3>) -> Result<TopkBufs, XFeatError> {
        self.preproc.run(img, &mut self.input)?;
        let out = self.model.run(&self.input)?;
        let desc_ptr = out
            .get("descriptors")
            .ok_or(XFeatError::MissingOutput("descriptors"))?
            .f32_ptr()?;
        let heat_ptr = out
            .get("heatmap")
            .ok_or(XFeatError::MissingOutput("heatmap"))?
            .f32_ptr()?;
        let rel_ptr = out
            .get("reliability")
            .ok_or(XFeatError::MissingOutput("reliability"))?
            .f32_ptr()?;
        self.postproc
            .launch_score_nms(heat_ptr, rel_ptr, &self.score_dev, self.h, self.w)?;
        let topk = self
            .postproc
            .launch_topk(desc_ptr, &self.score_dev, self.h, self.w)?;
        Ok(topk)
    }

    /// Synchronous one-shot inference on an image (was `extract`).
    ///
    /// [`submit`](Self::submit) + one stream sync + read. Letterboxes `img` into
    /// the model input; `img` must be device-resident RGBA (any resolution).
    pub fn run(&mut self, img: &Image<u8, 3>) -> Result<XFeatResult, XFeatError> {
        let bufs = self.submit(img)?;
        self.postproc.stream().synchronize()?;
        Ok(self.postproc.finish_topk(bufs))
    }
}
