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
