//! `XFeat` — complete pipeline: GPU preprocessing + TRT backbone + GPU post-processing.


use std::sync::Arc;
use cudarc::driver::CudaSlice;
use vrt::{Engine, ModelSession, CudaStream, Operator, ExecCtx, BoxError, VrtTensor};
use crate::postprocess::{XFeatPostproc, XFeatResult, XFeatError, TopkBufs};

// ── Params ────────────────────────────────────────────────────────────────────

/// Configuration for the XFeat feature extractor.
#[derive(Debug, Clone)]
pub struct XFeatParams {
    /// Maximum keypoints returned per frame.
    pub top_k:     usize,
    /// Minimum NMS score for a keypoint candidate to be kept.
    pub threshold: f32,
    /// Model input height — must be a multiple of 32.
    pub h:         usize,
    /// Model input width  — must be a multiple of 32.
    pub w:         usize,
}

impl XFeatParams {
    pub fn new(top_k: usize, threshold: f32, h: usize, w: usize) -> Self {
        Self { top_k, threshold, h, w }
    }
}


// ── Model ─────────────────────────────────────────────────────────────────────

/// XFeat model: TRT backbone + GPU post-processing.
///
/// Takes a pre-processed [`VrtTensor`] (CHW FP32, shape `[1,3,H,W]`) and returns
/// [`XFeatResult`] with keypoints, scores, and descriptors.
///
/// ## APIs
/// - `extract(tensor)` — synchronous one-shot use (image pairs, batch jobs)
/// - `Operator` impl — two-phase async use in a [`Pipeline`] (e.g. `rtsp_xfeat`)
///
/// [`Pipeline`]: vrt::Pipeline
pub struct XFeat {
    model:     ModelSession,
    postproc:  XFeatPostproc,
    score_dev: CudaSlice<f32>,  // pre-allocated h×w NMS score buffer
    h:         usize,
    w:         usize,
}

impl XFeat {
    /// Pipeline constructor: shares `stream` with other stages in a [`Pipeline`].
    ///
    /// All stages must share the same stream so one `cudaStreamSynchronize` per frame
    /// covers the entire graph.
    ///
    /// [`Pipeline`]: vrt::Pipeline
    pub fn with_stream(
        engine: Arc<Engine>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        let model     = ModelSession::new(Arc::clone(&engine), Arc::clone(&stream))?;
        let (h, w)    = (params.h, params.w);
        let postproc  = XFeatPostproc::new(stream.clone(), params.top_k, params.threshold)?;
        let score_dev: CudaSlice<f32> = unsafe { stream.alloc(h * w)? };

        Ok(XFeat { model, postproc, score_dev, h, w })
    }

    /// Access the postproc (e.g. to call `match_mutual_nn_gpu` between two results).
    pub fn postproc(&self) -> &XFeatPostproc { &self.postproc }

    /// Synchronous inference on a pre-processed CHW FP32 tensor.
    ///
    /// Runs backbone + sync + postproc in one call.  The tensor must already be on
    /// device (shape `[1, 3, H, W]`, values in `[0, 1]`).
    pub fn extract(&mut self, input: &VrtTensor) -> Result<XFeatResult, XFeatError> {
        let out = self.model.run(input)?;
        let desc_ptr = out.get("descriptors").ok_or(XFeatError::MissingOutput("descriptors"))?.f32_ptr()?;
        let heat_ptr = out.get("heatmap").ok_or(XFeatError::MissingOutput("heatmap"))?.f32_ptr()?;
        let rel_ptr  = out.get("reliability").ok_or(XFeatError::MissingOutput("reliability"))?.f32_ptr()?;
        // process() launches NMS→top-K→sample on the same stream after the
        // backbone (stream-ordered) and syncs internally — no separate sync.
        self.postproc.process(desc_ptr, heat_ptr, rel_ptr, self.h, self.w)
    }
}

// ── Stage impl ────────────────────────────────────────────────────────────────

/// `XFeat` as a pipeline stage: `VrtTensor → XFeatResult`.
///
/// Internally chains TRT backbone inference + GPU NMS/sampling using the two-phase
/// pipeline contract:
/// - `enqueue`: backbone async + NMS score kernel async
/// - `finalize` (after stream sync): D2H scores → top-K → descriptor sampling + L2-norm
impl Operator for XFeat {
    type Input   = VrtTensor;
    type Pending = TopkBufs;
    type Output  = XFeatResult;

    fn enqueue(&mut self, input: &VrtTensor, _ctx: &ExecCtx) -> Result<TopkBufs, BoxError> {
        let out = self.model.run(input)?;
        let desc_ptr = out.f32("descriptors")?;
        let heat_ptr = out.f32("heatmap")?;
        let rel_ptr  = out.f32("reliability")?;

        self.postproc.launch_score_nms(heat_ptr, rel_ptr, &self.score_dev, self.h, self.w)?;
        Ok(self.postproc.launch_topk(desc_ptr, &self.score_dev, self.h, self.w)?)
    }

    fn finalize(&mut self, pending: TopkBufs, _ctx: &ExecCtx) -> Result<XFeatResult, BoxError> {
        Ok(self.postproc.finish_topk(pending))
    }
}
