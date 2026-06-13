//! `XFeat` — complete pipeline: GPU preprocessing + TRT backbone + GPU post-processing.


use std::sync::Arc;
use cudarc::driver::CudaSlice;
use vrt::{Engine, Runtime, ModelSession, CudaStream, Operator, ExecCtx, BoxError, VrtTensor, TRTensorMap};
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

// ── Builder ───────────────────────────────────────────────────────────────────

/// Builder for [`XFeat`].
///
/// Accepts a [`Runtime`] and an engine file path; `build()` loads the engine
/// and creates a `Session` internally so all pipeline stages share one CUDA stream.
///
/// # Example
/// ```no_run
/// # use std::sync::Arc;
/// # use vrt::{Logger, Runtime};
/// # use vrt::logger::Severity;
/// # use vrt_xfeat::{XFeatBuilder, XFeatParams};
/// let logger  = Logger::new(Severity::Warning).unwrap();
/// let runtime = Runtime::new(logger).unwrap();
/// let params  = XFeatParams::new(4096, 0.05, 736, 1280);
/// let xfeat   = XFeatBuilder::new(runtime, "xfeat.engine", params).build().unwrap();
/// ```
pub struct XFeatBuilder {
    runtime:     Arc<Runtime>,
    engine_path: String,
    params:      XFeatParams,
}

impl XFeatBuilder {
    pub fn new(runtime: Arc<Runtime>, engine_path: impl Into<String>, params: XFeatParams) -> Self {
        Self { runtime, engine_path: engine_path.into(), params }
    }

    /// Load the engine and create a model session with all pipeline stages
    /// wired to one shared CUDA stream.
    pub fn build(self) -> Result<XFeat, XFeatError> {
        let engine = Engine::from_file(Arc::clone(&self.runtime), &self.engine_path)?;
        let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
        let model  = ModelSession::new(engine, stream.clone())?;
        let (h, w) = (self.params.h, self.params.w);

        let postproc  = XFeatPostproc::new(stream.clone(), self.params.top_k, self.params.threshold)?;
        let score_dev: CudaSlice<f32> = unsafe { stream.alloc(h * w)? };

        Ok(XFeat { model, postproc, score_dev, h, w })
    }
}

// ── XFeatInferStage ───────────────────────────────────────────────────────────

/// Pipeline stage: [`VrtTensor`] → [`XFeatResult`].
///
/// Backbone + GPU NMS + GPU top-K + descriptor sampling are all enqueued async
/// on the shared stream in `enqueue` (returning a [`TopkBufs`] handle); the
/// result is assembled in `finalize` after the pipeline's single sync.  No
/// internal `stream.sync()` — the top-K runs entirely on the GPU.
pub struct XFeatInferStage {
    model:    ModelSession,
    postproc: XFeatPostproc,
}

impl XFeatInferStage {
    /// Create an XFeat inference+postproc stage sharing `cuda_stream`.
    ///
    /// `top_k` and `threshold` are the keypoint selection parameters.
    pub fn new(
        engine:      Arc<Engine>,
        cuda_stream: Arc<CudaStream>,
        top_k:       usize,
        threshold:   f32,
    ) -> Result<Self, BoxError> {
        let model    = ModelSession::new(Arc::clone(&engine), Arc::clone(&cuda_stream))?;
        let postproc = XFeatPostproc::new(cuda_stream, top_k, threshold)?;
        Ok(Self { model, postproc })
    }

    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        self.model.cuda_stream()
    }
}

impl Operator for XFeatInferStage {
    type Input   = VrtTensor;
    type Pending = TopkBufs;
    type Output  = XFeatResult;

    fn enqueue(&mut self, input: &VrtTensor, _ctx: &ExecCtx) -> Result<TopkBufs, BoxError> {
        let out = self.model.run(input)?;
        let desc_ptr = out.f32("descriptors")?;
        let heat_ptr = out.f32("heatmap")?;
        let rel_ptr  = out.f32("reliability")?;

        let (h, w) = (input.dim(2), input.dim(3));
        // Per-frame NMS score buffer (this stage has no fixed h/w). Dropped at
        // the end of enqueue; cudarc's stream-ordered free runs after the
        // kernels that read it, so it stays valid for the launched work.
        let score_dev: CudaSlice<f32> = unsafe { self.postproc.stream().alloc(h * w)? };
        self.postproc.launch_score_nms(heat_ptr, rel_ptr, &score_dev, h, w)?;
        Ok(self.postproc.launch_topk(desc_ptr, &score_dev, h, w)?)
    }

    fn finalize(&mut self, pending: TopkBufs, _ctx: &ExecCtx) -> Result<XFeatResult, BoxError> {
        Ok(self.postproc.finish_topk(pending))
    }
}

// ── XFeatPostprocStage ────────────────────────────────────────────────────────

/// Pipeline stage: [`TRTensorMap`] → [`XFeatResult`].
///
/// Pairs with [`vrt::TrtInferStage`] (the backbone) in the pipeline:
/// ```text
/// NvmmPreprocessStage  →  TrtInferStage("image")  →  XFeatPostprocStage
/// NvmmFrame               TRTensorMap                 XFeatResult
/// ```
///
/// ## Two-phase execution
/// - `enqueue`: launches the NMS score kernel async onto the shared stream.
/// - `finalize` (after pipeline sync): D2H scores → top-K → descriptor
///   sampling + L2-norm → internal sync → stores [`XFeatResult`].
pub struct XFeatPostprocStage {
    postproc:  XFeatPostproc,
    score_dev: CudaSlice<f32>,  // pre-allocated; reused every frame
    h:         usize,
    w:         usize,
}

impl XFeatPostprocStage {
    /// Create the stage.
    ///
    /// `h` / `w` are the backbone input dimensions (multiples of 32) — must
    /// match the shapes the TRT engine was built with.
    pub fn new(
        stream:    Arc<CudaStream>,
        top_k:     usize,
        threshold: f32,
        h:         usize,
        w:         usize,
    ) -> Result<Self, BoxError> {
        let postproc = XFeatPostproc::new(stream.clone(), top_k, threshold)?;
        let score_dev: CudaSlice<f32> = unsafe { stream.alloc(h * w)? };
        Ok(Self { postproc, score_dev, h, w })
    }
}

impl Operator for XFeatPostprocStage {
    type Input   = TRTensorMap;
    type Pending = TopkBufs;
    type Output  = XFeatResult;

    fn enqueue(&mut self, input: &TRTensorMap, _ctx: &ExecCtx) -> Result<TopkBufs, BoxError> {
        let desc_ptr = input.f32("descriptors")?;
        let heat_ptr = input.f32("heatmap")?;
        let rel_ptr  = input.f32("reliability")?;

        self.postproc.launch_score_nms(heat_ptr, rel_ptr, &self.score_dev, self.h, self.w)?;
        Ok(self.postproc.launch_topk(desc_ptr, &self.score_dev, self.h, self.w)?)
    }

    fn finalize(&mut self, pending: TopkBufs, _ctx: &ExecCtx) -> Result<XFeatResult, BoxError> {
        Ok(self.postproc.finish_topk(pending))
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
    /// Convenience constructor: loads the engine from `engine_path` using `runtime`.
    ///
    /// Equivalent to `XFeatBuilder::new(runtime, engine_path, params).build()`.
    pub fn new(
        runtime:     Arc<Runtime>,
        engine_path: impl Into<String>,
        params:      XFeatParams,
    ) -> Result<Self, XFeatError> {
        XFeatBuilder::new(runtime, engine_path, params).build()
    }

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

    /// The CUDA stream used by this model.
    pub fn stream(&self) -> &vrt::Stream { self.model.stream() }

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
