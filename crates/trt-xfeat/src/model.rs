//! `XFeat` — complete pipeline: GPU preprocessing + TRT backbone + GPU post-processing.


use std::sync::Arc;
use cudarc::driver::CudaSlice;
use trt::{Engine, Runtime, Session, CudaStream, Stage, BoxError, TRTensor, TRTensorMap};
use crate::postprocess::{XFeatPostproc, XFeatResult, XFeatError};

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
/// # use trt::{Logger, Runtime};
/// # use trt::logger::Severity;
/// # use trt_xfeat::{XFeatBuilder, XFeatParams};
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

    /// Load the engine and create a session with all pipeline stages wired to its stream.
    pub fn build(self) -> Result<XFeat, XFeatError> {
        let engine  = Engine::from_file(Arc::clone(&self.runtime), &self.engine_path)?;
        let session = Session::new(Arc::clone(&engine))?;
        let stream  = session.stream().cuda_stream().clone();
        let (h, w)  = (self.params.h, self.params.w);

        let postproc  = XFeatPostproc::new(stream.clone(), self.params.top_k, self.params.threshold)?;
        let score_dev: CudaSlice<f32> = unsafe { stream.alloc(h * w)? };

        Ok(XFeat { session, postproc, score_dev, h, w, desc_ptr: None, result: None })
    }
}

// ── XFeatInferStage ───────────────────────────────────────────────────────────

/// Pipeline stage: [`TRTensor`] → [`XFeatResult`].
///
/// Runs TRT inference + GPU NMS/sampling async on the shared stream, then a
/// D2H sync for the top-K score selection (inherent to the XFeat algorithm).
/// All GPU kernels are enqueued in `enqueue`; `finalize` is a no-op.
///
/// ## Internal sync note
/// XFeat's top-K selection requires scores on the CPU, so `enqueue` syncs the
/// stream once internally.  The pipeline's outer sync after `enqueue` is
/// harmless (the stream is already idle).
pub struct XFeatInferStage {
    session:  Session,
    postproc: XFeatPostproc,
    result:   Option<XFeatResult>,
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
        let session  = Session::with_stream(Arc::clone(&engine), Arc::clone(&cuda_stream))?;
        let postproc = XFeatPostproc::new(cuda_stream, top_k, threshold)?;
        Ok(Self { session, postproc, result: None })
    }

    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        self.session.stream().cuda_stream().clone()
    }
}

impl Stage for XFeatInferStage {
    type Input  = TRTensor;
    type Output = XFeatResult;

    fn enqueue(&mut self, input: &TRTensor) -> Result<(), BoxError> {
        let shape   = input.shape_i64();
        let dev_ptr = input.as_mut_ptr();

        let views = unsafe {
            self.session.run_device_inputs_on_device(
                &[("image", dev_ptr, &shape)]
            )?
        };

        // Sync before GPU postproc reads TRT output (top-K requires CPU scores).
        self.session.stream().sync()?;

        let desc_ptr = views.get("descriptors").ok_or("no 'descriptors' output")?.f32_ptr()?;
        let heat_ptr = views.get("heatmap").ok_or("no 'heatmap' output")?.f32_ptr()?;
        let rel_ptr  = views.get("reliability").ok_or("no 'reliability' output")?.f32_ptr()?;

        let h = input.shape[2];
        let w = input.shape[3];
        self.result = Some(self.postproc.process(desc_ptr, heat_ptr, rel_ptr, h, w)?);
        Ok(())
    }

    fn output(&self) -> &XFeatResult {
        self.result.as_ref().expect("XFeatInferStage: output() called before enqueue")
    }
}

// ── XFeatPostprocStage ────────────────────────────────────────────────────────

/// Pipeline stage: [`TRTensorMap`] → [`XFeatResult`].
///
/// Pairs with [`trt::TrtInferStage`] (the backbone) in the pipeline:
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
    desc_ptr:  Option<*const f32>,
    result:    Option<XFeatResult>,
}

// SAFETY: desc_ptr is a device address owned by the upstream session's output
// buffer.  It is valid only from enqueue to the same frame's finalize — the
// session's next enqueue may reallocate it.  finalize() take()s it every frame,
// so it never dangles across frames.  Stream ordering serializes GPU access.
unsafe impl Send for XFeatPostprocStage {}

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
        Ok(Self { postproc, score_dev, h, w, desc_ptr: None, result: None })
    }
}

impl Stage for XFeatPostprocStage {
    type Input  = TRTensorMap;
    type Output = XFeatResult;

    fn enqueue(&mut self, input: &TRTensorMap) -> Result<(), BoxError> {
        let desc_ptr = input.f32("descriptors")?;
        let heat_ptr = input.f32("heatmap")?;
        let rel_ptr  = input.f32("reliability")?;

        self.postproc.launch_score_nms(heat_ptr, rel_ptr, &self.score_dev, self.h, self.w)?;

        self.desc_ptr = Some(desc_ptr);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), BoxError> {
        let desc_ptr = self.desc_ptr.take().ok_or("finalize called before enqueue")?;
        self.result = Some(
            self.postproc.process_topk_sample(desc_ptr, &self.score_dev, self.h, self.w)
                ?
        );
        Ok(())
    }

    fn output(&self) -> &XFeatResult {
        self.result.as_ref().expect("XFeatPostprocStage: output() called before finalize")
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// XFeat model: TRT backbone + GPU post-processing.
///
/// Takes a pre-processed [`TRTensor`] (CHW FP32, shape `[1,3,H,W]`) and returns
/// [`XFeatResult`] with keypoints, scores, and descriptors.
///
/// ## APIs
/// - `extract(tensor)` — synchronous one-shot use (image pairs, batch jobs)
/// - `Stage` impl — two-phase async use in a [`Pipeline`] (e.g. `rtsp_xfeat`)
///
/// [`Pipeline`]: trt::Pipeline
pub struct XFeat {
    session:   Session,
    postproc:  XFeatPostproc,
    score_dev: CudaSlice<f32>,  // pre-allocated h×w NMS score buffer
    h:         usize,
    w:         usize,
    // Two-phase state: set in enqueue, consumed in finalize
    desc_ptr:  Option<*const f32>,
    result:    Option<XFeatResult>,
}

// SAFETY: desc_ptr is a device address owned by this struct's own Session
// output buffer.  Valid from enqueue to the same frame's finalize (the next
// run may reallocate it); finalize() take()s it every frame so it never
// dangles across frames.  Stream ordering serializes GPU access.
unsafe impl Send for XFeat {}

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
    /// [`Pipeline`]: trt::Pipeline
    pub fn with_stream(
        engine: Arc<Engine>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        let session   = Session::with_stream(Arc::clone(&engine), Arc::clone(&stream))?;
        let (h, w)    = (params.h, params.w);
        let postproc  = XFeatPostproc::new(stream.clone(), params.top_k, params.threshold)?;
        let score_dev: CudaSlice<f32> = unsafe { stream.alloc(h * w)? };

        Ok(XFeat { session, postproc, score_dev, h, w, desc_ptr: None, result: None })
    }

    /// The CUDA stream used by this model.
    pub fn stream(&self) -> &trt::Stream { self.session.stream() }

    /// Access the postproc (e.g. to call `match_mutual_nn_gpu` between two results).
    pub fn postproc(&self) -> &XFeatPostproc { &self.postproc }

    /// Synchronous inference on a pre-processed CHW FP32 tensor.
    ///
    /// Runs backbone + sync + postproc in one call.  The tensor must already be on
    /// device (shape `[1, 3, H, W]`, values in `[0, 1]`).
    pub fn extract(&mut self, input: &TRTensor) -> Result<XFeatResult, XFeatError> {
        let shape   = input.shape_i64();
        let dev_ptr = input.as_mut_ptr();

        let views = unsafe {
            self.session.run_device_inputs_on_device(&[("image", dev_ptr, &shape)])?
        };
        self.session.stream().sync()?;

        let desc_ptr = views.get("descriptors").ok_or(XFeatError::MissingOutput("descriptors"))?.f32_ptr()?;
        let heat_ptr = views.get("heatmap").ok_or(XFeatError::MissingOutput("heatmap"))?.f32_ptr()?;
        let rel_ptr  = views.get("reliability").ok_or(XFeatError::MissingOutput("reliability"))?.f32_ptr()?;

        self.postproc.process(desc_ptr, heat_ptr, rel_ptr, self.h, self.w)
    }
}

// ── Stage impl ────────────────────────────────────────────────────────────────

/// `XFeat` as a pipeline stage: `TRTensor → XFeatResult`.
///
/// Internally chains TRT backbone inference + GPU NMS/sampling using the two-phase
/// pipeline contract:
/// - `enqueue`: backbone async + NMS score kernel async
/// - `finalize` (after stream sync): D2H scores → top-K → descriptor sampling + L2-norm
impl trt::Stage for XFeat {
    type Input  = TRTensor;
    type Output = XFeatResult;

    fn enqueue(&mut self, input: &TRTensor) -> Result<(), BoxError> {
        let shape   = input.shape_i64();
        let dev_ptr = input.as_mut_ptr();

        let views = unsafe {
            self.session.run_device_inputs_on_device(&[("image", dev_ptr, &shape)])?
        };

        let desc_ptr = views.get("descriptors").ok_or("no 'descriptors' output")?.f32_ptr()?;
        let heat_ptr = views.get("heatmap").ok_or("no 'heatmap' output")?.f32_ptr()?;
        let rel_ptr  = views.get("reliability").ok_or("no 'reliability' output")?.f32_ptr()?;

        self.postproc.launch_score_nms(heat_ptr, rel_ptr, &self.score_dev, self.h, self.w)?;

        self.desc_ptr = Some(desc_ptr);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), BoxError> {
        let desc_ptr = self.desc_ptr.take().ok_or("XFeat: finalize called before enqueue")?;
        self.result  = Some(
            self.postproc.process_topk_sample(desc_ptr, &self.score_dev, self.h, self.w)
                ?
        );
        Ok(())
    }

    fn output(&self) -> &XFeatResult {
        self.result.as_ref().expect("XFeat: output() called before finalize")
    }
}
