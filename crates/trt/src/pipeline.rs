use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use cudarc::driver::CudaStream;
use crate::buffer::Stream;
use crate::session::TensorView;
use crate::tensor::TRTensor;
use crate::{Engine, Session};

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

// ── Source ────────────────────────────────────────────────────────────────────

/// A source of pipeline input frames (e.g. [`RtspSource`](trt_gst::RtspSource)).
pub trait Source {
    type Frame;
    fn next_frame(&mut self) -> Option<Self::Frame>;
}

// ── Stage ─────────────────────────────────────────────────────────────────────

/// A typed, stream-bound pipeline stage.
///
/// ## Two-phase execution
/// 1. `enqueue` — queue all GPU work on the shared stream (non-blocking).
/// 2. `finalize` — called by the pipeline *after* stream sync; run CPU-side
///    work (D2H reads, NMS, etc.). Default: no-op.
///
/// The pipeline calls `output()` only after `finalize()` completes.
pub trait Stage {
    type Input;
    type Output;

    fn enqueue(&mut self, input: &Self::Input) -> Result<(), BoxError>;
    fn finalize(&mut self) -> Result<(), BoxError> { Ok(()) }
    fn output(&self) -> &Self::Output;
}

// ── Chain ─────────────────────────────────────────────────────────────────────

/// Two stages composed in sequence.  Created by [`Pipeline::chain`].
pub struct Chain<A, B> {
    pub(crate) a: A,
    pub(crate) b: B,
}

impl<A, B> Stage for Chain<A, B>
where
    A: Stage,
    B: Stage<Input = A::Output>,
{
    type Input  = A::Input;
    type Output = B::Output;

    fn enqueue(&mut self, input: &A::Input) -> Result<(), BoxError> {
        self.a.enqueue(input)?;
        self.b.enqueue(self.a.output())
    }

    fn finalize(&mut self) -> Result<(), BoxError> {
        self.a.finalize()?;
        self.b.finalize()
    }

    fn output(&self) -> &B::Output { self.b.output() }
}

// ── TRTensorMap ───────────────────────────────────────────────────────────────

/// Device-side output map from a TRT inference stage.
///
/// Keys are output tensor names; values are typed [`TensorView`]s carrying
/// the device pointer plus resolved shape/dtype/byte-length, so downstream
/// stages never re-derive dimensions out of band.
///
/// Views are valid until the owning session's next `enqueue` (or drop) —
/// consume them within the same frame, never store them across frames.
pub struct TRTensorMap(HashMap<String, TensorView>);

impl TRTensorMap {
    pub fn new(views: HashMap<String, TensorView>) -> Self { Self(views) }

    /// Typed view of a named output tensor.
    pub fn get(&self, name: &str) -> Option<&TensorView> {
        self.0.get(name)
    }

    /// Device pointer of a named FP32 output, dtype-checked.
    pub fn f32(&self, name: &str) -> Result<*const f32, BoxError> {
        self.0.get(name)
            .ok_or_else(|| format!("no output tensor '{name}'"))?
            .f32_ptr()
            .map_err(Into::into)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

// ── TrtInferStage ─────────────────────────────────────────────────────────────

/// Generic TRT backbone stage: [`TRTensor`] → [`TRTensorMap`].
///
/// Outputs stay on the GPU — a downstream postprocessing stage reads them
/// either via device kernels (no copy) or async D2H in its own `enqueue`.
pub struct TrtInferStage {
    session:    Session,
    input_name: String,
    outputs:    TRTensorMap,
}

impl TrtInferStage {
    /// Create a stage that shares `cuda_stream` with all other pipeline stages.
    pub fn new(
        engine:      Arc<Engine>,
        input_name:  impl Into<String>,
        cuda_stream: Arc<CudaStream>,
    ) -> crate::error::Result<Self> {
        let session = Session::with_stream(engine, cuda_stream)?;
        Ok(Self {
            session,
            input_name: input_name.into(),
            outputs: TRTensorMap::new(HashMap::new()),
        })
    }

    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        self.session.stream().cuda_stream().clone()
    }
}

impl Stage for TrtInferStage {
    type Input  = TRTensor;
    type Output = TRTensorMap;

    fn enqueue(&mut self, input: &TRTensor) -> Result<(), BoxError> {
        let shape = input.shape_i64();
        let dev_ptr = input.as_mut_ptr();
        let views = unsafe {
            self.session.run_device_inputs_on_device(
                &[(self.input_name.as_str(), dev_ptr, &shape)]
            )?
        };
        self.outputs = TRTensorMap::new(views);
        Ok(())
    }

    fn output(&self) -> &TRTensorMap { &self.outputs }
}

// ── PipelineTiming ────────────────────────────────────────────────────────────

/// Timing breakdown of one `next` call, in milliseconds.
///
/// | Field | Phase |
/// |-------|-------|
/// | `source_ms`   | Blocking wait for the next frame (RTSP decode + network) |
/// | `enqueue_ms`  | CPU time to submit all GPU work (kernel launches — always < 1 ms) |
/// | `gpu_ms`      | Actual GPU execution time measured by CUDA events (hardware timestamp delta) |
/// | `sync_ms`     | Wall-clock time in `cudaStreamSynchronize` (GPU time + CPU wake-up jitter) |
/// | `finalize_ms` | CPU post-processing after the sync (e.g. top-K selection) |
///
/// `gpu_ms` is the authoritative GPU metric.  `sync_ms` ≥ `gpu_ms` due to CPU scheduling.
#[derive(Debug, Clone, Default)]
pub struct PipelineTiming {
    pub source_ms:   f64,
    pub enqueue_ms:  f64,
    /// Actual GPU execution time from CUDA events — more accurate than `sync_ms`.
    pub gpu_ms:      f64,
    pub sync_ms:     f64,
    pub finalize_ms: f64,
}

impl PipelineTiming {
    /// Sum of all phases (uses wall-clock `sync_ms`, not `gpu_ms`).
    pub fn total_ms(&self) -> f64 {
        self.source_ms + self.enqueue_ms + self.sync_ms + self.finalize_ms
    }
}

impl std::fmt::Display for PipelineTiming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "source={:.1}ms  enqueue={:.2}ms  gpu={:.2}ms  sync={:.1}ms  finalize={:.2}ms  total={:.1}ms",
            self.source_ms, self.enqueue_ms, self.gpu_ms, self.sync_ms, self.finalize_ms, self.total_ms(),
        )
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// An assembled source + stage chain sharing one CUDA stream.
///
/// ## Execution model per frame
/// 1. `source.next_frame()` — block until a new frame arrives (CPU).
/// 2. `stage.enqueue(frame)` — queue all GPU work (non-blocking).
/// 3. `stream.sync()` — drain the GPU; one sync per frame.
/// 4. `stage.finalize()` — CPU postprocessing on the flushed results.
/// 5. Return `stage.output()`.
///
/// ## Building a pipeline
/// ```no_run
/// # use trt::{Pipeline, TrtInferStage};
/// # let (stream, source, preproc, backbone, postproc) = todo!();
/// let mut pipeline = Pipeline::new(stream, source)
///     .pipe(preproc)
///     .pipe(backbone)
///     .pipe(postproc);
///
/// while let Some(Ok((output, timing))) = pipeline.next() { }
/// ```
pub struct Pipeline<Src, Stg> {
    source: Src,
    stage:  Stg,
    stream: Arc<CudaStream>,
}

// ── Builder state (no stage attached yet) ────────────────────────────────────

impl<Src: Source> Pipeline<Src, ()> {
    /// Create a pipeline from a source.  Attach stages with [`.pipe()`](Pipeline::pipe).
    pub fn new(stream: Arc<CudaStream>, source: Src) -> Self {
        Pipeline { source, stage: (), stream }
    }

    /// Attach the first stage.  Its `Input` must match the source's `Frame` type.
    pub fn pipe<Stg>(self, stage: Stg) -> Pipeline<Src, Stg>
    where
        Stg: Stage<Input = Src::Frame>,
    {
        Pipeline { source: self.source, stage, stream: self.stream }
    }
}

// ── Runnable pipeline ─────────────────────────────────────────────────────────

impl<Src, Stg> Pipeline<Src, Stg>
where
    Src: Source,
    Stg: Stage<Input = Src::Frame>,
{
    /// Append a stage after all existing stages.
    ///
    /// The new stage's `Input` must match the current tail stage's `Output`.
    pub fn pipe<T>(self, next: T) -> Pipeline<Src, Chain<Stg, T>>
    where
        T: Stage<Input = Stg::Output>,
    {
        Pipeline {
            source: self.source,
            stage:  Chain { a: self.stage, b: next },
            stream: self.stream,
        }
    }

    /// Advance by one frame: source → enqueue → sync → finalize → output.
    ///
    /// Returns `None` when the source is exhausted.  On success returns the
    /// stage output and a per-phase [`PipelineTiming`] breakdown.
    ///
    /// GPU time is measured with CUDA events bracketing the `enqueue` call:
    /// both events are recorded on the shared stream so `gpu_ms` reflects
    /// only the GPU execution portion, free of CPU scheduling jitter.
    ///
    /// ## Error safety
    /// Any error return first drains the stream (best-effort sync).  This is
    /// load-bearing: a partial `enqueue` may have launched kernels that read
    /// stage-held resources (NVMM imports, texture objects, the source frame),
    /// and those are dropped/replaced as soon as this call returns.  Returning
    /// with work in flight would be a use-after-free on the GPU timeline.
    pub fn next(&mut self) -> Option<Result<(&Stg::Output, PipelineTiming), BoxError>> {
        let t0 = Instant::now();
        let frame = self.source.next_frame()?;
        let t1 = Instant::now();

        // Drain the stream before surfacing an error — see "Error safety" above.
        let fail = |stream: &Arc<CudaStream>, e: BoxError| {
            let _ = stream.synchronize();
            Some(Err(e))
        };

        // Place a start-marker on the stream before queuing any GPU work.
        // CU_EVENT_DEFAULT explicitly: cudarc's `None` means DISABLE_TIMING,
        // which makes elapsed_ms fail (and gpu_ms silently 0).
        let timing_flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let gpu_start = match self.stream.record_event(timing_flags) {
            Ok(e)  => e,
            Err(e) => return fail(&self.stream, e.into()),
        };

        if let Err(e) = self.stage.enqueue(&frame) {
            return fail(&self.stream, e);
        }

        // Place a stop-marker after all GPU work has been submitted.
        let gpu_stop = match self.stream.record_event(timing_flags) {
            Ok(e)  => e,
            Err(e) => return fail(&self.stream, e.into()),
        };

        let t2 = Instant::now();
        if let Err(e) = Stream::from_cuda_stream(self.stream.clone()).sync() {
            // The sync itself failed — retry once so resources aren't freed
            // under in-flight work; a sticky CUDA error will fail again fast.
            return fail(&self.stream, e.into());
        }
        let t3 = Instant::now();

        // Both events are complete after the stream sync; elapsed_ms reads the hardware delta.
        let gpu_ms = gpu_start.elapsed_ms(&gpu_stop).unwrap_or(0.0) as f64;

        if let Err(e) = self.stage.finalize() { return Some(Err(e)); }
        let t4 = Instant::now();

        let timing = PipelineTiming {
            source_ms:   ms(t0, t1),
            enqueue_ms:  ms(t1, t2),
            gpu_ms,
            sync_ms:     ms(t2, t3),
            finalize_ms: ms(t3, t4),
        };
        Some(Ok((self.stage.output(), timing)))
    }
}

fn ms(a: Instant, b: Instant) -> f64 {
    (b - a).as_secs_f64() * 1000.0
}
