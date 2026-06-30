use crate::buffer::Stream;
use crate::model::ModelSession;
use crate::tensor::VrtTensor;
use crate::Engine;
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

// ── Source / Sink ───────────────────────────────────────────────────────────

/// The **input boundary** of a pipeline: pulls frames in until exhausted
/// (e.g. [`RtspSource`](vrt_gst::RtspSource)).
///
/// A `Source` is a pull-generator, not an [`Operator`] — it has no per-frame
/// input and signals end-of-stream by returning `None`.
pub trait Source {
    type Frame;
    fn next_frame(&mut self) -> Option<Self::Frame>;
}

/// The **output boundary** of a pipeline: consumes each frame's result.
///
/// Symmetric to [`Source`].  Where a `Source` pulls frames in and [`Operator`]s
/// transform them, a `Sink` pushes the final result out — draw it, save it,
/// publish it to a message bus, or match it against a map for relocalization.
///
/// A `Sink` is **not** an `Operator`: a chained operator only sees the
/// upstream's `Pending` (the pre-sync handle), never the finalized `Output`,
/// which exists only at the pipeline boundary.  The sink is that boundary.
///
/// ```no_run
/// # use vrt::{Sink, FrameMeta, BoxError};
/// # struct Matches; struct Reloc { map: () }
/// impl Sink for Reloc {                 // the SLAM relocalization sink
///     type Input = Matches;             // each frame's XFeatResult, matched upstream
///     fn consume(&mut self, _m: Matches, frame: &FrameMeta) -> Result<(), BoxError> {
///         // match against self.map, estimate pose, publish …
///         let _ = frame.seq;
///         Ok(())
///     }
/// }
/// ```
pub trait Sink {
    type Input;
    fn consume(&mut self, input: Self::Input, frame: &FrameMeta) -> Result<(), BoxError>;
}

// ── ExecCtx ───────────────────────────────────────────────────────────────────

/// Per-frame metadata flowing alongside the data through every operator.
///
/// Carries the information tracking, multi-camera, and synchronized pipelines
/// need but that the tensor itself doesn't hold.  Sources fill what they know;
/// unknown fields stay `None`.
#[derive(Debug, Clone, Default)]
pub struct FrameMeta {
    /// Monotonic frame counter assigned by the pipeline.
    pub seq: u64,
    /// Presentation timestamp in nanoseconds, if the source provides one.
    pub pts_ns: Option<u64>,
    /// Camera / stream identifier for multi-source pipelines.
    pub source_id: Option<u32>,
}

/// Execution context threaded through [`Operator::enqueue`] / [`Operator::finalize`].
///
/// Gives operators the shared CUDA stream to launch work on and the current
/// frame's [`FrameMeta`].  It deliberately does **not** expose a `sync()` —
/// the pipeline owns the single per-frame `cudaStreamSynchronize`, and an
/// operator syncing inside `enqueue` would break the one-sync-per-frame model.
pub struct ExecCtx {
    stream: Arc<CudaStream>,
    frame: FrameMeta,
}

impl ExecCtx {
    pub fn new(stream: Arc<CudaStream>, frame: FrameMeta) -> Self {
        Self { stream, frame }
    }
    /// The shared stream to launch kernels / TRT enqueues on (do not sync it).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    /// This frame's metadata.
    pub fn frame(&self) -> &FrameMeta {
        &self.frame
    }
}

// ── Operator ──────────────────────────────────────────────────────────────────

/// A typed, stream-bound pipeline operator (formerly `Stage`).
///
/// ## Two-phase execution
/// 1. `enqueue` — queue all GPU work on `ctx`'s stream (non-blocking) and
///    return a [`Pending`](Operator::Pending) handle.  The handle is what the
///    *next* operator consumes during its own `enqueue` (so it must be valid
///    immediately, before any sync — typically device pointers / buffer views).
/// 2. `finalize` — called by the pipeline *after* the stream sync; consumes the
///    `Pending` and returns the final [`Output`](Operator::Output) (D2H reads,
///    top-K, NMS, …).
///
/// Returning the result from `finalize` (instead of stashing it in `self` and
/// exposing a panic-prone `output()`) removes the two-phase footgun: the
/// data's enqueue→finalize lifetime is visible in the type flow.
pub trait Operator {
    type Input;
    /// Inter-operator handle produced by `enqueue`, consumed by `finalize`.
    /// Must be `Send` and valid before the stream sync.
    type Pending: Send;
    /// Final result produced by `finalize`.
    type Output;

    fn enqueue(&mut self, input: &Self::Input, ctx: &ExecCtx) -> Result<Self::Pending, BoxError>;
    fn finalize(&mut self, pending: Self::Pending, ctx: &ExecCtx)
        -> Result<Self::Output, BoxError>;
}

// ── Chain ─────────────────────────────────────────────────────────────────────

/// Two operators composed in sequence.  Created by [`Pipeline::pipe`].
///
/// The downstream operator's `Input` is the upstream's `Pending` — i.e. stages
/// are wired by what `enqueue` hands forward, type-checked at compile time.
pub struct Chain<A, B> {
    pub(crate) a: A,
    pub(crate) b: B,
}

impl<A, B> Operator for Chain<A, B>
where
    A: Operator,
    B: Operator<Input = A::Pending>,
{
    type Input = A::Input;
    type Pending = (A::Pending, B::Pending);
    type Output = B::Output;

    fn enqueue(&mut self, input: &A::Input, ctx: &ExecCtx) -> Result<Self::Pending, BoxError> {
        let pa = self.a.enqueue(input, ctx)?;
        let pb = self.b.enqueue(&pa, ctx)?;
        Ok((pa, pb))
    }

    fn finalize(&mut self, (pa, pb): Self::Pending, ctx: &ExecCtx) -> Result<B::Output, BoxError> {
        self.a.finalize(pa, ctx)?; // upstream output discarded (non-terminal)
        self.b.finalize(pb, ctx)
    }
}

// ── Fork ──────────────────────────────────────────────────────────────────────

/// Fan-out: run two operators on the **same** input, producing both outputs.
///
/// The opposite of [`Chain`] (which feeds one into the next).  Both branches
/// receive the same `&Input` in `enqueue` and run on the shared stream; the
/// combined `Output` is the tuple `(A::Output, B::Output)`.
///
/// This is the structured replacement for ad-hoc side channels (e.g. the
/// camera viz `Arc<Mutex>`): a `Fork` lets one frame drive, say, detection on
/// one branch and keypoints on the other, or compute on one and a passthrough
/// for visualization on the other.
///
/// ```no_run
/// # use vrt::{Fork, Operator};
/// # fn build<A, B, I>(a: A, b: B) -> Fork<A, B>
/// # where A: Operator<Input = I>, B: Operator<Input = I> {
/// Fork::new(a, b)   // Input = I, Output = (A::Output, B::Output)
/// # }
/// ```
pub struct Fork<A, B> {
    a: A,
    b: B,
}

impl<A, B> Fork<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I, A, B> Operator for Fork<A, B>
where
    A: Operator<Input = I>,
    B: Operator<Input = I>,
{
    type Input = I;
    type Pending = (A::Pending, B::Pending);
    type Output = (A::Output, B::Output);

    fn enqueue(&mut self, input: &I, ctx: &ExecCtx) -> Result<Self::Pending, BoxError> {
        let pa = self.a.enqueue(input, ctx)?;
        let pb = self.b.enqueue(input, ctx)?;
        Ok((pa, pb))
    }

    fn finalize(
        &mut self,
        (pa, pb): Self::Pending,
        ctx: &ExecCtx,
    ) -> Result<Self::Output, BoxError> {
        let oa = self.a.finalize(pa, ctx)?;
        let ob = self.b.finalize(pb, ctx)?;
        Ok((oa, ob))
    }
}

// ── TRTensorMap ───────────────────────────────────────────────────────────────

/// Device-side output map from a TRT inference stage.
///
/// Keys are output tensor names; values are borrowed [`VrtTensor`]s carrying
/// the device pointer plus resolved shape/dtype/byte-length, so downstream
/// stages never re-derive dimensions out of band.
///
/// The tensors are valid until the owning session's next `enqueue` (or drop) —
/// consume them within the same frame, never store them across frames.
pub struct TRTensorMap(HashMap<String, VrtTensor>);

impl TRTensorMap {
    pub fn new(views: HashMap<String, VrtTensor>) -> Self {
        Self(views)
    }

    /// Borrowed tensor for a named output.
    pub fn get(&self, name: &str) -> Option<&VrtTensor> {
        self.0.get(name)
    }

    /// Device pointer of a named FP32 output, dtype-checked.
    pub fn f32(&self, name: &str) -> Result<*const f32, BoxError> {
        self.0
            .get(name)
            .ok_or_else(|| format!("no output tensor '{name}'"))?
            .f32_ptr()
            .map_err(Into::into)
    }
}

// ── TrtInferStage ─────────────────────────────────────────────────────────────

/// Generic TRT backbone stage: [`VrtTensor`] → [`TRTensorMap`].
///
/// Outputs stay on the GPU — a downstream postprocessing stage reads them
/// either via device kernels (no copy) or async D2H in its own `enqueue`.
pub struct TrtInferStage {
    model: ModelSession,
    input_name: String,
}

impl TrtInferStage {
    /// Create a stage that shares `cuda_stream` with all other pipeline stages.
    pub fn new(
        engine: Arc<Engine>,
        input_name: impl Into<String>,
        cuda_stream: Arc<CudaStream>,
    ) -> crate::error::Result<Self> {
        let model = ModelSession::new(engine, cuda_stream)?;
        Ok(Self {
            model,
            input_name: input_name.into(),
        })
    }
}

impl Operator for TrtInferStage {
    type Input = VrtTensor;
    type Pending = TRTensorMap;
    type Output = ();

    fn enqueue(&mut self, input: &VrtTensor, _ctx: &ExecCtx) -> Result<TRTensorMap, BoxError> {
        self.model
            .run_inputs(&[(self.input_name.as_str(), input)])
            .map_err(Into::into)
    }

    fn finalize(&mut self, _pending: TRTensorMap, _ctx: &ExecCtx) -> Result<(), BoxError> {
        Ok(())
    }
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
    pub source_ms: f64,
    pub enqueue_ms: f64,
    /// Actual GPU execution time from CUDA events — more accurate than `sync_ms`.
    pub gpu_ms: f64,
    pub sync_ms: f64,
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
/// # use vrt::{Pipeline, TrtInferStage};
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
    stage: Stg,
    stream: Arc<CudaStream>,
    seq: u64,
}

// ── Builder state (no stage attached yet) ────────────────────────────────────

impl<Src: Source> Pipeline<Src, ()> {
    /// Create a pipeline from a source.  Attach operators with [`.pipe()`](Pipeline::pipe).
    pub fn new(stream: Arc<CudaStream>, source: Src) -> Self {
        Pipeline {
            source,
            stage: (),
            stream,
            seq: 0,
        }
    }

    /// Attach the first operator.  Its `Input` must match the source's `Frame` type.
    pub fn pipe<Stg>(self, stage: Stg) -> Pipeline<Src, Stg>
    where
        Stg: Operator<Input = Src::Frame>,
    {
        Pipeline {
            source: self.source,
            stage,
            stream: self.stream,
            seq: 0,
        }
    }
}

// ── Runnable pipeline ─────────────────────────────────────────────────────────

impl<Src, Stg> Pipeline<Src, Stg>
where
    Src: Source,
    Stg: Operator<Input = Src::Frame>,
{
    /// Append an operator after all existing ones.
    ///
    /// The new operator's `Input` must match the current tail's `Pending` —
    /// i.e. what the tail's `enqueue` hands forward.
    pub fn pipe<T>(self, next: T) -> Pipeline<Src, Chain<Stg, T>>
    where
        T: Operator<Input = Stg::Pending>,
    {
        Pipeline {
            source: self.source,
            stage: Chain {
                a: self.stage,
                b: next,
            },
            stream: self.stream,
            seq: 0,
        }
    }

    /// Run the pipeline to exhaustion, pushing each frame's [`Output`] into
    /// `sink`.  Returns when the source ends, or on the first error (from a
    /// pipeline stage or the sink).
    ///
    /// This is the symmetric counterpart to the [`Source`] driving the front:
    /// `Source → [Operator…] → Sink`, with the pipeline owning the loop and the
    /// single per-frame sync.  For per-frame timing or custom control flow, use
    /// [`next`](Pipeline::next) directly instead.
    ///
    /// [`Output`]: Operator::Output
    pub fn drive<S>(&mut self, sink: &mut S) -> Result<(), BoxError>
    where
        S: Sink<Input = Stg::Output>,
    {
        while let Some(res) = self.next() {
            let (output, _timing) = res?;
            let meta = FrameMeta {
                seq: self.seq,
                ..FrameMeta::default()
            };
            sink.consume(output, &meta)?;
        }
        Ok(())
    }

    /// Advance by one frame: source → enqueue → sync → finalize → output.
    ///
    /// Returns `None` when the source is exhausted.  On success returns the
    /// operator's [`Output`](Operator::Output) **by value** and a per-phase
    /// [`PipelineTiming`] breakdown.
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
    #[allow(clippy::should_implement_trait)] // pairs Output with timing; Iterator's next() can't
    pub fn next(&mut self) -> Option<Result<(Stg::Output, PipelineTiming), BoxError>> {
        let t0 = Instant::now();
        let frame = self.source.next_frame()?;
        let t1 = Instant::now();

        self.seq += 1;
        let ctx = ExecCtx::new(
            self.stream.clone(),
            FrameMeta {
                seq: self.seq,
                ..FrameMeta::default()
            },
        );

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
            Ok(e) => e,
            Err(e) => return fail(&self.stream, e.into()),
        };

        let pending = match self.stage.enqueue(&frame, &ctx) {
            Ok(p) => p,
            Err(e) => return fail(&self.stream, e),
        };

        // Place a stop-marker after all GPU work has been submitted.
        let gpu_stop = match self.stream.record_event(timing_flags) {
            Ok(e) => e,
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

        let output = match self.stage.finalize(pending, &ctx) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let t4 = Instant::now();

        let timing = PipelineTiming {
            source_ms: ms(t0, t1),
            enqueue_ms: ms(t1, t2),
            gpu_ms,
            sync_ms: ms(t2, t3),
            finalize_ms: ms(t3, t4),
        };
        Some(Ok((output, timing)))
    }
}

fn ms(a: Instant, b: Instant) -> f64 {
    (b - a).as_secs_f64() * 1000.0
}

#[cfg(test)]
mod fork_tests {
    use super::*;
    use crate::buffer::Stream;

    // Trivial operators (no GPU work) to exercise the combinator wiring:
    // enqueue passes the value through as Pending; finalize transforms it.
    struct Doubler;
    impl Operator for Doubler {
        type Input = i32;
        type Pending = i32;
        type Output = i32;
        fn enqueue(&mut self, input: &i32, _ctx: &ExecCtx) -> Result<i32, BoxError> {
            Ok(*input)
        }
        fn finalize(&mut self, p: i32, _ctx: &ExecCtx) -> Result<i32, BoxError> {
            Ok(p * 2)
        }
    }
    struct Negator;
    impl Operator for Negator {
        type Input = i32;
        type Pending = i32;
        type Output = i32;
        fn enqueue(&mut self, input: &i32, _ctx: &ExecCtx) -> Result<i32, BoxError> {
            Ok(*input)
        }
        fn finalize(&mut self, p: i32, _ctx: &ExecCtx) -> Result<i32, BoxError> {
            Ok(-p)
        }
    }

    /// Fork runs both branches on the same input and returns both outputs.
    /// Needs a CUDA context only to build an ExecCtx; run on-device:
    ///   cargo test -p vision-rt -- --ignored
    #[test]
    #[ignore]
    fn fork_runs_both_branches() {
        let stream = Stream::new_standalone().unwrap().cuda_stream().clone();
        let ctx = ExecCtx::new(stream, FrameMeta::default());

        let mut fork = Fork::new(Doubler, Negator);
        let pending = fork.enqueue(&7, &ctx).unwrap();
        let (doubled, negated) = fork.finalize(pending, &ctx).unwrap();
        assert_eq!(doubled, 14);
        assert_eq!(negated, -7);
    }
}

#[cfg(test)]
mod sink_tests {
    use super::*;
    use crate::buffer::Stream;
    use std::sync::{Arc, Mutex};

    struct CountSource {
        n: u32,
        max: u32,
    }
    impl Source for CountSource {
        type Frame = u32;
        fn next_frame(&mut self) -> Option<u32> {
            if self.n >= self.max {
                return None;
            }
            self.n += 1;
            Some(self.n)
        }
    }
    struct Doubler;
    impl Operator for Doubler {
        type Input = u32;
        type Pending = u32;
        type Output = u32;
        fn enqueue(&mut self, input: &u32, _: &ExecCtx) -> Result<u32, BoxError> {
            Ok(*input)
        }
        fn finalize(&mut self, p: u32, _: &ExecCtx) -> Result<u32, BoxError> {
            Ok(p * 2)
        }
    }
    struct CollectSink {
        got: Arc<Mutex<Vec<(u64, u32)>>>,
    }
    impl Sink for CollectSink {
        type Input = u32;
        fn consume(&mut self, input: u32, frame: &FrameMeta) -> Result<(), BoxError> {
            self.got.lock().unwrap().push((frame.seq, input));
            Ok(())
        }
    }

    /// drive() pulls Source → Operator → Sink to exhaustion, with frame seq.
    /// Run on-device: cargo test -p vision-rt -- --ignored
    #[test]
    #[ignore]
    fn drive_source_to_sink() {
        let stream = Stream::new_standalone().unwrap().cuda_stream().clone();
        let got = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = Pipeline::new(stream, CountSource { n: 0, max: 3 }).pipe(Doubler);
        let mut sink = CollectSink { got: got.clone() };
        pipeline.drive(&mut sink).unwrap();
        // frames 1,2,3 doubled → 2,4,6, tagged with seq 1,2,3.
        assert_eq!(*got.lock().unwrap(), vec![(1, 2), (2, 4), (3, 6)]);
    }
}
