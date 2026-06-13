---
name: writing-pipeline-stages
description: Use when adding, modifying, or debugging an Operator (formerly Stage) in the vision-rt pipeline (preprocess, inference, postprocess) — covers the two-phase enqueue/finalize contract, ExecCtx, typed Pending, stream sharing, and Send safety for device pointers.
---

# Writing Pipeline Operators

## The Operator trait (crates/vrt/src/pipeline.rs)

```rust
pub trait Operator {
    type Input;
    type Pending: Send;   // handed to the NEXT operator's enqueue (pre-sync)
    type Output;          // produced by finalize (post-sync), returned by value
    fn enqueue(&mut self, input: &Self::Input, ctx: &ExecCtx) -> Result<Self::Pending, BoxError>;
    fn finalize(&mut self, pending: Self::Pending, ctx: &ExecCtx) -> Result<Self::Output, BoxError>;
}
```

Two phases, separated by ONE `cudaStreamSynchronize` the `Pipeline` issues —
never the operator itself:

1. `enqueue(&input, ctx)` — submit ALL GPU work to `ctx.stream()`, non-blocking.
   Return a **`Pending`** carrying whatever `finalize` needs (a device pointer,
   a borrowed `VrtTensor` view, …). This is the inter-operator currency: the
   next operator's `Input` **is** this `Pending`.
2. `finalize(pending, ctx)` — runs after the sync. Do CPU work (D2H, top-K, NMS)
   and **return** the `Output` by value. No `output()` method, no stored
   `Option<Output>`, no `expect()`.

Key shift from the old `Stage`: results flow through return values and the
typed `Pending`, not stashed mutable `self` state. `Chain` wires
`B::Input = A::Pending` and the compiler checks it.

**Never call `ctx.stream().synchronize()` inside `enqueue`** unless the
algorithm truly needs CPU data mid-frame (document why — see `XFeatInferStage`).
`ExecCtx` exposes the stream for launching but no `sync()` for this reason.

## Rules

- **One stream for everything.** Launch on `ctx.stream()`; construct sessions
  with the pipeline's `Arc<CudaStream>` (`Session::with_stream`). An operator
  with its own stream breaks the one-sync-per-frame model.
- **Pre-allocate device buffers in the constructor** (e.g. `score_dev`),
  reuse every frame. Never allocate in `enqueue`. To hand a reused output
  buffer downstream, return `self.output.view()` (a borrowed `VrtTensor`).
- **Raw device pointers belong in the `Pending` value, not `self`.** Wrap them
  in a small `Send` newtype (see `DescPending` in vrt-xfeat) with a SAFETY note
  — one honest `unsafe impl Send` on a data-only handle beats one on the whole
  operator. Resources that must outlive the sync (TextureGuard, CudaMemory
  import) still live in `self`, dropped in `finalize`.
- **Errors**: library APIs return per-crate thiserror enums (`TrtError`,
  `PreprocError`, `XFeatError`, `GstSourceError`, `HubError`); only the
  `Operator` trait uses `BoxError` (`Box<dyn Error + Send + Sync>`) so operator
  authors can use any error type — typed errors convert via plain `?`.
  Never introduce non-Send `Box<dyn Error>` returns (audit 2026-06-12).
- **FrameMeta**: `ctx.frame()` gives `{seq, pts_ns, source_id}` — use it for
  tracking / multi-camera correlation instead of threading your own counter.
- **Chaining is type-checked**: `pipeline.pipe(op)` requires
  `op::Input == previous::Pending`. If types don't line up, fix the operator
  types, don't add adapter copies.

## ModelSession — the TRT-backed operator shortcut

A new model operator holds a `ModelSession` (not a raw `Session`) and its
enqueue is essentially two lines + kernels:
```rust
let out = self.model.run(input)?;       // safe: no unsafe, auto input-name
let p = out.f32("output_name")?;        // dtype-checked device pointer
```
`ModelSession::new(engine, stream)` / `load(runtime, path)`; `run` (single
input) / `run_inputs` (multi); returns `TrtError` so typed-error operators
convert via `?`. This is the ~20-line-operator path — don't hand-roll
`run_device_inputs_on_device` + manual output extraction.

## Source / Sink boundaries

Operators are the transforms in the middle. The boundaries are separate traits:
- `Source` (input): pull-generator, `next_frame() -> Option<Frame>`, ends the loop.
- `Sink` (output): `consume(output, &FrameMeta)` — draw / save / publish / match
  against a map. A Sink is NOT an operator (a chained operator only sees the
  upstream's Pending, never the finalized Output — that's the pipeline boundary).
  `pipeline.drive(&mut sink)` runs Source → operators → Sink to exhaustion.
  The SLAM reloc matcher is a Sink.

## Separation of concerns

- Platform adapters (NVMM → VrtTensor) live in `vrt-gst` (e.g. `NvmmPreprocessStage`).
- Models (VrtTensor → result) live in their own crate (`vrt-xfeat`, `vrt-yolo`)
  and present as ONE stage even if internally backbone + postproc.
- Drop ordering matters with NVMM: release GPU texture objects in `finalize`
  BEFORE dropping the `CudaMemory` import (see `NvmmPreprocessStage` doc comment).

## Verifying a new stage

```bash
cargo check -p <crate>           # types line up?
cargo run --release -p <example> # gpu_ms in PipelineTiming shows real GPU cost
```

Watch `enqueue_ms` in the output: if it exceeds ~1ms, the stage is blocking
in enqueue (hidden sync or allocation) — fix it.
