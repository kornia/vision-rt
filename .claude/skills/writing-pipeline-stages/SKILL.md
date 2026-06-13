---
name: writing-pipeline-stages
description: Use when adding, modifying, or debugging a Stage in the vision-rt pipeline (preprocess, inference, postprocess stages) — covers the two-phase enqueue/finalize contract, stream sharing, and Send safety for device pointers.
---

# Writing Pipeline Stages

## The two-phase contract (crates/vrt/src/pipeline.rs)

Every `Stage` runs in two phases, separated by ONE `cudaStreamSynchronize` issued by the `Pipeline` — never by the stage itself:

1. `enqueue(&input)` — submit ALL GPU work to the **shared stream**, non-blocking.
   Store any device pointers needed later in `self` (as `Option<*const f32>` etc.).
2. `finalize()` — runs after the pipeline sync. Do CPU work here: D2H reads,
   top-K selection, NMS. Store the result in `self.result: Option<Output>`.
3. `output()` — `self.result.as_ref().expect(...)`.

**Never call `stream.sync()` inside `enqueue`** unless the algorithm truly
requires CPU data mid-stage (document why if so — see `XFeatInferStage` legacy note).

## Rules

- **One stream for everything.** Construct stages with the pipeline's
  `Arc<CudaStream>` (`Session::with_stream`, `XFeatPostproc::new(stream, ...)`).
  A stage with its own stream breaks the one-sync-per-frame model.
- **Pre-allocate device buffers in the constructor** (e.g. `score_dev`),
  reuse every frame. Never allocate in `enqueue`.
- **Raw device pointers in struct fields need `unsafe impl Send`** with a
  safety comment: stream ordering enforces exclusive access. See `XFeat` and
  `XFeatPostprocStage` for the pattern.
- **Errors**: library APIs return per-crate thiserror enums (`TrtError`,
  `PreprocError`, `XFeatError`, `GstSourceError`, `HubError`); only the
  `Stage` trait uses `BoxError` (`Box<dyn Error + Send + Sync>`) so operator
  authors can use any error type — typed errors convert via plain `?`.
  Never introduce non-Send `Box<dyn Error>` returns (audit 2026-06-12).
- **Chaining is type-checked**: `pipeline.pipe(stage)` requires
  `stage::Input == previous::Output`. If types don't line up, fix the stage
  types, don't add adapter copies.

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
