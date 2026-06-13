# pipeline

Reference for the vision-rt pipeline architecture.

## Operator trait (crates/vrt/src/pipeline.rs)

```
enqueue(&input, ctx) -> Pending  — queue GPU work on ctx.stream() (non-blocking);
                                    Pending is handed to the next operator's enqueue
finalize(Pending, ctx) -> Output — CPU work after stream sync (D2H, top-K, NMS);
                                    returns the result BY VALUE (no output() method)
```

The pipeline calls ONE `cudaStreamSynchronize` per frame between enqueue and
finalize. `ExecCtx` carries the shared stream + per-frame `FrameMeta`
(seq/pts_ns/source_id); it exposes no sync(). `Pipeline::next()` returns
`(Output, PipelineTiming)` by value.

## Current rtsp_xfeat pipeline

```
RtspSource          →  NvmmPreprocessStage    →  XFeat
NvmmFrame              VrtTensor (CHW FP32)      XFeatResult
RTSP → nvv4l2decoder   NVMM DMA-BUF → CUDA       TRT backbone
VIC resize 1280×720    letterbox to 1280×736      + GPU NMS + top-K
```

## PipelineTiming fields

| Field | What it measures |
|-------|-----------------|
| `source_ms`   | Wait for RTSP frame (network + decode) |
| `enqueue_ms`  | CPU submission time (always <1ms) |
| `gpu_ms`      | **Actual GPU time** (CUDA event delta — use this) |
| `sync_ms`     | Wall-clock in cudaStreamSynchronize (gpu_ms + CPU jitter) |
| `finalize_ms` | CPU top-K / NMS after sync |

## Adding a new operator

1. Implement `Operator<Input=PrevPending, Pending=.., Output=YourType>`
2. Place GPU work in `enqueue`, CPU work in `finalize`
3. `.pipe(your_op)` — the compiler enforces Input==prev::Pending
