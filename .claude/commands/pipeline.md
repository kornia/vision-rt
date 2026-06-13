# pipeline

Reference for the vision-rt pipeline architecture.

## Stage trait (crates/vrt/src/pipeline.rs)

```
enqueue(&input)  — queue all GPU work on the shared stream (non-blocking)
finalize()       — CPU work after stream sync (D2H reads, top-K, NMS)
output()         — return result (only valid after finalize)
```

The pipeline calls ONE `cudaStreamSynchronize` per frame between enqueue and finalize.

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

## Adding a new stage

1. Implement `Stage<Input=PrevOutput, Output=YourType>` in your crate
2. Place GPU work in `enqueue`, CPU work in `finalize`
3. `.pipe(your_stage)` — the compiler enforces type-chaining
