---
name: jetson-benchmarking
description: Use when measuring, comparing, or reporting performance numbers on this Jetson — FPS, latency, GPU time, before/after optimization claims. Enforces power-mode discipline and correct interpretation of PipelineTiming fields.
---

# Jetson Benchmarking Discipline

## Before ANY timing run

```bash
sudo nvpmodel -q                      # must say MAXN_SUPER (mode 2)
sudo nvpmodel -m 2 && sudo jetson_clocks   # if not
```

Numbers taken in the default 15W mode are ~3× slower and **not comparable**
to MAXN numbers. Never mix them in a before/after claim.

## Reading PipelineTiming (crates/vrt/src/pipeline.rs)

| Field | Meaning | Use for |
|-------|---------|---------|
| `source_ms` | Blocking wait for RTSP frame | Detecting camera-bound pipelines (≈ frame interval when GPU is fast) |
| `enqueue_ms` | CPU kernel-launch time | Should be <1ms; higher = hidden sync/alloc in a stage |
| `gpu_ms` | **CUDA-event hardware time** | The authoritative GPU cost. Use this for optimization claims |
| `sync_ms` | Wall-clock in cudaStreamSynchronize | gpu_ms + CPU wake-up jitter; don't quote as "GPU time" |
| `finalize_ms` | CPU postproc after sync | top-K / NMS cost |

A camera-paced pipeline shows large `source_ms` (~33ms at 30fps) — the GPU
is idle waiting. Real GPU headroom = `gpu_ms`, not `1000/fps`.

## Method

- Discard the first ~20 frames (TRT warm-up, CUDA context init, cache fill).
- Report averages over ≥100 frames AND peaks; examples already print both
  every 100 frames.
- For pure-GPU comparisons (engine A vs B), use
  `/usr/src/tensorrt/bin/trtexec --loadEngine=... --shapes=image:1x3xHxW`
  which removes the camera entirely.
- Watch thermals during long runs: `tegrastats` — a thermally throttled run
  invalidates the numbers (look for falling GR3D_FREQ).

## Reference numbers (MAXN_SUPER, this machine)

- XFeat backbone FP16 @ 640×640: ~3ms GPU
- Decode+VIC path: free of GPU (NVDEC + VIC fixed-function blocks)
- 1080p camera, VIC-resized to 1280×720, padded 1280×736: backbone ~10ms
