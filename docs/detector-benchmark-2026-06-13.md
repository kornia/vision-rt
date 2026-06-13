# XFeat detector — benchmark & profile (2026-06-13)

End-to-end profile of the XFeat detector path (GPU letterbox → TRT backbone →
GPU top-K), isolated from the camera transport via a synthetic static-image
source (`examples/xfeat_bench`). Current power mode (NOT MAXN_SUPER — sudo
unavailable; expect ~2–3× faster at MAXN).

## Latency (1280×720 frame → 1280×736 model, top_k=4096)

| phase | mean (ms) | note |
|-------|-----------|------|
| enqueue  | 8.72 | CPU launch — blocks on the pageable result D2H (see below) |
| gpu      | 8.72 | CUDA events: letterbox + backbone + top-K (true GPU time) |
| sync     | 0.01 | wall cudaStreamSynchronize (already drained) |
| finalize | 0.01 | host read of count/scores/xy |
| **end-to-end** | **8.7** | **≈ 114 fps** |

trtexec backbone GPU-only: **7.45 ms** @ 1280×736, **3.15 ms** @ 640×640 (opt).
So preproc + GPU top-K add ~1.2 ms; the detector is **GPU-bound by the backbone**.
Dropping to the 640×640 opt shape ~halves backbone time (→ ~320 fps GPU).

## Extra copies: none

nsys GPU MemOps: only **one-time H2D** (frame upload) + **~3 D2H/frame** (the
necessary count/scores/xy result reads). Zero device-to-device copies, no
redundant transfers. The device path source→preproc→backbone→top-K is
genuinely zero-copy (texture over device RGBA, TRT setTensorAddress, kernels
read device outputs directly, top-K stays on device).

## CUDA bottlenecks

**Fixed — cudarc event tracking.** cudarc tracks cross-stream buffer hazards by
default (an event per alloc + a wait/record per op). Our pipeline is
single-stream with one sync/frame, so this was pure overhead. Disabled it
(`CudaContext::disable_event_tracking`, safe under our single-stream invariant):

| CUDA API | before | after |
|----------|--------|-------|
| cuStreamWaitEvent | 2920 | **0** |
| cuEventRecord     | 1691 | 160 (our own gpu_ms timing) |
| cuEventCreate     | 1141 | 161 |

~95 wasted event calls/frame eliminated. Headline fps unchanged (detector is
GPU-bound, so the CPU overhead was hidden under GPU work) — but correct hygiene,
and it matters for CPU-bound models or multi-pipeline use.

**Fixed — pinned result D2H.** The async D2H of results now lands in reused
**pinned, cacheable** host buffers (`PinnedBuffer`, `cudaHostAlloc` flags=0 —
not cudarc's write-combined `alloc_pinned`, which is slow to read back). This
makes `cudaMemcpyAsync` genuinely asynchronous, so the per-frame profile is now:

| phase | pageable (before) | pinned (after) |
|-------|-------------------|----------------|
| enqueue  | 8.64 (blocked on GPU) | **0.79** (CPU launch only) |
| sync     | ~0 | **7.81** (the GPU wait, correctly attributed) |
| gpu      | 8.64 | 8.55 |
| finalize | 0.01 | 0.03 |

End-to-end total unchanged (~116 fps — GPU-bound), but the host thread is now
**free for ~7.8 ms during GPU compute** instead of blocked — the pipeline's
"async tail" is genuinely async. This enables CPU/GPU overlap and lets one CPU
thread drive multiple GPU streams; the cudaHostAlloc helpers live in trt-sys.

## Verified

GPU correctness tests (top-K, matching) and the reloc example pass with event
tracking off — confirming no synchronization regression from the single-stream
assumption.
