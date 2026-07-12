---
name: vrt-pipeline-compose
description: Use when building or composing a real-time CV pipeline from vrt model crates — wiring RfDetrSeg / DepthAnything / XFeat together, running N models on one frame, going fast, or handling multiple cameras. Covers the async / caller-owned / one-stream idiom, not kernel internals.
---

# Composing a Fast vrt CV Pipeline

There is **no `Pipeline` / `Operator` framework** — a pipeline is a plain loop
that calls model methods. Speed comes from one discipline: **enqueue everything
async on one stream, sync once, read only what you need.** Get that right and
you are running at the GPU's true wall-clock; get it wrong (a hidden sync, a
per-frame alloc, a host copy you didn't need) and you pay for it every frame.

## The core loop (one image, N models, one stream)

```rust
// ONE shared stream for the whole app. Every model + the source enqueue on it.
let stream = CudaContext::new(0)?.default_stream();          // or vrt::Stream::new_standalone()
let mut det   = RfDetrSeg::from_engine_file(seg_engine, stream.clone(), conf)?;
let mut depth = DepthAnything::from_engine_file(depth_engine, stream.clone())?;

// Allocate each model's caller-owned result ONCE, reuse every frame.
let mut d = det.alloc_result()?;
let mut z = depth.alloc_result()?;

loop {
    let frame = source.next_frame()?;      // device Image<u8,3>, enqueues its copy
    let img = frame.image();

    det.submit(img, &mut d)?;              // enqueue only — NO sync, NO host copy
    depth.submit(img, &mut z)?;            // SAME &img by reference, same stream
    let zs = z.depth_image()               // depth-at-mask sampling is a DepthImage builtin
        .sample_masks(d.masks_slice(), d.mask_size(), d.count_slice(), &stream)?; // fusion enqueued LAST
    stream.synchronize()?;                 // the ONE sync drains source + both models + fusion

    let n = d.count();                     // reads pinned scalar — post-sync, cheap
    // host copies happen ONLY here, on request: d.detections()?, d.masks_host()?,
    // z.depth_host()?, stream.clone_dtoh(&zs)?
}
```

Ground truth: `crates/vrt-depth-anything/examples/detect_depth.rs:39-45` (single
frame) and `examples/rtsp_rfdetr_seg/src/main.rs:63-97` (RTSP loop + profiler).

## Why this is fast (and the rules that keep it fast)

- **No hidden syncs.** `submit` only *enqueues* preproc → backbone → decode
  kernels and returns. The stream is an ordered FIFO, so enqueue order *is* the
  dependency edge — a fusion kernel enqueued after two `submit`s is guaranteed
  to see both finished outputs, with **no CUDA events and no second stream**
  (single serial stream by design; event tracking is deliberately disabled — see
  `rust-cuda-patterns`). One `stream.synchronize()` per frame is the whole cost.
- **Buffers reused.** `alloc_result()` **once**, outside the loop; `submit`
  refills it. The reused `input: Tensor<f32,4>` inside each model is reallocated
  only on a size change. The only per-frame alloc is a tiny `count` scalar.
- **Zero-copy device image shared.** Pass the **same** `&Image<u8,3>` by
  reference to every `submit`. Each preprocessor only *reads* it and writes its
  *own* `input` — no aliasing, no divergence. Never `.to_host()` the frame in the
  hot path.
- **Results stay GPU-resident.** `d.dets_slice()` / `d.masks_slice()` /
  `z.depth_slice()` hand back `CudaSlice`s so fusion kernels consume them **on
  device**. Host transfer happens **only when you call** `detections()` /
  `masks_host()` / `depth_host()` / `clone_dtoh`.

## Fusion: detect + depth on one stream

Depth-at-mask/box sampling are **builtins on the typed device images in
`vrt-types`**: `z.depth_image().sample_masks(masks, mask_wh, live_count, &stream)`
(and `sample_boxes`, plus `Mask::sample_depth(&depth, &stream)` for a single mask;
`live_count` = the detector's on-device survivor count, e.g. `d.count_slice()`, which
gates stale capacity slots on the GPU) enqueue a GPU fusion kernel that reads the
detector's device masks **and** the depth map,
returning one z per slot valid after the single sync
(`crates/vrt-types/src/lib.rs`; call site `detect_depth.rs`). The slot count is
derived from the mask buffer — you do **not** pass a survivor count, so it never
depends on a pre-sync `count()` (which would lag a frame); just `zip` the z buffer
against `d.instances()?`, which truncates to the live count. Sample from the
**instance mask**, not the box — the box bleeds background depth. Coordinates line up
for free: every model decodes back to **source pixel space** and the full-frame
`Stretch` preprocess makes cross-grid scaling a plain `grid/src` ratio (mask grid →
depth grid → source all differ only by a scalar).

## Reading the profiler (is it actually async?)

The per-stage timers in `examples/rtsp_rfdetr_seg/src/main.rs:59-114` are the
diagnostic:

| Stage | What it measures | Healthy |
|-------|------------------|---------|
| `source` | `recv(camera)` + enqueue the un-pitch copy | small |
| `enqueue` (submit) | CPU kernel-launch cost | **≪ sync** |
| `sync` (GPU) | `stream.synchronize()` — the **real GPU wall** | the number that matters |
| `read`/`decode` | on-demand host copies / CPU post | only what you asked for |

`enqueue ≪ sync` means the launches are truly async and the GPU is the
bottleneck (correct). If `enqueue ≈ sync`, a hidden sync leaked into `submit`.
Report the **`sync`** figure as GPU time — and only at MAXN (see
`jetson-benchmarking` for power-mode discipline and PipelineTiming semantics).

## Multiple cameras

One physical GPU → CUDA compute is **serial** no matter what; `Session` is `Send`
but **`!Sync`**. But NVDEC decode + VIC resize are separate fixed-function blocks,
so N cameras *decode concurrently* while only inference serializes. See CLAUDE.md
**"Multiple cameras"** for the full treatment. Short version:

- **Pattern A — round-robin, one stream, shared models (default on Orin Nano).**
  Loop cameras, each with its own reused result buffers, one sync per
  camera-frame. Memory-light (one engine copy), no code changes. Throughput
  ≈ `1/(N × per-frame GPU ms)` — honest, you're GPU-bound anyway. Right default.
- **Pattern B — thread+stream per camera.** Own model instances per thread
  (`!Sync`) → **N× engine memory** (~2–3 cameras with seg+depth on 7.4 GB). GPU
  still serializes, so little throughput gain — only for independent per-camera
  latency. Usually avoid.
- **Pattern C — batched engine.** Re-export at `batch=N`, one enqueue. Best GPU
  utilization, but needs batch-aware decode/fusion kernels; **not** supported by
  the current fixed-`batch=1` exports.

**Async cameras + batching:** batching couples camera timing — carry
`(camera_id, timestamp)` per slot and **demux** the batched output back
per-camera; never block a batch on the slowest camera. Each camera runs its own
tracker stepping by its own frame `dt`. For truly async cameras, **A usually
wins** (batching's launch-overhead payoff is small next to model compute).

## Pitfalls

- **Keep every buffer alive until the sync.** The GPU reads the frame's, each
  model's `input`, and each result buffer's device pointers *during* the sync —
  dropping any before `stream.synchronize()` is UB.
- **`submit` all models from the SAME frame before advancing the source.**
  Otherwise a later model reads a different frame's pixels.
- **Don't re-`submit`/`run` a session before consuming its output view.** TRT
  output views alias session memory that the next `run` overwrites; results that
  must outlive a re-run are copied into the caller-owned buffer (that is why
  `DepthResult` copies the depth map out — `crates/vrt-depth-anything/src/lib.rs:60`).
- **Enqueue fusion LAST**, after every model it reads has been `submit`ted.
- **Single serial stream by design** — don't reach for multi-stream/events to
  "parallelize"; the one GPU serializes compute regardless.

## Related skills

- `jetson-benchmarking` — power mode + how to report the numbers.
- `rust-cuda-patterns` — the cudarc launch side, streams, pinned readback.
- `cuda-kernel-craft` — writing the fusion kernel itself.
- `vrt-add-model-crate` — add a new model to compose into a pipeline.
- `trt-engine-rebuild` — build the `.engine` files these constructors load.
