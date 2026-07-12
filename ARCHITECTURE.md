# Architecture

`vision-rt` is a set of **plain Rust libraries** for real-time vision on **Jetson
Orin** (aarch64, SM87) — no orchestration framework. You construct a type, then call
methods in a loop. The design borrows its async shape from NVIDIA **VPI**: a *payload*
holds pre-compiled state, *caller-owned buffers* are filled by `submit`, one
`stream.synchronize()` completes the work, then you read. Every design choice below —
the single stream, the on-device engine cache, the machine-locked artifacts — follows
from targeting one board family rather than a portable runtime.

## Crate stack

```
trt-sys      pure-C shim over TensorRT C++ (bindgen never sees C++); TRT_STUB off-Jetson
   ↓
vrt          safe core: Logger→Runtime→Engine→Session (Arc chain), ModelSession,
             cuda launch cfgs, PinnedBuffer/Stream, error types
   ↓
vrt-hub      weights (HF Hub, sha256-pinned) + on-device EngineCache (onnx→engine,
   ↓         optional guarded prebuilt-engine download)
   │
   ├─ model crates — each a payload + caller-owned output over `vrt`:
   │    vrt-rfdetr         object detection (NMS-free) + on-device GPU decode
   │    vrt-rfdetr-seg     instance segmentation (boxes + per-instance masks)
   │    vrt-rfdetr-kpts    human pose (box + 17 COCO keypoints)
   │    vrt-depth-anything metric depth + depth-at-mask/box fusion kernels
   │    vrt-xfeat          keypoints + descriptors + GPU mutual-NN matching
   │
   └─ model-free leaves (no TensorRT, no GPU model of their own):
        vrt-types    CameraIntrinsics/Extrinsics, GPU Undistorter, depth-at-mask sampling
        vrt-track    3D multi-object tracker (CPU: ByteTrack assoc + depth-gated Kalman)
        vrt-viz      CPU render (masks / boxes / BEV) + H.264 / WebSocket live view
```

`vrt-types` is a dependency-light leaf shared by the model crates and the trackers;
`vrt-track` and `vrt-viz` are pure-CPU and depend only on `vrt-types` (+ kornia image
types), so they carry no TensorRT/CUDA weight and are natural upstream candidates.

## The one-stream / one-sync model

Everything for a frame runs on **one shared `CudaStream`**. The **library never syncs
for you** — `submit` enqueues all GPU work **async** and returns; the caller issues the
single `cudaStreamSynchronize`, then reads. There are **no hidden syncs**: the backbone
(`Session::run_device_inputs_on_device`), preprocessing (kornia `Preprocessor`), and
post-processing kernels all enqueue and return; counts / detections / keypoints reach
the host via **async pinned D2H** that completes at the caller's sync. Every
`synchronize()` is explicit, in the caller's code.

## VPI-style API (payload + caller-owned output)

| Role | Type (example) | Notes |
|------|------|-------|
| Payload (created once) | `RfDetrSeg`, `DepthAnything`, `XFeat` | own kernels + scratch, reused every frame |
| Output buffer (caller-owned) | `SegResult`, `DepthResult`, `XFeatResult` | pre-allocated (`alloc_result`), reused |
| Submit (async) | `model.submit(&img, &mut result)` | writes into `result`, **no sync** |
| Sync | `stream.synchronize()` | caller-issued; one call covers all submitted work |
| Read | `result.detections()`, `.depth_image()`, `.kpts_to_host()` | valid after the sync |

There is **no sync convenience method**: the caller always owns the sync. Holding
several result buffers lets **multiple frames stay outstanding** under one sync.

## Composing multiple models (one image, one stream)

The single-model idiom extends to **N models on the same frame** with no framework.
Build every model on **one shared `Arc<CudaStream>`**; pass the **same** device
`Image<u8,3>` **by reference** to each `submit` (each preprocessor only reads it and
writes its own reused input tensor — no aliasing); enqueue any **fusion kernel last**;
then **one** `stream.synchronize()` drains everything.

The stream is an ordered FIFO, so **enqueue order *is* the dependency edge**: a fusion
kernel enqueued after two models' `submit`s is guaranteed to see both models' finished
outputs — no CUDA events, no second stream (event tracking is deliberately disabled).
Caller responsibilities: `submit` all models from the **same** frame before advancing
the source, and keep each frame / input / result buffer alive until the sync (the GPU
reads their device pointers during it). Coordinates line up because every model decodes
back to **source-pixel space**, and a full-frame `Stretch` preprocess makes cross-grid
scaling a plain `grid/src` ratio.

Worked example — the flagship `rtsp_track` pipeline: `RfDetrSeg` + `DepthAnything` on
one stream, then a `sample_masks` fusion kernel turns each instance mask into a metric
depth, all in one sync; the CPU `vrt-track` tracker and `vrt-viz` rendering follow.

## Metric 3D and the world frame

`vrt-types` carries the geometry that lifts 2D perception into metric 3D:

- **`CameraIntrinsics` (`fx, fy, cx, cy`)** — `from_hfov` builds them from a spec'd
  field of view; `unproject(px, py, z)` back-projects a pixel + metric depth to a
  camera-frame `(X, Y, Z)`.
- **`Undistorter`** — a one-shot GPU remap (`k1` barrel model) applied **before**
  seg/depth, so boxes, masks, and metric-3D are all in a rectified pinhole frame.
- **depth-at-mask sampling** — a GPU kernel that reduces each instance mask over the
  metric depth map to a per-object range (the fusion step above).

The `vrt-track` Kalman state is **3D** (`px, py, pz` + velocity), fed the mask-sampled
metric depth via `Detection::with_depth`. Association is **depth-gated** (a match is
rejected when metric ranges disagree beyond tolerance), so objects that overlap in the
image but sit at different distances don't swap IDs. `Track::metric_position(intr)`
returns camera-frame metres; `world_position(intr, extr)` applies a
**`CameraExtrinsics { r, t }`** pose to put tracks in a shared world frame — the basis
for the world-frame bird's-eye view and for multi-camera fusion.

## Multiple cameras

One physical GPU → CUDA compute is **serial**, but **NVDEC decode + VIC resize are
separate fixed-function blocks**, so N cameras *decode concurrently* and only model
inference serialises. Three patterns:

- **A — round-robin, one stream, shared models** (default on Orin Nano). One stream +
  one set of model instances; loop cameras, each with its own reused result buffers; one
  sync per camera-frame. Memory-light, works today; throughput ≈ `1/(N × per-frame GPU
  ms)` — honest, since you're GPU-bound anyway. Right for a handful of cameras.
- **B — stream + thread per camera.** Each camera gets its own thread, stream, and
  **own** model instances (`Session` is `!Sync`) → **N× engine memory**, and the single
  GPU still serialises compute → little throughput gain. Only for independent per-camera
  latency; usually avoid.
- **C — batched engine.** Re-export at `batch=N` (or dynamic batch), stack frames into
  one enqueue → best GPU utilisation, one engine + N× activations. Needs batch-aware
  decode/fusion kernels; not supported by the current fixed-`batch=1` exports. Clean for
  genlocked/same-fps cameras; for truly async cameras pattern **A** usually wins.

**Unified multi-camera view.** Patterns A/B/C run **independent per-camera pipelines**.
To fuse cameras into **one coordinate system** you supply each camera's pose
(`CameraExtrinsics`) → every camera's tracks live in one world frame, enabling a shared
world-frame BEV and cross-camera re-ID (by world-position proximity in overlapping FoV,
or by appearance for non-overlapping views). The crate DAG is already per-instance
(`vrt-types` leaf ← `vrt-track` ← `vrt-viz`), so multi-cam is a driver that instantiates
N pipelines + supplies poses, not a rework.

## XFeat data flow (worked example, `vrt-xfeat`)

```
Image<u8,3> (any size, device)
  → Preprocessor::stretch      resize to floor-of-32 (mh,mw)=(H/32*32,W/32*32), /255   [upstream XFeat]
  → TRT backbone               descriptors (1,64,mh/8,mw/8), heatmap (1,1,mh,mw), reliability
  → xfeat_score_nms            5×5 NMS, score = heatmap×reliability
  → histogram → cutoff → select   GPU top-K (capacity = result's top_k), atomic-append order
  → xfeat_sample_descs → l2_norm  64-D descriptors, unit-norm
  → async D2H of the count scalar → result.count_pin
XFeatResult: device kpts/descs/scores + count + scale(rw,rh).
kpts_to_host applies scale → keypoints in ORIGINAL pixels.
```

Matching is a **separate** concern (`matching::Matcher`): `xfeat_match_argmax` run twice
(both directions) gives mutual-NN pairs by cosine (dot, since descriptors are
unit-norm), sharing the extractor's stream. All CUDA kernels are NVRTC-JIT-compiled once
(kornia `CudaKernel`); arch is auto-detected (no hardcoded sm_87).

## Live view (`vrt-viz`)

A pure-CPU render + streaming leaf that takes **host** RGB buffers + `vrt_track::Track`s
(the caller does any GPU→host copy): `render_main` tints instance masks per track id and
draws boxes + `id depth`; `render_bev` draws a world-frame top-down floor plan (metre
grid, camera at the FoV apex, footprints sized by real width, motion trails).
`StreamServer` / `LiveStream` (feature `h264`) encode both views to **H.264** (software
x264 — Orin Nano has no NVENC) and broadcast them over one **WebSocket**; the served
page decodes them in the browser via **WebCodecs** with a small jitter buffer. Encoding
runs on a worker thread off the capture loop.

## Models & engines

ONNX is the portable artifact (HF `kornia/*`, sha256-pinned). **Engines are
machine-locked** (TRT version + GPU arch) — built on-device by `EngineCache` (cached
under `~/.cache/vision-rt/engines/…`), or a prebuilt engine downloaded **only** when its
`trt_version`+`sm` match the local box (`ModelHub::get_engine`). `trt-sys` parses the
installed TRT version into `TENSORRT_VERSION` (feeds cache keys) and warns if it's
outside the tested 10.3.x range.

## Errors & safety

Per-crate `thiserror` enums (`TrtError`, `HubError`, `RfDetrError`, `SegError`,
`KptsError`, `DepthError`, `XFeatError`, `TrackError`, `VizError`, `TypeError`);
`vrt::BoxError` for constructors that aggregate kinds. `Session` is `Send` but **not `Sync`** (drive one per thread from
a shared `Arc<Engine>`). Device pointers from TRT outputs are borrowed `OutputView`s
valid until the next inference call or the session drops — the single per-frame sync
serialises access.
