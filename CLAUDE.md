# vision-rt

Standalone Rust TensorRT inference + real-time vision **algorithm libraries**
for Jetson Orin (aarch64, SM87, TensorRT 10.3.x, CUDA 12.6). Pure libraries — no
orchestration framework, no bubbaloop dependency. Sensor drivers live in the
separate `sensor-rt` repo; GPU image/tensor types come from `kornia-rs`
(pinned git dep, `cudarc` feature).

## Workspace layout

Package `vrt` (core) + `vrt-*` / `trt-sys` satellites. Short crate names — code
uses `use vrt::`, `use vrt_xfeat::`, `use trt_sys::`. Errors: per-crate
`thiserror` enums; `vrt::BoxError` for algorithm constructors that aggregate kinds.

This repo is being open-sourced under the kornia org incrementally, one model
crate per PR.

| Crate | Role |
|-------|------|
| `crates/trt-sys` | Raw FFI: pure-C shim over TensorRT C++ (bindgen never sees C++ headers) |
| `crates/vrt` | Safe core: Logger→Runtime→Engine→Session Arc chain, `ModelSession`, `cuda` launch helpers |
| `crates/vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |
| `crates/vrt-xfeat` | XFeat keypoints: backbone + GPU NMS/top-K/descriptor sampling/mutual-NN. Crate-local `examples/` (`xfeat_match`, `xfeat_bench`) + `scripts/export_xfeat_backbone.py` |
| `crates/vrt-rfdetr` | RF-DETR object detector (NMS-free) + on-device GPU decode |
| `crates/vrt-rfdetr-kpts` | RF-DETR human pose: box + 17 COCO keypoints + confidence (CPU decode) |

## Architecture in one paragraph

Each model is a plain type that owns a kornia `Preprocessor` and shares **one
CUDA stream** with the rest of the app: `run()` = enqueue all GPU work async →
ONE `cudaStreamSynchronize` → CPU post-process. `ModelSession` wraps the
Session and takes a kornia `Tensor<f32,4>` device input. `XFeat` offers
convenience constructors (`from_hub`/`from_onnx`/`from_engine_file`) over the
`vrt-hub` weight-fetch + engine-cache. No `Pipeline`/`Operator` framework —
composition is just calling methods in a loop.

## Composing multiple models (one image, one stream)

The single-model idiom extends to running **N models on the same frame** with no
framework. Build every model on **one shared `Arc<CudaStream>`**; pass the **same**
device `Image<u8,3>` **by reference** to each `submit` (each preprocessor only reads
it and writes its own reused `input` tensor — no aliasing, no divergence); enqueue
any **fusion kernel last**; then **one** `stream.synchronize()` drains everything.

The stream is an ordered FIFO, so enqueue order *is* the dependency edge: a fusion
kernel enqueued after two models' `submit`s is guaranteed to see both models'
finished outputs — no CUDA events, no second stream (single serial stream by design;
event tracking is deliberately disabled). Caller responsibilities: `submit` all
models from the **same** frame before advancing the source, and keep each frame /
`input` / result buffer alive until the sync (the GPU reads their device pointers
during it). Coordinates line up because every model decodes back to **source pixel
space** and the full-frame `Stretch` preprocess makes cross-grid scaling a plain
`grid/src` ratio. Worked example: `vrt-depth-anything`'s `detect_depth` (RF-DETR-Seg
+ Depth Anything V2 on one stream → per-instance mask-sampled metric depth, one sync).

## Multiple cameras

One physical GPU → CUDA compute is **serial** whatever you do; `Session` is `Send`
but **`!Sync`** (drive each model from one thread). But **NVDEC decode + VIC resize
are separate fixed-function blocks**, so N cameras *decode concurrently* — only model
inference serializes. Three patterns:

- **A — round-robin, one stream, shared models (default on Orin Nano).** One stream +
  one set of model instances; loop cameras, each with its own reused result buffers;
  one sync per camera-frame. Memory-light (one copy of each engine), works today, no
  code changes. Throughput ≈ `1/(N × per-frame GPU ms)` — honest, since you're
  GPU-bound anyway. Right default for a handful of cameras.
- **B — stream + thread per camera.** Each camera gets its own thread, stream, and
  **own** model instances (`!Sync`) → **N× engine memory**; on 7.4 GB that's ~2–3
  cameras with seg+depth. The single GPU still serializes compute, so little
  throughput gain — only worth it for independent per-camera latency. Usually avoid.
- **C — batched engine.** Re-export at `batch=N` (or dynamic batch), stack N frames
  into one enqueue → best GPU utilization. One engine + N× activations (cheaper than
  B). Needs batch-aware decode/fusion kernels; not supported by the current
  fixed-`batch=1` exports.

**Async cameras + batching:** batching **couples camera timing** — you must assemble
N frames. Different-fps/phase cameras break the natural batch. It's fine that a
batch's slots have different timestamps (perception is per-frame; no cross-camera
temporal fusion), but you **must** carry `(camera_id, timestamp)` per slot and
**demux** the batched output back to per-camera. Never block a batch on the slowest
camera. Assemble by: **latest-frame on a fixed tick** (loose sync, similar fps,
accept ≤1-interval staleness), **ragged dynamic batch** (batch only the cameras ready
this tick, `min=1..max=N`), or **don't batch → pattern A**. Per-camera state is
unaffected: each camera runs its **own** tracker stepping the Kalman by **its own
frame `dt`** (that camera's timestamps), not the batch cadence — IDs don't cross
cameras without explicit multi-camera re-ID. For genlocked/same-fps cameras C is
clean; for truly async cameras **A usually wins** (batching's launch-overhead payoff
is small next to the model compute).

## Hard constraints

- **RAM 7.4 GB (Orin Nano): build with `-j2` / `CARGO_BUILD_JOBS=2`** — parallel
  template builds OOM-kill the box.
- `.engine` files are machine-locked (TRT version + SM87). Rebuild on-device with
  trtexec at `/usr/src/tensorrt/bin/trtexec` — never copy across hosts.
- Benchmark only at MAXN: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

## Commands

```bash
cargo build --release -j2                              # full build (capped jobs)
cargo test -p vrt-hub                                  # CPU-only unit tests
cargo test -p vrt-xfeat --release -- --ignored         # GPU kernel tests (on-device)
TRT_STUB=1 cargo clippy --all-targets -- -D warnings   # off-Jetson check (no CUDA/TRT)
```

Off-Jetson / CI: `TRT_STUB=1` makes `trt-sys` use committed bindings —
`cargo check`/`clippy`/`doc` work with nothing native compiled. kornia builds
via cudarc's `fallback-*` features (no CUDA needed to check).

## Detailed knowledge

Project skills in `.claude/skills/` cover engine rebuilds, benchmarking
discipline, Rust↔CUDA patterns, CUDA kernel craft, and model tensor semantics.
They auto-activate; trust them over re-deriving from code.
