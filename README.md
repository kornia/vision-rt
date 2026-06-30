# vision-rt

Real-time neural-vision **algorithm libraries** for NVIDIA Jetson — TensorRT
inference, GPU pre/post-processing, and 2D/3D tracking, exposed as plain Rust
types with synchronous methods. No orchestration framework: threading,
messaging, and microservices are the application's job.

Sensor drivers (RTSP / OAK-D cameras) live in the separate
[`sensor-rt`](https://github.com/edgarriba/sensor-rt) workspace; the GPU
image/tensor types come from [`kornia-rs`](https://github.com/kornia/kornia-rs).

**Target platform:** Jetson Orin (aarch64), JetPack 6.x, TensorRT 10.3.x, CUDA 12.6.

## Workspace

| Crate (package) | Path | Role |
|---|---|---|
| `trt-sys` | `crates/trt-sys` | Raw FFI: pure-C shim over TensorRT C++ (bindgen), optional in-process engine builder (`builder` feature) |
| `vision-rt` | `crates/vrt` | Safe core: `Logger→Runtime→Engine→Session`, `ModelSession` inference, `Intrinsics`, `VrtDepthMap`, `stamp` (FrameMeta/Stamped/Clock), `cuda` launch helpers |
| `vrt-xfeat` | `crates/vrt-xfeat` | XFeat keypoints: TRT backbone + GPU NMS / top-K / descriptor sampling / mutual-NN matching |
| `vrt-rfdetr` | `crates/vrt-rfdetr` | RF-DETR object detector (NMS-free), on-device GPU decode |
| `vrt-rfdetr-kpts` | `crates/vrt-rfdetr-kpts` | RF-DETR human pose: per person box + 17 COCO keypoints with confidence |
| `vrt-track` | `crates/vrt-track` | Generic 2D/3D BoT-SORT tracker (nalgebra only — no GPU/TRT dependency) |
| `vrt-lift` | `crates/vrt-lift` | 2D→3D lift: back-projection through intrinsics + depth + anthropometric bone priors |
| `vrt-reid` | `crates/vrt-reid` | OSNet appearance re-identification embeddings |
| `vrt-hub` | `crates/vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |

In Rust the crates keep short names: `use vrt::…`, `use vrt_xfeat::…`.

## Execution model

Models own their kornia `Preprocessor` and share **one CUDA stream** — one
`cudaStreamSynchronize` per `run`. A loop is just construct-then-call:

```rust
use vrt::{Engine, Logger, Runtime, Stream, logger::Severity};
use vrt_rfdetr::RfDetr;

let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
let engine  = Engine::from_file(runtime, &engine_path)?;      // machine-locked .engine
let stream  = Stream::new_standalone()?.cuda_stream().clone();

let mut detr = RfDetr::new(engine, stream, 0.5)?;             // conf threshold 0.5
let detections = detr.run(&image)?;                          // Vec<Detection>, one sync/frame
```

`vrt-track` is independent of the GPU stack and composes downstream:

```rust
use vrt_track::{Box2DTracker, TrackerConfig};

let mut tracker = Box2DTracker::new(TrackerConfig::default());
let tracks = tracker.update(&observations, dt_secs, None);   // stable IDs across frames
```

## Models & engines

- **ONNX is the portable artifact** — distributed via Hugging Face Hub with
  sha256 pins (`vrt-hub`), never committed to this repo.
- **Engines are machine-locked** (TRT version + GPU arch) and built
  **on-device** into `~/.cache/vision-rt/engines/…`. First run builds (minutes,
  once); every run after is a cache hit.

## Examples

```bash
cargo run --release -p rfdetr_bench       -- <model>   # RF-DETR detector latency
cargo run --release -p rfdetr_kpts_check  -- <model>   # RF-DETR pose sanity check
cargo run --release -p track_bench        -- <model>   # detector → re-id → tracker pipeline
cargo run --release -p xfeat_match        -- model.onnx map.jpg query.jpg out.png
```

Set MAXN power mode before benchmarking: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

## Building

On Jetson everything builds out of the box (TRT headers via JetPack). **Cap the
job count** — the Orin Nano OOM-kills parallel template builds:

```bash
cargo build --release -j2
cargo test -p vrt-track -p vrt-lift -p vrt-hub       # CPU-only unit tests
cargo test -p vrt-xfeat --release -- --ignored       # GPU kernel tests (on-device)
```

Off-Jetson (no TensorRT/CUDA): `TRT_STUB=1 cargo check` / `clippy` work using a
committed bindings snapshot — nothing native is compiled or linked. This is what
CI runs on hosted runners. Env overrides: `TRT_INCLUDE_DIR`, `TRT_LIB_DIR`, `CUDA_HOME`.

## License

Apache-2.0
