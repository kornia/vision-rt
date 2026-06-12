# vision-rt

Rust TensorRT bindings and real-time neural-vision pipelines for NVIDIA
Jetson — composable operators (in the spirit of NVIDIA VPI, oriented around
neural networks) with zero-copy camera ingest and one-sync-per-frame
execution.

Target platform: Jetson Orin (aarch64), JetPack 6.x, TensorRT 10.3.x,
CUDA 12.6.

## Workspace

| Crate (crates.io name) | Path | Role |
|---|---|---|
| `vrt-sys` | `crates/vrt-sys` | Raw FFI: pure-C shim over TensorRT C++, bindgen, optional in-process engine builder (`builder` feature, links nvonnxparser) |
| `vision-rt` | `crates/vrt` | Safe wrapper: Logger→Runtime→Engine→Session, typed `Pipeline`/`Stage`, `TensorView`, CUDA-event timing, `cuda::Kernels` nvrtc helper |
| `vrt-preproc` | `crates/vrt-preproc` | GPU letterbox RGBA→CHW (hardware bilinear via texture objects) |
| `vrt-xfeat` | `crates/vrt-xfeat` | XFeat keypoints: TRT backbone + GPU NMS / top-K compaction / descriptor sampling / mutual-NN matching |
| `vrt-yolo` | `crates/vrt-yolo` | YOLO11/v8: CPU letterbox, decode, NMS |
| `vrt-gst` | `crates/vrt-gst` | GStreamer RTSP source: NVMM DMA-BUF → CUDA zero-copy, VIC hardware resize (Jetson-only, not published) |
| `nvbuf-sys` | `crates/nvbuf-sys` | NvBufSurface helpers (Jetson-only, not published) |
| `vrt-hub` | `crates/vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |

In Rust code the crates keep short names: `use vrt::…`, `use vrt_xfeat::…`.

## Execution model

A `Pipeline` chains typed `Stage`s on **one shared CUDA stream**. Per frame:

```
source → enqueue (all GPU work, async) → one cudaStreamSynchronize → finalize (CPU postproc)
```

`.pipe()` is compile-time type-checked (`stage.Input == previous.Output`).
GPU time is measured with CUDA events (`PipelineTiming.gpu_ms`).

## Models & engines

- **ONNX is the portable artifact** — distributed via Hugging Face Hub with
  sha256 pins (`vrt-hub`), never committed to this repo.
- **Engines are machine-locked** (TRT version + GPU arch) and built
  **on-device** into `~/.cache/vision-rt/engines/<name>-<onnx_sha8>-trt<ver>-sm<cc>.engine`
  — first run builds (~minutes, once), every run after is a cache hit.

## Examples (live RTSP cameras)

```bash
# XFeat keypoint detection: pass ONNX (auto-builds engine) or a .engine
cargo run --release -p rtsp_xfeat -- model.onnx rtsp://camera/stream /tmp/out

# YOLO detection
cargo run --release -p rtsp_yolo -- yolo.engine rtsp://camera/stream
```

Set MAXN power mode before benchmarking: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

## Building

On Jetson everything builds out of the box (TRT headers via JetPack):

```bash
cargo build --release
cargo test -p vrt-yolo -p vrt-hub          # CPU-only unit tests
cargo test -p vrt-xfeat --release -- --ignored # GPU kernel tests (on-device)
```

Off-Jetson (no TensorRT): `TRT_STUB=1 cargo check` / `clippy` work using a
committed bindings snapshot — nothing native is compiled or linked.
Env overrides for custom installs: `TRT_INCLUDE_DIR`, `TRT_LIB_DIR`, `CUDA_HOME`.

## License

Apache-2.0
