# vision-rt

Real-time neural-vision **algorithm libraries** for NVIDIA Jetson — TensorRT
inference and GPU pre/post-processing, exposed as plain Rust types with
synchronous methods. No orchestration framework: threading, messaging, and
microservices are the application's job.

The GPU image/tensor types come from
[`kornia-rs`](https://github.com/kornia/kornia-rs); models are a workspace of
per-model crates (`vrt-xfeat`, more to come) over a shared safe core.

**Target platform:** Jetson Orin (aarch64), JetPack 6.x, TensorRT 10.3.x, CUDA 12.6.

## Workspace

| Crate (package) | Path | Role |
|---|---|---|
| `trt-sys` | `crates/trt-sys` | Raw FFI: pure-C shim over the TensorRT C++ API (bindgen), optional in-process engine builder (`builder` feature) |
| `vrt` | `crates/vrt` | Safe core: `Logger→Runtime→Engine→Session`, `ModelSession` inference, `cuda` launch helpers |
| `vrt-hub` | `crates/vrt-hub` | Model weights (Hugging Face Hub, sha256-pinned) + on-device engine cache |
| `vrt-xfeat` | `crates/vrt-xfeat` | XFeat keypoints: TRT backbone + GPU NMS / top-K / descriptor sampling / mutual-NN matching |

In Rust the crates keep short names: `use vrt::…`, `use vrt_xfeat::…`.

## Execution model

Models own their kornia `Preprocessor` and share **one CUDA stream** — one
`cudaStreamSynchronize` per `run`. A loop is just construct-then-call.

The quickest way to stand up XFeat is to let it fetch and build everything itself
(feature `hub`), or hand it a model path you already have:

```rust
use std::sync::Arc;
use vrt_xfeat::{XFeat, XFeatParams};

let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
let params = XFeatParams::new(/*top_k*/ 2048, /*threshold*/ 0.05, /*h*/ 640, /*w*/ 640);

// A) Auto-pull weights from Hugging Face (kornia/xfeat) + build/cache the engine.
//    Feature `hub`. First run needs network; later runs are cache hits.
let mut xfeat = XFeat::from_hub(stream.clone(), params.clone())?;

// B) Build from a local ONNX (feature `hub` → trtexec, or `builder` → in-process):
let mut xfeat = XFeat::from_onnx("xfeat_backbone.onnx", stream.clone(), params.clone())?;

// C) Load a prebuilt .engine (no feature needed):
let mut xfeat = XFeat::from_engine_file("xfeat.engine", stream.clone(), params.clone())?;

let result = xfeat.run(&image)?; // one sync per frame → keypoints + descriptors
```

Or drive the whole `Logger→Runtime→Engine` chain yourself and pass an `Engine`
to `XFeat::new` — see `examples/xfeat_match`.

## Models & engines

- **ONNX is the portable artifact** — distributed via Hugging Face Hub with
  sha256 pins (`vrt-hub`), never committed to this repo. XFeat weights live at
  the [`kornia/xfeat`](https://huggingface.co/kornia/xfeat) HF repo (a
  backbone-only export of **XFeat**, Potje et al. CVPR 2024 —
  [verlab/accelerated_features](https://github.com/verlab/accelerated_features);
  all model credit to the original authors). If a repo is private/gated, export `HF_TOKEN`.
- **Engines are machine-locked** (TRT version + GPU arch) and built
  **on-device** into `~/.cache/vision-rt/engines/…`. First run builds (minutes,
  once); every run after is a cache hit.

## Examples

Examples live inside the `vrt-xfeat` crate (`crates/vrt-xfeat/examples/`):

```bash
# .onnx → engine built once on device, then feature-matched across two images
cargo run --release -p vrt-xfeat --example xfeat_match -- xfeat_backbone.onnx map.jpg query.jpg out.png
cargo run --release -p vrt-xfeat --example xfeat_bench -- xfeat_backbone.onnx image.jpg 100
```

Set MAXN power mode before benchmarking: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

## Building

On Jetson everything builds out of the box (TRT headers via JetPack). **Cap the
job count** — the Orin Nano OOM-kills parallel template builds:

```bash
cargo build --release -j2
cargo test  -p vrt-hub                                # CPU-only unit tests
cargo test  -p vrt-xfeat --release -- --ignored       # GPU kernel tests (on-device)
```

Off-Jetson (no TensorRT/CUDA): `TRT_STUB=1 cargo check` / `clippy` work using a
committed bindings snapshot — nothing native is compiled or linked. This is what
CI runs on hosted runners. Env overrides: `TRT_INCLUDE_DIR`, `TRT_LIB_DIR`, `CUDA_HOME`.

## License

Apache-2.0
