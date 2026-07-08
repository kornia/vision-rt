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

This repo is the **open-source xfeat chain** under the kornia org (step 1). Other
model crates (rfdetr, rfdetr-kpts, track, lift, reid, depth) live in the private
`edgarriba/vision-rt` and land here in later steps.

| Crate | Role |
|-------|------|
| `crates/trt-sys` | Raw FFI: pure-C shim over TensorRT C++ (bindgen never sees C++ headers) |
| `crates/vrt` | Safe core: Logger→Runtime→Engine→Session Arc chain, `ModelSession`, `cuda` launch helpers |
| `crates/vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |
| `crates/vrt-xfeat` | XFeat keypoints: backbone + GPU NMS/top-K/descriptor sampling/mutual-NN. Crate-local `examples/` (`xfeat_match`, `xfeat_bench`) + `scripts/export_xfeat_backbone.py` |

## Architecture in one paragraph

Each model is a plain type that owns a kornia `Preprocessor` and shares **one
CUDA stream** with the rest of the app: `run()` = enqueue all GPU work async →
ONE `cudaStreamSynchronize` → CPU post-process. `ModelSession` wraps the
Session and takes a kornia `Tensor<f32,4>` device input. `XFeat` offers
convenience constructors (`from_hub`/`from_onnx`/`from_engine_file`) over the
`vrt-hub` weight-fetch + engine-cache. No `Pipeline`/`Operator` framework —
composition is just calling methods in a loop.

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
