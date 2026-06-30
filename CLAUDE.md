# vision-rt

Standalone Rust TensorRT inference + real-time vision **algorithm libraries**
for Jetson Orin (aarch64, SM87, TensorRT 10.3.x, CUDA 12.6). Pure libraries — no
orchestration framework, no bubbaloop dependency. Sensor drivers live in the
separate `sensor-rt` repo; GPU image/tensor types come from `kornia-rs`
(pinned git dep, `cudarc` feature).

## Workspace layout

Package `vision-rt` (core) + `vrt-*` / `trt-sys` satellites. `[lib] name` keeps
short names — code uses `use vrt::`, `use vrt_xfeat::`, `use trt_sys::`.
Errors: per-crate `thiserror` enums; `vrt::BoxError` for algorithm constructors
that aggregate kinds.

| Crate | Role |
|-------|------|
| `crates/trt-sys` | Raw FFI: pure-C shim over TensorRT C++ (bindgen never sees C++ headers) |
| `crates/vrt` | Safe core: Logger→Runtime→Engine→Session Arc chain, `ModelSession`, `Intrinsics`, `VrtDepthMap`, `stamp`, `cuda` launch helpers |
| `crates/vrt-xfeat` | XFeat keypoints: backbone + GPU NMS/top-K/descriptor sampling/mutual-NN |
| `crates/vrt-rfdetr` | RF-DETR object detector (NMS-free) + on-device GPU decode |
| `crates/vrt-rfdetr-kpts` | RF-DETR human pose: box + 17 COCO keypoints + confidence |
| `crates/vrt-track` | Generic 2D/3D BoT-SORT tracker (nalgebra only, no GPU/TRT) |
| `crates/vrt-lift` | 2D→3D lift via intrinsics + depth + anthropometric bone priors |
| `crates/vrt-reid` | OSNet appearance re-identification embeddings |
| `crates/vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |
| `examples/` | `rfdetr_bench`, `rfdetr_kpts_check`, `track_bench`, `xfeat_match`, `xfeat_bench` |

## Architecture in one paragraph

Each model is a plain type that owns a kornia `Preprocessor` and shares **one
CUDA stream** with the rest of the app: `run()` = enqueue all GPU work async →
ONE `cudaStreamSynchronize` → CPU post-process. `ModelSession` wraps the
Session and takes a kornia `Tensor<f32,4>` device input. Detectors/keypoints
feed `vrt-track` (identity) and `vrt-lift` (3D); provenance travels via
`stamp` (`FrameMeta`/`Stamped`). No `Pipeline`/`Operator` framework — composition
is just calling methods in a loop.

## Hard constraints

- **RAM 7.4 GB (Orin Nano): build with `-j2` / `CARGO_BUILD_JOBS=2`** — parallel
  template builds OOM-kill the box.
- `.engine` files are machine-locked (TRT version + SM87). Rebuild on-device with
  trtexec at `/usr/src/tensorrt/bin/trtexec` — never copy across hosts.
- Benchmark only at MAXN: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

## Commands

```bash
cargo build --release -j2                              # full build (capped jobs)
cargo test -p vrt-track -p vrt-lift -p vrt-hub         # CPU-only unit tests
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
