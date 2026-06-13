# vision-rt

Standalone Rust TensorRT bindings + real-time vision pipelines for Jetson Orin
(aarch64, SM87, TensorRT 10.3.0.30, CUDA 12.6). Zero bubbaloop dependency in
the library crates.

## Workspace layout

Package names: `vision-rt` (core) + `vrt-*` satellites (crates.io: `trt`/`vrt` taken), but
`[lib] name` keeps the short names — code uses `use vrt::`, `use vrt_xfeat::`.
Errors: per-crate thiserror enums; `BoxError` only in the `Operator` trait.

| Crate | Role |
|-------|------|
| `crates/vrt-sys` | Raw FFI: pure-C shim over TensorRT C++ (bindgen never sees C++ headers) |
| `crates/vrt` | Safe wrapper: Logger→Runtime→Engine→Session Arc chain, `Pipeline`/`Operator` |
| `crates/vrt-preproc` | GPU letterbox RGBA→CHW (nvrtc JIT kernel) |
| `crates/vrt-xfeat` | XFeat keypoints: backbone + GPU NMS/top-K/descriptor sampling |
| `crates/vrt-yolo` | YOLO11/v8: CPU letterbox + decode + NMS |
| `crates/vrt-gst` | GStreamer RTSP source, NVMM zero-copy → CUDA, VIC resize |
| `crates/nvbuf-sys` | NVMM DMA-BUF → cudaImportExternalMemory helpers |
| `examples/` | `rtsp_yolo`, `rtsp_xfeat` — both run live on RTSP cameras |

## Architecture in one paragraph

A `Pipeline` chains typed `Operator`s (`.pipe()`, compile-time type-checked) on
ONE shared CUDA stream. Each frame: `source → enqueue (all GPU work, async) →
one cudaStreamSynchronize → finalize (CPU postproc)`. GPU time is measured
with CUDA events (`PipelineTiming.gpu_ms` — the authoritative metric).
Platform adapters (NVMM→tensor) live in `vrt-gst`; models (tensor→result)
present as one stage each.

## Hard constraints

- `.engine` files are machine-locked (TRT version + SM87). Rebuild with
  trtexec at `/usr/src/tensorrt/bin/trtexec` — never copy across hosts.
- Model input H/W must be multiples of 32 (`pad32`).
- Benchmarks only at MAXN_SUPER: `sudo nvpmodel -m 2 && sudo jetson_clocks`.
- Cameras are H.264 RTSP; pipeline string in `vrt-gst` is H.264-only.

## Commands

```bash
cargo check                              # fast validation
cargo build --release -p rtsp_xfeat      # build one example
cargo test -p vrt-yolo                   # CPU-only unit tests
/usr/src/tensorrt/bin/trtexec --loadEngine=<eng> --verbose 2>&1 | grep -i "tensor\|profile"  # inspect engine I/O
```

## Examples

```bash
cargo run --release -p rtsp_xfeat -- models/xfeat/xfeat_backbone_fp16.engine <rtsp_url> [save_dir]
cargo run --release -p rtsp_yolo  -- <yolo_engine> <rtsp_url>
```

## Detailed knowledge

Project skills in `.claude/skills/` cover: pipeline stages, engine rebuilds,
GStreamer/NVMM debugging, benchmarking discipline, Rust↔CUDA patterns,
CUDA kernel craft, and model tensor semantics. They auto-activate; trust
them over re-deriving from code.
