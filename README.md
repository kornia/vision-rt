# trt-rs

Standalone Rust bindings for TensorRT 10.3.x on Jetson Orin (aarch64).

Organized as a Cargo workspace with three crates:

- **trt-sys** — raw FFI bindings via a hand-written C shim and `bindgen`
- **trt** — safe, idiomatic Rust wrapper (YOLO post-processing included via the `yolo` feature)
- **trt-detector-node** — optional bubbaloop processor node that runs a TensorRT YOLO detector on camera frames

This repo is completely standalone and lives outside the bubbaloop repo. The bubbaloop integration is an optional dependency in `trt-detector-node` only.

Target platform: Jetson Orin, JetPack 6.x, TensorRT 10.3.x.
