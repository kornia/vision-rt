# trt-sys

Raw FFI bindings for NVIDIA TensorRT, via a pure-C shim over the TensorRT C++
API (bindgen never sees C++ headers). Targets the TensorRT that ships with
JetPack on Jetson Orin (10.3.x, aarch64). Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

- Compiles small C++ shims (`logger_shim`, `trt_bridge`, and `builder_shim`
  under the `builder` feature) with `cc`, then generates `btrt_*` bindings with
  bindgen.
- Links `nvinfer`, `nvinfer_plugin`, `cudart` (+ `nvonnxparser` with `builder`).
- Exports `TENSORRT_VERSION` (parsed from `NvInferVersion.h`) — used downstream
  for engine-cache keys. Warns at build time if the installed TRT is outside the
  tested 10.3.x range.

**Off-Jetson:** set `TRT_STUB=1` (or build on docs.rs) to skip the native
compile/link and use committed pregenerated bindings — `cargo check`/`clippy`/
`doc` work with no CUDA/TensorRT installed. Anything that links or runs still
needs a real TensorRT install.

License: Apache-2.0
