# Updating for a new TensorRT version

---

## Files to change in `vrt-sys`

### `vrt-sys/src/trt_bridge.cpp`

This is the C++ bridge that wraps TRT's abstract C++ API and exposes a flat C surface (`btrt_*` functions). Each function has a comment referencing the exact TRT header and method it wraps.

When TRT renames or changes a method, the C++ compiler reports an error here on `cargo build -p vrt-sys`. Fix the error, then proceed.

### `vrt-sys/src/logger_shim.cpp`

The ONLY hand-written C++ that cannot be replaced by code generation: `ShimLogger : public nvinfer1::ILogger`. Only change this file if TRT changes the `ILogger::log()` virtual signature.

### `vrt-sys/include/trt_bridge.h`

The pure-C header that `bindgen` uses to generate `OUT_DIR/bridge_bindings.rs`. Only change this if the C API surface changes (new `btrt_*` function, changed return type, etc.). `bindgen` regenerates `bridge_bindings.rs` automatically on every build — never edit it by hand.

### `vrt-sys/src/lib.rs`

Nothing — `TENSORRT_VERSION` is parsed from `NvInferVersion.h` at build time.

---

## TRT 8 → TRT 10 migration table

| Old API (TRT 8)                              | New API (TRT 10)                              |
|----------------------------------------------|-----------------------------------------------|
| `obj->destroy()`                             | `delete obj` (standard C++ RAII)              |
| `Dims` with `int32_t` values                 | `Dims64` with `int64_t` values                |
| `builder->setMaxWorkspaceSize(n)`            | `config->setMemoryPoolLimit(kWORKSPACE, n)`   |
| `context->enqueueV2(bindings, stream, null)` | `context->enqueueV3(stream)` (named tensors)  |
| `kEXPLICIT_BATCH` flag in `createNetworkV2` | Removed — explicit batch always assumed       |

TRT 10.x minor releases: the named-tensor I/O API (`setTensorAddress`, `getIOTensorName`) is stable across 10.x. Check TRT release notes for deprecated symbols.

---

## Update checklist

1. Install new TRT headers:
   ```
   apt install tensorrt   # or download tar and copy headers
   dpkg -l | grep tensorrt
   cat /usr/include/aarch64-linux-gnu/NvInferVersion.h | grep NV_TENSORRT
   ```

2. Build `vrt-sys` — the C++ compiler catches API breakage:
   ```
   cargo build -p vrt-sys
   ```
   Fix any errors in `trt_bridge.cpp` (and rarely `logger_shim.cpp`).

4. Run the unit tests (no GPU required):
   ```
   cargo test -p vrt-yolo -p vision-rt
   ```

5. Rebuild all `.engine` files — TRT engines are tied to the exact runtime version:
   ```
   /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.fp16.engine --fp16
   ```
