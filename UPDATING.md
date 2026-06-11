# Updating for a new TensorRT version

This document is for a developer who has just installed a new TensorRT version and wants to update this library to match.

---

## 1. Files to change in `trt-sys`

### `trt-sys/src/shim.cpp`

This is the C++ shim that wraps TensorRT's abstract C++ API (virtual classes, `IRuntime`, `ICudaEngine`, `IExecutionContext`, etc.) and exposes a plain-C surface.

Each function has a comment referencing the exact TRT C++ method it wraps. When TRT adds, renames, or changes a method, update the function body here.

After each change, verify compilation:
```
cargo build -p trt-sys
```

### `trt-sys/include/shim.h`

The pure-C header that `bindgen` consumes to auto-generate `trt-sys/src/bindings.rs`.

Only change this file if the *C API surface itself* needs to change — i.e. you are adding a new exported function, changing a return type, or renaming a parameter. `bindgen` re-runs on every `cargo build` and overwrites `bindings.rs` automatically; you never edit `bindings.rs` by hand.

### `trt-sys/build.rs`

Contains three constants near the top:

```rust
const TENSORRT_REQUIRED_MAJOR: u32 = 10;
const TENSORRT_REQUIRED_MINOR: u32 = 3;
const TENSORRT_REQUIRED_PATCH: u32 = 0;
```

Update these to the new TensorRT version. The build script reads `NvInferVersion.h` at compile time and emits a clear error if the installed headers do not match, so you get an explicit failure rather than a silent mismatch.

---

## 2. What changes between TRT major versions (historical notes)

### TRT 8 → TRT 10

| Old API (TRT 8) | New API (TRT 10) |
|---|---|
| `obj->destroy()` on every TRT object | `delete obj` (standard C++ RAII) |
| `Dims` with `int32_t` values | `Dims64` with `int64_t` values |
| `builder->setMaxWorkspaceSize(n)` | `config->setMemoryPoolLimit(kWORKSPACE, n)` |
| `context->enqueueV2(bindings, stream, nullptr)` | `context->enqueueV3(stream)` (no bindings array; tensors set by name) |
| `kEXPLICIT_BATCH` flag in `createNetworkV2` | Flag removed; explicit batch is now always assumed |

### TRT 10.x minor versions

The named-tensor I/O API (`setTensorAddress`, `getTensorName`) is stable across 10.x. Minor releases are mostly additive. Check the TRT 10.x release notes for deprecated symbols and update `shim.cpp` accordingly.

---

## 3. Checklist

1. Install the new TensorRT headers on the Jetson (via `apt` or the TensorRT tar package). Verify with:
   ```
   dpkg -l | grep tensorrt
   cat /usr/include/NvInferVersion.h | grep NV_TENSORRT
   ```

2. Update `TENSORRT_REQUIRED_MAJOR` / `_MINOR` / `_PATCH` in `trt-sys/build.rs`.

3. Rebuild the sys crate to catch header-level breakage first:
   ```
   cargo build -p trt-sys
   ```

4. Fix any `shim.cpp` compilation errors. These are almost always API signature changes (see the table above for common patterns).

5. Run the safe-wrapper smoke tests:
   ```
   cargo test -p trt
   ```

6. Any `.engine` file serialized with the old TensorRT version is **not compatible** with the new runtime. Rebuild all engines on the Jetson using `trtexec`:
   ```
   trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
   ```
