# Updating for a new TensorRT version (Jetson / JetPack)

TensorRT ships as part of **JetPack** on Jetson Orin, so a TRT bump usually means
a JetPack upgrade. Engines are machine-locked to the exact TRT runtime + GPU arch
(SM87), so **every `.engine` must be rebuilt on-device** after the bump — see the
checklist. Off-Jetson, `TRT_STUB=1` lets `cargo check`/`clippy` run against the
committed bindings without any of this.

---

## The TensorRT shim lives in `kornia/tensorrt-rs`

`tensorrt-rs` — the C++ bridge, the pure-C header bindgen reads, and the `build.rs` that
parses `TENSORRT_VERSION` — is a git dependency from
[`kornia/tensorrt-rs`](https://github.com/kornia/tensorrt-rs). When a TRT bump breaks
the shim, the fix is a PR there; its
[`UPDATING.md`](https://github.com/kornia/tensorrt-rs/blob/main/UPDATING.md) lists the
files to change. Pull the fix in here with `cargo update -p tensorrt-rs`.

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

1. Confirm the TRT headers the new JetPack installed (Jetson paths are `aarch64`):
   ```
   dpkg -l | grep -i tensorrt
   grep NV_TENSORRT /usr/include/aarch64-linux-gnu/NvInferVersion.h
   ```

2. Build `tensorrt-rs` — the C++ compiler catches API breakage:
   ```
   cargo build -p tensorrt-rs
   ```
   Errors point into `trt_bridge.cpp` (rarely `logger_shim.cpp`) in the
   `kornia/tensorrt-rs` checkout cargo fetched; fix them there, then
   `cargo update -p tensorrt-rs`.

3. Run the CPU unit tests (no GPU required):
   ```
   cargo test -p vrt-hub
   ```

4. Rebuild all `.engine` files on-device — engines are tied to the exact TRT
   runtime version + SM87, so stale caches must be dropped:
   ```
   rm -rf ~/.cache/vision-rt/engines/*
   /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.fp16.engine --fp16
   ```
   (The `vrt-hub` `EngineCache` rebuilds automatically on next run — the new
   `TENSORRT_VERSION` changes the cache key, so old engines are ignored.)
