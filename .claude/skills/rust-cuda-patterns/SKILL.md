---
name: rust-cuda-patterns
description: Use when writing or modifying CUDA code from Rust in this repo — kornia CudaKernel JIT, cudarc launches/memory/events, device pointers from TRT, or the trt-sys C FFI shim. Covers this repo's conventions, not general CUDA.
---

# Rust↔CUDA Patterns in vision-rt

Two distinct FFI layers exist — don't mix them:

1. **cudarc 0.19** (workspace dep: `default-features = false`, features
   `cuda-version-from-build-system, driver, nvrtc, std, fallback-dynamic-loading,
   fallback-latest`) — device memory, streams, events, kernel launches. One
   cudarc across the workspace, matched to kornia-rs's set so `CudaStream`/
   `CudaSlice` types unify for zero-copy interop.
2. **trt-sys C shim** (`trt_bridge.h/cpp`) — TensorRT only (no C ABI exists).
   New CUDA work goes through cudarc/kornia, NOT new shim functions.

## Kernel authoring — kornia `CudaKernel`

Kernels are CUDA C strings JIT-compiled once via kornia's `CudaKernel`
(`kornia_tensor::CudaKernel`), which handles arch detection (no hardcoded sm_87)
and PTX caching. Grid math comes from `vrt::cuda::{cfg_1d, cfg_2d, cfg_per_item}`.

```rust
use kornia_tensor::CudaKernel;
use vrt::cuda::cfg_2d;
use cudarc::driver::{CudaSlice, DevicePtr, sys::CUdeviceptr};

const KERNELS_SRC: &str = r#"
extern "C" __global__ void my_kernel(const float* __restrict__ in, float* out, int w, int h) { ... }
"#;

// Constructor: compile the whole suite once; destructure the named functions.
let [my_kernel]: [CudaKernel; 1] =
    CudaKernel::compile_many(stream.context(), KERNELS_SRC, &["my_kernel"])?
        .try_into().unwrap_or_else(|_| unreachable!());

// Per-launch (async on the shared stream):
let out_raw: CUdeviceptr = out.device_ptr(stream.as_ref()).0;   // CudaSlice → ptr
my_kernel
    .launch_builder(&stream)
    .arg(&in_slice)          // CudaSlice<T>, or &raw scalar (&w_i32), or &CUdeviceptr
    .arg(&out_raw)
    .arg(&w_i).arg(&h_i)
    .launch_cfg(cfg_2d(w, h))?;   // (32,8) block, ceil-div grid
```

- `cfg_2d(w,h)` for image kernels, `cfg_1d(n, block)` for flat arrays,
  `cfg_per_item(items, threads)` for one-block-per-item (e.g. per keypoint).
- `extern "C"` on every kernel — nvrtc mangles names otherwise.
- `__restrict__` + `__ldg()` for read-only inputs (helps Orin's L1/tex path).
- Compile ONCE in the constructor, never per-frame.
- **Raw device pointers from TRT** arrive via `TRTensorMap::get(name)` →
  `OutputView::f32_ptr()?` (dtype-checked `*const f32`). Cast to `CUdeviceptr` at
  the launch site: `ptr as usize as CUdeviceptr`. For a `CudaSlice`, use
  `slice.device_ptr(stream).0`.

## Memory / stream rules

- `stream.alloc::<f32>(n)` is unsafe (uninitialized) — fine for buffers fully
  written by a kernel; use `alloc_zeros` when partial writes are possible.
- Copies are stream-ordered; CPU reads of a host `Vec` filled by a D2H copy are
  only valid after a sync. For low-latency count/scores readback use a reused
  `vrt::PinnedBuffer` + async D2H, then one `stream.synchronize()` per frame.
- One shared `Arc<CudaStream>` across a model's whole run (preproc + backbone +
  postproc) → a single sync per frame. TRT enqueues onto it via
  `Session::with_stream` (see `vrt::model::ModelSession`).
- CUDA events for timing: record with `CU_EVENT_DEFAULT`, not `None` — cudarc's
  default is `DISABLE_TIMING`, which makes `elapsed_ms` fail (a `.unwrap_or(0.0)`
  then silently reports 0 GPU time — this bug shipped once). Prefer the
  jetson-benchmarking skill's harness for reported numbers.

## trt-sys shim rules (only when touching TensorRT FFI)

- Header `trt_bridge.h` is pure C (opaque handles + stdint) — bindgen never sees
  C++/TRT headers. Keep it that way.
- TRT 10: destroy with `delete` (no `->destroy()`), named-tensor API
  (`setTensorAddress`/`enqueueV3`), `Dims64` int64 dims.
- The Rust logger callback must be `catch_unwind`-wrapped (panic across FFI is UB).
- Destruction order context → engine → runtime → logger is enforced by the Arc
  chain in `vrt` — never hold raw shim handles outside those wrappers.
- Off-Jetson: `TRT_STUB=1` swaps in committed bindings, no native compile/link.

## FP16

Engines are built `--fp16` but I/O tensors stay FP32 (TRT inserts casts).
Kernels therefore read TRT outputs as `float*`. Don't add `half` handling unless
an engine is rebuilt with FP16 I/O bindings.
