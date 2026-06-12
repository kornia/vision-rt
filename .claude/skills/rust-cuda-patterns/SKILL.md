---
name: rust-cuda-patterns
description: Use when writing or modifying CUDA code from Rust in this repo — cudarc kernel launches, nvrtc JIT compilation, device memory, CUDA events, FP16, or the trt-sys C++ FFI shim. Covers this repo's specific conventions, not general CUDA.
---

# Rust↔CUDA Patterns in trt-rs

Two distinct FFI layers exist — don't mix them:

1. **cudarc 0.17** (`features = ["cuda-12060"]`) — all kernel work, memory,
   streams, events. Used by `trt-preproc`, `trt-xfeat`.
2. **trt-sys C shim** (`trt_bridge.h/cpp`) — TensorRT only (no C ABI exists).
   Has its own minimal `btrt_cuda_*` helpers so `trt` core stays cudarc-free.
   New CUDA code goes through cudarc, NOT new shim functions.

## Kernel authoring convention — use `trt::cuda::Kernels`

Kernels are CUDA C strings JIT-compiled at construction time. The repo helper
handles arch detection (no hardcoded sm_87), compile options, and grid math:

```rust
use cudarc::driver::PushKernelArg;            // gives .arg()
use trt::cuda::{Kernels, cfg_2d, cfg_per_item};

const KERNELS_SRC: &str = r#"
extern "C" __global__ void my_kernel(const float* __restrict__ in, ...) { ... }
"#;

// In the constructor (~10ms once; CUDA caches PTX; arch auto-detected):
let kernels = Kernels::compile(stream.clone(), KERNELS_SRC)?;
let func    = kernels.function("my_kernel")?;

// Per-frame launch (async on the shared stream):
unsafe {
    stream.launch_builder(&func)
        .arg(&in_slice)      // CudaSlice<T> or &raw scalar
        .arg(&w).arg(&h)
        .launch(cfg_2d(w, h))?;   // (32,8) block, ceil-div grid
}
```

- `cfg_2d(w, h)` for image kernels, `cfg_1d(n, block)` for flat arrays,
  `cfg_per_item(items, threads)` for one-block-per-item (e.g. per keypoint).
- `extern "C"` on every kernel — nvrtc mangles names otherwise.
- `__restrict__` + `__ldg()` for read-only inputs (helps Orin's L1/tex path).
- Compile ONCE in the constructor, never per-frame.
- Raw device pointers from TRT arrive as typed `trt::TensorView`s — use
  `.f32_ptr()?` (dtype-checked); cast to `CUdeviceptr` only at the launch site.

## Memory / stream rules

- `stream.alloc::<f32>(n)` is unsafe (uninitialized) — fine for buffers fully
  written by a kernel; use `alloc_zeros` when partial writes are possible.
- `memcpy_stod` / `memcpy_dtov` are stream-ordered; CPU reads of a `Vec`
  filled by `memcpy_dtov` are only valid after a sync.
- One shared `Arc<CudaStream>` across all pipeline stages (see
  writing-pipeline-stages skill). TRT enqueues onto the same stream via
  `Session::with_stream`.
- CUDA events for timing: `stream.record_event(None)` → after sync,
  `start.elapsed_ms(&stop)`. Both events must be on the same context.

## trt-sys shim rules (only when touching TensorRT FFI)

- Header `trt_bridge.h` is pure C (opaque handles + stdint) — bindgen never
  sees C++/TRT headers. Keep it that way.
- TRT 10: destroy with `delete` (no `->destroy()`), named-tensor API
  (`setTensorAddress`/`enqueueV3`), `Dims64` int64 dims.
- The Rust logger callback must be `catch_unwind`-wrapped (panic across FFI is UB).
- Destruction order context → engine → runtime → logger is enforced by the
  Arc chain in `trt` — never hold raw shim handles outside those wrappers.

## FP16

Engines are built `--fp16` but I/O tensors stay FP32 (TRT inserts casts).
Kernels therefore read TRT outputs as `float*`. Don't add `half` handling
unless an engine is rebuilt with FP16 I/O bindings.
