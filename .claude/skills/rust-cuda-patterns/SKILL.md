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

## Kernel authoring convention (see trt-preproc, trt-xfeat/postprocess.rs)

Kernels are CUDA C strings JIT-compiled at construction time with nvrtc:

```rust
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use cudarc::driver::{LaunchConfig, PushKernelArg};   // PushKernelArg gives .arg()

const KERNELS_SRC: &str = r#"
extern "C" __global__ void my_kernel(const float* __restrict__ in, ...) { ... }
"#;

// In the constructor (~10ms once; CUDA caches PTX):
let opts   = CompileOptions { arch: Some("sm_87"), ..Default::default() };
let ptx    = compile_ptx_with_opts(KERNELS_SRC, opts).map_err(|e| format!("nvrtc: {e:?}"))?;
let module = stream.context().load_module(ptx)?;
let func   = module.load_function("my_kernel")?;

// Per-frame launch (async on the shared stream):
unsafe {
    stream.launch_builder(&func)
        .arg(&in_slice)      // CudaSlice<T> or &raw scalar
        .arg(&w).arg(&h)
        .launch(LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 0 })?;
}
```

- `extern "C"` on every kernel — nvrtc mangles names otherwise.
- `__restrict__` + `__ldg()` for read-only inputs (helps Orin's L1/tex path).
- Compile ONCE in the constructor, never per-frame.
- Raw device pointers from TRT cross into kernels as `CUdeviceptr`
  (`cudarc::driver::sys::CUdeviceptr`) — cast `*const f32 as CUdeviceptr`.

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
