//! CUDA kernel authoring helpers: nvrtc JIT compilation + launch-config math.
//!
//! Wraps the boilerplate every GPU operator otherwise repeats (~40 lines per
//! operator measured across trt-preproc / trt-xfeat): compile options, arch
//! selection, PTX load, function lookup, and ceil-div grid sizing.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use vrt::cuda::{Kernels, cfg_2d};
//! # let stream: Arc<cudarc::driver::CudaStream> = todo!();
//! const SRC: &str = r#"extern "C" __global__ void my_op(float* out, int w, int h) { /*...*/ }"#;
//!
//! let kernels = Kernels::compile(stream.clone(), SRC)?;   // arch auto-detected
//! let my_op   = kernels.function("my_op")?;
//! let (w, h)  = (1280usize, 736usize);
//! let wi = w as i32; let hi = h as i32;
//! # let out: cudarc::driver::CudaSlice<f32> = todo!();
//! unsafe {
//!     stream.launch_builder(&my_op)
//!         .arg(&out).arg(&wi).arg(&hi)
//!         .launch(cfg_2d(w, h))?;
//! }
//! # Ok::<(), vrt::TrtError>(())
//! ```

use std::sync::Arc;
use cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use crate::error::{Result, TrtError};

/// Repo-standard 2D image block: x = warp size for coalescing, 256 threads.
pub const BLOCK_2D: (u32, u32) = (32, 8);

/// A JIT-compiled CUDA module bound to a stream's device.
///
/// Compile once in your operator's constructor (~10 ms; CUDA caches the PTX),
/// then look up functions by name.  The target architecture is read from the
/// stream's device — no hardcoded `sm_XX`.
pub struct Kernels {
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
}

impl Kernels {
    /// JIT-compile CUDA C `src` for the device behind `stream`.
    ///
    /// Kernels must be declared `extern "C" __global__` (nvrtc mangles
    /// names otherwise).  The CUDA toolkit include dir (`$CUDA_HOME/include`,
    /// default `/usr/local/cuda/include`) is on the search path, so kernels
    /// may `#include` CUDA headers (e.g. `<cuda_texture_types.h>`).
    pub fn compile(stream: Arc<CudaStream>, src: &str) -> Result<Self> {
        let (major, minor) = stream.context().compute_capability()
            .map_err(TrtError::from)?;
        let cuda_inc = std::env::var("CUDA_HOME")
            .unwrap_or_else(|_| "/usr/local/cuda".into()) + "/include";
        let opts = CompileOptions {
            options:       vec![format!("--gpu-architecture=sm_{major}{minor}")],
            include_paths: vec![cuda_inc],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(src, opts)
            .map_err(|e| TrtError::Nvrtc(format!("{e:?}")))?;
        let module = stream.context().load_module(ptx)
            .map_err(TrtError::from)?;
        Ok(Self { module, stream })
    }

    /// Look up a compiled `extern "C" __global__` function by name.
    pub fn function(&self, name: &str) -> Result<CudaFunction> {
        self.module.load_function(name)
            .map_err(TrtError::from)
    }

    /// The stream kernels from this module launch on.
    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
}

/// Ceil-div 2D launch config covering `w × h` with the standard
/// [`BLOCK_2D`] block (consecutive threads → consecutive x → coalesced).
pub fn cfg_2d(w: usize, h: usize) -> LaunchConfig {
    let (bx, by) = BLOCK_2D;
    LaunchConfig {
        grid_dim:  ((w as u32).div_ceil(bx), (h as u32).div_ceil(by), 1),
        block_dim: (bx, by, 1),
        shared_mem_bytes: 0,
    }
}

/// Ceil-div 1D launch config covering `n` items with `block` threads/block.
pub fn cfg_1d(n: usize, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim:  ((n as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One block per item, `threads` threads per block — for per-row/per-item
/// kernels (e.g. one block per keypoint, one thread per descriptor channel).
pub fn cfg_per_item(items: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim:  (items as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
