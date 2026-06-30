//! CUDA launch-config helpers (ceil-div grid sizing).
//!
//! Kernel *authoring* (nvrtc JIT compile + launch) now lives in kornia's
//! [`CudaKernel`](kornia_tensor::CudaKernel) / `CudaLaunchBuilder`. What remains
//! here is the small launch-geometry math shared by the GPU operators — the
//! `LaunchConfig` builders passed to `CudaLaunchBuilder::launch_cfg` for kernels
//! whose grid/block shape `launch_1d` can't express (2-D image kernels,
//! one-block-per-item reductions, fixed-block tiled kernels).

use cudarc::driver::LaunchConfig;

/// Repo-standard 2D image block: x = warp size for coalescing, 256 threads.
pub const BLOCK_2D: (u32, u32) = (32, 8);

/// Ceil-div 2D launch config covering `w × h` with the standard
/// [`BLOCK_2D`] block (consecutive threads → consecutive x → coalesced).
pub fn cfg_2d(w: usize, h: usize) -> LaunchConfig {
    let (bx, by) = BLOCK_2D;
    LaunchConfig {
        grid_dim: ((w as u32).div_ceil(bx), (h as u32).div_ceil(by), 1),
        block_dim: (bx, by, 1),
        shared_mem_bytes: 0,
    }
}

/// Ceil-div 1D launch config covering `n` items with `block` threads/block.
pub fn cfg_1d(n: usize, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One block per item, `threads` threads per block — for per-row/per-item
/// kernels (e.g. one block per keypoint, one thread per descriptor channel).
pub fn cfg_per_item(items: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (items as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
