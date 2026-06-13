---
name: cuda-kernel-craft
description: Use when writing, optimizing, or reviewing CUDA kernel code itself (the __global__ functions) — block/grid sizing, memory coalescing, divergence, shared memory, precision, and Jetson Orin (SM87) specifics. Complements rust-cuda-patterns (which covers the Rust launch side).
---

# CUDA Kernel Craft (Jetson Orin SM87)

## Orin hardware facts that shape kernel design

- Ampere-class iGPU, **unified memory**: CPU and GPU share LPDDR5. There is
  no PCIe — "H2D copies" are memory-to-memory, but bandwidth (~200 GB/s) is
  shared with the CPU, decoder, and VIC. Image-processing kernels here are
  almost always **bandwidth-bound, not compute-bound** — optimize bytes
  moved, not FLOPs.
- Warp size 32, max 1024 threads/block. FP16 arithmetic is double-rate, but
  our kernels read FP32 TRT outputs, so FP32 throughout is correct.
- Texture/`__ldg` path is effective on Orin for read-only data with 2D
  locality (see `xfeat_score_nms`'s 25-neighbour window).

## Repo conventions for block/grid sizing

- **2D image kernels**: `block = (32, 8)` — 256 threads, x-dim = warp size so
  consecutive threads read consecutive addresses (coalesced). Grid covers the
  image with ceil-div: `((w + 31)/32, (h + 7)/8)`. Used by every image kernel
  in the repo; don't invent new shapes without a measured reason.
- **Per-item kernels** (one item per block): `grid = (K,1,1)`,
  `block = (64,1,1)` matching the 64-D descriptor — each thread owns one
  channel (see `xfeat_sample_descs`, `xfeat_l2_norm`).
- Always bounds-check first: `if (x >= W || y >= H) return;` — grids overshoot.

## Memory access

- **Coalescing rule**: thread `x` and thread `x+1` should touch addresses 4
  bytes apart. For CHW tensors, index `[c][y][x]` with x innermost — looping
  over channels inside a thread is fine, striding x across threads is not.
- Mark read-only pointer params `const float* __restrict__` and read hot
  reused data with `__ldg(&p[i])` — lets the compiler use the read-only
  cache. All existing kernels do this.
- Avoid read-modify-write to global memory in loops; accumulate in registers,
  write once at the end (see `xfeat_l2_norm` pattern: sum in register,
  one `rsqrtf`, then scale-and-store).
- Shared memory: justified only when a block reuses the same global data many
  times (e.g. tiled windows). The 5×5 NMS deliberately uses `__ldg` instead —
  simpler and the read-only cache already captures the overlap. Measure
  before adding `__shared__` complexity.

## Control flow & precision

- Early-`return` divergence is cheap when spatially coherent (e.g. most
  pixels fail the NMS threshold together). Avoid divergence that differs
  per-lane within a warp in hot loops.
- Branchless `min/max/clamp` (`fminf`, `fmaxf`) over `if` for range clamps.
- Use float literals (`0.0f`, `114.0f/255.0f`) — a bare `0.5` is double and
  forces FP64 ops, which are 1/32 rate on Orin.
- `rsqrtf`, `__fdividef` are fine for normalization (descriptor precision
  tolerates fast-math); do NOT fast-math coordinate computations that feed
  bilinear sampling — sub-pixel keypoint accuracy matters.
- Bilinear sampling convention is **align_corners=False**
  (`src = (dst + 0.5) * scale - 0.5`) to match PyTorch `grid_sample` — any
  new resampling kernel must use the same convention or descriptors shift.

## Correctness checklist for new kernels

1. Bounds check at the top.
2. No assumption that W/H are multiples of block dims (ceil-div grid).
3. Output fully written for every in-bounds thread (or buffer pre-zeroed
   with `alloc_zeros`) — `stream.alloc` is uninitialized.
4. No inter-block dependencies — there is no global sync inside a kernel.
   If a reduction needs all blocks' results, split into two kernels or do
   the final step on CPU (the top-K does exactly this).
5. Test against a scalar CPU reference on a small synthetic input where
   the expected output is hand-computable (edge pixels included).

## Optimizing: measure first

- `gpu_ms` in `PipelineTiming` is per-frame whole-pipeline GPU time; to
  isolate one kernel, bracket it with `stream.record_event(None)` pairs.
- For deep dives: `sudo /opt/nvidia/nsight-compute/ncu --set basic <binary>`
  (kernel-level) or `nsys profile` (timeline). Run at MAXN_SUPER.
- A kernel at < 0.2ms on 1280×736 is in the noise next to the ~10ms
  backbone — don't optimize it; fuse it or leave it.
- GPU top-K without a CPU round trip: histogram-cutoff (bin scores → scan
  for the K-th threshold → atomic-gather above it). Approximate at the
  boundary bin but avoids the mid-frame device→host→device sort. See
  xfeat_topk_* in vrt-xfeat. Output is atomic-append order, not sorted.
- Fusing beats micro-tuning here: `xfeat_score_nms` fuses NMS + score
  multiply into one pass to halve traffic. Look for fusion (one read, one
  write) before tweaking block sizes.
