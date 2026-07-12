//! GPU lens **undistortion** as a precomputed remap.
//!
//! Barrel/pincushion distortion is corrected with a single radial coefficient `k1`
//! (Brown–Conrady, first term). The remap LUT — for each *undistorted* output pixel,
//! the *distorted* source coordinate to sample — is built once on the host from the
//! [`CameraIntrinsics`] + `k1` and uploaded; each frame is then a single bilinear
//! remap kernel (~1 ms at 720p). Apply it **before** detection/depth so everything
//! downstream (boxes, masks, metric-3D) lives in a clean pinhole space.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig};
use kornia_image::Image;
use kornia_tensor::CudaKernel;

use crate::{CameraIntrinsics, TypeError};

// One thread per output pixel: bilinear-sample the source at the mapped (distorted)
// coordinate; out-of-bounds → black (the undistort pulls the borders in).
const REMAP_SRC: &str = r#"
extern "C" __global__ void undistort_remap(
    const unsigned char* __restrict__ src, int W, int H,
    const float* __restrict__ map, unsigned char* __restrict__ dst
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    long o = (long)y * W + x;
    float sx = map[2 * o], sy = map[2 * o + 1];
    if (sx < 0.0f || sy < 0.0f || sx > W - 1 || sy > H - 1) {
        dst[o * 3] = 0; dst[o * 3 + 1] = 0; dst[o * 3 + 2] = 0;
        return;
    }
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float fx = sx - x0, fy = sy - y0;
    for (int c = 0; c < 3; ++c) {
        float v = (1 - fx) * (1 - fy) * src[((long)y0 * W + x0) * 3 + c]
                +      fx  * (1 - fy) * src[((long)y0 * W + x1) * 3 + c]
                + (1 - fx) *      fy  * src[((long)y1 * W + x0) * 3 + c]
                +      fx  *      fy  * src[((long)y1 * W + x1) * 3 + c];
        dst[o * 3 + c] = (unsigned char)(v + 0.5f);
    }
}
"#;

/// A reusable GPU undistorter for a fixed resolution + intrinsics + `k1`.
pub struct Undistorter {
    map: CudaSlice<f32>, // [2*W*H] distorted source (x,y) per output pixel
    kernel: CudaKernel,
    w: usize,
    h: usize,
}

impl Undistorter {
    /// Build the remap LUT for `w×h` from `intr` + radial `k1` (barrel → negative).
    /// The distortion model is `p_distorted = p·(1 + k1·r²)` in normalized coords.
    pub fn new(
        intr: &CameraIntrinsics,
        k1: f32,
        w: usize,
        h: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, TypeError> {
        let mut map = vec![0f32; 2 * w * h];
        for v in 0..h {
            for u in 0..w {
                let x = (u as f32 - intr.cx) / intr.fx;
                let y = (v as f32 - intr.cy) / intr.fy;
                let d = 1.0 + k1 * (x * x + y * y);
                let o = 2 * (v * w + u);
                map[o] = intr.fx * (x * d) + intr.cx; // distorted source x
                map[o + 1] = intr.fy * (y * d) + intr.cy; // distorted source y
            }
        }
        let map = stream.clone_htod(&map)?;
        let kernel = CudaKernel::compile(stream.context(), REMAP_SRC, "undistort_remap")?;
        Ok(Self { map, kernel, w, h })
    }

    /// Undistort `src` into `dst` (both device `Image<u8,3>` of the configured size).
    /// Enqueues on `stream`; valid after the caller's next sync.
    pub fn apply(
        &self,
        src: &Image<u8, 3>,
        dst: &mut Image<u8, 3>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), TypeError> {
        let s = stream.as_ref();
        let src_raw = src
            .as_cudaslice()
            .ok_or(TypeError::NotOnDevice)?
            .device_ptr(s)
            .0 as CUdeviceptr;
        let dst_raw = dst
            .as_cudaslice_mut()
            .ok_or(TypeError::NotOnDevice)?
            .device_ptr(s)
            .0 as CUdeviceptr;
        let map_raw = self.map.device_ptr(s).0 as CUdeviceptr;
        let (wi, hi) = (self.w as i32, self.h as i32);
        let cfg = LaunchConfig {
            grid_dim: (self.w.div_ceil(16) as u32, self.h.div_ceil(16) as u32, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        self.kernel
            .launch_builder(stream)
            .arg(&src_raw)
            .arg(&wi)
            .arg(&hi)
            .arg(&map_raw)
            .arg(&dst_raw)
            .launch_cfg(cfg)?;
        Ok(())
    }
}
