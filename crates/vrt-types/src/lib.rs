//! Shared vision types for the `vision-rt` workspace.
//!
//! Dependency-light home for the data types that model crates pass between each
//! other, so the `Detection` triple stops being copy-pasted per crate and the
//! depth/segmentation crates share one `Mask` / `DepthImage` vocabulary. This
//! crate depends only on `kornia-image` / `kornia-tensor` (no `vrt` / `trt-sys`),
//! so it is a leaf every model crate can depend on **downward** — and a clean
//! candidate to upstream into kornia-rs.
//!
//! - [`Detection`] — a detected object: COCO class + score + `xyxy` box in source pixels.
//! - [`Mask`] — a binary instance mask (`Image<u8,1>`, `1` = foreground), host or device.
//! - [`DepthImage`] — a **metric** depth map in meters (`Image<f32,1>`), host or device.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_tensor::CudaKernel;

/// Errors from the device-resident sampling builtins ([`DepthImage::sample_masks`]
/// / [`sample_boxes`](DepthImage::sample_boxes), [`Mask::sample_depth`]).
///
/// [`sample_boxes`]: DepthImage::sample_boxes
#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error(transparent)]
    Image(#[from] ImageError),
    #[error("kornia CUDA: {0}")]
    Cuda(#[from] kornia_tensor::CudaError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("expected a device-resident image (backed by a CudaSlice)")]
    NotOnDevice,
    #[error("invalid sampling argument: {0}")]
    InvalidArg(&'static str),
}

/// A detected object in original-image coordinate space.
///
/// The shared detector-output triple (currently re-defined in each `vrt-rfdetr*`
/// crate — those migrate onto this type separately).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// COCO category id (1–90); background is never emitted.
    pub class_id: u32,
    pub score: f32,
    /// Bounding box `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
}

// Typed single-channel image newtypes, mirroring kornia's `color_spaces` macro
// (`kornia-image/src/color_spaces.rs`) — a zero-cost `#[repr(transparent)]`
// wrapper that Derefs to the inner `Image` so all image ops still apply, but the
// type name carries mask / depth semantics instead of a color space.
macro_rules! typed_image {
    ($name:ident, $ty:ty, $ch:expr, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Zero-cost `#[repr(transparent)]` wrapper over the inner `Image` — it
        /// Derefs to the image, so every `kornia-image` op still applies.
        #[repr(transparent)]
        pub struct $name(pub Image<$ty, $ch>);

        impl $name {
            #[doc = concat!("Create a host ", stringify!($name), " from size + row-major data.")]
            pub fn from_size_vec(size: ImageSize, data: Vec<$ty>) -> Result<Self, ImageError> {
                Ok(Self(Image::new(size, data)?))
            }

            #[doc = concat!("Create a host ", stringify!($name), " filled with `val`.")]
            pub fn from_size_val(size: ImageSize, val: $ty) -> Result<Self, ImageError> {
                Ok(Self(Image::from_size_val(size, val)?))
            }

            /// Unwrap into the underlying `Image`.
            pub fn into_inner(self) -> Image<$ty, $ch> {
                self.0
            }

            /// Borrow the underlying `Image`.
            pub fn as_image(&self) -> &Image<$ty, $ch> {
                &self.0
            }

            #[doc = concat!("Borrow the backing `CudaSlice` if this ", stringify!($name), " is device-resident (else `None`).")]
            pub fn as_cudaslice(&self) -> Option<&cudarc::driver::CudaSlice<$ty>> {
                // Deref to the inner `Tensor` (Image → Tensor3), then to its CudaSlice.
                self.0.as_cudaslice()
            }

            #[doc = concat!("Mutably borrow the backing `CudaSlice` if this ", stringify!($name), " is device-resident (else `None`).")]
            pub fn as_cudaslice_mut(&mut self) -> Option<&mut cudarc::driver::CudaSlice<$ty>> {
                self.0.as_cudaslice_mut()
            }

            #[doc = concat!("Allocate a zero-initialised device-resident ", stringify!($name), ".")]
            pub fn zeros_cuda(
                size: ImageSize,
                stream: &Arc<CudaStream>,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::zeros_cuda(size, stream)?))
            }

            #[doc = concat!("Upload to a device-resident ", stringify!($name), " (H2D).")]
            pub fn to_cuda(&self, stream: &Arc<CudaStream>) -> Result<Self, ImageError> {
                Ok(Self(self.0.to_cuda_image(stream)?))
            }

            #[doc = concat!("Copy a device-resident ", stringify!($name), " back to host (D2H).")]
            pub fn to_host(&self, stream: &Arc<CudaStream>) -> Result<Self, ImageError> {
                Ok(Self(self.0.to_host_image(stream)?))
            }
        }

        impl Deref for $name {
            type Target = Image<$ty, $ch>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl AsRef<Image<$ty, $ch>> for $name {
            fn as_ref(&self) -> &Image<$ty, $ch> {
                &self.0
            }
        }
    };
}

typed_image!(
    Mask,
    u8,
    1,
    "A binary instance mask (`1` = foreground, `0` = background), host or device."
);
typed_image!(
    DepthImage,
    f32,
    1,
    "A **metric** depth map in meters (per-pixel `z`), host or device."
);

// ── Device-resident depth-sampling builtins ────────────────────────────────────
//
// GPU depth-at-box / depth-at-mask fusion, hosted on the typed images so any crate
// with a `DepthImage` + a `Mask` (or a detector's packed masks/boxes) can sample
// per-instance metric depth without pulling in a model crate. The kernels are the
// depth-fusion pair that used to live in `vrt-depth-anything`.

// One block per box: average the depth over the box's inner-50% central patch
// (avoids DPT depth bleeding across object edges). Box xyxy is in source pixels;
// the Stretch preprocess is full-frame, so source→map is a plain linear scale.
//
// One block per instance: average the depth over the instance mask's foreground
// pixels. Mask grid (mmh,mmw) and depth grid (dmh,dmw) both span the full frame,
// so a mask pixel maps to a depth pixel by a plain scale. `masks` is packed per
// surviving slot ([count*mmh*mmw], 1 = foreground).
const SAMPLE_SRC: &str = r#"
extern "C" __global__ void depth_box(
    const float* __restrict__ depth, int dmh, int dmw,
    const float* __restrict__ boxes, int stride, int n,
    const int* __restrict__ live,
    float src_w, float src_h,
    float* __restrict__ out_z
) {
    int b = blockIdx.x;
    if (b >= n) return;
    // On-device survivor gate: slots >= the live count are stale (never written by
    // the detector this frame) — leave their pre-zeroed out_z at 0, don't sample.
    if (b >= *live) return;
    const float* box = boxes + (long)b * stride;
    float sx = (float)dmw / src_w, sy = (float)dmh / src_h;
    float x1 = box[0] * sx, y1 = box[1] * sy, x2 = box[2] * sx, y2 = box[3] * sy;
    float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
    float hw = (x2 - x1) * 0.25f, hh = (y2 - y1) * 0.25f;  // inner 50%
    int ix0 = max(0, (int)floorf(cx - hw)), ix1 = min(dmw - 1, (int)ceilf(cx + hw));
    int iy0 = max(0, (int)floorf(cy - hh)), iy1 = min(dmh - 1, (int)ceilf(cy + hh));
    int W = ix1 - ix0 + 1, H = iy1 - iy0 + 1, N = (W > 0 && H > 0) ? W * H : 0;

    float sum = 0.0f; int cnt = 0;
    for (int p = threadIdx.x; p < N; p += blockDim.x) {
        int x = ix0 + p % W, y = iy0 + p / W;
        float d = depth[(long)y * dmw + x];
        if (d > 0.0f) { sum += d; ++cnt; }
    }
    __shared__ float ssum[256]; __shared__ int scnt[256];
    int t = threadIdx.x;
    ssum[t] = sum; scnt[t] = cnt; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { ssum[t] += ssum[t + s]; scnt[t] += scnt[t + s]; }
        __syncthreads();
    }
    if (t == 0) out_z[b] = scnt[0] > 0 ? ssum[0] / scnt[0] : 0.0f;
}

extern "C" __global__ void depth_mask(
    const float* __restrict__ depth, int dmh, int dmw,
    const unsigned char* __restrict__ masks, int mmh, int mmw,
    int count,
    const int* __restrict__ live,
    float* __restrict__ out_z
) {
    int m = blockIdx.x;
    if (m >= count) return;
    // On-device survivor gate: slots >= the live count are stale (never written by
    // the detector this frame) — leave their pre-zeroed out_z at 0, don't sample.
    if (m >= *live) return;
    const unsigned char* mask = masks + (long)m * mmh * mmw;
    int L = mmh * mmw;
    float sx = (float)dmw / mmw, sy = (float)dmh / mmh;

    float sum = 0.0f; int cnt = 0;
    for (int p = threadIdx.x; p < L; p += blockDim.x) {
        if (mask[p] != 0) {
            int mx = p % mmw, my = p / mmw;
            int dx = min(dmw - 1, (int)(mx * sx)), dy = min(dmh - 1, (int)(my * sy));
            float d = depth[(long)dy * dmw + dx];
            if (d > 0.0f) { sum += d; ++cnt; }
        }
    }
    __shared__ float ssum[256]; __shared__ int scnt[256];
    int t = threadIdx.x;
    ssum[t] = sum; scnt[t] = cnt; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { ssum[t] += ssum[t + s]; scnt[t] += scnt[t + s]; }
        __syncthreads();
    }
    if (t == 0) out_z[m] = scnt[0] > 0 ? ssum[0] / scnt[0] : 0.0f;
}
"#;

// One block per item, `threads` threads/block — inlined from `vrt::cuda::cfg_per_item`
// (`vrt` is not a dependency of this leaf crate).
fn cfg_per_item(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The two compiled sampling kernels, JIT-compiled once (lazily) and shared.
struct SampleKernels {
    depth_box: CudaKernel,
    depth_mask: CudaKernel,
}

// A free builtin has no `self` to hold the compiled kernel, so cache it in a
// process-global. vision-rt uses a single CUDA context for the process lifetime, so
// one cache suffices — no per-context map. The nvrtc compile runs once (first call);
// `CudaKernel` is `Send + Sync`, so the static is sound.
static SAMPLE_K: OnceLock<Mutex<Option<Arc<SampleKernels>>>> = OnceLock::new();

fn sample_kernels(stream: &Arc<CudaStream>) -> Result<Arc<SampleKernels>, TypeError> {
    let cell = SAMPLE_K.get_or_init(|| Mutex::new(None));
    // Recover from a poisoned lock: a panic mid-compile shouldn't wedge the cache.
    let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(k) = guard.as_ref() {
        return Ok(k.clone());
    }
    let mut it =
        CudaKernel::compile_many(stream.context(), SAMPLE_SRC, &["depth_box", "depth_mask"])?
            .into_iter();
    let kernels = Arc::new(SampleKernels {
        depth_box: it.next().expect("compile_many returns one kernel per name"),
        depth_mask: it.next().expect("compile_many returns one kernel per name"),
    });
    *guard = Some(kernels.clone());
    Ok(kernels)
}

impl DepthImage {
    /// Sample per-**instance** metric depth over each instance mask's foreground
    /// pixels (isolates the object — no background bleed, unlike the box). `masks`
    /// is a device `[slots*mmh*mmw]` u8 buffer packed per slot (e.g. a detector's
    /// `masks_slice()`), `mask_wh = (mmw, mmh)`. `live` is the detector's **on-device**
    /// survivor count (a `[1]` i32 device slice, e.g. `SegResult::count_slice()`): the
    /// kernel reads it **on the GPU** so no host sync is needed, and every slot `>=`
    /// the live count is left at `0` instead of sampling its stale (previous-frame)
    /// mask contents. So the returned `[slots]` z buffer is safe to read in full — the
    /// trailing entries are deterministically `0`, not garbage — though zipping against
    /// `instances()` / `detections()` is still the natural consume. Enqueues on the
    /// shared stream; z (meters) is valid after the caller's single
    /// `stream.synchronize()`.
    pub fn sample_masks(
        &self,
        masks: &CudaSlice<u8>,
        mask_wh: (usize, usize),
        live: &CudaSlice<i32>,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, TypeError> {
        let (mmw, mmh) = mask_wh;
        let slots = masks.len() / (mmw * mmh).max(1);
        let z = stream.alloc_zeros::<f32>(slots)?;
        if slots > 0 {
            let depth = self.as_cudaslice().ok_or(TypeError::NotOnDevice)?;
            let (dmh, dmw) = (self.size().height as i32, self.size().width as i32);
            let (mmhi, mmwi, cnt) = (mmh as i32, mmw as i32, slots as i32);
            let kernels = sample_kernels(stream)?;
            kernels
                .depth_mask
                .launch_builder(stream)
                .arg(depth)
                .arg(&dmh)
                .arg(&dmw)
                .arg(masks)
                .arg(&mmhi)
                .arg(&mmwi)
                .arg(&cnt)
                .arg(live)
                .arg(&z)
                .launch_cfg(cfg_per_item(slots, 256))?;
        }
        Ok(z)
    }

    /// Sample per-**box** metric depth (mean of the box's inner-50% central patch).
    /// `boxes` is a device `[slots*stride]` buffer with `x1,y1,x2,y2` (source pixels)
    /// in the first 4 lanes — e.g. a detector's `dets_slice()` (`stride=6`); `src_wh`
    /// is the original-image `(width, height)`. `live` is the detector's **on-device**
    /// survivor count (a `[1]` i32 device slice): the kernel gates on it on the GPU, so
    /// slots `>=` the live count are left at `0` (not sampled from stale box coords) and
    /// the returned `[slots]` z buffer is safe to read in full. Enqueues on the shared
    /// stream; z (meters) is valid after the single sync.
    pub fn sample_boxes(
        &self,
        boxes: &CudaSlice<f32>,
        stride: usize,
        live: &CudaSlice<i32>,
        src_wh: (f32, f32),
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, TypeError> {
        let (sw, sh) = src_wh;
        // Misconfig is an error, not a silent all-zero z: stride==0 would read every
        // box from offset 0, and a non-positive src makes the source→map scale
        // inf → NaN box coords. Both yield plausible-but-wrong depths downstream.
        if stride == 0 {
            return Err(TypeError::InvalidArg("sample_boxes: stride must be > 0"));
        }
        if sw <= 0.0 || sh <= 0.0 {
            return Err(TypeError::InvalidArg(
                "sample_boxes: src_wh must be positive",
            ));
        }
        let slots = boxes.len() / stride;
        let z = stream.alloc_zeros::<f32>(slots)?;
        if slots > 0 {
            let depth = self.as_cudaslice().ok_or(TypeError::NotOnDevice)?;
            let (dmh, dmw) = (self.size().height as i32, self.size().width as i32);
            let (st, ni) = (stride as i32, slots as i32);
            let kernels = sample_kernels(stream)?;
            kernels
                .depth_box
                .launch_builder(stream)
                .arg(depth)
                .arg(&dmh)
                .arg(&dmw)
                .arg(boxes)
                .arg(&st)
                .arg(&ni)
                .arg(live)
                .arg(&sw)
                .arg(&sh)
                .arg(&z)
                .launch_cfg(cfg_per_item(slots, 256))?;
        }
        Ok(z)
    }
}

impl Mask {
    /// Single-mask convenience: mean metric depth over this one mask's foreground.
    /// Runs the `depth_mask` fusion with `count = 1`, then D2H-copies and syncs the
    /// single value. Returns the scalar `z` in meters (`0.0` if the mask is empty).
    /// For many masks at once, prefer [`DepthImage::sample_masks`] (one launch).
    pub fn sample_depth(
        &self,
        depth: &DepthImage,
        stream: &Arc<CudaStream>,
    ) -> Result<f32, TypeError> {
        let this = self.as_cudaslice().ok_or(TypeError::NotOnDevice)?;
        let mask_wh = (self.size().width, self.size().height);
        let live = stream.clone_htod(&[1i32])?; // exactly one live slot
        let z = depth.sample_masks(this, mask_wh, &live, stream)?; // single mask → 1 slot
        let v = stream.clone_dtoh(&z)?;
        stream.synchronize()?;
        Ok(v[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_plain_data() {
        let d = Detection {
            class_id: 17,
            score: 0.9,
            bbox: [1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(d, d);
        assert_eq!(d.bbox[2], 3.0);
    }

    #[test]
    fn mask_host_ctor_and_deref() {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let m = Mask::from_size_vec(size, vec![1, 0, 0, 1]).unwrap();
        // Derefs to Image<u8,1>.
        assert_eq!(m.size().width, 2);
        assert_eq!(m.as_slice(), &[1, 0, 0, 1]);
    }

    #[test]
    fn depth_host_ctor_and_deref() {
        let size = ImageSize {
            width: 2,
            height: 1,
        };
        let z = DepthImage::from_size_val(size, 2.5).unwrap();
        assert_eq!(z.size().width, 2);
        assert_eq!(z.as_slice(), &[2.5, 2.5]);
    }
}
