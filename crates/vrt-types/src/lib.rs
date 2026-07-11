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
use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageError, ImageSize};

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
