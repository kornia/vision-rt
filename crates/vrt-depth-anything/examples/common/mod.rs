//! Shared helpers for the `vrt-depth-anything` examples (included via
//! `#[path = "common/mod.rs"] mod common;`).

use kornia_image::Image;
use kornia_imgproc::color::{apply_colormap, ColormapType};
use vrt_types::DepthImage;

/// Colorize a metric depth map (meters) to RGB with the **Turbo** colormap,
/// normalizing over the map's finite `[min, max]` range. Host path — vision-rt
/// enables `cudarc`, not `gpu-cuda`, so the colormap runs on the CPU.
pub fn depth_to_turbo(dmap: &DepthImage) -> Result<Image<u8, 3>, kornia_image::ImageError> {
    let vals = dmap.as_slice();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in vals {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let span = (hi - lo).max(1e-6);
    let gray: Vec<u8> = vals
        .iter()
        .map(|&v| (((v - lo) / span).clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let gray = Image::<u8, 1>::new(dmap.size(), gray)?;
    let mut rgb = Image::<u8, 3>::from_size_val(dmap.size(), 0)?;
    apply_colormap(&gray, &mut rgb, ColormapType::Turbo)?;
    Ok(rgb)
}
