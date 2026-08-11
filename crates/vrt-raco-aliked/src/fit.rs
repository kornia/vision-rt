//! Sizing an arbitrary image for the extractor engine, in ONE place.
//!
//! RaCo needs both dimensions on a multiple of [`DIM_DIVISOR`], and TensorRT rejects anything
//! outside the engine's built shape profile. Every tool that feeds this crate therefore needs the
//! same two decisions — how far to downscale, and how to reach the grid — and they must agree
//! *exactly*, not approximately.
//!
//! They must agree because keypoint INDICES travel between tools. `aliked_batch` writes a `.vrtk`
//! of keypoints and `lightglue_batch` writes a `.vrtm` of index pairs addressing that same
//! ordering, and the two extract independently. Same engine plus same resize is what makes those
//! two orderings the same set; a divergence addresses the wrong keypoint while staying perfectly
//! well-formed, so nothing downstream can detect it. The consumer measured that exact failure once
//! from a different cause — 4,062,208 correspondences fed, 62 inliers per surviving pair, and a map
//! with a quarter of the expected points — which is why this is a shared function rather than two
//! copies with a comment asking them to match.

use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_fast_u8_aa;

use crate::DIM_DIVISOR;

/// Fallback cap, used ONLY after the engine rejects a frame's natural size.
///
/// The shipped extractor engines carry `images:1x3x256x256|2x3x512x512|2x3x640x640`, so a side over
/// 640 is rejected by TensorRT rather than handled — which is how the first run of `aliked_batch`
/// produced zero files on 480x853 previews.
///
/// But the engine is an argument, and a caller may pass one built with a larger profile. Capping at
/// 640 unconditionally then throws away exactly the resolution that engine exists to provide:
/// measured on the consumer, feeding 1080x1920 through a 640 cap reproduces the preview path's
/// precision (keypoints found on a 352x640 image, multiplied by ~3 to reach source coordinates,
/// which multiplies their localisation error by ~3 too) while looking like a full-resolution run.
/// So callers try the natural size FIRST and fall back to this on [`ShapeRejected`].
///
/// [`ShapeRejected`]: crate::RaCoAlikedError::ShapeRejected
pub const FALLBACK_MAX_SIDE: usize = 640;

/// An image sized for the engine, with the geometry needed to map coordinates back.
pub struct Scaled {
    pub image: Image<u8, 3>,
    /// The scale actually applied, per axis — `resized_dim / original_dim`, **not** the scale that
    /// was requested. Rounding the destination to whole pixels means the two differ, and the
    /// difference lands directly in every rescaled intrinsic and every keypoint coordinate.
    pub scale_x: f64,
    pub scale_y: f64,
}

impl Scaled {
    /// Map a keypoint from engine pixels back to SOURCE pixels.
    ///
    /// The half-pixel terms are not decoration. kornia's bilinear resize maps source to destination
    /// as `dst = s * src + (s - 1) / 2` — pixel *centres*, not corners — so the inverse is
    /// `src = (dst + 0.5) / s - 0.5`. Dropping them costs a constant bias of `(1/s - 1)/2` px,
    /// which at a 3x downscale is a full pixel of systematic error on every keypoint in the run,
    /// in the same direction, on every frame. Bias that correlated does not average out in a bundle
    /// adjust; it moves the solution.
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x + 0.5) / self.scale_x as f32 - 0.5,
            (y + 0.5) / self.scale_y as f32 - 0.5,
        )
    }
}

/// Errors from [`fit_to_engine`].
#[derive(Debug, thiserror::Error)]
pub enum FitError {
    #[error(transparent)]
    Image(#[from] ImageError),
    #[error("{w}x{h} scaled to {rw}x{rh} is under one {DIM_DIVISOR}px cell")]
    TooSmall {
        w: usize,
        h: usize,
        rw: usize,
        rh: usize,
    },
}

/// Downscale to fit `max_side` with a **uniform** scale, then crop to a multiple of
/// [`DIM_DIVISOR`].
///
/// Flooring each axis to a multiple of 32 independently — the obvious implementation — makes
/// `scale_x != scale_y` by up to 2.9% on Oxford `bark`, turning an isotropic pixel threshold into
/// an ellipse whose axes differ per sequence. At 1080x1920 it is a 2.3% horizontal squash
/// (`0.32593` against `0.33333`), which is not a rounding detail: it is a systematic shear applied
/// to every keypoint before the geometry ever sees it. Requesting one scale for both axes and
/// cropping the remainder keeps them within a rounding step of each other, and cropping from the
/// right/bottom leaves the coordinate origin — and therefore every kept keypoint's coordinate —
/// untouched.
///
/// The residual anisotropy is not swept under the rug: rounding the destination to whole pixels
/// still leaves `rw/w != rh/h` in the last decimal, so both are returned and every consumer is
/// expected to use both (see [`Scaled::to_source`]).
pub fn fit_to_engine(
    src: &Image<u8, 3>,
    max_side: usize,
    interpolation: InterpolationMode,
) -> Result<Scaled, FitError> {
    let (w, h) = (src.cols(), src.rows());
    let scale = (max_side as f64 / w.max(h) as f64).min(1.0);
    let (rw, rh) = (
        ((w as f64 * scale).round() as usize).max(1),
        ((h as f64 * scale).round() as usize).max(1),
    );
    let (cw, ch) = (
        rw / DIM_DIVISOR * DIM_DIVISOR,
        rh / DIM_DIVISOR * DIM_DIVISOR,
    );
    if cw == 0 || ch == 0 {
        return Err(FitError::TooSmall { w, h, rw, rh });
    }

    // Skip the resample entirely when the natural size already fits — the full-resolution path is
    // now the common one, and a 1080x1920 frame is 6 MB of pointless copy per call on a 7.4 GB
    // board. Cropping alone still reaches the grid.
    let resized;
    let fitted = if (rw, rh) == (w, h) {
        src
    } else {
        // On u8 directly: the f32 `resize` needs two full-resolution f32 buffers, 12 bytes/pixel.
        //
        // `true` requests antialiasing but only the separable kernels honour it. Under `bilinear`
        // this is a fixed 2-tap sampler and a >2x reduction DOES alias; that is the state the
        // published tables were measured in, so it is recorded rather than silently changed.
        let mut dst = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: rw,
                height: rh,
            },
            0,
        )?;
        resize_fast_u8_aa::<3>(src, &mut dst, interpolation, true)?;
        resized = dst;
        &resized
    };

    if (cw, ch) == (rw, rh) {
        return Ok(Scaled {
            image: fitted.clone(),
            scale_x: rw as f64 / w as f64,
            scale_y: rh as f64 / h as f64,
        });
    }

    // Crop top-left anchored, so pixel coordinates are unchanged by the crop.
    let f = fitted.as_slice();
    let mut cropped = Vec::with_capacity(cw * ch * 3);
    for y in 0..ch {
        let row = y * rw * 3;
        cropped.extend_from_slice(&f[row..row + cw * 3]);
    }
    Ok(Scaled {
        image: Image::<u8, 3>::new(
            ImageSize {
                width: cw,
                height: ch,
            },
            cropped,
        )?,
        // The scale from the resize step — cropping does not change it.
        scale_x: rw as f64 / w as f64,
        scale_y: rh as f64 / h as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize) -> Image<u8, 3> {
        Image::from_size_val(
            ImageSize {
                width: w,
                height: h,
            },
            7,
        )
        .unwrap()
    }

    /// The regression this function exists for: independent floor-to-32 sheared 1080x1920 by 2.3%.
    #[test]
    fn scales_stay_isotropic() {
        let s = fit_to_engine(&img(1080, 1920), 640, InterpolationMode::Bilinear).unwrap();
        assert_eq!((s.image.cols(), s.image.rows()), (352, 640));
        assert!(
            (s.scale_x - s.scale_y).abs() < 1e-3,
            "sx {} vs sy {}",
            s.scale_x,
            s.scale_y
        );
    }

    /// Natural size: crop to the grid, never squash — and the scale stays exactly 1.
    #[test]
    fn natural_size_crops_rather_than_stretches() {
        let s = fit_to_engine(&img(1080, 1920), 1920, InterpolationMode::Bilinear).unwrap();
        assert_eq!((s.image.cols(), s.image.rows()), (1056, 1920));
        assert_eq!((s.scale_x, s.scale_y), (1.0, 1.0));
        // A keypoint at the far right of the kept region maps back to itself, not to 1082 — which
        // is what the stretching version returned for a 1080-wide source.
        let (x, _) = s.to_source(1055.0, 0.0);
        assert!((x - 1055.0).abs() < 1e-3, "{x}");
    }

    /// The half-pixel term, isolated: at 1/3 scale it is a full pixel of bias.
    #[test]
    fn inverse_carries_the_half_pixel_term() {
        let s = fit_to_engine(&img(1080, 1920), 640, InterpolationMode::Bilinear).unwrap();
        let (x, _) = s.to_source(0.0, 0.0);
        let naive = 0.0 / s.scale_x as f32;
        assert!((x - naive).abs() > 0.9, "bias {} px", (x - naive).abs());
    }

    #[test]
    fn rejects_what_cannot_reach_one_cell() {
        assert!(matches!(
            fit_to_engine(&img(100, 20), 64, InterpolationMode::Bilinear),
            Err(FitError::TooSmall { .. })
        ));
    }
}
