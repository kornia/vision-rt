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

/// The image extensions the bridge tools accept, compared case-insensitively.
///
/// Shared for the same reason the sizing is: `aliked_batch` scans a directory while
/// `lightglue_batch` builds filenames from frame indices, so two different lists let one tool see a
/// frame set the other cannot. That is not a cosmetic disagreement — it is the dead-frame failure
/// the matcher exits non-zero on, reached by writing `.jpg` in one place and `jpg|jpeg|png` in the
/// other.
pub const FRAME_EXTS: [&str; 3] = ["jpg", "jpeg", "png"];

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
    /// A host image reached [`fit_to_engine_cuda`]. Reported rather than silently uploaded: an
    /// implicit transfer per frame is exactly the cost this path exists to remove, so it should
    /// fail loudly instead of quietly performing it.
    #[error("fit_to_engine_cuda needs device-resident images; upload the source with `to_cuda`")]
    NotDeviceResident,
    #[error("CUDA: {0}")]
    Cuda(String),
}

/// The geometry of one fit: resize to `rw x rh`, then crop to `cw x ch`.
///
/// Split out as its own type because the host and device paths MUST agree on it exactly. They
/// exchange nothing at runtime, so if the arithmetic were written twice the two could drift and a
/// build that mixed them would place keypoints from one geometry into coordinates computed for the
/// other — well-formed, and wrong. Everything below is integer arithmetic on dimensions, so both
/// paths reach identical numbers by construction rather than by review.
struct Plan {
    rw: usize,
    rh: usize,
    cw: usize,
    ch: usize,
}

impl Plan {
    fn new(w: usize, h: usize, max_side: usize) -> Result<Self, FitError> {
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
        Ok(Self { rw, rh, cw, ch })
    }
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
    let Plan { rw, rh, cw, ch } = Plan::new(w, h, max_side)?;
    let (scale_x, scale_y) = (rw as f64 / w as f64, rh as f64 / h as f64);

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
            scale_x,
            scale_y,
        });
    }

    // Crop top-left anchored, so pixel coordinates are unchanged by the crop. `crop_image` rather
    // than a row loop: it is the same operation, and kornia's carries the NEON strided-row copy and
    // the rayon split that a scalar loop here would not — on the crop-only path this copy IS the
    // cost of the fit.
    let mut cropped = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: cw,
            height: ch,
        },
        0,
    )?;
    kornia_imgproc::crop::crop_image(fitted, &mut cropped, 0, 0)?;
    Ok(Scaled {
        image: cropped,
        // The scale from the resize step — cropping does not change it.
        scale_x,
        scale_y,
    })
}

/// [`fit_to_engine`] with the resize and the crop run **on the GPU**.
///
/// `src` must already be device-resident (`Image::to_cuda`), and the returned image is too — so it
/// goes straight into [`RaCoAliked::submit`](crate::RaCoAliked::submit) with no round trip.
///
/// ## When this is the faster path, and when it is not
///
/// Use it when the frame is **already on the device** — a camera surface, or any pipeline whose
/// previous stage left its output in device memory. There it is a clear win: measured on an Orin
/// Nano at 1080x1920 -> 640, the fit itself costs 0.53 ms against the host's 1.13 ms, because a
/// >2x antialiased downscale is exactly the work a GPU is for.
///
/// Do NOT reach for it when the frame starts on the host, as a JPEG-fed batch tool's does. The
/// host path uploads the FITTED image; this one has to upload the RAW frame, and at the 640 cap
/// that is 6.2 MB against 0.68 MB. Measured end to end on the same board:
///
/// ```text
///                                     host fit + upload   device fit + upload
///   1080x1920 natural (crop only)          1.95 ms          3.33 ms (2.24 + 1.10)
///   1080x1920 -> 640 (resize + crop)       1.13 ms          1.62 ms (0.53 + 1.09)
/// ```
///
/// The upload is the whole difference, and it is not avoidable: the extractor needs device memory
/// either way, so the only question is whether the bytes crossing the bus are the big ones or the
/// small ones. Both numbers are a few percent of the ~79 ms the extractor itself takes, so this is
/// a choice about which resource to spend rather than a bottleneck in either direction.
///
/// ## Why this cannot drift from the host path
///
/// (Which matters more than the timing: the two paths write coordinates into files that are
/// compared against each other.)
///
/// The resize is the SAME `resize_fast_u8_aa` call. kornia routes a device/device pair to its CUDA
/// u8 kernels and documents the result as bit-identical — "the coordinate/weight tables come from
/// the same host builders the CPU uses" — with `resize_u8_path` as a single shared routing
/// decision, so a device pair always runs the GPU twin of the kernel a host pair would run. A
/// mixed host/device pair is a typed error, never a silent transfer. The geometry comes from the
/// same [`Plan`]. That leaves the crop as the only genuinely separate implementation — `crop_image`
/// on the host, an identity affine warp on the device — and `gpu_fit.rs` pins those two together
/// byte-for-byte precisely because they are not the same code.
pub fn fit_to_engine_cuda(
    src: &Image<u8, 3>,
    max_side: usize,
    interpolation: InterpolationMode,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<Scaled, FitError> {
    let (w, h) = (src.cols(), src.rows());
    let Plan { rw, rh, cw, ch } = Plan::new(w, h, max_side)?;
    let (scale_x, scale_y) = (rw as f64 / w as f64, rh as f64 / h as f64);

    // Same skip as the host path: at natural size there is nothing to resample, only to crop.
    let resized;
    let fitted = if (rw, rh) == (w, h) {
        src
    } else {
        let mut dst = Image::<u8, 3>::zeros_cuda(
            ImageSize {
                width: rw,
                height: rh,
            },
            stream,
        )?;
        // Device in, device out — this dispatches to the CUDA kernels. If `src` were host-resident
        // kornia would return a residency error rather than quietly falling back, which is what
        // makes "did this actually run on the GPU?" a question the types answer.
        resize_fast_u8_aa::<3>(src, &mut dst, interpolation, true)?;
        resized = dst;
        &resized
    };

    // Always through the crop, even when `(cw, ch) == (rw, rh)` and it degenerates to a straight
    // copy. The host path can return the resized buffer by move there; here the borrow may point at
    // `src`, which the caller owns and may reuse, so a copy is needed either way — and one path is
    // worth more than saving a D2D memcpy the resize already dwarfs.
    let mut out = Image::<u8, 3>::zeros_cuda(
        ImageSize {
            width: cw,
            height: ch,
        },
        stream,
    )?;
    crop_top_left_device(fitted, &mut out, stream)?;
    Ok(Scaled {
        image: out,
        scale_x,
        scale_y,
    })
}

/// Top-left crop on the device, in ONE launch, by one of two routes.
///
/// When the width is unchanged the kept region is a contiguous prefix, so the whole crop is a
/// single `memcpy_dtod`. Otherwise the rows are strided and it goes through an identity affine
/// warp. Both are stream-ordered, so this needs no sync of its own — it lands in order with the
/// resize before it and the extractor after it.
///
/// Geometry is read from the buffers rather than passed in: this is the one function whose entire
/// purpose is that host and device geometry cannot diverge, so it should not be possible to hand it
/// dimensions that disagree with the images it is also handed.
fn crop_top_left_device(
    src: &Image<u8, 3>,
    dst: &mut Image<u8, 3>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<(), FitError> {
    let src_w = src.cols();
    let (cw, ch) = (dst.cols(), dst.rows());
    // When the width is unchanged the kept region is a contiguous PREFIX of the buffer, so the
    // whole crop is one copy. This is not a rare special case: it is every frame whose width
    // already sits on the 32 px grid, which includes the 480-wide previews and every 640 fallback.
    if cw == src_w {
        let s = src.0.as_cudaslice().ok_or(FitError::NotDeviceResident)?;
        let d = dst
            .0
            .as_cudaslice_mut()
            .ok_or(FitError::NotDeviceResident)?;
        let n = cw * ch * 3;
        let sv = s.slice(0..n);
        let mut dv = d.slice_mut(0..n);
        return stream
            .memcpy_dtod(&sv, &mut dv)
            .map_err(|e| FitError::Cuda(e.to_string()));
    }

    // Strided crop, in ONE launch. The obvious form — one `memcpy_dtod` per row — was measured at
    // 11.6 ms for a 1920-row frame against 2.1 ms for the entire host path: at ~5 us of launch
    // overhead apiece, 1920 launches are the whole cost, and the copy itself is free by comparison.
    //
    // An identity affine is a pure copy: every destination pixel samples its own integer source
    // coordinate, where the bilinear weights collapse to the top-left tap exactly. `warp_affine_u8`
    // is used rather than a hand-rolled kernel for the same reason as the resize — it carries
    // kornia's residency dispatch and its host/device bit-identity guarantee, so the device crop
    // cannot drift from the host one. `gpu_fit.rs` asserts that byte-for-byte.
    let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    kornia_imgproc::warp::warp_affine_u8::<3>(src, dst, &m).map_err(FitError::Image)
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
