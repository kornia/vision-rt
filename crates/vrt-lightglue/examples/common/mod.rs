//! Shared helpers for the benchmark examples (included via
//! `#[path = "common/mod.rs"] mod common;`).
//!
//! `eval_oxford`, `eval_imc` and `prep_oxford` must resample images and parse ground truth
//! *identically* or their numbers are not comparable, so that logic lives here once rather
//! than in three copies that can drift apart silently.

// Each example compiles this module separately and uses a subset of it.
#![allow(dead_code)]

use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_fast_u8_aa;
use vrt_raco_aliked::DIM_DIVISOR;
use vrt_xfeat::{Descriptors, MatchResult, Matcher, XFeat, XFeatParams, XFeatResult};

/// Read a whitespace-separated list of floats, naming the file on any failure.
///
/// Every token must parse. `filter_map(|t| t.parse().ok())` is the tempting form and it
/// is the same trap [`arg_or`] documents: a corrupt ground-truth file would silently lose
/// the unparseable tokens, and the callers only check the *count*, so a 10-token file with
/// one bad entry would sail through as a valid 3x3 built from the wrong numbers.
pub fn read_floats(path: &Path) -> Result<Vec<f64>, vrt::BoxError> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.split_whitespace()
        .map(|t| {
            t.parse()
                .map_err(|_| format!("{}: cannot parse {t:?} as a number", path.display()).into())
        })
        .collect()
}

/// Parse an optional positional argument, or fail loudly.
///
/// `.and_then(|s| s.parse().ok()).unwrap_or(default)` is the tempting form and it is a
/// trap: these binaries take several optional positionals, so one misplaced argument
/// silently reverts to a default and prints a plausible table nobody can distinguish from
/// a real one.
pub fn arg_or<T: FromStr>(
    args: &[String],
    i: usize,
    name: &str,
    default: T,
) -> Result<T, vrt::BoxError> {
    match args.get(i) {
        None => Ok(default),
        Some(s) => s
            .parse()
            .map_err(|_| format!("argument {i} ({name}): cannot parse {s:?}").into()),
    }
}

/// `Mat3F64` is column-major (glam); every ground-truth file here is row-major.
///
/// Fallible rather than slicing-and-panicking: the callers read these from files, so a
/// short one is bad input, not a bug, and `copy_from_slice`'s panic names neither the
/// file nor the expected length.
pub fn mat3_from_row_major(v: &[f64]) -> Result<Mat3F64, vrt::BoxError> {
    let a: [f64; 9] = v
        .try_into()
        .map_err(|_| format!("expected 9 floats for a 3x3 matrix, got {}", v.len()))?;
    Ok(Mat3F64::from_cols_array(&a).transpose())
}

/// Parse the resampling kernel name.
///
/// **Only `bicubic` and `lanczos` anti-alias.** `resize_fast_u8_aa`'s `antialias` flag
/// widens the *separable* kernel by the downscale factor; kornia documents `Nearest` and
/// `Bilinear` as "unaffected by this flag", and they route to fixed 1-tap / 2-tap
/// samplers ([`resize_u8_path`] in kornia-imgproc `resize/mod.rs`). Downscaling by more
/// than ~2x under `bilinear` or `nearest` therefore aliases the high-frequency texture keypoint
/// detectors fire on. All four are accepted because the published tables were measured
/// under `lanczos`, which is a separable path and so does honour the flag.
///
/// All four share the same half-pixel geometry, so [`resize_matrix`] is valid for every
/// one of them: `Nearest`'s `floor((i + 0.5) * scale)` is exactly `round` of the
/// `align_corners=False` centre, not a different convention.
pub fn parse_interpolation(s: &str) -> Result<InterpolationMode, vrt::BoxError> {
    match s {
        "nearest" => Ok(InterpolationMode::Nearest),
        "bilinear" => Ok(InterpolationMode::Bilinear),
        "bicubic" => Ok(InterpolationMode::Bicubic),
        "lanczos" => Ok(InterpolationMode::Lanczos),
        other => Err(format!("unknown interpolation {other}").into()),
    }
}

/// The pixel-coordinate map induced by [`resize_to_fit`], as a homogeneous matrix.
///
/// `resize_fast_u8_aa` samples at `src = a*dst + (a - 1)/2` with `a = src/dst` — the
/// `align_corners=False` half-pixel centre every one of its kernels uses (`bilinear.rs`
/// `bilinear_tap`, `common.rs` `center`, `nearest.rs` `nearest_index`, all in
/// kornia-imgproc `resize/`) — so the forward map is `dst = s*src + (s - 1)/2`. The
/// translation term is small — a fifth of a pixel at these scales — but it is not zero,
/// and a bare `diag(s, s, 1)` silently biases every rescaled intrinsic and homography.
///
/// Takes both axes because the destination size is an integer: `round(w*s)/w` and
/// `round(h*s)/h` are not equal in general even when a single `s` was requested. On
/// Oxford `bark` (765x512 -> 640x428) they differ by 8e-4, which is 0.28 px at the
/// bottom of the image against a 2.5 px threshold — on the one sequence the rotation
/// claim rests on.
pub fn resize_matrix(sx: f64, sy: f64) -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(sx, 0.0, 0.0),
        Vec3F64::new(0.0, sy, 0.0),
        Vec3F64::new((sx - 1.0) / 2.0, (sy - 1.0) / 2.0, 1.0),
    )
}

/// An image resized for the engines, with the geometry needed to follow it.
pub struct Scaled {
    pub image: Image<u8, 3>,
    /// The scale actually applied, per axis — `resized_dim / original_dim`, **not** the
    /// scale that was requested. Rounding the destination to whole pixels means the two
    /// differ, and the difference lands directly in every rescaled intrinsic.
    pub scale_x: f64,
    pub scale_y: f64,
}

/// Downscale to fit `max_side` with a **uniform** scale, then crop to a multiple of
/// [`DIM_DIVISOR`].
///
/// Flooring each axis to a multiple of 32 independently — the obvious implementation —
/// makes `sx != sy` by up to 2.9% on Oxford `bark`, turning an isotropic pixel threshold
/// into an ellipse whose axes differ per sequence. Requesting one scale for both axes and
/// cropping the remainder keeps them within a rounding step of each other, and cropping
/// from the right/bottom leaves the coordinate origin untouched.
///
/// The residual anisotropy is not swept under the rug: rounding the destination to whole
/// pixels leaves `rw/w != rh/h`, so both are returned and every consumer uses both.
pub fn resize_to_fit(
    src: &Image<u8, 3>,
    max_side: usize,
    interpolation: InterpolationMode,
) -> Result<Scaled, vrt::BoxError> {
    // The output feeds RaCo *and* XFeat, and XFeat floors to its own hardcoded 32. If
    // RaCo's divisor ever moves, images sized on RaCo's grid stop being multiples of
    // XFeat's and every XFeat keypoint in the baseline column shifts — silently, because
    // XFeat just rescales by a ratio that is no longer 1.
    const XFEAT_GRID: usize = 32;
    assert!(
        DIM_DIVISOR.is_multiple_of(XFEAT_GRID),
        "DIM_DIVISOR {DIM_DIVISOR} must stay a multiple of XFeat's {XFEAT_GRID}px grid"
    );
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
        return Err(
            format!("{w}x{h} scaled to {rw}x{rh} is under one {DIM_DIVISOR}px cell").into(),
        );
    }

    let mut resized = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: rw,
            height: rh,
        },
        0,
    )?;
    // On u8 directly: the f32 `resize` needed two full-resolution f32 buffers, 12
    // bytes/pixel, on a 7.4 GB box.
    //
    // `true` requests antialiasing but only the separable kernels honour it — see
    // [`parse_interpolation`]. Under `bilinear` this is a fixed 2-tap
    // sampler and a >2x reduction DOES alias; that is the state the published tables
    // were measured in, so it is recorded rather than silently changed.
    resize_fast_u8_aa::<3>(src, &mut resized, interpolation, true)?;

    // Crop top-left anchored, so pixel coordinates are unchanged by the crop.
    let f = resized.as_slice();
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
        // The applied scale, from the resize step — cropping does not change it.
        scale_x: rw as f64 / w as f64,
        scale_y: rh as f64 / h as f64,
    })
}

/// One pair's precision, from its `(matches, inliers)`.
///
/// The same expression [`Tally::add`] folds in, so a printed per-pair row and the macro
/// average over those rows cannot disagree about what "precision" means. A pair with no
/// matches is 0%, not undefined — the matcher returned nothing and that is a result.
pub fn pair_pct((m, i): (usize, usize)) -> f64 {
    if m == 0 {
        0.0
    } else {
        100.0 * i as f64 / m as f64
    }
}

/// One matcher column's running score over a set of pairs.
///
/// Shared by both harnesses: `eval_oxford` previously carried the same four quantities as
/// seven loose scalars, so the two evaluations could drift on what "macro precision" meant
/// while both claiming to report it.
#[derive(Default, Clone, Copy)]
pub struct Tally {
    pub pairs: usize,
    pub matches: usize,
    pub inliers: usize,
    /// Sum of per-pair precisions, for the macro average.
    pub pct_sum: f64,
}

impl Tally {
    /// Fold in one pair's `(matches, inliers)`.
    ///
    /// A pair with no matches still counts as a pair and contributes 0% — a matcher that
    /// returns nothing has not earned a missing row.
    pub fn add(&mut self, scored: (usize, usize)) {
        self.pairs += 1;
        self.matches += scored.0;
        self.inliers += scored.1;
        self.pct_sum += pair_pct(scored);
    }

    /// Fold in a whole sub-tally, for a totals row.
    pub fn add_tally(&mut self, other: &Tally) {
        self.pairs += other.pairs;
        self.matches += other.matches;
        self.inliers += other.inliers;
        self.pct_sum += other.pct_sum;
    }

    /// Pooled ("micro") precision: inliers over matches across the whole set.
    ///
    /// Weighted by match volume, so a single high-match pair can dominate — and the
    /// columns differ severalfold in volume, so it weights them differently too. Read it
    /// next to [`macro_pct`](Self::macro_pct), never alone.
    pub fn pct(&self) -> f64 {
        if self.matches == 0 {
            0.0
        } else {
            100.0 * self.inliers as f64 / self.matches as f64
        }
    }

    /// Mean of the per-pair precisions: every pair counts once.
    pub fn macro_pct(&self) -> f64 {
        if self.pairs == 0 {
            0.0
        } else {
            self.pct_sum / self.pairs as f64
        }
    }
}

/// XFeat plus its native 64-D mutual-NN, the independent baseline both evaluations
/// compare against.
///
/// Model, matcher and all three result buffers live together because they are only ever
/// valid as a set — holding them as separate `Option`s lets one be updated without the
/// others and fail by silently taking the "no baseline" path instead of not compiling.
pub struct XFeatBaseline {
    xfeat: XFeat,
    matcher: Matcher,
    left: XFeatResult,
    right: XFeatResult,
    matches: MatchResult,
    capacity: usize,
}

impl XFeatBaseline {
    /// `capacity` is the keypoint budget. Pass the RaCo engine's `k` so both columns are
    /// measured at the same budget — hardcoding a larger one hands the baseline more
    /// keypoints than the method it is a baseline for, and nothing in the output says so.
    pub fn load(
        engine: &str,
        stream: &Arc<CudaStream>,
        capacity: usize,
    ) -> Result<Self, vrt::BoxError> {
        let xfeat =
            XFeat::from_engine_file(engine, stream.clone(), XFeatParams::new(capacity, 0.05))?;
        let matcher = Matcher::new(stream.clone())?;
        Ok(Self {
            left: xfeat.alloc_result()?,
            right: xfeat.alloc_result()?,
            matches: matcher.alloc_result(capacity)?,
            xfeat,
            matcher,
            capacity,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Enqueue extraction for both images. Separate from [`Self::finish`] so the caller can
    /// submit this alongside the other models and pay for one stream synchronise, not two.
    /// `sync_between` must be `Some` when the two frames differ in size: each `submit`
    /// sets the execution context's input shape and can reallocate session-owned output
    /// buffers, and those are host-side calls rather than stream-ordered ones, so the
    /// second would mutate a context whose first enqueue is still in flight.
    pub fn submit(
        &mut self,
        left: &Image<u8, 3>,
        right: &Image<u8, 3>,
        sync_between: Option<&Arc<CudaStream>>,
    ) -> Result<(), vrt::BoxError> {
        self.xfeat.submit(left, &mut self.left)?;
        if let Some(stream) = sync_between {
            stream.synchronize()?;
        }
        self.xfeat.submit(right, &mut self.right)?;
        Ok(())
    }

    /// Match what [`Self::submit`] extracted, returning `(pairs, left keypoints, right
    /// keypoints)`. The caller must have synchronised the stream since `submit`, because
    /// the keypoint counts driving the match are read back from the device.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        &mut self,
        stream: &Arc<CudaStream>,
        min_cossim: f32,
    ) -> Result<(Vec<(usize, usize)>, Vec<(f32, f32)>, Vec<(f32, f32)>), vrt::BoxError> {
        self.matcher.submit(
            Descriptors::new(&self.left.descs, self.left.count(), self.left.desc_dim()),
            Descriptors::new(&self.right.descs, self.right.count(), self.right.desc_dim()),
            min_cossim,
            &mut self.matches,
        )?;
        stream.synchronize()?;
        let to_xy = |flat: Vec<f32>| -> Vec<(f32, f32)> {
            flat.chunks_exact(2).map(|p| (p[0], p[1])).collect()
        };
        Ok((
            self.matches.pairs(),
            to_xy(self.left.kpts_to_host()?),
            to_xy(self.right.kpts_to_host()?),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major in, column-major out. A symmetric fixture cannot catch a dropped
    /// transpose, so this uses an intrinsics matrix — the shape that actually flows
    /// through these harnesses, and the one whose silent transposition would rewrite
    /// every published number without failing anything.
    #[test]
    fn mat3_from_row_major_transposes() {
        let m = mat3_from_row_major(&[799.4, 0.0, 524.0, 0.0, 799.4, 289.0, 0.0, 0.0, 1.0])
            .expect("9 floats is the valid length");
        // Column-major storage: `x_axis` is the first COLUMN of the logical matrix.
        assert_eq!(m.x_axis.x, 799.4);
        assert_eq!(m.x_axis.y, 0.0);
        assert_eq!(
            m.z_axis.x, 524.0,
            "cx must land in the top-right, not the bottom-left"
        );
        assert_eq!(m.z_axis.y, 289.0);
        // A wrong length is an error, not a panic.
        assert!(mat3_from_row_major(&[1.0, 2.0, 3.0]).is_err());
        // And it must act like K on a point.
        let p = m * Vec3F64::new(1.0, 2.0, 1.0);
        assert!((p.x - (799.4 + 524.0)).abs() < 1e-9);
        assert!((p.y - (2.0 * 799.4 + 289.0)).abs() < 1e-9);
    }

    /// The half-pixel term is exactly what a bare `diag(s, s, 1)` gets wrong.
    /// `resize_to_fit` must report the scale it *applied*, not the one it asked for.
    ///
    /// This calls `resize_to_fit`. An earlier version re-derived the formula inline and
    /// asserted on its own copy, so reverting the function to a single uniform scale left
    /// it green — it could not fail for the bug it was written to catch.
    #[test]
    fn resize_to_fit_reports_the_applied_scale_per_axis() {
        // Oxford bark: 765x512 -> 640x428, cropped to 640x416.
        let src = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 765,
                height: 512,
            },
            0,
        )
        .unwrap();
        let out = resize_to_fit(&src, 640, InterpolationMode::Bilinear).unwrap();

        assert_eq!(out.image.cols(), 640);
        assert_eq!(out.image.rows(), 416, "cropped to the 32px grid");

        let requested = 640.0 / 765.0;
        assert!((out.scale_x - requested).abs() < 1e-12, "x is exact here");
        assert!(
            (out.scale_y - requested).abs() > 1e-6,
            "y must differ from the requested scale — that is the whole bug"
        );
        assert!(
            (out.scale_y - 428.0 / 512.0).abs() < 1e-12,
            "y is the applied scale"
        );
        // ~0.28 px at the bottom of the cropped image, against a ~2.5 px threshold.
        assert!(((out.scale_y - requested) * 416.0).abs() > 0.2);
    }

    /// A square image scales exactly on both axes — the case that must NOT report a
    /// spurious difference, so the test above is measuring something real.
    #[test]
    fn resize_to_fit_is_exact_when_the_scale_divides() {
        let src = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 1280,
                height: 640,
            },
            0,
        )
        .unwrap();
        let out = resize_to_fit(&src, 640, InterpolationMode::Bilinear).unwrap();
        assert_eq!((out.scale_x, out.scale_y), (0.5, 0.5));
    }

    #[test]
    fn resize_matrix_carries_the_half_pixel_offset() {
        let m = resize_matrix(0.5, 0.5);
        let p = m * Vec3F64::new(0.0, 0.0, 1.0);
        assert!((p.x - (-0.25)).abs() < 1e-12, "got {}", p.x);
        assert!((p.y - (-0.25)).abs() < 1e-12);
        // Identity scale must be exactly the identity, offset included.
        let i = resize_matrix(1.0, 1.0);
        assert_eq!(i.z_axis.x, 0.0);
        assert_eq!(i.z_axis.y, 0.0);
    }
}
