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
use kornia_image::Image;
use kornia_imgproc::interpolation::InterpolationMode;
use vrt_raco_aliked::{fit_to_engine, DIM_DIVISOR};
// Re-exported, not redefined: `eval_imc`/`prep_oxford` name it through this module.
pub use vrt_raco_aliked::Scaled;
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

/// Downscale to fit `max_side` with a **uniform** scale, then crop to a multiple of
/// [`DIM_DIVISOR`].
///
/// A thin wrapper over [`vrt_raco_aliked::fit_to_engine`], which owns the policy and
/// documents why the scale has to be uniform. The policy lives in the library rather than
/// here because the batch bridge tools in `vrt-raco-aliked` and `vrt-lightglue` need
/// byte-identical sizing: they exchange keypoint INDICES, so two implementations that
/// agree today and drift tomorrow produce a well-formed file addressing the wrong points.
///
/// What this wrapper adds is the one constraint the library cannot know about — that the
/// benchmark harness feeds the same image to XFeat as well.
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
    Ok(fit_to_engine(src, max_side, interpolation)?)
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

/// Squared Sampson error of a correspondence against a fundamental matrix.
///
/// Mirrors `kornia_3d::pose::sampson_distance` (kornia-3d/src/pose/fundamental.rs), which
/// is not depended on here only because that crate costs 74 transitive dependencies for
/// this function and one pose composition. Note it is the error **squared**, as there;
/// callers take the root so a pixel threshold means pixels.
pub fn sampson_squared(f: &Mat3F64, x1: (f64, f64), x2: (f64, f64)) -> f64 {
    let (a, b) = (Vec3F64::new(x1.0, x1.1, 1.0), Vec3F64::new(x2.0, x2.1, 1.0));
    let (fx1, ftx2) = (*f * a, f.transpose() * b);
    let err = b.dot(fx1);
    let denom = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
    if denom <= 1e-12 {
        return err * err;
    }
    err * err / denom
}

/// Relative pose `a -> b` for world-to-camera poses, i.e. `kornia_3d::pose::Pose3d::between`.
///
/// With `p_cam = R p_world + t`, composing `b` with `a.inverse()` gives this.
pub fn relative_pose(
    (ra, ta): (Mat3F64, Vec3F64),
    (rb, tb): (Mat3F64, Vec3F64),
) -> (Mat3F64, Vec3F64) {
    let r = rb * ra.transpose();
    (r, tb - r * ta)
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
    /// Differently-sized frames are safe: `XFeat::submit` drains the stream itself when
    /// the model size changes, so callers do not have to know that reconfiguring the
    /// execution context is a host-side operation.
    pub fn submit(
        &mut self,
        left: &Image<u8, 3>,
        right: &Image<u8, 3>,
    ) -> Result<(), vrt::BoxError> {
        self.xfeat.submit(left, &mut self.left)?;
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
            Descriptors::from_xfeat(&self.left),
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
