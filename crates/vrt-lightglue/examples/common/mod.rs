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
use kornia_imgproc::resize::resize;
use vrt_raco_aliked::DIM_DIVISOR;
use vrt_xfeat::{Descriptors, MatchResult, Matcher, XFeat, XFeatParams, XFeatResult};

/// Read a whitespace-separated list of floats.
pub fn read_floats(path: &Path) -> Result<Vec<f64>, vrt::BoxError> {
    Ok(std::fs::read_to_string(path)?
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect())
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
pub fn mat3_from_row_major(v: &[f64]) -> Mat3F64 {
    let mut a = [0.0; 9];
    a.copy_from_slice(v);
    Mat3F64::from_cols_array(&a).transpose()
}

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
/// `resize` samples at `src = a*dst + (a - 1)/2` with `a = src/dst` (half-pixel centres,
/// kornia-imgproc `resize/mod.rs`), so the forward map is `dst = s*src + (s - 1)/2`. The
/// translation term is small — a fifth of a pixel at these scales — but it is not zero,
/// and a bare `diag(s, s, 1)` silently biases every rescaled intrinsic and homography.
pub fn resize_matrix(scale: f64) -> Mat3F64 {
    let off = (scale - 1.0) / 2.0;
    Mat3F64::from_cols(
        Vec3F64::new(scale, 0.0, 0.0),
        Vec3F64::new(0.0, scale, 0.0),
        Vec3F64::new(off, off, 1.0),
    )
}

/// An image resized for the engines, with the geometry needed to follow it.
pub struct Scaled {
    pub image: Image<u8, 3>,
    /// Uniform scale applied to both axes.
    pub scale: f64,
}

/// Downscale to fit `max_side` with a **uniform** scale, then crop to a multiple of
/// [`DIM_DIVISOR`].
///
/// Flooring each axis to a multiple of 32 independently — the obvious implementation —
/// makes `sx != sy`: on Oxford `bark` that is a 2.9% horizontal squash, and it turns an
/// isotropic pixel threshold into an ellipse whose axes differ per sequence. Scaling
/// uniformly and cropping the remainder keeps one scalar scale, and cropping from the
/// right/bottom leaves the coordinate origin untouched, so ground-truth homographies and
/// intrinsics need only that scalar.
pub fn resize_to_fit(
    src: &Image<u8, 3>,
    max_side: usize,
    interpolation: InterpolationMode,
) -> Result<Scaled, vrt::BoxError> {
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

    let src_f32 = Image::<f32, 3>::new(
        src.size(),
        src.as_slice().iter().map(|&v| v as f32).collect(),
    )?;
    let mut dst_f32 = Image::<f32, 3>::from_size_val(
        ImageSize {
            width: rw,
            height: rh,
        },
        0.0,
    )?;
    resize(&src_f32, &mut dst_f32, interpolation)?;

    // Crop top-left anchored, so pixel coordinates are unchanged by the crop.
    let f = dst_f32.as_slice();
    let mut cropped = Vec::with_capacity(cw * ch * 3);
    for y in 0..ch {
        let row = y * rw * 3;
        cropped.extend(
            f[row..row + cw * 3]
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8),
        );
    }
    Ok(Scaled {
        image: Image::<u8, 3>::new(
            ImageSize {
                width: cw,
                height: ch,
            },
            cropped,
        )?,
        scale,
    })
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
        let dim = self.matcher.dim();
        self.matcher.submit(
            Descriptors::new(&self.left.descs, self.left.count(), dim),
            Descriptors::new(&self.right.descs, self.right.count(), dim),
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
        let m = mat3_from_row_major(&[799.4, 0.0, 524.0, 0.0, 799.4, 289.0, 0.0, 0.0, 1.0]);
        // Column-major storage: `x_axis` is the first COLUMN of the logical matrix.
        assert_eq!(m.x_axis.x, 799.4);
        assert_eq!(m.x_axis.y, 0.0);
        assert_eq!(
            m.z_axis.x, 524.0,
            "cx must land in the top-right, not the bottom-left"
        );
        assert_eq!(m.z_axis.y, 289.0);
        // And it must act like K on a point.
        let p = m * Vec3F64::new(1.0, 2.0, 1.0);
        assert!((p.x - (799.4 + 524.0)).abs() < 1e-9);
        assert!((p.y - (2.0 * 799.4 + 289.0)).abs() < 1e-9);
    }

    /// The half-pixel term is exactly what a bare `diag(s, s, 1)` gets wrong.
    #[test]
    fn resize_matrix_carries_the_half_pixel_offset() {
        let m = resize_matrix(0.5);
        let p = m * Vec3F64::new(0.0, 0.0, 1.0);
        assert!((p.x - (-0.25)).abs() < 1e-12, "got {}", p.x);
        assert!((p.y - (-0.25)).abs() < 1e-12);
        // Identity scale must be exactly the identity, offset included.
        let i = resize_matrix(1.0);
        assert_eq!(i.z_axis.x, 0.0);
        assert_eq!(i.z_axis.y, 0.0);
    }
}
