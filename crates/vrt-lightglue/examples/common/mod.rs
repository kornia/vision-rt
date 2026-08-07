//! Shared helpers for the benchmark examples (included via
//! `#[path = "common/mod.rs"] mod common;`).
//!
//! `eval_oxford`, `eval_imc` and `prep_oxford` must resample images and parse ground truth
//! *identically* or their numbers are not comparable, so that logic lives here once rather
//! than in three copies that can drift apart silently.

// Each example compiles this module separately and uses a subset of it.
#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaStream;

use kornia_algebra::Mat3F64;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize;
use vrt_raco_aliked::DIM_DIVISOR;
use vrt_xfeat::{MatchResult, Matcher, XFeat, XFeatParams, XFeatResult};

/// Read a whitespace-separated list of floats.
pub fn read_floats(path: &Path) -> Result<Vec<f64>, vrt::BoxError> {
    Ok(std::fs::read_to_string(path)?
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect())
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

/// Largest size within `max_side` whose sides are multiples of RaCo's `DIM_DIVISOR`.
pub fn target_size(w: usize, h: usize, max_side: usize) -> (usize, usize) {
    let f = (max_side as f64 / w.max(h) as f64).min(1.0);
    let fit = |n: usize| (((n as f64 * f) as usize) / DIM_DIVISOR * DIM_DIVISOR).max(DIM_DIVISOR);
    (fit(w), fit(h))
}

/// Downscale to fit `max_side`, returning the image and the exact `(sx, sy)` applied so
/// intrinsics or homographies can follow it.
///
/// Goes through `resize`, the f32 path, so the interpolation kernel is selectable and the
/// filter does not round to u8 internally. The conversion is irrelevant next to the
/// inference this feeds.
pub fn resize_to_fit(
    src: &Image<u8, 3>,
    max_side: usize,
    interpolation: InterpolationMode,
) -> Result<(Image<u8, 3>, f64, f64), vrt::BoxError> {
    let (w, h) = (src.cols(), src.rows());
    let (nw, nh) = target_size(w, h, max_side);

    let src_f32 = Image::<f32, 3>::new(
        src.size(),
        src.as_slice().iter().map(|&v| v as f32).collect(),
    )?;
    let mut dst_f32 = Image::<f32, 3>::from_size_val(
        ImageSize {
            width: nw,
            height: nh,
        },
        0.0,
    )?;
    resize(&src_f32, &mut dst_f32, interpolation)?;

    let dst = Image::<u8, 3>::new(
        dst_f32.size(),
        dst_f32
            .as_slice()
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
    )?;
    Ok((dst, nw as f64 / w as f64, nh as f64 / h as f64))
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
}

impl XFeatBaseline {
    /// Keypoint budget for the baseline; also the match-result capacity.
    const CAPACITY: usize = 4096;

    pub fn load(engine: &str, stream: &Arc<CudaStream>) -> Result<Self, vrt::BoxError> {
        let xfeat = XFeat::from_engine_file(
            engine,
            stream.clone(),
            XFeatParams::new(Self::CAPACITY, 0.05),
        )?;
        let matcher = Matcher::new(stream.clone())?;
        Ok(Self {
            left: xfeat.alloc_result()?,
            right: xfeat.alloc_result()?,
            matches: matcher.alloc_result(Self::CAPACITY)?,
            xfeat,
            matcher,
        })
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
        self.matcher.submit(
            &self.left.descs,
            self.left.count(),
            &self.right.descs,
            self.right.count(),
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
