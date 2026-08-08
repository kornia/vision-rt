//! Isolate **in-plane rotation** from every other nuisance, and sweep it to 180°.
//!
//! The Oxford `bark` sequence is the usual evidence for rotation robustness, and it cannot
//! settle the question: every large-rotation pair in it also carries a large zoom
//! (`bark/3` 150° + 1.85x, `bark/4` 120° + 2.48x, `bark/6` 153° + 4.09x). A failure there
//! could be the rotation, the scale, or the two together. `bark/5` shows 3x zoom alone is
//! fine at 23°, which is suggestive but not conclusive.
//!
//! This runs one image against rotated copies of itself. The source is scaled so its
//! **diagonal** fits a square canvas and pasted centrally, so rotating the canvas about
//! its centre loses no content at any angle — the only difference between the reference
//! and the query is the rotation, and the ground-truth homography is exact by
//! construction rather than estimated.
//!
//! Resampling is the one confound left: a rotated image is interpolated where the original
//! is not, so a few tenths of a percent at 0° is the harness, not the model. 0° is included
//! precisely to measure that floor.
//!
//! Usage:
//!   cargo run --release -p vrt-lightglue --example eval_rotation -- \
//!       <image> <raco.engine> <lightglue.engine> \
//!       [xfeat.engine] [step_deg] [inlier_px] [nn_cossim] [xfeat_cossim]

use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::warp::warp_perspective_u8;
use kornia_io::functional::read_image_any_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::{RaCoAliked, DIM_DIVISOR};
use vrt_xfeat::{Descriptors, Matcher};

#[path = "common/mod.rs"]
mod common;
use common::{arg_or, pair_pct, XFeatBaseline};

/// Square canvas side. Must fit the engines' shape profile and the 32px grid.
const CANVAS: usize = 640;

/// Uniform scale about the canvas centre — the other half of what `bark` confounds.
fn scale_about_centre(z: f64, centre: f64) -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(z, 0.0, 0.0),
        Vec3F64::new(0.0, z, 0.0),
        Vec3F64::new(centre * (1.0 - z), centre * (1.0 - z), 1.0),
    )
}

/// Rotation about the canvas centre, mapping reference pixels to rotated pixels.
fn rotation_about_centre(deg: f64, centre: f64) -> Mat3F64 {
    let (s, c) = deg.to_radians().sin_cos();
    // Column-major: columns are the images of the basis vectors.
    Mat3F64::from_cols(
        Vec3F64::new(c, s, 0.0),
        Vec3F64::new(-s, c, 0.0),
        Vec3F64::new(
            centre - c * centre + s * centre,
            centre - s * centre - c * centre,
            1.0,
        ),
    )
}

/// Scale the source so its diagonal fits the canvas, then paste it centrally.
///
/// Fitting the *diagonal* rather than the long side is what makes the sweep lossless: a
/// 45° rotation of a canvas-filling image would push its corners outside.
fn inscribe(src: &Image<u8, 3>) -> Result<Image<u8, 3>, vrt::BoxError> {
    let (w, h) = (src.cols() as f64, src.rows() as f64);
    let diag = (w * w + h * h).sqrt();
    let s = (CANVAS as f64 - 2.0) / diag;
    let (rw, rh) = (((w * s) as usize).max(1), ((h * s) as usize).max(1));

    let mut small = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: rw,
            height: rh,
        },
        0,
    )?;
    kornia_imgproc::resize::resize_fast_u8_aa::<3>(
        src,
        &mut small,
        kornia_imgproc::interpolation::InterpolationMode::Lanczos,
        true,
    )?;

    let mut canvas = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: CANVAS,
            height: CANVAS,
        },
        0,
    )?;
    let (ox, oy) = ((CANVAS - rw) / 2, (CANVAS - rh) / 2);
    let (srcbuf, dst) = (small.as_slice(), canvas.as_slice_mut());
    for y in 0..rh {
        let s0 = y * rw * 3;
        let d0 = ((oy + y) * CANVAS + ox) * 3;
        dst[d0..d0 + rw * 3].copy_from_slice(&srcbuf[s0..s0 + rw * 3]);
    }
    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sweep generates its query image with the same matrix it then scores against,
    /// so a convention error would partly cancel — the numbers would stay high while the
    /// *label* on each row was wrong. These check the matrix against hand-computed
    /// geometry instead, which the sweep itself cannot do.
    #[test]
    fn rotation_matrix_matches_hand_computed_geometry() {
        let c = 100.0;
        let apply = |deg: f64, x: f64, y: f64| {
            let p = rotation_about_centre(deg, c) * Vec3F64::new(x, y, 1.0);
            (p.x / p.z, p.y / p.z)
        };
        // The centre is the fixed point at every angle.
        for deg in [0.0, 37.0, 90.0, 180.0] {
            let (x, y) = apply(deg, c, c);
            assert!((x - c).abs() < 1e-9 && (y - c).abs() < 1e-9, "deg {deg}");
        }
        // 90°: the point right of centre goes below it (y grows downward in image space).
        let (x, y) = apply(90.0, c + 10.0, c);
        assert!((x - c).abs() < 1e-9 && (y - (c + 10.0)).abs() < 1e-9);
        // 180°: every point maps to its reflection through the centre.
        let (x, y) = apply(180.0, c + 10.0, c - 4.0);
        assert!((x - (c - 10.0)).abs() < 1e-9 && (y - (c + 4.0)).abs() < 1e-9);
        // 0° is exactly the identity, so the 0° row measures only resampling.
        let (x, y) = apply(0.0, 7.0, 11.0);
        assert!((x - 7.0).abs() < 1e-12 && (y - 11.0).abs() < 1e-12);
    }

    #[test]
    fn scale_matrix_matches_hand_computed_geometry() {
        let c = 100.0;
        let p = scale_about_centre(0.5, c) * Vec3F64::new(c + 40.0, c, 1.0);
        assert!(((p.x / p.z) - (c + 20.0)).abs() < 1e-9);
        assert!(((p.y / p.z) - c).abs() < 1e-9);
    }
}

/// (matches, inliers) under the exact ground-truth rotation.
fn score(
    pairs: &[(usize, usize)],
    lk: &[(f32, f32)],
    rk: &[(f32, f32)],
    h: &Mat3F64,
    thresh: f32,
) -> (usize, usize) {
    let inl = pairs
        .iter()
        .filter(|(i, j)| {
            let p = *h * Vec3F64::new(lk[*i].0 as f64, lk[*i].1 as f64, 1.0);
            let (dx, dy) = ((p.x / p.z) as f32 - rk[*j].0, (p.y / p.z) as f32 - rk[*j].1);
            (dx * dx + dy * dy).sqrt() <= thresh
        })
        .count();
    (pairs.len(), inl)
}

fn main() -> Result<(), vrt::BoxError> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!(
            "Usage: eval_rotation <image> <raco.engine> <lightglue.engine> \
             [xfeat.engine] [step_deg] [inlier_px] [nn_cossim] [xfeat_cossim]"
        );
        std::process::exit(1);
    }
    let step: f64 = arg_or(&a, 5, "step_deg", 15.0)?;
    if !(step.is_finite() && step > 0.0) {
        return Err(format!("step_deg must be positive and finite, got {step}").into());
    }
    let thresh: f32 = arg_or(&a, 6, "inlier_px", 3.0)?;
    // Exposed for the same reason the other two harnesses expose them: this table backs
    // "the invariance is the matcher's, not the descriptors'", and a claim resting on an
    // ungated baseline that cannot be re-run with a tuned one is not checkable.
    let nn_cossim: f32 = arg_or(&a, 7, "nn_cossim", -1.0)?;
    let xf_cossim: f32 = arg_or(&a, 8, "xfeat_cossim", -1.0)?;
    assert_eq!(CANVAS % DIM_DIVISOR, 0, "canvas must sit on the model grid");

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(&a[2], stream.clone())?;
    let mut glue = LightGlue::from_engine_file(&a[3], stream.clone())?;
    let mnn = Matcher::with_dim(stream.clone(), vrt_raco_aliked::DESC_DIM)?;
    let k = raco.num_keypoints();
    // No `.filter(exists)`: a mistyped path must fail, not silently print an XFeat
    // column of 0.0% at every angle that reads exactly like a measured total failure —
    // which is the column the rotation claim rests on.
    let mut xf = a
        .get(4)
        .map(|p| XFeatBaseline::load(p, &stream, k))
        .transpose()?;

    let reference = inscribe(read_image_any_rgb8(&a[1])?.as_ref())?;
    let dev_ref = reference.to_cuda(&stream)?;

    let (mut l, mut r) = (raco.alloc_result()?, raco.alloc_result()?);
    let mut lg_out = glue.alloc_result()?;
    let mut mnn_out = mnn.alloc_result(k)?;

    println!(
        "pure in-plane rotation, {CANVAS}x{CANVAS} canvas (source inscribed by its \
         diagonal, so no content is lost at any angle)"
    );
    println!(
        "RaCo k{k}, LightGlue k{}, inlier <= {thresh}px\n",
        glue.num_keypoints()
    );
    println!(
        // 23 = the `{:>8} {:>6} {:>6.1}%` group each column prints below.
        "{:>6} {:>23} {:>23} {:>23}",
        "deg", "LightGlue+ m/inl/%", "RaCo-ALIKED NN m/inl/%", "XFeat NN m/inl/%"
    );

    let mut deg = 0.0;
    while deg <= 180.0 + 1e-9 {
        let h = rotation_about_centre(deg, (CANVAS as f64 - 1.0) / 2.0);
        let m: [f32; 9] = {
            // warp_perspective_u8 wants row-major f32; Mat3F64 is column-major.
            let c = h.transpose().to_cols_array();
            std::array::from_fn(|i| c[i] as f32)
        };
        let mut rotated = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: CANVAS,
                height: CANVAS,
            },
            0,
        )?;
        warp_perspective_u8::<3>(&reference, &mut rotated, &m)?;
        let dev_rot = rotated.to_cuda(&stream)?;

        raco.submit(&dev_ref, &mut l)?;
        raco.submit(&dev_rot, &mut r)?;
        glue.submit(&l, &r, &mut lg_out)?;
        mnn.submit(
            Descriptors::new(l.descs_slice(), l.count(), l.desc_dim()),
            Descriptors::new(r.descs_slice(), r.count(), r.desc_dim()),
            nn_cossim,
            &mut mnn_out,
        )?;
        if let Some(x) = &mut xf {
            x.submit(&dev_ref, &dev_rot)?;
        }
        stream.synchronize()?;

        let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
        let lg = score(&lg_out.pairs(0.0)?, &lk, &rk, &h, thresh);
        let nn = score(&mnn_out.pairs(), &lk, &rk, &h, thresh);
        let xs = match &mut xf {
            Some(x) => {
                let (pairs, xlk, xrk) = x.finish(&stream, xf_cossim)?;
                score(&pairs, &xlk, &xrk, &h, thresh)
            }
            None => (0, 0),
        };

        println!(
            "{deg:>6.0} {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}%",
            lg.0,
            lg.1,
            pair_pct(lg),
            nn.0,
            nn.1,
            pair_pct(nn),
            xs.0,
            xs.1,
            pair_pct(xs)
        );
        deg += step;
    }

    // Scale, swept over the same range bark spans (up to 4.09x), so the two nuisances can
    // be compared on one image instead of inferred from pairs that mix them.
    println!(
        "\n{:>6} {:>23} {:>23} {:>23}",
        "zoom", "LightGlue+ m/inl/%", "RaCo-ALIKED NN m/inl/%", "XFeat NN m/inl/%"
    );
    for zoom in [1.0f64, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] {
        let h = scale_about_centre(1.0 / zoom, (CANVAS as f64 - 1.0) / 2.0);
        let m: [f32; 9] = {
            let c = h.transpose().to_cols_array();
            std::array::from_fn(|i| c[i] as f32)
        };
        let mut scaled = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: CANVAS,
                height: CANVAS,
            },
            0,
        )?;
        warp_perspective_u8::<3>(&reference, &mut scaled, &m)?;
        let dev = scaled.to_cuda(&stream)?;

        raco.submit(&dev_ref, &mut l)?;
        raco.submit(&dev, &mut r)?;
        glue.submit(&l, &r, &mut lg_out)?;
        mnn.submit(
            Descriptors::new(l.descs_slice(), l.count(), l.desc_dim()),
            Descriptors::new(r.descs_slice(), r.count(), r.desc_dim()),
            nn_cossim,
            &mut mnn_out,
        )?;
        if let Some(x) = &mut xf {
            x.submit(&dev_ref, &dev)?;
        }
        stream.synchronize()?;

        let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
        let lg = score(&lg_out.pairs(0.0)?, &lk, &rk, &h, thresh);
        let nn = score(&mnn_out.pairs(), &lk, &rk, &h, thresh);
        let xs = match &mut xf {
            Some(x) => {
                let (pairs, xlk, xrk) = x.finish(&stream, xf_cossim)?;
                score(&pairs, &xlk, &xrk, &h, thresh)
            }
            None => (0, 0),
        };
        println!(
            "{zoom:>5.1}x {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}%",
            lg.0,
            lg.1,
            pair_pct(lg),
            nn.0,
            nn.1,
            pair_pct(nn),
            xs.0,
            xs.1,
            pair_pct(xs)
        );
    }
    Ok(())
}
