//! Evaluate matchers on the **IMC 2021 phototourism validation** set — real 3D scenes
//! with ground-truth camera poses, stratified by co-visibility.
//!
//! The Oxford/VGG benchmark this crate also ships is planar: one homography maps every
//! pixel, so a match has a single correct destination. Phototourism is not planar. There
//! is no homography, only a camera pair, and the strongest statement ground truth can
//! make about a match is that it must lie on the **epipolar line**. So a match `(i, j)`
//! counts as an inlier iff its **Sampson error** against `F = K2⁻ᵀ [t]ₓ R K1⁻¹`, built
//! from the ground-truth poses, is under `inlier_px`.
//!
//! Sampson error comes from `kornia_3d::pose::sampson_distance`, the same error family the
//! IMC challenge's own DEGENSAC configuration uses. That makes these numbers *comparable
//! in kind* to the published leaderboard -- not identical to it, since the leaderboard's
//! matching score is computed under its own thresholding conventions. Treat the comparison
//! as indicative.
//!
//! Note `sampson_distance` returns the error **squared**, despite the name; the square root
//! is taken here so `inlier_px` is in pixels.
//!
//! **This is a weaker test than Oxford's** and the two sets of numbers are not
//! comparable. Epipolar agreement is necessary but not sufficient: a match sitting on
//! the correct line at the wrong depth passes. It still separates matchers well, because
//! a wrong match lands on the right line only by coincidence.
//!
//! Pairs are grouped by co-visibility band (0.1 = barely overlapping, 0.5 = strongly
//! overlapping). Reporting a single mean over an unstratified pool hides the hard regime,
//! which is the regime worth measuring.
//!
//! Prepare metadata once with `scripts/prep_imc.py`, then:
//!   cargo run --release -p vrt-lightglue --example eval_imc -- \
//!       <phototourism_dir> <raco.engine> <lightglue.engine> \
//!       [inlier_px] [min_cossim] [xfeat.engine] [nearest|bilinear|bicubic|lanczos]

use std::collections::BTreeMap;
use std::path::Path;

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize;
use kornia_io::functional::read_image_any_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::{RaCoAliked, DESC_DIM};
use vrt_xfeat::{Matcher, XFeat, XFeatParams};

/// Mutual-NN similarity gate. Tuned for XFeat's 64-D descriptors; ALIKED's 128-D live on
/// a different scale, so this is a CLI argument — a gate set for the wrong descriptor
/// family silently returns zero matches rather than erroring.
const DEFAULT_MIN_COSSIM: f32 = 0.0;

/// Long side the images are resized to. The published engines' shape profile tops out at
/// 640, and RaCo needs both sides to be multiples of 32.
const MAX_SIDE: usize = 640;

use kornia_3d::pose::{essential_from_fundamental, sampson_distance, Pose3d};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// `Mat3F64` is column-major (glam); the files are row-major, so transpose on the way in.
fn mat3_from_row_major(v: &[f64]) -> Mat3F64 {
    let mut a = [0.0; 9];
    a.copy_from_slice(v);
    Mat3F64::from_cols_array(&a).transpose()
}

/// Camera calibration as written by `scripts/prep_imc.py`: K (9), R (9), T (3).
///
/// The extrinsics are a `kornia_3d::pose::Pose3d`, whose convention (`p_cam = R p_world +
/// t`) is exactly the dataset's, so the relative pose is `Pose3d::between` rather than
/// hand-rolled matrix algebra.
struct Calib {
    k: Mat3F64,
    pose: Pose3d,
}

impl Calib {
    fn load(path: &Path) -> Result<Self, vrt::BoxError> {
        let v: Vec<f64> = std::fs::read_to_string(path)?
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if v.len() != 21 {
            return Err(format!("{}: expected 21 floats, got {}", path.display(), v.len()).into());
        }
        Ok(Self {
            k: mat3_from_row_major(&v[0..9]),
            pose: Pose3d::new(
                mat3_from_row_major(&v[9..18]),
                Vec3F64::new(v[18], v[19], v[20]),
            ),
        })
    }

    /// Resizing the image rescales the intrinsics with it. Skipping this is silent: the
    /// epipolar geometry stays self-consistent and simply describes the wrong camera.
    fn scaled_k(&self, sx: f64, sy: f64) -> Mat3F64 {
        Mat3F64::from_diagonal(Vec3F64::new(sx, sy, 1.0)) * self.k
    }
}

/// Ground-truth fundamental matrix mapping `a`'s pixels to epipolar lines in `b`.
///
/// Sampson error is invariant to F's scale (numerator and denominator both scale with it),
/// so no normalisation is needed here.
fn fundamental(a: &Calib, ka: &Mat3F64, b: &Calib, kb: &Mat3F64) -> Option<Mat3F64> {
    let (r, t) = Pose3d::between(&a.pose, &b.pose).to_rt();
    // A near-zero baseline is a pure rotation: F degenerates and every match would pass
    // the epipolar test vacuously, so report it rather than scoring the pair.
    if t.length() < 1e-9 {
        return None;
    }
    // E = [t]x R.
    let skew = Mat3F64::from_cols(
        Vec3F64::new(0.0, t.z, -t.y),
        Vec3F64::new(-t.z, 0.0, t.x),
        Vec3F64::new(t.y, -t.x, 0.0),
    );
    let e = skew * r;
    // Inverse of kornia's `essential_from_fundamental` (E = K2^T F K1).
    let f = kb.inverse().transpose() * e * ka.inverse();
    debug_assert!(
        essential_from_fundamental(&f, ka, kb)
            .to_cols_array()
            .iter()
            .zip(e.to_cols_array())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "F must round-trip through kornia's E<->F conversion"
    );
    Some(f)
}

fn score(
    pairs: &[(usize, usize)],
    lk: &[(f32, f32)],
    rk: &[(f32, f32)],
    f: &Mat3F64,
    thresh: f64,
) -> (usize, usize) {
    let inl = pairs
        .iter()
        .filter(|(i, j)| {
            let (p1, p2) = (lk[*i], rk[*j]);
            // sampson_distance returns the SQUARED error despite the name.
            let d2 = sampson_distance(
                f,
                &Vec2F64::new(p1.0 as f64, p1.1 as f64),
                &Vec2F64::new(p2.0 as f64, p2.1 as f64),
            );
            d2.sqrt() <= thresh
        })
        .count();
    (pairs.len(), inl)
}

/// Read through `kornia_io` and downscale through `kornia_imgproc`, returning the image
/// alongside the exact scale factors so the intrinsics can follow it.
///
/// Uses `resize` (the f32 path) with a caller-selectable kernel, matching
/// `examples/prep_oxford` so both benchmarks resample identically. It costs a
/// u8 -> f32 -> u8 round trip per image, which is irrelevant next to the inference it
/// feeds, and the filter no longer rounds to u8 internally.
fn load_scaled(
    path: &Path,
    interpolation: InterpolationMode,
) -> Result<(Image<u8, 3>, f64, f64), vrt::BoxError> {
    let src = read_image_any_rgb8(path)?;
    let (w, h) = (src.cols(), src.rows());
    let f = (MAX_SIDE as f64 / w.max(h) as f64).min(1.0);
    let nw = (((w as f64 * f) as usize) / 32 * 32).max(32);
    let nh = (((h as f64 * f) as usize) / 32 * 32).max(32);

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

#[derive(Default, Clone, Copy)]
struct Tally {
    pairs: usize,
    matches: usize,
    inliers: usize,
}

impl Tally {
    fn add(&mut self, (m, i): (usize, usize)) {
        self.pairs += 1;
        self.matches += m;
        self.inliers += i;
    }
    fn pct(&self) -> f32 {
        if self.matches == 0 {
            0.0
        } else {
            100.0 * self.inliers as f32 / self.matches as f32
        }
    }
}

fn main() -> Result<(), vrt::BoxError> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!(
            "Usage: eval_imc <phototourism_dir> <raco.engine> <lightglue.engine> \
             [inlier_px] [min_cossim] [xfeat.engine] [nearest|bilinear|bicubic|lanczos]"
        );
        std::process::exit(1);
    }
    let root = Path::new(&a[1]);
    let thresh: f64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let min_cossim: f32 = a
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MIN_COSSIM);
    let interpolation = match a.get(7).map(String::as_str).unwrap_or("bilinear") {
        "nearest" => InterpolationMode::Nearest,
        "bilinear" => InterpolationMode::Bilinear,
        "bicubic" => InterpolationMode::Bicubic,
        "lanczos" => InterpolationMode::Lanczos,
        other => return Err(format!("unknown interpolation {other}").into()),
    };

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(&a[2], stream.clone())?;
    let mut glue = LightGlue::from_engine_file(&a[3], stream.clone())?;
    let mnn = Matcher::with_dim(stream.clone(), DESC_DIM)?;
    let mut xf = match a.get(6) {
        Some(p) => Some((
            XFeat::from_engine_file(p, stream.clone(), XFeatParams::new(4096, 0.05))?,
            Matcher::new(stream.clone())?,
        )),
        None => None,
    };

    let k = raco.num_keypoints();
    let (mut l, mut r) = (raco.alloc_result()?, raco.alloc_result()?);
    let mut lg_out = glue.alloc_result()?;
    let mut mnn_out = mnn.alloc_result(k)?;
    let mut xf_res = match &xf {
        Some((x, m)) => Some((x.alloc_result()?, x.alloc_result()?, m.alloc_result(4096)?)),
        None => None,
    };

    println!(
        "IMC2021 phototourism val — RaCo k{k}, LightGlue k{}, mutual-NN {DESC_DIM}-D \
         (cossim>={min_cossim}), Sampson <= {thresh}px, long side {MAX_SIDE}",
        glue.num_keypoints()
    );

    let manifest = std::fs::read_to_string(root.join("imc_manifest.txt"))?;
    let mut bands: BTreeMap<String, (Tally, Tally, Tally)> = BTreeMap::new();
    let mut skipped = 0usize;

    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (scene, band, i1, i2) = (f[0], f[1].to_string(), f[2], f[3]);
        let base = root.join(scene).join("set_100");

        let img_dir = base.join("images");
        let (im1, sx1, sy1) = load_scaled(&img_dir.join(format!("{i1}.jpg")), interpolation)?;
        let (im2, sx2, sy2) = load_scaled(&img_dir.join(format!("{i2}.jpg")), interpolation)?;
        let c1 = Calib::load(&base.join("calib_txt").join(format!("{i1}.txt")))?;
        let c2 = Calib::load(&base.join("calib_txt").join(format!("{i2}.txt")))?;
        let (k1, k2) = (c1.scaled_k(sx1, sy1), c2.scaled_k(sx2, sy2));
        let Some(fmat) = fundamental(&c1, &k1, &c2, &k2) else {
            // Coincident centres give a degenerate F; every match would "pass".
            skipped += 1;
            continue;
        };

        let (dl, dr) = (im1.to_cuda(&stream)?, im2.to_cuda(&stream)?);
        raco.submit(&dl, &mut l)?;
        raco.submit(&dr, &mut r)?;
        glue.submit(&l, &r, &mut lg_out)?;
        mnn.submit(
            l.descs_slice(),
            k,
            r.descs_slice(),
            k,
            min_cossim,
            &mut mnn_out,
        )?;
        stream.synchronize()?;

        let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
        let e = bands.entry(band).or_default();
        e.0.add(score(&lg_out.pairs(0.0)?, &lk, &rk, &fmat, thresh));
        e.1.add(score(&mnn_out.pairs(), &lk, &rk, &fmat, thresh));

        if let (Some((x, m)), Some((xl, xr, xo))) = (&mut xf, &mut xf_res) {
            x.submit(&dl, xl)?;
            x.submit(&dr, xr)?;
            stream.synchronize()?;
            m.submit(&xl.descs, xl.count(), &xr.descs, xr.count(), min_cossim, xo)?;
            stream.synchronize()?;
            let (fl, fr) = (xl.kpts_to_host()?, xr.kpts_to_host()?);
            let xlk: Vec<(f32, f32)> = fl.chunks_exact(2).map(|p| (p[0], p[1])).collect();
            let xrk: Vec<(f32, f32)> = fr.chunks_exact(2).map(|p| (p[0], p[1])).collect();
            e.2.add(score(&xo.pairs(), &xlk, &xrk, &fmat, thresh));
        }
    }

    println!(
        "\n{:<8} {:>5} {:>22} {:>22} {:>22}",
        "covis", "pairs", "LightGlue+ m/inl/%", "RaCo-ALIKED NN m/inl/%", "XFeat NN m/inl/%"
    );
    let mut tot = (Tally::default(), Tally::default(), Tally::default());
    for (band, (lg, nn, x)) in &bands {
        println!(
            "{:<8} {:>5} {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}%",
            band,
            lg.pairs,
            lg.matches,
            lg.inliers,
            lg.pct(),
            nn.matches,
            nn.inliers,
            nn.pct(),
            x.matches,
            x.inliers,
            x.pct()
        );
        for (dst, src) in [(&mut tot.0, lg), (&mut tot.1, nn), (&mut tot.2, x)] {
            dst.pairs += src.pairs;
            dst.matches += src.matches;
            dst.inliers += src.inliers;
        }
    }
    println!(
        "{:<8} {:>5} {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}% {:>8} {:>6} {:>6.1}%",
        "all",
        tot.0.pairs,
        tot.0.matches,
        tot.0.inliers,
        tot.0.pct(),
        tot.1.matches,
        tot.1.inliers,
        tot.1.pct(),
        tot.2.matches,
        tot.2.inliers,
        tot.2.pct()
    );
    if skipped > 0 {
        println!("\n{skipped} pairs skipped (near-zero baseline; F is degenerate)");
    }
    Ok(())
}
