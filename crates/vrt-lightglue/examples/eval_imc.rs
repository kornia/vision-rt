//! Evaluate matchers on the **IMC 2021 phototourism validation** set — real 3D scenes
//! with ground-truth camera poses, stratified by co-visibility.
//!
//! The Oxford/VGG benchmark this crate also ships is planar: one homography maps every
//! pixel, so a match has a single correct destination. Phototourism is not planar. There
//! is no homography, only a camera pair, and the strongest statement ground truth can
//! make about a match is that it must lie on the **epipolar line**. So a match `(i, j)`
//! counts as an inlier iff the symmetric epipolar distance
//!
//! ```text
//!   |x2ᵀ F x1| · ( 1/‖(F x1)_xy‖ + 1/‖(Fᵀ x2)_xy‖ )
//! ```
//!
//! is under `inlier_px`, with `F = K2⁻ᵀ [t]ₓ R K1⁻¹` built from the ground-truth poses.
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
//!       <phototourism_dir> <raco.engine> <lightglue.engine> [inlier_px] [min_cossim] [xfeat.engine]

use std::collections::BTreeMap;
use std::path::Path;

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_fast_rgb;
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

#[path = "common/geometry.rs"]
mod geometry;
use geometry::{inv3, matvec3, mul3, scale3, skew3, sym_epipolar, transpose3, M3};

/// Camera calibration as written by `scripts/prep_imc.py`: K (9), R (9), T (3).
struct Calib {
    k: M3,
    r: M3,
    t: [f64; 3],
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
        let (mut k, mut r) = ([0.0; 9], [0.0; 9]);
        k.copy_from_slice(&v[0..9]);
        r.copy_from_slice(&v[9..18]);
        Ok(Self {
            k,
            r,
            t: [v[18], v[19], v[20]],
        })
    }

    /// Resizing the image rescales the intrinsics with it. Skipping this is silent: the
    /// epipolar geometry stays self-consistent and simply describes the wrong camera.
    fn scaled_k(&self, sx: f64, sy: f64) -> M3 {
        mul3(&scale3(sx, sy), &self.k)
    }
}

/// Ground-truth fundamental matrix mapping `a`'s pixels to epipolar lines in `b`.
fn fundamental(a: &Calib, ka: &M3, b: &Calib, kb: &M3) -> Option<M3> {
    // world -> cam is x_c = R x_w + T, so the relative pose from a to b is
    let r_ab = mul3(&b.r, &transpose3(&a.r));
    let ra_ta = matvec3(&r_ab, &a.t);
    let t = [b.t[0] - ra_ta[0], b.t[1] - ra_ta[1], b.t[2] - ra_ta[2]];
    let e = mul3(&skew3(&t), &r_ab);
    let f = mul3(&mul3(&transpose3(&inv3(kb)?), &e), &inv3(ka)?);
    // Normalise so the threshold comparison is not at the mercy of E's arbitrary scale.
    let n = f.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if n < 1e-12 {
        return None;
    }
    Some(f.map(|v| v / n))
}

fn score(
    pairs: &[(usize, usize)],
    lk: &[(f32, f32)],
    rk: &[(f32, f32)],
    f: &M3,
    thresh: f64,
) -> (usize, usize) {
    let inl = pairs
        .iter()
        .filter(|(i, j)| sym_epipolar(f, lk[*i], rk[*j]) <= thresh)
        .count();
    (pairs.len(), inl)
}

/// Read through `kornia_io` and downscale through `kornia_imgproc`, returning the image
/// alongside the exact scale factors so the intrinsics can follow it.
fn load_scaled(path: &Path) -> Result<(Image<u8, 3>, f64, f64), vrt::BoxError> {
    let src = read_image_any_rgb8(path)?;
    let (w, h) = (src.cols(), src.rows());
    let f = (MAX_SIDE as f64 / w.max(h) as f64).min(1.0);
    let nw = (((w as f64 * f) as usize) / 32 * 32).max(32);
    let nh = (((h as f64 * f) as usize) / 32 * 32).max(32);
    let mut dst = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: nw,
            height: nh,
        },
        0,
    )?;
    resize_fast_rgb(src.as_ref(), &mut dst, InterpolationMode::Bilinear)?;
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
             [inlier_px] [min_cossim] [xfeat.engine]"
        );
        std::process::exit(1);
    }
    let root = Path::new(&a[1]);
    let thresh: f64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let min_cossim: f32 = a
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MIN_COSSIM);

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
         (cossim>={min_cossim}), symmetric epipolar <= {thresh}px, long side {MAX_SIDE}",
        glue.num_keypoints()
    );

    let manifest = std::fs::read_to_string(root.join("imc_manifest.txt"))?;
    let mut bands: BTreeMap<String, (Tally, Tally, Tally)> = BTreeMap::new();
    let mut skipped = 0usize;

    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (scene, band, i1, i2) = (f[0], f[1].to_string(), f[2], f[3]);
        let base = root.join(scene).join("set_100");

        let (im1, sx1, sy1) = load_scaled(&base.join("images").join(format!("{i1}.jpg")))?;
        let (im2, sx2, sy2) = load_scaled(&base.join("images").join(format!("{i2}.jpg")))?;
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
        glue.submit_match(&l, &r, &mut lg_out)?;
        mnn.submit_match(
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
            m.submit_match(&xl.descs, xl.count(), &xr.descs, xr.count(), min_cossim, xo)?;
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
        println!("\n{skipped} pairs skipped (degenerate ground-truth geometry)");
    }
    Ok(())
}
