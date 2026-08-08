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
//!       [inlier_px] [nn_cossim] [xfeat.engine] [xfeat_cossim] [nearest|bilinear|bicubic|lanczos]
//!
//! Arguments 1-7 mean the same thing here as in `eval_oxford`, so a command line
//! transfers between the two harnesses; the interpolation kernel is the one extra,
//! because only this one resizes at evaluation time.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use kornia_image::Image;
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_io::functional::read_image_any_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::{RaCoAliked, DESC_DIM};
use vrt_xfeat::{Descriptors, Matcher};

/// Mutual-NN similarity gates, one per descriptor family. The gate is descriptor-specific
/// and unforgiving, so the two families get separate knobs rather than sharing one that is
/// necessarily wrong for at least one of them.
///
/// 0.0 is the *loosest gate reported*, not literally ungated: it still drops a mutual-NN
/// pair whose best cosine is negative. Truly ungated is -1.0 (what the GPU/CPU agreement
/// tests use). The distinction is immaterial for L2-normalised descriptors — an
/// anti-correlated best match over thousands of candidates is noise — but the tables call
/// this row "ungated", so what it actually admits is recorded here.
const DEFAULT_NN_COSSIM: f32 = -1.0;
const DEFAULT_XF_COSSIM: f32 = -1.0;

/// Long side the images are resized to. The published engines' shape profile tops out at
/// 640, and RaCo needs both sides to be multiples of 32.
const MAX_SIDE: usize = 640;

use kornia_3d::pose::{sampson_distance, Pose3d};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

#[path = "common/mod.rs"]
mod common;
use common::{
    arg_or, mat3_from_row_major, parse_interpolation, read_floats, resize_matrix, resize_to_fit,
    Tally, XFeatBaseline,
};

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
        let v = read_floats(path)?;
        if v.len() != 21 {
            return Err(format!("{}: expected 21 floats, got {}", path.display(), v.len()).into());
        }
        Ok(Self {
            k: mat3_from_row_major(&v[0..9])?,
            pose: Pose3d::new(
                mat3_from_row_major(&v[9..18])?,
                Vec3F64::new(v[18], v[19], v[20]),
            ),
        })
    }

    /// Resizing the image rescales the intrinsics with it. Skipping this is silent: the
    /// epipolar geometry stays self-consistent and simply describes the wrong camera.
    fn scaled_k(&self, scale: (f64, f64)) -> Mat3F64 {
        resize_matrix(scale.0, scale.1) * self.k
    }
}

/// Relative pose `a`-camera → `b`-camera, or `None` when the baseline is degenerate.
///
/// Split from [`fundamental`] so the caller can reject a pair *before* decoding and
/// resizing its two JPEGs — and, more importantly, so `fundamental` takes one image's
/// pose and its own K together rather than four loose arguments two of which could be
/// swapped without the types noticing.
fn relative_pose(a: &Calib, b: &Calib) -> Option<(Mat3F64, Vec3F64)> {
    let (r, t) = Pose3d::between(&a.pose, &b.pose).to_rt();
    // A near-zero baseline is a pure rotation: F degenerates and every match would pass
    // the epipolar test vacuously, so report it rather than scoring the pair.
    //
    // The threshold is relative to the cameras' own translation magnitudes, because
    // phototourism poses are in arbitrary SfM units: a fixed 1e-9 is meaningless against a
    // scene whose coordinates happen to be large, and vacuous against one where they are
    // small. This scales with the reconstruction instead.
    //
    // MEASURED on the current 90-pair set: the smallest baseline is 1.2% of its scene's
    // mean camera-centre spread (sacre_coeur, 0.047 against 4.06), so nothing here is
    // near-degenerate and this guard does not fire. It exists for the pair that would be.
    // A tighter gate — rejecting weak-but-usable baselines — would need the scene's depth
    // range, which this metadata does not carry.
    let scale = a.pose.translation.length() + b.pose.translation.length();
    if t.length() <= 1e-6 * scale.max(f64::MIN_POSITIVE) {
        return None;
    }
    Some((r, t))
}

/// Ground-truth fundamental matrix mapping image-1 pixels to epipolar lines in image 2,
/// from the relative pose and the two *scaled* intrinsics.
///
/// Sampson error is invariant to F's scale (numerator and denominator both scale with it),
/// so no normalisation is needed here.
fn fundamental(r: &Mat3F64, t: &Vec3F64, k1: &Mat3F64, k2: &Mat3F64) -> Mat3F64 {
    // E = [t]x R.
    let skew = Mat3F64::from_cols(
        Vec3F64::new(0.0, t.z, -t.y),
        Vec3F64::new(-t.z, 0.0, t.x),
        Vec3F64::new(t.y, -t.x, 0.0),
    );
    // Inverse of kornia's `essential_from_fundamental` (E = K2^T F K1).
    //
    // There was a `debug_assert` here round-tripping F back through that function. It was
    // an algebraic tautology — it reduces to K^T K^-T = I — so it passed for three
    // deliberately wrong conventions (factors swapped, images flipped, skew transposed)
    // while reading like a verified invariant, and `--release` compiled it out anyway.
    // The convention is pinned by the benchmark numbers, not by an assert.
    k2.inverse().transpose() * (skew * *r) * k1.inverse()
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

/// Read through `kornia_io`, then downscale so the engine's shape profile is satisfied.
fn load_scaled(
    path: &Path,
    interpolation: InterpolationMode,
) -> Result<(Image<u8, 3>, (f64, f64)), vrt::BoxError> {
    let src = read_image_any_rgb8(path)?;
    let scaled = resize_to_fit(src.as_ref(), MAX_SIDE, interpolation)?;
    Ok((scaled.image, (scaled.scale_x, scaled.scale_y)))
}

fn main() -> Result<(), vrt::BoxError> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!(
            "Usage: eval_imc <phototourism_dir> <raco.engine> <lightglue.engine> \
             [inlier_px] [nn_cossim] [xfeat.engine] [xfeat_cossim] \
             [nearest|bilinear|bicubic|lanczos]"
        );
        std::process::exit(1);
    }
    let root = Path::new(&a[1]);
    // Default matches the headline table in the READMEs; 3 px is the sensitivity check.
    let thresh: f64 = arg_or(&a, 4, "inlier_px", 1.0)?;
    let nn_cossim: f32 = arg_or(&a, 5, "nn_cossim", DEFAULT_NN_COSSIM)?;
    // Indices 1-7 match eval_oxford exactly, so one command line works against both.
    let xf_cossim: f32 = arg_or(&a, 7, "xfeat_cossim", DEFAULT_XF_COSSIM)?;
    let interpolation = parse_interpolation(a.get(8).map(String::as_str).unwrap_or("lanczos"))?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(&a[2], stream.clone())?;
    let mut glue = LightGlue::from_engine_file(&a[3], stream.clone())?;
    let mnn = Matcher::with_dim(stream.clone(), DESC_DIM)?;
    let k = raco.num_keypoints();
    // Same keypoint budget as RaCo, so the columns are comparable.
    let mut xf = a
        .get(6)
        .map(|p| XFeatBaseline::load(p, &stream, k))
        .transpose()?;

    let (mut l, mut r) = (raco.alloc_result()?, raco.alloc_result()?);
    let mut lg_out = glue.alloc_result()?;
    let mut mnn_out = mnn.alloc_result(k)?;

    println!(
        "IMC2021 phototourism val — RaCo k{k}, LightGlue k{}, {DESC_DIM}-D mutual-NN \
         (cossim>={nn_cossim}), XFeat {}, 64-D mutual-NN (cossim>={xf_cossim})",
        glue.num_keypoints(),
        match &xf {
            Some(x) => format!("k{}", x.capacity()),
            None => "absent — column reports 0".to_string(),
        }
    );
    println!(
        "Sampson <= {thresh}px, measured in the resized frame (long side {MAX_SIDE}). \
         Unlike Oxford this is not converted to original resolution: Sampson mixes both \
         images' frames, so no single scale converts it."
    );

    let mut cache: HashMap<std::path::PathBuf, (Image<u8, 3>, (f64, f64))> = HashMap::new();
    let manifest = std::fs::read_to_string(root.join("imc_manifest.txt"))?;
    let mut bands: BTreeMap<String, (Tally, Tally, Tally)> = BTreeMap::new();
    let mut skipped = 0usize;

    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let [scene, band, i1, i2] = f[..] else {
            return Err(format!(
                "imc_manifest.txt: expected 4 fields, got {}: {line:?}",
                f.len()
            )
            .into());
        };
        let band = band.to_string();
        let base = root.join(scene).join("set_100");

        let c1 = Calib::load(&base.join("calib_txt").join(format!("{i1}.txt")))?;
        let c2 = Calib::load(&base.join("calib_txt").join(format!("{i2}.txt")))?;
        // Poses first: a degenerate pair is rejected before its two JPEGs are decoded and
        // resized, which is the expensive half of the loop.
        let Some((rmat, tvec)) = relative_pose(&c1, &c2) else {
            // Coincident centres give a degenerate F; every match would "pass".
            skipped += 1;
            continue;
        };

        let img_dir = base.join("images");
        // The 90 pairs are drawn from ~135 distinct images, so decoding and resampling per
        // pair repeats roughly a quarter of the work. Cache by path.
        for id in [i1, i2] {
            let path = img_dir.join(format!("{id}.jpg"));
            if !cache.contains_key(&path) {
                cache.insert(path.clone(), load_scaled(&path, interpolation)?);
            }
        }
        let (im1, s1) = cache[&img_dir.join(format!("{i1}.jpg"))].clone();
        let (im2, s2) = cache[&img_dir.join(format!("{i2}.jpg"))].clone();
        let fmat = fundamental(&rmat, &tvec, &c1.scaled_k(s1), &c2.scaled_k(s2));

        let (dl, dr) = (im1.to_cuda(&stream)?, im2.to_cuda(&stream)?);
        // Phototourism aspect ratios vary, so the two frames usually differ in size. Each
        // `submit` sets the context's input shape and may reallocate the session's output
        // buffers — host-side calls that are NOT stream-ordered, so issuing the second
        // while the first enqueue is still in flight mutates a live context. Every other
        // caller in the repo feeds same-sized frames; this is the one that does not.
        let differing = im1.size() != im2.size();
        raco.submit(&dl, &mut l)?;
        if differing {
            stream.synchronize()?;
        }
        raco.submit(&dr, &mut r)?;
        glue.submit(&l, &r, &mut lg_out)?;
        mnn.submit(
            Descriptors::new(l.descs_slice(), l.count(), l.desc_dim()),
            Descriptors::new(r.descs_slice(), r.count(), r.desc_dim()),
            nn_cossim,
            &mut mnn_out,
        )?;
        // XFeat depends only on the uploaded images, so enqueue it here rather than after
        // the readback: one synchronise for every model instead of two.
        if let Some(x) = &mut xf {
            x.submit(&dl, &dr, differing.then_some(&stream))?;
        }
        stream.synchronize()?;

        let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
        let e = bands.entry(band).or_default();
        e.0.add(score(&lg_out.pairs(0.0)?, &lk, &rk, &fmat, thresh));
        e.1.add(score(&mnn_out.pairs(), &lk, &rk, &fmat, thresh));

        // Always tally, even with no XFeat engine, so `Tally::pairs` means the same
        // thing in every column and in both harnesses.
        let xf_score = match &mut xf {
            Some(x) => {
                let (pairs, xlk, xrk) = x.finish(&stream, xf_cossim)?;
                score(&pairs, &xlk, &xrk, &fmat, thresh)
            }
            None => (0, 0),
        };
        e.2.add(xf_score);
    }

    // 23 = the `{:>8} {:>6} {:>6.1}%` group each column prints below; the header must be
    // measured against that, not eyeballed, or every column heading sits one off.
    println!(
        "\n{:<8} {:>5} {:>23} {:>23} {:>23}",
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
        tot.0.add_tally(lg);
        tot.1.add_tally(nn);
        tot.2.add_tally(x);
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
    // Pooled precision is match-volume weighted; the macro average gives every pair one
    // vote. Reporting only the first lets one dense pair speak for a whole band.
    println!(
        "\nmacro-average precision (mean over pairs, each weighted equally):\n\
         {:<8} {:>5} {:>23.1} {:>23.1} {:>23.1}",
        "all",
        tot.0.pairs,
        tot.0.macro_pct(),
        tot.1.macro_pct(),
        tot.2.macro_pct()
    );
    for (band, (lg, nn, x)) in &bands {
        println!(
            "{:<8} {:>5} {:>23.1} {:>23.1} {:>23.1}",
            band,
            lg.pairs,
            lg.macro_pct(),
            nn.macro_pct(),
            x.macro_pct()
        );
    }
    if skipped > 0 {
        println!("\n{skipped} pairs skipped (near-zero baseline; F is degenerate)");
    }
    Ok(())
}
