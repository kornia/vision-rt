//! Evaluate matchers on the **Oxford/VGG affine** benchmark — real images, real
//! ground-truth homographies.
//!
//! Every accuracy number this crate has published so far came from one synthetic pair
//! with a synthetic transform, which cannot show how a matcher behaves on real texture,
//! real blur, or real scale change. This runs the same comparison on `bark` and `boat`
//! (rotation + zoom) and `graf` (viewpoint), scoring each match against the sequence's
//! ground-truth homography.
//!
//! A match `(i, j)` is an **inlier** iff `H · left[i]` lands within `inlier_px` of
//! `right[j]`, the standard protocol for this benchmark. The threshold is expressed at
//! **original image resolution** and converted per pair using the scale `prep_oxford`
//! recorded, so every sequence is scored under one common criterion. Reported per pair:
//!
//! * **matches** — how many correspondences survived the matcher's own filtering
//! * **inlier %** — of those, how many are geometrically correct
//! * **inliers** — the product, which is what a pose solver actually consumes
//!
//! Usage:
//!   cargo run --release -p vrt-lightglue --example eval_oxford -- \
//!       <dataset_dir> <raco.engine> <lightglue.engine> \
//!       [inlier_px] [nn_cossim] [xfeat.engine] [xfeat_cossim]
//!
//! `dataset_dir` holds `manifest.txt` with `seq left right homography sx sy` per line; produce
//! it with `scripts/get_oxford.sh` followed by `examples/prep_oxford`, which also
//! photometrically verifies the rescaled ground truth before you trust any number here.

use std::path::Path;

use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_io::functional::read_image_any_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::{RaCoAliked, DESC_DIM};
use vrt_xfeat::{Descriptors, Matcher};

#[path = "common/mod.rs"]
mod common;
use common::{arg_or, mat3_from_row_major, read_floats, XFeatBaseline};

/// Mutual-NN similarity gates, one per descriptor family.
///
/// The gate is descriptor-specific and unforgiving: XFeat's tuned 0.82 applied to ALIKED's
/// 128-D descriptors returns **zero** matches on 10 of these 15 pairs. Both default to
/// ungated so the two mutual-NN columns measure the descriptors rather than a threshold
/// picked for one of them, and both are separately overridable.
const DEFAULT_NN_COSSIM: f32 = 0.0;
const DEFAULT_XF_COSSIM: f32 = 0.0;

fn warp(h: &Mat3F64, x: f32, y: f32) -> (f32, f32) {
    let p = *h * Vec3F64::new(x as f64, y as f64, 1.0);
    // A degenerate w means the point maps to infinity; push it far away so it can never
    // be counted as an inlier rather than producing a NaN that silently compares false.
    if p.z.abs() < 1e-12 {
        return (f32::MAX, f32::MAX);
    }
    ((p.x / p.z) as f32, (p.y / p.z) as f32)
}

/// (matches, inliers, inlier %) for a match set scored against the ground truth.
fn score(
    pairs: &[(usize, usize)],
    lk: &[(f32, f32)],
    rk: &[(f32, f32)],
    h: &Mat3F64,
    thresh: f32,
    scale: (f32, f32),
) -> (usize, usize, f32) {
    let inl = pairs
        .iter()
        .filter(|(i, j)| {
            let (ex, ey) = warp(h, lk[*i].0, lk[*i].1);
            // Undo the resize so the threshold means the same thing on every sequence.
            let dx = (ex - rk[*j].0) / scale.0;
            let dy = (ey - rk[*j].1) / scale.1;
            (dx * dx + dy * dy).sqrt() <= thresh
        })
        .count();
    let pct = if pairs.is_empty() {
        0.0
    } else {
        100.0 * inl as f32 / pairs.len() as f32
    };
    (pairs.len(), inl, pct)
}

fn main() -> Result<(), vrt::BoxError> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!(
            "Usage: eval_oxford <dataset_dir> <raco.engine> <lightglue.engine> \
             [inlier_px] [nn_cossim] [xfeat.engine] [xfeat_cossim]"
        );
        std::process::exit(1);
    }
    let root = Path::new(&a[1]);
    let thresh: f32 = arg_or(&a, 4, "inlier_px", 3.0)?;
    let nn_cossim: f32 = arg_or(&a, 5, "nn_cossim", DEFAULT_NN_COSSIM)?;
    let xf_cossim: f32 = arg_or(&a, 7, "xfeat_cossim", DEFAULT_XF_COSSIM)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(&a[2], stream.clone())?;
    let mut glue = LightGlue::from_engine_file(&a[3], stream.clone())?;
    let mnn = Matcher::with_dim(stream.clone(), DESC_DIM)?;
    // XFeat + its native 64-D mutual-NN, as the baseline the crate READMEs compare to.
    // Optional so the eval still runs without an XFeat engine on hand.
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
        "Oxford/VGG affine — RaCo k{k}, LightGlue k{}, {DESC_DIM}-D mutual-NN \
         (cossim>={nn_cossim}), XFeat {}, 64-D mutual-NN (cossim>={xf_cossim})",
        glue.num_keypoints(),
        match &xf {
            Some(x) => format!("k{}", x.capacity()),
            None => "absent — column reports 0".to_string(),
        }
    );
    println!("inlier <= {thresh}px measured at ORIGINAL image resolution");
    println!(
        "{:<14} {:>24} {:>24} {:>24}",
        "", "LightGlue+ m/inl/%", "RaCo mutual-NN m/inl/%", "XFeat mutual-NN m/inl/%"
    );

    let manifest = std::fs::read_to_string(root.join("manifest.txt"))?;
    let (mut lg_tot, mut mnn_tot, mut xf_tot) = (0usize, 0usize, 0usize);
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (seq, left_n, right_n, hf) = (f[0], f[1], f[2], f[3]);
        // prep_oxford records the per-axis scale it applied to the right image. Scoring in
        // the resized frame with a fixed pixel threshold would make the criterion 11%
        // looser on boat than on bark, so the error is converted back to original pixels
        // instead — per axis, because the two scales differ by a rounding step.
        let missing = "manifest is missing its two scale columns — regenerate with prep_oxford";
        let sx: f32 = f.get(4).ok_or(missing)?.parse()?;
        let sy: f32 = f.get(5).ok_or(missing)?.parse()?;
        let dir = root.join(seq);

        let hv = read_floats(&dir.join(hf))?;
        if hv.len() < 9 {
            return Err(format!("{seq}/{hf}: expected 9 floats, got {}", hv.len()).into());
        }
        let h = mat3_from_row_major(&hv[..9]);

        let left = read_image_any_rgb8(dir.join(left_n))?.to_cuda(&stream)?;
        let right = read_image_any_rgb8(dir.join(right_n))?.to_cuda(&stream)?;

        raco.submit(&left, &mut l)?;
        raco.submit(&right, &mut r)?;
        glue.submit(&l, &r, &mut lg_out)?;
        // XFeat depends only on the uploaded images, so enqueue it before the readback:
        // one synchronise for every model instead of two.
        if let Some(x) = &mut xf {
            x.submit(&left, &right)?;
        }
        mnn.submit(
            Descriptors::new(l.descs_slice(), k, l.desc_dim()),
            Descriptors::new(r.descs_slice(), k, r.desc_dim()),
            nn_cossim,
            &mut mnn_out,
        )?;
        stream.synchronize()?;

        let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
        let (lm, li, lp) = score(&lg_out.pairs(0.0)?, &lk, &rk, &h, thresh, (sx, sy));
        let (mm, mi, mp) = score(&mnn_out.pairs(), &lk, &rk, &h, thresh, (sx, sy));
        lg_tot += li;
        mnn_tot += mi;

        // XFeat on the same pair, its own keypoints and its own 64-D matcher.
        let (xm, xi, xp) = match &mut xf {
            Some(x) => {
                let (pairs, xlk, xrk) = x.finish(&stream, xf_cossim)?;
                score(&pairs, &xlk, &xrk, &h, thresh, (sx, sy))
            }
            None => (0, 0, 0.0),
        };
        xf_tot += xi;

        println!(
            "{:<14} {:>7} {:>6} {:>6.1}% {:>7} {:>6} {:>6.1}% {:>7} {:>6} {:>6.1}%",
            format!("{seq}/{}", right_n.trim_end_matches(".png")),
            lm,
            li,
            lp,
            mm,
            mi,
            mp,
            xm,
            xi,
            xp
        );
    }
    println!(
        "\ntotal correct correspondences: LightGlue+ {lg_tot}, RaCo mutual-NN {mnn_tot}, XFeat {xf_tot}"
    );
    Ok(())
}
