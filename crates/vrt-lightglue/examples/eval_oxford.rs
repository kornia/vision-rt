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
use common::{arg_or, mat3_from_row_major, pair_pct, read_floats, Tally, XFeatBaseline};

/// Mutual-NN similarity gates, one per descriptor family.
///
/// The gate is descriptor-specific and unforgiving: XFeat's tuned 0.82 applied to ALIKED's
/// 128-D descriptors returns **zero** matches on 10 of these 15 pairs. Both default to the
/// loosest gate so the two mutual-NN columns measure the descriptors rather than a
/// threshold picked for one of them, and both are separately overridable.
///
/// 0.0 is the loosest gate *reported*, not literally ungated: it still drops a pair whose
/// best cosine is negative (truly ungated is -1.0). Immaterial for L2-normalised
/// descriptors, but the tables call this row "ungated", so the difference is recorded.
const DEFAULT_NN_COSSIM: f32 = -1.0;
const DEFAULT_XF_COSSIM: f32 = -1.0;

fn warp(h: &Mat3F64, x: f32, y: f32) -> (f32, f32) {
    let p = *h * Vec3F64::new(x as f64, y as f64, 1.0);
    // A degenerate w means the point maps to infinity; push it far away so it can never
    // be counted as an inlier rather than producing a NaN that silently compares false.
    if p.z.abs() < 1e-12 {
        return (f32::MAX, f32::MAX);
    }
    ((p.x / p.z) as f32, (p.y / p.z) as f32)
}

/// (matches, inliers) for a match set scored against the ground truth.
///
/// Same shape as `eval_imc`'s `score`, so both harnesses feed one shared [`Tally`] and the
/// precision is derived in exactly one place ([`common::pair_pct`]).
fn score(
    pairs: &[(usize, usize)],
    lk: &[(f32, f32)],
    rk: &[(f32, f32)],
    h: &Mat3F64,
    thresh: f32,
    scale: (f32, f32),
) -> (usize, usize) {
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
    (pairs.len(), inl)
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

    let manifest_path = root.join("manifest.txt");
    let manifest = std::fs::read_to_string(&manifest_path).map_err(|e| {
        format!(
            "{}: {e} — produce it with prep_oxford",
            manifest_path.display()
        )
    })?;
    let (mut lg, mut mnn_t, mut xf_t) = (Tally::default(), Tally::default(), Tally::default());
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let [seq, left_n, right_n, hf, sx_s, sy_s] = f[..] else {
            return Err(format!(
                "manifest.txt: expected 6 fields (seq left right H sx sy), got {}: {line:?} \
                 — regenerate with prep_oxford",
                f.len()
            )
            .into());
        };
        // prep_oxford records the per-axis scale it applied to the right image. Scoring in
        // the resized frame with a fixed pixel threshold would make the criterion 11%
        // looser on boat than on bark, so the error is converted back to original pixels
        // instead — per axis, because the two scales differ by a rounding step.
        let (sx, sy): (f32, f32) = (sx_s.parse()?, sy_s.parse()?);
        let dir = root.join(seq);

        let hv = read_floats(&dir.join(hf))?;
        let h = mat3_from_row_major(&hv).map_err(|e| format!("{seq}/{hf}: {e}"))?;

        let left = read_image_any_rgb8(dir.join(left_n))?.to_cuda(&stream)?;
        let right = read_image_any_rgb8(dir.join(right_n))?.to_cuda(&stream)?;

        raco.submit(&left, &mut l)?;
        raco.submit(&right, &mut r)?;
        glue.submit(&l, &r, &mut lg_out)?;
        // XFeat depends only on the uploaded images, so enqueue it before the readback:
        // one synchronise for every model instead of two.
        if let Some(x) = &mut xf {
            // Oxford sequences are uniform within a sequence, so no sync is needed here.
            x.submit(
                &left,
                &right,
                (left.size() != right.size()).then_some(&stream),
            )?;
        }
        mnn.submit(
            Descriptors::new(l.descs_slice(), l.count(), l.desc_dim()),
            Descriptors::new(r.descs_slice(), r.count(), r.desc_dim()),
            nn_cossim,
            &mut mnn_out,
        )?;
        stream.synchronize()?;

        let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
        let lg_s = score(&lg_out.pairs(0.0)?, &lk, &rk, &h, thresh, (sx, sy));
        let mnn_s = score(&mnn_out.pairs(), &lk, &rk, &h, thresh, (sx, sy));

        // XFeat on the same pair, its own keypoints and its own 64-D matcher.
        let xf_s = match &mut xf {
            Some(x) => {
                let (pairs, xlk, xrk) = x.finish(&stream, xf_cossim)?;
                score(&pairs, &xlk, &xrk, &h, thresh, (sx, sy))
            }
            None => (0, 0),
        };
        lg.add(lg_s);
        mnn_t.add(mnn_s);
        xf_t.add(xf_s);

        println!(
            "{:<14} {:>7} {:>6} {:>6.1}% {:>7} {:>6} {:>6.1}% {:>7} {:>6} {:>6.1}%",
            format!("{seq}/{}", right_n.trim_end_matches(".png")),
            lg_s.0,
            lg_s.1,
            pair_pct(lg_s),
            mnn_s.0,
            mnn_s.1,
            pair_pct(mnn_s),
            xf_s.0,
            xf_s.1,
            pair_pct(xf_s)
        );
    }
    println!(
        "\ntotal correct correspondences: LightGlue+ {}, RaCo mutual-NN {}, XFeat {}",
        lg.inliers, mnn_t.inliers, xf_t.inliers
    );
    // Every pair weighted equally, so a single dense pair cannot carry the summary.
    println!(
        "macro-average precision over {} pairs: LightGlue+ {:.1}%, RaCo mutual-NN \
         {:.1}%, XFeat {:.1}%",
        lg.pairs,
        lg.macro_pct(),
        mnn_t.macro_pct(),
        xf_t.macro_pct()
    );
    Ok(())
}
