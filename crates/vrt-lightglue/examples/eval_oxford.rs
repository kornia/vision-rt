//! Evaluate matchers on the **Oxford/VGG affine** benchmark — real images, real
//! ground-truth homographies.
//!
//! Every accuracy number this crate has published so far came from one synthetic pair
//! with a synthetic transform, which cannot show how a matcher behaves on real texture,
//! real blur, or real scale change. This runs the same comparison on `bark` and `boat`
//! (rotation + zoom) and `graf` (viewpoint), scoring each match against the sequence's
//! ground-truth homography.
//!
//! A match `(i, j)` is an **inlier** iff `H · left[i]` lands within `INLIER_PX` of
//! `right[j]`, the standard protocol for this benchmark. Reported per pair:
//!
//! * **matches** — how many correspondences survived the matcher's own filtering
//! * **inlier %** — of those, how many are geometrically correct
//! * **inliers** — the product, which is what a pose solver actually consumes
//!
//! Usage:
//!   cargo run --release -p vrt-lightglue --example eval_oxford -- \
//!       <dataset_dir> <raco.engine> <lightglue.engine> [inlier_px]
//!
//! `dataset_dir` holds `manifest.txt` with `seq left right homography` per line; produce
//! it with `scripts/get_oxford.sh` followed by `examples/prep_oxford`, which also
//! photometrically verifies the rescaled ground truth before you trust any number here.

use std::path::Path;

use kornia_io::functional::read_image_any_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::{RaCoAliked, DESC_DIM};
use vrt_xfeat::{Matcher, XFeat, XFeatParams};

/// Mutual-NN similarity gate. Tuned for XFeat's 64-D descriptors; ALIKED's 128-D live on
/// a different scale, so this is a CLI argument — a gate set for the wrong descriptor
/// family silently returns zero matches rather than erroring.
const DEFAULT_MIN_COSSIM: f32 = 0.82;

/// Row-major 3x3 ground-truth homography mapping left pixels to right pixels.
type Homography = [f64; 9];

fn warp(h: &Homography, x: f32, y: f32) -> (f32, f32) {
    let (x, y) = (x as f64, y as f64);
    let w = h[6] * x + h[7] * y + h[8];
    // A degenerate w means the point maps to infinity; push it far away so it can never
    // be counted as an inlier rather than producing a NaN that silently compares false.
    if w.abs() < 1e-12 {
        return (f32::MAX, f32::MAX);
    }
    (
        ((h[0] * x + h[1] * y + h[2]) / w) as f32,
        ((h[3] * x + h[4] * y + h[5]) / w) as f32,
    )
}

/// (matches, inliers, inlier %) for a match set scored against the ground truth.
fn score(
    pairs: &[(usize, usize)],
    lk: &[(f32, f32)],
    rk: &[(f32, f32)],
    h: &Homography,
    thresh: f32,
) -> (usize, usize, f32) {
    let inl = pairs
        .iter()
        .filter(|(i, j)| {
            let (ex, ey) = warp(h, lk[*i].0, lk[*i].1);
            let (dx, dy) = (ex - rk[*j].0, ey - rk[*j].1);
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
        eprintln!("Usage: eval_oxford <dataset_dir> <raco.engine> <lightglue.engine> [inlier_px] [min_cossim] [xfeat.engine]");
        std::process::exit(1);
    }
    let root = Path::new(&a[1]);
    let thresh: f32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let min_cossim: f32 = a
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MIN_COSSIM);

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(&a[2], stream.clone())?;
    let mut glue = LightGlue::from_engine_file(&a[3], stream.clone())?;
    let mnn = Matcher::with_dim(stream.clone(), DESC_DIM)?;
    // XFeat + its native 64-D mutual-NN, as the baseline the crate READMEs compare to.
    // Optional so the eval still runs without an XFeat engine on hand.
    let xfeat_engine = a.get(6).cloned();
    let mut xf = match &xfeat_engine {
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
        "Oxford/VGG affine — RaCo k{k}, LightGlue k{}, mutual-NN {DESC_DIM}-D (cossim>={min_cossim}), inlier <= {thresh}px",
        glue.num_keypoints()
    );
    println!(
        "{:<14} {:>24} {:>24} {:>24}",
        "", "LightGlue+ m/inl/%", "RaCo mutual-NN m/inl/%", "XFeat mutual-NN m/inl/%"
    );

    let manifest = std::fs::read_to_string(root.join("manifest.txt"))?;
    let (mut lg_tot, mut mnn_tot, mut xf_tot) = (0usize, 0usize, 0usize);
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (seq, left_n, right_n, hf) = (f[0], f[1], f[2], f[3]);
        let dir = root.join(seq);

        let hv: Vec<f64> = std::fs::read_to_string(dir.join(hf))?
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        let mut h: Homography = [0.0; 9];
        h.copy_from_slice(&hv[..9]);

        let left = read_image_any_rgb8(dir.join(left_n))?.to_cuda(&stream)?;
        let right = read_image_any_rgb8(dir.join(right_n))?.to_cuda(&stream)?;

        raco.submit(&left, &mut l)?;
        raco.submit(&right, &mut r)?;
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
        let (lm, li, lp) = score(&lg_out.pairs(0.0)?, &lk, &rk, &h, thresh);
        let (mm, mi, mp) = score(&mnn_out.pairs(), &lk, &rk, &h, thresh);
        lg_tot += li;
        mnn_tot += mi;

        // XFeat on the same pair, its own keypoints and its own 64-D matcher.
        let (xm, xi, xp) = match (&mut xf, &mut xf_res) {
            (Some((x, m)), Some((xl, xr, xo))) => {
                x.submit(&left, xl)?;
                x.submit(&right, xr)?;
                stream.synchronize()?;
                m.submit_match(&xl.descs, xl.count(), &xr.descs, xr.count(), min_cossim, xo)?;
                stream.synchronize()?;
                let flat_l = xl.kpts_to_host()?;
                let flat_r = xr.kpts_to_host()?;
                let xlk: Vec<(f32, f32)> = flat_l.chunks_exact(2).map(|p| (p[0], p[1])).collect();
                let xrk: Vec<(f32, f32)> = flat_r.chunks_exact(2).map(|p| (p[0], p[1])).collect();
                score(&xo.pairs(), &xlk, &xrk, &h, thresh)
            }
            _ => (0, 0, 0.0),
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
