//! Benchmark RaCo-ALIKED + LightGlue+ against XFeat + mutual-NN on the *same* image pair.
//!
//! Reports per-stage and end-to-end latency for both pipelines, plus a match-quality
//! comparison. Speed alone would be misleading: the two produce very different numbers
//! of correspondences, so this also reports how many survive and how many are
//! geometrically consistent.
//!
//! **Quality metric.** Two modes:
//!
//! * With a ground-truth affine (a 6-number file `a11 a12 a13 a21 a22 a23` mapping left
//!   pixels to right pixels), a match is an inlier iff it lands within `INLIER_PX` of
//!   where the transform says it should. This is a true correctness measure.
//! * Without one, consistency is measured against the pair's own median displacement.
//!   Exact for a pure-translation pair, a proxy otherwise — and it only rewards a
//!   matcher for agreeing with itself, so read it alongside the raw count: 10 matches
//!   at 100% is worse than 700 at 98%.
//!
//! A pure-translation pair does **not** discriminate these two pipelines — RaCo's claim
//! is *rotation* robustness, so use a rotated pair with a ground-truth affine to test
//! the case that actually separates them.
//!
//! For a fair comparison both engines should be built with min=opt=max at the
//! benchmark resolution, so neither is penalised for running off its optimum profile.
//!
//! Usage:
//!   cargo run --release -p vrt-lightglue --example bench_vs_xfeat -- \
//!       <raco_extractor.engine> <lightglue.engine> <xfeat_backbone.engine> \
//!       <left.png> <right.png> [iters] [gt_affine.txt]

use std::sync::Arc;
use std::time::Instant;

use kornia_image::Image;
use kornia_io::functional::read_image_any_rgb8;
use vrt::CudaStream;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::RaCoAliked;
use vrt_xfeat::{Descriptors, Matcher, XFeat, XFeatParams};

#[path = "common/mod.rs"]
mod common;
use common::read_floats;

const XFEAT_THRESHOLD: f32 = 0.05;
const MIN_COSSIM: f32 = 0.82;
const LG_MIN_SCORE: f32 = 0.0; // LightGlue already filtered inside the graph
const WARMUP: usize = 5;
const INLIER_PX: f32 = 2.0;

fn main() -> Result<(), vrt::BoxError> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!(
            "Usage: bench_vs_xfeat <raco.engine> <lightglue.engine> <xfeat.engine> \
             <left.png> <right.png> [iters] [gt_affine.txt]"
        );
        std::process::exit(1);
    }
    let iters: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(20);
    // `.and_then` would turn a mistyped path into a silent fallback to the
    // median-displacement proxy, which only rewards a matcher for agreeing with itself.
    let gt = a.get(7).map(|p| load_affine(p)).transpose()?;
    match &gt {
        Some(m) => println!("ground-truth affine: [{:?}]", m),
        None => println!("no ground truth — scoring against median displacement"),
    }

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let left_src = read_image_any_rgb8(&a[4])?;
    let right_src = read_image_any_rgb8(&a[5])?;
    let left = left_src.to_cuda(&stream)?;
    let right = right_src.to_cuda(&stream)?;
    println!(
        "pair {}x{}, {iters} iters after {WARMUP} warmup\n",
        left_src.width(),
        left_src.height()
    );

    // XFeat's keypoint budget is taken from the RaCo engine rather than fixed, so the
    // two pipelines are compared at the SAME budget whichever kN asset is loaded — and
    // so XFeat's own O(K^2) mutual-NN cost scales into view alongside LightGlue's.
    let raco = bench_raco(&stream, &a[1], &a[2], &left, &right, iters, &gt)?;
    let xfeat = bench_xfeat(&stream, &a[3], &left, &right, iters, &gt, raco.k)?;

    println!("\n{:-<74}", "");
    println!("keypoint budget K = {}", raco.k);
    println!(
        "{:<28} {:>10} {:>10} {:>12} {:>10} {:>8}",
        "", "extract x2", "match", "E2E median", "E2E min", "matches"
    );
    for r in [&raco, &xfeat] {
        println!(
            "{:<28} {:>9.1}ms {:>9.1}ms {:>11.1}ms {:>9.1}ms {:>8}",
            r.name, r.extract_ms, r.match_ms, r.total_ms, r.total_min_ms, r.matches
        );
    }
    println!("{:-<74}", "");
    let (f, s) = (&raco, &xfeat);
    println!(
        "end-to-end: RaCo+LightGlue is {:.1}x (median) / {:.1}x (min) {} than XFeat+mutual-NN",
        if f.total_ms > s.total_ms {
            f.total_ms / s.total_ms
        } else {
            s.total_ms / f.total_ms
        },
        if f.total_min_ms > s.total_min_ms {
            f.total_min_ms / s.total_min_ms
        } else {
            s.total_min_ms / f.total_min_ms
        },
        if f.total_ms > s.total_ms {
            "SLOWER"
        } else {
            "faster"
        }
    );
    for r in [&raco, &xfeat] {
        println!(
            "{:<28} {:>4} matches, median disp dx={:>7.1} dy={:>7.1}, {:>5.1}% inliers within {INLIER_PX}px",
            r.name, r.matches, r.dx, r.dy, r.inlier_pct
        );
    }
    Ok(())
}

struct Row {
    name: &'static str,
    /// Keypoint budget this row ran at (RaCo's baked-in K; XFeat's top_k).
    k: usize,
    extract_ms: f64,
    match_ms: f64,
    total_ms: f64,
    /// Best sample. On a box running other GPU work, the median is inflated by
    /// contention; the minimum is the closest thing to an uncontended measurement.
    total_min_ms: f64,
    matches: usize,
    dx: f32,
    dy: f32,
    inlier_pct: f32,
}

/// Median of the collected samples (medians, not means — one scheduler hiccup on a
/// loaded Jetson otherwise dominates the average).
fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn minimum(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Ground-truth affine mapping left pixels to right pixels, if one was supplied.
type Affine = [f32; 6];

/// Load a 2x3 affine, failing on anything it cannot parse.
///
/// Dropping unparseable tokens and checking only the count is the trap `read_floats`
/// exists to close: one typo shifts the remaining values into the wrong slots and still
/// satisfies `len() >= 6`, producing a wrong-but-plausible inlier percentage.
fn load_affine(path: &str) -> Result<Affine, vrt::BoxError> {
    let v = read_floats(std::path::Path::new(path))?;
    if v.len() < 6 {
        return Err(format!("{path}: expected 6 affine values, got {}", v.len()).into());
    }
    Ok(std::array::from_fn(|i| v[i] as f32))
}

/// Fraction of matches landing within `INLIER_PX` of where `gt` says they should.
fn quality_gt(pairs: &[(usize, usize)], lk: &[(f32, f32)], rk: &[(f32, f32)], gt: &Affine) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let n = pairs
        .iter()
        .filter(|(i, j)| {
            let (x, y) = lk[*i];
            let (ex, ey) = (
                gt[0] * x + gt[1] * y + gt[2] - rk[*j].0,
                gt[3] * x + gt[4] * y + gt[5] - rk[*j].1,
            );
            (ex * ex + ey * ey).sqrt() <= INLIER_PX
        })
        .count();
    100.0 * n as f32 / pairs.len() as f32
}

/// Median displacement and the fraction of matches within `INLIER_PX` of it.
fn quality(pairs: &[(usize, usize)], lk: &[(f32, f32)], rk: &[(f32, f32)]) -> (f32, f32, f32) {
    if pairs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let (mut dxs, mut dys): (Vec<f32>, Vec<f32>) = pairs
        .iter()
        .map(|(i, j)| (lk[*i].0 - rk[*j].0, lk[*i].1 - rk[*j].1))
        .unzip();
    dxs.sort_by(f32::total_cmp);
    dys.sort_by(f32::total_cmp);
    let (dx, dy) = (dxs[dxs.len() / 2], dys[dys.len() / 2]);
    let inliers = pairs
        .iter()
        .filter(|(i, j)| {
            let (ex, ey) = (lk[*i].0 - rk[*j].0 - dx, lk[*i].1 - rk[*j].1 - dy);
            (ex * ex + ey * ey).sqrt() <= INLIER_PX
        })
        .count();
    (dx, dy, 100.0 * inliers as f32 / pairs.len() as f32)
}

fn bench_raco(
    stream: &Arc<CudaStream>,
    raco_engine: &str,
    lg_engine: &str,
    left: &Image<u8, 3>,
    right: &Image<u8, 3>,
    iters: usize,
    gt: &Option<Affine>,
) -> Result<Row, vrt::BoxError> {
    let mut raco = RaCoAliked::from_engine_file(raco_engine, stream.clone())?;
    let mut glue = LightGlue::from_engine_file(lg_engine, stream.clone())?;
    let (mut l, mut r) = (raco.alloc_result()?, raco.alloc_result()?);
    let mut m = glue.alloc_result()?;

    for _ in 0..WARMUP {
        raco.submit(left, &mut l)?;
        raco.submit(right, &mut r)?;
        glue.submit(&l, &r, &mut m)?;
        stream.synchronize()?;
    }

    let (mut ext, mut mat, mut tot) = (vec![], vec![], vec![]);
    for _ in 0..iters {
        // Per-stage attribution needs a sync per stage; the end-to-end pass below is
        // the number that matters in a real pipeline (one sync covers everything).
        let t = Instant::now();
        raco.submit(left, &mut l)?;
        raco.submit(right, &mut r)?;
        stream.synchronize()?;
        ext.push(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        glue.submit(&l, &r, &mut m)?;
        stream.synchronize()?;
        mat.push(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        raco.submit(left, &mut l)?;
        raco.submit(right, &mut r)?;
        glue.submit(&l, &r, &mut m)?;
        stream.synchronize()?;
        tot.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let pairs = m.pairs(LG_MIN_SCORE)?;
    let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
    let (dx, dy, mut inlier_pct) = quality(&pairs, &lk, &rk);
    if let Some(m) = gt {
        inlier_pct = quality_gt(&pairs, &lk, &rk, m);
    }
    println!(
        "RaCo+LightGlue: {} kpts/img, {} matches",
        raco.num_keypoints(),
        pairs.len()
    );
    Ok(Row {
        name: "RaCo-ALIKED + LightGlue+",
        k: raco.num_keypoints(),
        extract_ms: median(&ext),
        match_ms: median(&mat),
        total_ms: median(&tot),
        total_min_ms: minimum(&tot),
        matches: pairs.len(),
        dx,
        dy,
        inlier_pct,
    })
}

fn bench_xfeat(
    stream: &Arc<CudaStream>,
    engine: &str,
    left: &Image<u8, 3>,
    right: &Image<u8, 3>,
    iters: usize,
    gt: &Option<Affine>,
    top_k: usize,
) -> Result<Row, vrt::BoxError> {
    let mut xf = XFeat::from_engine_file(
        engine,
        stream.clone(),
        XFeatParams::new(top_k, XFEAT_THRESHOLD),
    )?;
    let matcher = Matcher::new(stream.clone())?;
    let (mut l, mut r) = (xf.alloc_result()?, xf.alloc_result()?);
    let mut m = matcher.alloc_result(top_k)?;

    // XFeat's keypoint count is threshold-dependent and read back from the device, so
    // matching genuinely needs the extraction sync first — unlike RaCo, whose K is
    // fixed at export time. That extra sync is part of its end-to-end cost.
    for _ in 0..WARMUP {
        xf.submit(left, &mut l)?;
        xf.submit(right, &mut r)?;
        stream.synchronize()?;
        matcher.submit(
            Descriptors::new(&l.descs, l.count(), l.desc_dim()),
            Descriptors::new(&r.descs, r.count(), r.desc_dim()),
            MIN_COSSIM,
            &mut m,
        )?;
        stream.synchronize()?;
    }

    let (mut ext, mut mat, mut tot) = (vec![], vec![], vec![]);
    for _ in 0..iters {
        let t = Instant::now();
        xf.submit(left, &mut l)?;
        xf.submit(right, &mut r)?;
        stream.synchronize()?;
        ext.push(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        matcher.submit(
            Descriptors::new(&l.descs, l.count(), l.desc_dim()),
            Descriptors::new(&r.descs, r.count(), r.desc_dim()),
            MIN_COSSIM,
            &mut m,
        )?;
        stream.synchronize()?;
        mat.push(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        xf.submit(left, &mut l)?;
        xf.submit(right, &mut r)?;
        stream.synchronize()?;
        matcher.submit(
            Descriptors::new(&l.descs, l.count(), l.desc_dim()),
            Descriptors::new(&r.descs, r.count(), r.desc_dim()),
            MIN_COSSIM,
            &mut m,
        )?;
        stream.synchronize()?;
        tot.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let pairs = m.pairs();
    let flat_l = l.kpts_to_host()?;
    let flat_r = r.kpts_to_host()?;
    let lk: Vec<(f32, f32)> = flat_l.chunks_exact(2).map(|p| (p[0], p[1])).collect();
    let rk: Vec<(f32, f32)> = flat_r.chunks_exact(2).map(|p| (p[0], p[1])).collect();
    let (dx, dy, mut inlier_pct) = quality(&pairs, &lk, &rk);
    if let Some(m) = gt {
        inlier_pct = quality_gt(&pairs, &lk, &rk, m);
    }
    println!(
        "XFeat: {} / {} kpts (top_k {top_k}, threshold {XFEAT_THRESHOLD}), {} matches",
        l.count(),
        r.count(),
        pairs.len()
    );
    Ok(Row {
        name: "XFeat + mutual-NN",
        k: top_k,
        extract_ms: median(&ext),
        match_ms: median(&mat),
        total_ms: median(&tot),
        total_min_ms: minimum(&tot),
        matches: pairs.len(),
        dx,
        dy,
        inlier_pct,
    })
}
