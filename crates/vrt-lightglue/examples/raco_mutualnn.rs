//! RaCo-ALIKED + **GPU mutual-NN** matching — the cheap alternative to LightGlue.
//!
//! LightGlue's attention is O(K²) and dominates the pipeline at large K (126 ms a pair at
//! k3072). A plain mutual-nearest-neighbour search over the same 128-D descriptors is a
//! single tiled argmax kernel. This example runs that path end to end so it can be
//! profiled and compared against `raco_lightglue_match`.
//!
//! Everything stays on one stream, so a profiler should see exactly one H2D (the source
//! images) and one D2H (the match readback) per pair — no intermediate host round-trips.
//!
//! Usage:
//!   cargo run --release -p vrt-lightglue --example raco_mutualnn -- \
//!       <raco_extractor.engine> <left.png> <right.png> [min_cossim] [iters]

use std::time::Instant;

use kornia_io::functional::read_image_any_rgb8;
use vrt_raco_aliked::{RaCoAliked, DESC_DIM};
use vrt_xfeat::{Descriptors, Matcher};

#[path = "common/mod.rs"]
mod common;
use common::arg_or;

fn main() -> Result<(), vrt::BoxError> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("Usage: raco_mutualnn <raco.engine> <left.png> <right.png> [min_cossim] [iters]");
        std::process::exit(1);
    }
    let min_cossim: f32 = arg_or(&a, 4, "min_cossim", 0.0)?;
    // At least one timed pass: `min` over an empty loop leaves `best` at infinity and the
    // run prints "inf ms" as though it had measured something.
    let iters: usize = arg_or(&a, 5, "iters", 20)?.max(1);

    // One shared stream: extraction and matching are a single continuous queue, and one
    // synchronize drains both.
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(&a[1], stream.clone())?;
    let matcher = Matcher::with_dim(stream.clone(), DESC_DIM)?;

    let left = read_image_any_rgb8(&a[2])?.to_cuda(&stream)?;
    let right = read_image_any_rgb8(&a[3])?.to_cuda(&stream)?;

    let (mut l, mut r) = (raco.alloc_result()?, raco.alloc_result()?);
    let k = raco.num_keypoints();
    let mut m = matcher.alloc_result(k)?;

    let mut once = |l: &mut _, r: &mut _, m: &mut _| -> Result<(), vrt::BoxError> {
        raco.submit(&left, l)?;
        raco.submit(&right, r)?;
        matcher.submit(
            Descriptors::new(l.descs_slice(), l.count(), l.desc_dim()),
            Descriptors::new(r.descs_slice(), r.count(), r.desc_dim()),
            min_cossim,
            m,
        )?;
        stream.synchronize()?;
        Ok(())
    };

    for _ in 0..5 {
        once(&mut l, &mut r, &mut m)?; // warm-up
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        once(&mut l, &mut r, &mut m)?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let pairs = m.pairs();
    println!("K={k} descriptors {DESC_DIM}-D, min_cossim={min_cossim}");
    println!("{} mutual-NN matches", pairs.len());
    println!("end-to-end (extract x2 + match), min of {iters}: {best:.1} ms");

    // Same cheap correctness signal the other examples print: for a pure-translation pair
    // the median displacement recovers the shift.
    let (lk, rk) = (l.keypoints_host()?, r.keypoints_host()?);
    if !pairs.is_empty() {
        let mut dxs: Vec<f32> = pairs.iter().map(|(a, b)| lk[*a].0 - rk[*b].0).collect();
        let mut dys: Vec<f32> = pairs.iter().map(|(a, b)| lk[*a].1 - rk[*b].1).collect();
        dxs.sort_by(f32::total_cmp);
        dys.sort_by(f32::total_cmp);
        let mid = dxs.len() / 2;
        println!(
            "median match displacement: dx={:.1} dy={:.1}",
            dxs[mid], dys[mid]
        );
    }
    Ok(())
}
