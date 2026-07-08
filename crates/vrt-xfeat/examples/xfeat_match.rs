//! XFeat feature matching — a relocalization demo.
//!
//! Detects XFeat keypoints+descriptors in a **map** image and a **query**
//! image, then matches them by mutual nearest-neighbour on the descriptors.
//! The match count is a relocalization signal: many mutual matches ⇒ the query
//! frame sees the same place as the map keyframe.
//!
//! This is the offline, two-image shape of what runs online in a robot:
//! ```text
//!   map keyframe   ──extract once──►  map XFeatResult  (stored)
//!   live camera ─► preproc ─► XFeat ─► live XFeatResult ─► match vs map ─► pose
//! ```
//! In the streaming pipeline the per-frame extraction is
//! `camera → preproc → XFeat` (see `rtsp_xfeat`); matching the result against
//! the stored map is the relocalization step shown here.
//!
//! Usage:
//!   cargo run --release -p xfeat_match -- \
//!       models/xfeat/xfeat_backbone.onnx  map.jpg  query.jpg  [out.png]

use std::sync::Arc;

use kornia_image::{Image, ImageSize, InterpolationMode};
use kornia_imgproc::resize::resize_fast_rgb;
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt::logger::Severity;
use vrt::{CudaStream, Engine, Logger, Runtime};
use vrt_xfeat::{XFeat, XFeatParams, XFeatResult};

const MODEL_W: u32 = 640; // multiple of 32 (XFeat downsamples ×8)
const MODEL_H: u32 = 640;
const TOP_K: usize = 2048;
const THRESHOLD: f32 = 0.05;
const MIN_COSSIM: f32 = 0.82; // descriptor cosine-similarity gate
                              // Raw mutual-NN match count is only a coarse relocalization signal: a few tens
                              // of false matches survive between unrelated scenes (repeated texture), while
                              // the same place yields hundreds–thousands. Production SLAM filters these with
                              // geometric verification (essential-matrix RANSAC) and counts inliers; here we
                              // just threshold the raw count to separate "same place" from "different place".
const RELOC_MIN_MATCHES: usize = 150;

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: xfeat_match <model.onnx|engine> <map_image> <query_image> [out.png]");
        std::process::exit(1);
    }
    let (model_path, map_path, query_path) = (&args[1], &args[2], &args[3]);
    let out_path = args.get(4).map(String::as_str).unwrap_or("xfeat_match.png");

    // .onnx → on-device engine cache (one-time build); .engine → used directly.
    let profile = vrt_hub::EngineProfile {
        input: Some((
            "image".into(),
            vec![1, 3, 240, 320],
            vec![1, 3, 640, 640],
            vec![1, 3, 1088, 1920],
        )),
        fp16: true,
        workspace_mb: 2048,
    };
    let engine_path =
        vrt_hub::EngineCache::default().resolve("xfeat-backbone", model_path, &profile)?;

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;
    let params = XFeatParams::new(TOP_K, THRESHOLD, MODEL_H as usize, MODEL_W as usize);

    // One shared stream for XFeat (one sync per extract). XFeat letterboxes internally.
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut xfeat = XFeat::new(Arc::clone(&engine), stream.clone(), params)?;

    // Extract features from both images (resized to the model size for viz parity).
    let (map_res, map_img) = extract(&mut xfeat, &stream, map_path)?;
    let (query_res, query_img) = extract(&mut xfeat, &stream, query_path)?;

    // Match: mutual nearest-neighbour on the L2-normalised descriptors.
    let matches = xfeat
        .postproc()
        .match_mutual_nn_gpu(&map_res, &query_res, MIN_COSSIM)?;

    println!("map:   {} keypoints", map_res.len());
    println!("query: {} keypoints", query_res.len());
    println!(
        "matches (mutual-NN, cossim ≥ {MIN_COSSIM}): {}",
        matches.len()
    );
    println!(
        "relocalization: {}",
        if matches.len() >= RELOC_MIN_MATCHES {
            "RELOCALIZED ✓"
        } else {
            "not enough matches ✗"
        }
    );

    // Keypoints live on the GPU — download both sets to host for drawing.
    let map_kpts = map_res.kpts_to_host(&stream)?;
    let query_kpts = query_res.kpts_to_host(&stream)?;
    save_match_viz(
        &map_img,
        &query_img,
        &map_kpts,
        &query_kpts,
        &matches,
        out_path,
    )?;
    println!("saved {out_path}");
    Ok(())
}

/// Load `path`, resize to the model size, run XFeat, and return the result
/// alongside the resized RGB image (model-space coords align with it).
fn extract(
    xfeat: &mut XFeat,
    stream: &Arc<CudaStream>,
    path: &str,
) -> Result<(XFeatResult, Image<u8, 3>), vrt::BoxError> {
    let src = read_image_any_rgb8(path)?;
    let mut resized = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: MODEL_W as usize,
            height: MODEL_H as usize,
        },
        0,
    )?;
    resize_fast_rgb(&src, &mut resized, InterpolationMode::Bilinear)?;

    // Upload to device, then hand XFeat the device image. The source is already
    // model-sized, so XFeat's internal letterbox is the identity.
    let dev = Image(resized.0.to_cuda(stream)?);
    let result = xfeat.run(&dev)?; // letterbox + backbone + sync
    Ok((result, resized))
}

// ── Visualization ─────────────────────────────────────────────────────────────

/// Side-by-side map | query with green lines between matched keypoints.
fn save_match_viz(
    map_img: &Image<u8, 3>,
    query_img: &Image<u8, 3>,
    map_kpts: &[f32],
    query_kpts: &[f32],
    matches: &[(usize, usize)],
    out_path: &str,
) -> Result<(), vrt::BoxError> {
    let (w, h) = (MODEL_W as usize, MODEL_H as usize);
    let cw = w * 2;
    let (mp, qp) = (map_img.as_slice(), query_img.as_slice());

    // RGB canvas: map on the left half, query on the right.
    let mut canvas = vec![0u8; cw * h * 3];
    for y in 0..h {
        for x in 0..w {
            let s = (y * w + x) * 3;
            let l = (y * cw + x) * 3;
            let r = (y * cw + (x + w)) * 3;
            canvas[l..l + 3].copy_from_slice(&mp[s..s + 3]);
            canvas[r..r + 3].copy_from_slice(&qp[s..s + 3]);
        }
    }

    for &(mi, qi) in matches {
        let (mx, my) = (map_kpts[mi * 2], map_kpts[mi * 2 + 1]);
        let (qx, qy) = (query_kpts[qi * 2], query_kpts[qi * 2 + 1]);
        draw_line(
            &mut canvas,
            cw,
            h,
            mx as i32,
            my as i32,
            qx as i32 + w as i32,
            qy as i32,
            [40, 220, 40],
        );
    }

    let out = Image::<u8, 3>::new(
        ImageSize {
            width: cw,
            height: h,
        },
        canvas,
    )?;
    write_image_png_rgb8(out_path, &out)?;
    Ok(())
}

/// Bresenham line over an interleaved RGB byte buffer (`w`×`h`), clipped.
fn draw_line(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    let (iw, ih) = (w as i32, h as i32);
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && x < iw && y >= 0 && y < ih {
            let p = (y as usize * w + x as usize) * 3;
            buf[p..p + 3].copy_from_slice(&color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}
