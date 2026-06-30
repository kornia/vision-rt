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

use image::{Rgb, RgbImage};
use vrt::logger::Severity;
use vrt::{CudaStream, DType, Engine, Logger, Runtime, VrtTensor};
use vrt_preproc::Preprocessor;
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

    // One shared stream for XFeat + the preprocessor (one sync per extract).
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut xfeat = XFeat::with_stream(Arc::clone(&engine), stream.clone(), params)?;
    let mut preproc = Preprocessor::new(stream.clone(), MODEL_W, MODEL_H, MODEL_W, MODEL_H)?;

    // Extract features from both images (resized to the model size for viz parity).
    let (map_res, map_img) = extract(&mut xfeat, &mut preproc, &stream, map_path)?;
    let (query_res, query_img) = extract(&mut xfeat, &mut preproc, &stream, query_path)?;

    // Match: mutual nearest-neighbour on the L2-normalised descriptors.
    let matches = xfeat
        .postproc()
        .match_mutual_nn_gpu(&map_res, &query_res, MIN_COSSIM)?;

    println!("map:   {} keypoints", map_res.scores.len());
    println!("query: {} keypoints", query_res.scores.len());
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

    save_match_viz(
        &map_img, &query_img, &map_res, &query_res, &matches, out_path,
    )?;
    println!("saved {out_path}");
    Ok(())
}

/// Load `path`, resize to the model size, run XFeat, and return the result
/// alongside the resized RGB image (model-space coords align with it).
fn extract(
    xfeat: &mut XFeat,
    preproc: &mut Preprocessor,
    stream: &Arc<CudaStream>,
    path: &str,
) -> Result<(XFeatResult, RgbImage), vrt::BoxError> {
    let img = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(
        &img,
        MODEL_W,
        MODEL_H,
        image::imageops::FilterType::Triangle,
    );

    // RGB → RGBA (the preprocessor samples an RGBA pitch-linear surface).
    let mut rgba = Vec::with_capacity((MODEL_W * MODEL_H * 4) as usize);
    for px in resized.pixels() {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }

    // H2D + letterbox (identity here: src == dst == model size) into a CHW tensor.
    let tensor = VrtTensor::alloc(
        stream,
        [1, 3, MODEL_H as usize, MODEL_W as usize],
        DType::F32,
    )?;
    let _guard = preproc.process(&rgba, MODEL_W * 4, tensor.as_mut_ptr() as *mut f32)?;
    let result = xfeat.extract(&tensor)?; // syncs internally
    Ok((result, resized))
}

// ── Visualization ─────────────────────────────────────────────────────────────

/// Side-by-side map | query with green lines between matched keypoints.
fn save_match_viz(
    map_img: &RgbImage,
    query_img: &RgbImage,
    map_res: &XFeatResult,
    query_res: &XFeatResult,
    matches: &[(usize, usize)],
    out_path: &str,
) -> Result<(), vrt::BoxError> {
    let (w, h) = (MODEL_W, MODEL_H);
    let mut canvas = RgbImage::new(w * 2, h);
    for y in 0..h {
        for x in 0..w {
            canvas.put_pixel(x, y, *map_img.get_pixel(x, y));
            canvas.put_pixel(x + w, y, *query_img.get_pixel(x, y));
        }
    }

    for &(mi, qi) in matches {
        let (mx, my) = (map_res.kpts_cpu[mi * 2], map_res.kpts_cpu[mi * 2 + 1]);
        let (qx, qy) = (query_res.kpts_cpu[qi * 2], query_res.kpts_cpu[qi * 2 + 1]);
        draw_line(
            &mut canvas,
            mx as i32,
            my as i32,
            qx as i32 + w as i32,
            qy as i32,
            Rgb([40, 220, 40]),
        );
    }

    canvas.save(out_path)?;
    Ok(())
}

/// Bresenham line, clipped to the canvas.
fn draw_line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
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
