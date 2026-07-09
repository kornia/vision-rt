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

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt::logger::Severity;
use vrt::{CudaStream, Engine, Logger, Runtime};
use vrt_xfeat::{Matcher, XFeat, XFeatParams};

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
    let params = XFeatParams::new(TOP_K, THRESHOLD);

    // One shared CUDA stream. Two XFeat instances (separate buffers, same engine
    // + stream) so both extractions can be outstanding under ONE sync.
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut xf_map = XFeat::new(Arc::clone(&engine), stream.clone(), params.clone())?;
    let mut xf_query = XFeat::new(Arc::clone(&engine), stream.clone(), params)?;

    // Load both frames to device (native resolution).
    let (map_dev, map_img) = load(&stream, map_path)?;
    let (query_dev, query_img) = load(&stream, query_path)?;

    // Continuous flow: submit BOTH extractions async on the shared stream, then a
    // SINGLE synchronize() covers both; keypoints come back in source pixels.
    let map_pending = xf_map.submit(&map_dev)?;
    let query_pending = xf_query.submit(&query_dev)?;
    stream.synchronize()?;
    let map_res = xf_map.finish(map_pending);
    let query_res = xf_query.finish(query_pending);

    // Match on the same stream (decoupled Matcher), also async: submit → sync → finish.
    let matcher = Matcher::new(stream.clone())?;
    let match_pending = matcher.submit_match(&map_res, &query_res, MIN_COSSIM)?;
    stream.synchronize()?;
    let matches = matcher.finish_match(match_pending);

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

/// Load `path` at native resolution: return the device `Image` (backbone input)
/// and the host `Image` (for drawing). XFeat resizes to floor-32 internally and
/// returns keypoints in source pixels, so they align with the host image.
fn load(
    stream: &Arc<CudaStream>,
    path: &str,
) -> Result<(Image<u8, 3>, Image<u8, 3>), vrt::BoxError> {
    let src = read_image_any_rgb8(path)?; // Rgb8 (derefs to Image<u8,3>)
    let dev = Image(src.0.to_cuda(stream)?); // device Image<u8,3>
    Ok((dev, src.0))
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
    // Images may differ in size (native resolutions): map left, query right,
    // canvas = (map_w + query_w) × max(heights).
    let (mw, mh) = (map_img.width(), map_img.height());
    let (qw, qh) = (query_img.width(), query_img.height());
    let cw = mw + qw;
    let ch = mh.max(qh);
    let (mp, qp) = (map_img.as_slice(), query_img.as_slice());

    let mut canvas = vec![0u8; cw * ch * 3];
    for y in 0..mh {
        for x in 0..mw {
            let s = (y * mw + x) * 3;
            let d = (y * cw + x) * 3;
            canvas[d..d + 3].copy_from_slice(&mp[s..s + 3]);
        }
    }
    for y in 0..qh {
        for x in 0..qw {
            let s = (y * qw + x) * 3;
            let d = (y * cw + (x + mw)) * 3;
            canvas[d..d + 3].copy_from_slice(&qp[s..s + 3]);
        }
    }

    for &(mi, qi) in matches {
        let (mx, my) = (map_kpts[mi * 2], map_kpts[mi * 2 + 1]);
        let (qx, qy) = (query_kpts[qi * 2], query_kpts[qi * 2 + 1]);
        draw_line(
            &mut canvas,
            cw,
            ch,
            mx as i32,
            my as i32,
            qx as i32 + mw as i32,
            qy as i32,
            [40, 220, 40],
        );
    }

    let out = Image::<u8, 3>::new(
        ImageSize {
            width: cw,
            height: ch,
        },
        canvas,
    )?;
    write_image_png_rgb8(out_path, &out)?;
    Ok(())
}

/// Bresenham line over an interleaved RGB byte buffer (`w`×`h`), clipped.
#[allow(clippy::too_many_arguments)]
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
