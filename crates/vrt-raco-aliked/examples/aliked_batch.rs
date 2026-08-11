//! Batch RaCo-ALIKED extraction over a directory of JPEGs, written out as `.vrtk` files.
//!
//! The out-of-process bridge flux-map's `aliked` feature backend uses, mirroring `xfeat_batch`.
//! It exists because the two sides cannot be linked: this crate pins a different kornia than
//! flux-map, so they meet on files instead.
//!
//! Usage:
//!   aliked_batch <raco_aliked_extractor_kN.engine> <img_dir> <out_dir>
//!
//! Reads `<img_dir>/kfNNNN.jpg` and writes `<out_dir>/kfNNNN.vrtk` for each, preserving the
//! caller's numbering — the index IS the keyframe identity on the other side, so a gap must stay a
//! gap rather than shifting everything after it.
//!
//! ## `.vrtk` layout (little-endian throughout)
//!
//! ```text
//!   0  "VRTK"
//!   4  u32  n     keypoints in this file
//!   8  u32  dim   descriptor dimension (128 for ALIKED)
//!  12  u32  0     reserved
//!  16  n * (2 + dim) f32   x, y, then dim descriptor values, per keypoint
//! ```
//!
//! Keypoints are in SOURCE-image pixels, not model pixels. The extractor resizes to a multiple of
//! 32 internally and reports its own scale factors; not undoing that would hand flux-map
//! coordinates in a frame it has no knowledge of, and every downstream pose would be quietly wrong
//! by the resize ratio.
//!
//! ## Ordering
//!
//! Written in the extractor's native order. flux-map subsamples with an even stride when it caps
//! the count, precisely because a detector's output order is not a quality ranking — truncating
//! `xfeat_batch`'s raster-ordered output cropped the image and collapsed a run to 378 map points.
//! Nothing here should sort or truncate; leave that decision to the consumer.

use std::io::Write;
use std::path::Path;

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_io::functional::read_image_any_rgb8;
use vrt_raco_aliked::{RaCoAliked, DIM_DIVISOR};

/// Largest side the shipped extractor engines accept.
///
/// Their shape profile is `images:1x3x256x256|2x3x512x512|2x3x640x640` (vrt-hub), so anything with
/// a side over 640 is REJECTED by TensorRT rather than silently handled. flux-map's keyframe
/// previews are 480x853 portrait, which fails on the long axis — the first thing this tool was
/// pointed at, and it produced zero files.
const MAX_SIDE: usize = 640;

/// Downscale so the long side fits `MAX_SIDE` and both sides land on the model's 32 px grid.
///
/// Returns the image and the ACTUAL per-axis scales, which differ from the requested one because
/// the destination is rounded to whole pixels — and both are needed, because keypoints have to come
/// back in the CALLER's coordinate frame. flux-map indexes the returned xy against the thumb it
/// sent (to sample colours and build tracks), so a coordinate left in resized space is not a small
/// error: every keypoint lands in the wrong place by the scale ratio, and nothing downstream can
/// tell.
fn fit_to_engine(src: &Image<u8, 3>) -> Result<(Image<u8, 3>, f64, f64), vrt::BoxError> {
    let (w, h) = (src.cols(), src.rows());
    let scale = (MAX_SIDE as f64 / w.max(h) as f64).min(1.0);
    let rw = (((w as f64 * scale).round() as usize) / DIM_DIVISOR * DIM_DIVISOR).max(DIM_DIVISOR);
    let rh = (((h as f64 * scale).round() as usize) / DIM_DIVISOR * DIM_DIVISOR).max(DIM_DIVISOR);
    if rw == w && rh == h {
        return Ok((src.clone(), 1.0, 1.0));
    }
    let mut dst = Image::<u8, 3>::from_size_val(ImageSize { width: rw, height: rh }, 0)?;
    // On u8 directly: the f32 path needs two full-resolution f32 buffers, 12 bytes/pixel, on a
    // 7.4 GB board that is usually also holding a reconstruction.
    kornia_imgproc::resize::resize_fast_u8(src, &mut dst, InterpolationMode::Bilinear)?;
    Ok((dst, rw as f64 / w as f64, rh as f64 / h as f64))
}

fn write_vrtk(path: &Path, kpts: &[(f32, f32)], descs: &[f32], dim: usize) -> std::io::Result<()> {
    let n = kpts.len();
    let mut buf: Vec<u8> = Vec::with_capacity(16 + n * (2 + dim) * 4);
    buf.extend_from_slice(b"VRTK");
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    for (i, (x, y)) in kpts.iter().enumerate() {
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
        let row = &descs[i * dim..(i + 1) * dim];
        for v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(path)?.write_all(&buf)
}

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: aliked_batch <extractor.engine> <img_dir> <out_dir>");
        std::process::exit(1);
    }
    let (engine, img_dir, out_dir) = (&args[1], Path::new(&args[2]), Path::new(&args[3]));
    std::fs::create_dir_all(out_dir)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(engine, stream.clone())?;
    let k = raco.num_keypoints();

    let mut names: Vec<_> = std::fs::read_dir(img_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jpg" || x == "jpeg" || x == "png"))
        .collect();
    names.sort();
    eprintln!("aliked_batch: {} frames, K={k}, dim={}", names.len(), vrt_raco_aliked::DESC_DIM);

    let mut done = 0usize;
    for p in &names {
        let stem = match p.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        // One frame at a time, synced per frame. The API is async and several frames could be in
        // flight, but each result owns a K x 128 device buffer and this runs on a 7.4 GB board that
        // is usually also holding a reconstruction — throughput here is not the bottleneck the
        // build cares about, and an OOM in the feature pass would be.
        let src = match read_image_any_rgb8(p) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  skip {}: {e}", p.display());
                continue;
            }
        };
        // Fit the engine's shape profile FIRST, on the host, then upload once.
        let (fitted, sx, sy) = fit_to_engine(&src)?;
        let dev = fitted.to_cuda(&stream)?;
        let mut out = raco.alloc_result()?;
        if let Err(e) = raco.submit(&dev, &mut out) {
            eprintln!("  skip {}: {e}", p.display());
            continue;
        }
        stream.synchronize()?;

        // `keypoints_host` already returns SOURCE-image pixels — the extractor applies its own
        // resize scale on the way out. Verified against `scale()`, which reports the ratio it used.
        // Back into the CALLER's frame: undo this tool's own downscale. `keypoints_host` already
        // undoes the extractor's internal 32 px fit, but it knows nothing about the resize above.
        let kpts: Vec<(f32, f32)> = out
            .keypoints_host()?
            .into_iter()
            .map(|(x, y)| (x / sx as f32, y / sy as f32))
            .collect();
        let descs = out.descriptors_host()?;
        let dim = out.desc_dim();
        if descs.len() < kpts.len() * dim {
            eprintln!("  skip {}: descriptor buffer short", p.display());
            continue;
        }
        write_vrtk(&out_dir.join(format!("{stem}.vrtk")), &kpts, &descs, dim)?;
        done += 1;
        if done % 50 == 0 {
            eprintln!("  {done}/{}", names.len());
        }
    }
    eprintln!("aliked_batch: wrote {done} of {} files", names.len());
    Ok(())
}
