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
//!  12  u32  fp    extractor-engine fingerprint (0 = unknown)
//!  16  n * (2 + dim) f32   x, y, then dim descriptor values, per keypoint
//! ```
//!
//! Word 12 was a zero reserved field. It now carries [`engine_fingerprint`] of the engine that
//! produced the keypoints, so the matcher's `.vrtm` can be checked against the file its indices
//! address instead of the two being assumed to match. Readers that ignored the reserved word are
//! unaffected; readers that checked it for zero were checking nothing.
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

use kornia_imgproc::interpolation::InterpolationMode;
use kornia_io::functional::read_image_any_rgb8;
use vrt_raco_aliked::{engine_fingerprint, fit_to_engine, RaCoAliked, FALLBACK_MAX_SIDE};

fn write_vrtk(
    path: &Path,
    kpts: &[(f32, f32)],
    descs: &[f32],
    dim: usize,
    fp: u32,
) -> std::io::Result<()> {
    let n = kpts.len();
    let mut buf: Vec<u8> = Vec::with_capacity(16 + n * (2 + dim) * 4);
    buf.extend_from_slice(b"VRTK");
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&fp.to_le_bytes());
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
    // Stamped into every `.vrtk` so the matcher's indices can be checked against the keypoints they
    // address rather than assumed to belong to them.
    let fp = engine_fingerprint(engine);

    let mut names: Vec<_> = std::fs::read_dir(img_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        // Case-INSENSITIVE: a `.JPG` directory used to match nothing here, which produced zero
        // files and, before this tool exited non-zero, a map built from empty feature sets.
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x.to_ascii_lowercase())
                .is_some_and(|x| matches!(x.as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    names.sort();
    eprintln!(
        "aliked_batch: {} frames, K={k}, dim={}, engine fp={fp:08x}",
        names.len(),
        vrt_raco_aliked::DESC_DIM
    );

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
        // Natural size first (floored to the model grid), and only fall back to the 640 cap if the
        // engine actually rejects it. `ShapeRejected` is a distinct typed variant, so this asks the
        // ENGINE what it accepts instead of assuming a profile it may not have.
        let mut fitted_pair = None;
        for (attempt, cap) in [src.cols().max(src.rows()), FALLBACK_MAX_SIDE]
            .into_iter()
            .enumerate()
        {
            let scaled = match fit_to_engine(&src, cap, InterpolationMode::Bilinear) {
                Ok(v) => v,
                Err(e) => {
                    if attempt == 1 {
                        eprintln!("  skip {}: {e}", p.display());
                    }
                    continue;
                }
            };
            let dev = scaled.image.to_cuda(&stream)?;
            let mut out = raco.alloc_result()?;
            match raco.submit(&dev, &mut out) {
                Ok(()) => {
                    stream.synchronize()?;
                    fitted_pair = Some((out, scaled));
                    break;
                }
                Err(vrt_raco_aliked::RaCoAlikedError::ShapeRejected { .. }) if attempt == 0 => {
                    // Engine has the smaller profile; retry under the cap.
                }
                Err(e) => {
                    eprintln!("  skip {}: {e}", p.display());
                    break;
                }
            }
        }
        let Some((out, scaled)) = fitted_pair else {
            continue;
        };

        // `keypoints_host` already returns pixels in the image THIS tool handed the extractor — it
        // applies its own internal 32 px fit on the way out — but it knows nothing about the
        // downscale above. `to_source` undoes that one, half-pixel term included, and putting the
        // inverse next to the forward map in the library is what keeps the two from drifting.
        let kpts: Vec<(f32, f32)> = out
            .keypoints_host()?
            .into_iter()
            .map(|(x, y)| scaled.to_source(x, y))
            .collect();
        let descs = out.descriptors_host()?;
        let dim = out.desc_dim();
        // Unreachable by the library's contract — `keypoints_host` yields exactly K pairs and
        // `descriptors_host` exactly K*dim. Kept as an ERROR rather than a skip because if it ever
        // fires the contract has changed underneath this tool, and `write_vrtk` would then slice
        // out of bounds and panic mid-directory, leaving a half-written feature set behind.
        if descs.len() < kpts.len() * dim {
            return Err(format!(
                "{}: extractor returned {} descriptor floats for {} keypoints at dim {dim} — \
                 the library's buffer contract has changed",
                p.display(),
                descs.len(),
                kpts.len()
            )
            .into());
        }
        write_vrtk(
            &out_dir.join(format!("{stem}.vrtk")),
            &kpts,
            &descs,
            dim,
            fp,
        )?;
        done += 1;
        if done.is_multiple_of(50) {
            eprintln!("  {done}/{}", names.len());
        }
    }
    eprintln!("aliked_batch: wrote {done} of {} files", names.len());
    // A silent zero-file success is read by the consumer as "every frame has no features", and it
    // builds a map from empty feature sets. Exit non-zero so the caller's `status.success()` check
    // is enough — it cannot see stderr.
    if names.is_empty() || done != names.len() {
        return Err(format!("wrote {done} of {} files", names.len()).into());
    }
    Ok(())
}
