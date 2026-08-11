//! Batch RaCo-ALIKED + LightGlue matching over a PAIR LIST, for flux-map's `aliked` backend.
//!
//! The matcher half of the out-of-process bridge. `aliked_batch` answers "what is in this frame";
//! this answers "which keypoints in frame A are the same points as in frame B", which is the half
//! that carries most of LightGlue's advantage — ALIKED descriptors through a plain nearest-neighbour
//! ratio test give up most of what the pipeline is for.
//!
//! Usage:
//!   lightglue_batch <extractor.engine> <matcher.engine> <img_dir> <pairs.txt> <out.vrtm>
//!
//! `pairs.txt` is one `A B` per line, keyframe indices matching `<img_dir>/kfNNNN.jpg`.
//!
//! ## Output: one `.vrtm` for the whole run
//!
//! ```text
//!   0  "VRTM"
//!   4  u32   n_pairs
//!   8  per pair:  u32 a, u32 b, u32 n_matches, then n_matches * (u32 ia, u32 ib, f32 score)
//! ```
//!
//! One file rather than one per pair: a 459-keyframe build schedules ~7800 pairs, and 7800 tiny
//! files is a filesystem tax with no benefit — the consumer reads the whole thing once.
//!
//! ## Why an LRU of extraction results
//!
//! Each keyframe appears in ~17 pairs (a 12-window plus the long-span ladder). Extracting per pair
//! would run the backbone ~34x per frame; caching every frame instead would hold
//! `n * K * 128 * 4` bytes of device memory — 700 MB at 459 frames, on a board that is usually also
//! holding a reconstruction. A small LRU over pairs sorted by first index gets the window hits
//! (which are the bulk) while keeping the resident set bounded.
//!
//! Keypoints are NOT written here — the consumer already has them from `aliked_batch`, and the
//! indices in this file refer to that ordering. Emitting them again would double the file and
//! create a second source of truth for the same coordinates.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use kornia_imgproc::interpolation::InterpolationMode;
use kornia_io::functional::read_image_any_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::{fit_to_engine, RaCoAliked, RaCoAlikedResult, FALLBACK_MAX_SIDE};

/// Extraction results kept on device. 24 covers a 12-wide window on both sides of the cursor.
const CACHE: usize = 24;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "Usage: lightglue_batch <extractor.engine> <matcher.engine> <img_dir> <pairs.txt> <out.vrtm>"
        );
        std::process::exit(1);
    }
    let (ext_p, mat_p) = (&args[1], &args[2]);
    let img_dir = Path::new(&args[3]);
    let pairs_txt = Path::new(&args[4]);
    let out_p = Path::new(&args[5]);

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(ext_p, stream.clone())?;
    let mut glue = LightGlue::from_engine_file(mat_p, stream.clone())?;
    if raco.num_keypoints() != glue.num_keypoints() {
        return Err(format!(
            "extractor K={} but matcher K={} — they must be exported together",
            raco.num_keypoints(),
            glue.num_keypoints()
        )
        .into());
    }

    let mut pairs: Vec<(usize, usize)> = std::fs::read_to_string(pairs_txt)?
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect();
    // Sorted so the LRU sees locality: consecutive pairs share their first index.
    pairs.sort_unstable();
    eprintln!("lightglue_batch: {} pairs, K={}", pairs.len(), raco.num_keypoints());

    let path_for = |i: usize| -> PathBuf { img_dir.join(format!("kf{i:04}.jpg")) };
    let mut cache: HashMap<usize, RaCoAlikedResult> = HashMap::new();
    // Frames that failed once. Without this a bad frame is re-decoded, re-resized and re-uploaded on
    // every one of its ~17 pairs, and at full resolution that is a 1080x1920 decode each time.
    let mut dead: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut order: Vec<usize> = Vec::new();
    let mut matches = glue.alloc_result()?;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"VRTM");
    out.extend_from_slice(&0u32.to_le_bytes()); // patched with the real count at the end
    let mut written = 0u32;
    let mut total_matches = 0usize;

    for (n, &(a, b)) in pairs.iter().enumerate() {
        for idx in [a, b] {
            if cache.contains_key(&idx) || dead.contains(&idx) {
                continue;
            }
            let p = path_for(idx);
            let Ok(src) = read_image_any_rgb8(&p) else { dead.insert(idx); continue };
            // Natural size first, 640 only if the engine rejects it — the SAME `fit_to_engine` and
            // the same fallback constant `aliked_batch` uses, because the two must see the same
            // pixels to detect the same keypoints in the same order.
            let mut got = None;
            for (attempt, cap) in [src.cols().max(src.rows()), FALLBACK_MAX_SIDE].into_iter().enumerate() {
                let Ok(scaled) = fit_to_engine(&src, cap, InterpolationMode::Bilinear) else { continue };
                let Ok(dev) = scaled.image.to_cuda(&stream) else { continue };
                let Ok(mut r) = raco.alloc_result() else { continue };
                match raco.submit(&dev, &mut r) {
                    Ok(()) => { stream.synchronize()?; got = Some(r); break; }
                    Err(vrt_raco_aliked::RaCoAlikedError::ShapeRejected { .. }) if attempt == 0 => {}
                    Err(_) => break,
                }
            }
            let Some(r) = got else { dead.insert(idx); continue };
            cache.insert(idx, r);
            order.push(idx);
            // Evict the least recently inserted that is not one of the two in flight.
            while order.len() > CACHE {
                if let Some(pos) = order.iter().position(|&k| k != a && k != b) {
                    let k = order.remove(pos);
                    cache.remove(&k);
                } else {
                    break;
                }
            }
        }
        let (Some(ra), Some(rb)) = (cache.get(&a), cache.get(&b)) else {
            continue;
        };
        if glue.submit(ra, rb, &mut matches).is_err() {
            continue;
        }
        stream.synchronize()?;
        // `pairs(min_score)` returns (index in A, index in B) into the extractor's own ordering,
        // which is exactly the ordering `aliked_batch` wrote — so these indices are directly usable
        // by the consumer without any remapping.
        let m = matches.pairs(0.0)?;
        let scores = matches.scores_host()?;
        out.extend_from_slice(&(a as u32).to_le_bytes());
        out.extend_from_slice(&(b as u32).to_le_bytes());
        out.extend_from_slice(&(m.len() as u32).to_le_bytes());
        for (ia, ib) in &m {
            out.extend_from_slice(&(*ia as u32).to_le_bytes());
            out.extend_from_slice(&(*ib as u32).to_le_bytes());
            let s = scores.get(*ia).copied().unwrap_or(0.0);
            out.extend_from_slice(&s.to_le_bytes());
        }
        written += 1;
        total_matches += m.len();
        if n % 200 == 0 {
            eprintln!("  {n}/{} pairs, {total_matches} matches so far", pairs.len());
        }
    }
    out[4..8].copy_from_slice(&written.to_le_bytes());
    // Written UNCONDITIONALLY before any error return: a transient failure at pair 7000 of 7800
    // used to propagate out of main and discard the whole run's GPU work.
    std::fs::File::create(out_p)?.write_all(&out)?;
    eprintln!(
        "lightglue_batch: {written} of {} pairs matched, {total_matches} correspondences, {} bytes",
        pairs.len(),
        out.len()
    );
    if pairs.is_empty() || written == 0 {
        return Err(format!("matched {written} of {} pairs", pairs.len()).into());
    }
    Ok(())
}
