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
//!   0  "VRM2"
//!   4  u32   n_pairs
//!   8  u32   fp        extractor-engine fingerprint, matching the `.vrtk` files (0 = unknown)
//!  12  u32   k         keypoints per frame the extractor emits
//!  16  per pair:  u32 a, u32 b, u32 n_matches, then n_matches * (u32 ia, u32 ib, f32 score)
//! ```
//!
//! The magic moved from `VRTM` to `VRM2` deliberately. `fp` and `k` let the consumer verify that
//! the indices in this file address the keypoints in the `.vrtk` files it holds, rather than
//! assuming it — run the two tools with different extractor engines (k1024 for one, k3072 for the
//! other) and the old format succeeded while emitting indices up to 3071 into files holding 1024
//! keypoints. Keeping the old magic and appending fields would have made a stale reader misparse
//! `fp` as its first pair record; a new magic makes an old reader stop.
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
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use kornia_imgproc::interpolation::InterpolationMode;
use kornia_io::functional::read_image_any_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::{
    engine_fingerprint, fit_to_engine, RaCoAliked, RaCoAlikedResult, FALLBACK_MAX_SIDE,
};

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
    // Written into the `.vrtm` header. The indices in that file address keypoints `aliked_batch`
    // wrote, and the ONLY thing making those the same keypoints is that both tools ran this same
    // extractor — an assumption the files could not previously state, let alone check.
    let fp = engine_fingerprint(ext_p);
    if raco.num_keypoints() != glue.num_keypoints() {
        return Err(format!(
            "extractor K={} but matcher K={} — they must be exported together",
            raco.num_keypoints(),
            glue.num_keypoints()
        )
        .into());
    }

    // Every non-blank line must parse. `filter_map(..ok()?)` was the previous form and it is a trap:
    // a comma-separated or otherwise malformed file yielded an EMPTY pair list, a well-formed
    // 8-byte `.vrtm`, and exit 0 — a complete no-op indistinguishable from a run with no work.
    let text = std::fs::read_to_string(pairs_txt)?;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (ln, l) in text.lines().enumerate() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let mut it = l.split_whitespace();
        let parsed = (|| -> Option<(usize, usize)> {
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })();
        let Some((a, b)) = parsed else {
            return Err(format!(
                "{}:{}: expected two whitespace-separated frame indices, got {l:?}",
                pairs_txt.display(),
                ln + 1
            )
            .into());
        };
        // A self-pair matches a frame against itself: K trivial correspondences with zero
        // parallax, which contribute nothing to the geometry and dilute every average computed
        // over the output.
        if a == b {
            return Err(format!(
                "{}:{}: self-pair {a} {b} — a frame cannot constrain its own pose",
                pairs_txt.display(),
                ln + 1
            )
            .into());
        }
        // Normalised so (a,b) and (b,a) dedup against each other. Emitting both would feed the
        // consumer the same correspondences twice and double-count those observations.
        pairs.push((a.min(b), a.max(b)));
    }
    // Sorted so the cache sees locality: consecutive pairs share their first index.
    pairs.sort_unstable();
    let before = pairs.len();
    pairs.dedup();
    if pairs.len() != before {
        eprintln!(
            "lightglue_batch: dropped {} duplicate pairs",
            before - pairs.len()
        );
    }
    eprintln!(
        "lightglue_batch: {} pairs, K={}, engine fp={fp:08x}",
        pairs.len(),
        raco.num_keypoints()
    );

    // `aliked_batch` accepts jpg/jpeg/png in any case; hardcoding `.jpg` here made this the one
    // tool that would silently find nothing in a directory the other one read fine — producing a
    // valid 0-pair `.vrtm` and exit 0.
    let path_for = |i: usize| -> PathBuf {
        let stem = format!("kf{i:04}");
        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"] {
            let p = img_dir.join(format!("{stem}.{ext}"));
            if p.exists() {
                return p;
            }
        }
        img_dir.join(format!("{stem}.jpg"))
    };
    let mut cache: HashMap<usize, RaCoAlikedResult> = HashMap::new();
    // Frames that failed once. Without this a bad frame is re-decoded, re-resized and re-uploaded on
    // every one of its ~17 pairs, and at full resolution that is a 1080x1920 decode each time.
    let mut dead: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut order: Vec<usize> = Vec::new();
    let mut matches = glue.alloc_result()?;

    // Streamed, not accumulated. The previous form built the whole run in a `Vec` — ~112 MB at 7800
    // pairs x ~1200 matches x 12 B, with doubling transients near 224 MB on a 7.4 GB board that is
    // usually also holding a reconstruction. The count is patched by seeking back at the end.
    let mut w = std::io::BufWriter::new(std::fs::File::create(out_p)?);
    w.write_all(b"VRM2")?;
    w.write_all(&0u32.to_le_bytes())?; // n_pairs, patched at the end
    w.write_all(&fp.to_le_bytes())?;
    w.write_all(&(raco.num_keypoints() as u32).to_le_bytes())?;
    let mut written = 0u32;
    let mut total_matches = 0usize;
    // Pairs the MATCHER rejected, as distinct from pairs skipped for a dead frame. The two mean
    // different things at exit and are counted apart for that reason.
    let mut failed_pairs = 0usize;

    // The loop is wrapped so that NOTHING inside it can return from `main` before the header count
    // is patched and the file is flushed. A transient failure at pair 7000 of 7800 used to
    // propagate straight out and discard the whole run's GPU work; the `?`s below now unwind only
    // as far as this closure.
    let mut run = || -> Result<(), vrt::BoxError> {
        for (n, &(a, b)) in pairs.iter().enumerate() {
            for idx in [a, b] {
                if dead.contains(&idx) {
                    continue;
                }
                if cache.contains_key(&idx) {
                    // Refresh recency. Without this `order` is insertion order and the eviction is
                    // FIFO despite its name — harmless at CACHE=24 against a ~13-frame window, but it
                    // silently stops being harmless the moment either number moves.
                    if let Some(pos) = order.iter().position(|&k| k == idx) {
                        let k = order.remove(pos);
                        order.push(k);
                    }
                    continue;
                }
                let p = path_for(idx);
                let Ok(src) = read_image_any_rgb8(&p) else {
                    dead.insert(idx);
                    continue;
                };
                // Natural size first, 640 only if the engine rejects it — the SAME `fit_to_engine`
                // and the same fallback constant `aliked_batch` uses, because the two must see the
                // same pixels to detect the same keypoints in the same order. Host fit for the same
                // reason `aliked_batch` uses one: the frame starts as a JPEG, so uploading the
                // fitted image beats uploading the raw one.
                let mut got = None;
                for (attempt, cap) in [src.cols().max(src.rows()), FALLBACK_MAX_SIDE]
                    .into_iter()
                    .enumerate()
                {
                    let Ok(scaled) = fit_to_engine(&src, cap, InterpolationMode::Bilinear) else {
                        continue;
                    };
                    let Ok(dev) = scaled.image.to_cuda(&stream) else {
                        continue;
                    };
                    let Ok(mut r) = raco.alloc_result() else {
                        continue;
                    };
                    match raco.submit(&dev, &mut r) {
                        // Counted, not propagated: a sync failure here costs this frame, and the pairs
                        // that need it, but the pairs already matched are still worth writing out.
                        Ok(()) => match stream.synchronize() {
                            Ok(()) => {
                                got = Some(r);
                                break;
                            }
                            Err(e) => {
                                eprintln!("  frame {idx}: sync failed: {e}");
                                break;
                            }
                        },
                        Err(vrt_raco_aliked::RaCoAlikedError::ShapeRejected { .. })
                            if attempt == 0 => {}
                        Err(_) => break,
                    }
                }
                let Some(r) = got else {
                    dead.insert(idx);
                    continue;
                };
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
            if let Err(e) = glue.submit(ra, rb, &mut matches) {
                if failed_pairs == 0 {
                    eprintln!("  pair {a}-{b} failed to match: {e}");
                }
                failed_pairs += 1;
                continue;
            }
            // Every readback below is counted rather than propagated, for the same reason as above: a
            // pair that cannot be read back is one lost edge, not grounds to discard the run.
            let readback = stream
                .synchronize()
                .map_err(|e| -> vrt::BoxError { e.into() })
                .and_then(|()| Ok((matches.pairs(0.0)?, matches.scores_host()?)));
            let (m, scores) = match readback {
                Ok(v) => v,
                Err(e) => {
                    if failed_pairs == 0 {
                        eprintln!("  pair {a}-{b} failed to read back: {e}");
                    }
                    failed_pairs += 1;
                    continue;
                }
            };
            // `pairs(min_score)` returns (index in A, index in B) into the extractor's own ordering,
            // which is exactly the ordering `aliked_batch` wrote — so these indices are directly usable
            // by the consumer without any remapping.
            //
            // The `?`s here are output-file writes. Those DO abort: if the destination is unwritable
            // there is nothing left to salvage, and the closure still lets the header be patched.
            w.write_all(&(a as u32).to_le_bytes())?;
            w.write_all(&(b as u32).to_le_bytes())?;
            w.write_all(&(m.len() as u32).to_le_bytes())?;
            for (ia, ib) in &m {
                w.write_all(&(*ia as u32).to_le_bytes())?;
                w.write_all(&(*ib as u32).to_le_bytes())?;
                let s = scores.get(*ia).copied().unwrap_or(0.0);
                w.write_all(&s.to_le_bytes())?;
            }
            written += 1;
            total_matches += m.len();
            if n.is_multiple_of(200) {
                eprintln!(
                    "  {n}/{} pairs, {total_matches} matches so far",
                    pairs.len()
                );
            }
        }
        Ok(())
    };
    let loop_result = run();

    // Finish the file BEFORE reporting any error from the loop: whatever pairs were matched are
    // real work, and a caller that reruns should not have to redo them to find that out.
    let finish = || -> Result<u64, vrt::BoxError> {
        let mut f = w.into_inner()?;
        f.seek(SeekFrom::Start(4))?;
        f.write_all(&written.to_le_bytes())?;
        f.seek(SeekFrom::End(0))?;
        let len = f.stream_position()?;
        f.sync_all()?;
        Ok(len)
    };
    let bytes = finish();
    eprintln!(
        "lightglue_batch: {written} of {} pairs matched, {total_matches} correspondences, \
         {} dead frames, {failed_pairs} failed pairs, {} bytes",
        pairs.len(),
        dead.len(),
        match &bytes {
            Ok(n) => n.to_string(),
            Err(e) => format!("UNWRITTEN ({e})"),
        }
    );
    // The file is the deliverable; if it could not be completed the run failed regardless of how
    // well the matching went, and that takes precedence over the loop's own error.
    bytes?;
    loop_result?;
    if pairs.is_empty() {
        return Err("no pairs to match".into());
    }
    // A DEAD FRAME is not a degraded result, it is a disagreement. `aliked_batch` exits non-zero
    // unless it wrote a `.vrtk` for every frame, so if the feature pass succeeded then every frame
    // in this directory has keypoints — and a frame this tool could not extract means the two tools
    // no longer share a frame set. The indices in this file address the OTHER tool's keypoint
    // ordering, so there is no safe partial answer to give: report which frames and stop.
    if !dead.is_empty() {
        let mut d: Vec<_> = dead.iter().copied().collect();
        d.sort_unstable();
        d.truncate(20);
        return Err(format!(
            "{} frames could not be extracted here but have keypoints from the feature pass \
             (first: {d:?}) — the two tools disagree on the frame set and the match indices \
             would address the wrong keypoints",
            dead.len()
        )
        .into());
    }
    // A FAILED PAIR is a degraded result: the map loses one graph edge and carries on. One transient
    // failure should not discard a 40-minute build, so this tolerates a small fraction and fails on
    // anything systemic. The file is already on disk either way, so a caller that disagrees with
    // this threshold can still use what was produced.
    const TOLERATED_PAIR_FAILURES: f64 = 0.01;
    if failed_pairs as f64 > TOLERATED_PAIR_FAILURES * pairs.len() as f64 {
        return Err(format!(
            "{failed_pairs} of {} pairs failed to match, over the {:.0}% this tolerates — the \
             matcher is failing systemically, not transiently",
            pairs.len(),
            TOLERATED_PAIR_FAILURES * 100.0
        )
        .into());
    }
    if failed_pairs > 0 {
        eprintln!("lightglue_batch: WARNING — {failed_pairs} pairs lost, map loses those edges");
    }
    Ok(())
}
