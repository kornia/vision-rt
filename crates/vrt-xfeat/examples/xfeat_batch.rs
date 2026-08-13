//! Batch XFeat extraction over a directory of frames, written out as `.vrtk` files.
//!
//! The out-of-process bridge for an XFeat feature backend, the sibling of
//! `vrt-raco-aliked`'s `aliked_batch`. It exists for the same reason: the consumer pins a
//! different kornia than this workspace, so the two cannot be linked and meet on files instead.
//!
//! Usage:
//!   xfeat_batch <xfeat.engine> <img_dir> <out_dir>
//!
//! Reads every frame in `<img_dir>` and writes `<out_dir>/<stem>.vrtk`, preserving the caller's
//! naming — for a `kfNNNN.jpg` set the index IS the keyframe identity on the other side, so a gap
//! must stay a gap rather than shifting everything after it.
//!
//! ## `.vrtk` layout (little-endian throughout)
//!
//! ```text
//!   0  "VRTK"
//!   4  u32  n     keypoints in this file
//!   8  u32  dim   descriptor dimension (64 for XFeat)
//!  12  u32  fp    extractor-engine fingerprint (0 = unknown)
//!  16  n * (2 + dim) f32   x, y, then dim descriptor values, per keypoint
//! ```
//!
//! Byte-identical to what `aliked_batch` writes, only with `dim = 64`, so one reader serves both
//! backends.
//!
//! ## Ordering
//!
//! Written in the extractor's native order, which for XFeat is GPU atomic-append order — not a
//! quality ranking. Nothing here sorts or truncates: a consumer that caps the count should
//! subsample with an even stride, because truncating a non-ranked list crops rather than thins.
//! `top_k` and the NMS threshold are the only quantity knobs, and they are applied on the GPU
//! where the scores are.

use std::io::Write;
use std::path::Path;

use kornia_io::functional::read_image_any_rgb8;
use vrt::engine_fingerprint;
use vrt_xfeat::{XFeat, XFeatParams};

/// Matches `xfeat_detect` / `xfeat_bench` in this crate rather than inventing a third default.
/// XFeat's top-K is a cap, not a target — a frame yielding fewer local maxima above `THRESHOLD`
/// simply returns fewer, so raising the cap costs device memory (`top_k × 66` floats) and nothing
/// else.
const TOP_K: usize = 4096;
const THRESHOLD: f32 = 0.05;

/// Frame extensions, compared **case-insensitively**, matching `aliked_batch`.
///
/// This does NOT make uppercase frames decode: `read_image_any_rgb8` matches the extension
/// case-sensitively itself, so a `.PNG` is still rejected — measured, on a staged directory of
/// 7 `.png` plus 1 `.PNG`. What it changes is which failure you get. Case-SENSITIVE here (what
/// the tool this was ported from did) drops those files before they are counted, so an
/// all-uppercase directory is 0 frames found, 0 written; case-INSENSITIVE counts them, names each
/// one on stderr, and trips the `done != names.len()` exit below. The consumer cannot see stderr —
/// it checks the exit status — and it reads a silent zero-file success as "every frame has no
/// features", then builds a map from empty feature sets.
const FRAME_EXTS: [&str; 3] = ["jpg", "jpeg", "png"];

fn is_frame_file(p: &Path) -> bool {
    p.extension()
        .and_then(|x| x.to_str())
        .map(|x| x.to_ascii_lowercase())
        .is_some_and(|x| FRAME_EXTS.contains(&x.as_str()))
}

fn write_vrtk(
    path: &Path,
    kpts: &[f32],
    descs: &[f32],
    dim: usize,
    fp: u32,
) -> std::io::Result<()> {
    let n = kpts.len() / 2;
    let mut buf: Vec<u8> = Vec::with_capacity(16 + n * (2 + dim) * 4);
    buf.extend_from_slice(b"VRTK");
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&fp.to_le_bytes());
    for i in 0..n {
        buf.extend_from_slice(&kpts[i * 2].to_le_bytes());
        buf.extend_from_slice(&kpts[i * 2 + 1].to_le_bytes());
        for v in &descs[i * dim..(i + 1) * dim] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(path)?.write_all(&buf)
}

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: xfeat_batch <xfeat.engine> <img_dir> <out_dir>");
        std::process::exit(1);
    }
    let (engine, img_dir, out_dir) = (&args[1], Path::new(&args[2]), Path::new(&args[3]));
    std::fs::create_dir_all(out_dir)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    // Prebuilt `.engine` only, deliberately: the fingerprint below identifies the file on disk, so
    // a path that might instead be an ONNX resolved through the engine cache would fingerprint a
    // file this tool never names.
    let mut xfeat =
        XFeat::from_engine_file(engine, stream.clone(), XFeatParams::new(TOP_K, THRESHOLD))?;
    // Stamped into every `.vrtk`. Word 12 was a zero reserved field; it now says which engine
    // produced the keypoints, so a consumer holding indices into these files can check they came
    // from the extractor it thinks they did instead of assuming it. Readers that ignored the
    // reserved word are unaffected; readers that checked it for zero were checking nothing.
    let fp = engine_fingerprint(engine);

    let mut names: Vec<_> = std::fs::read_dir(img_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_frame_file(p))
        .collect();
    names.sort();
    eprintln!(
        "xfeat_batch: {} frames, top_k={TOP_K}, thr={THRESHOLD}, dim={}, engine fp={fp:08x}",
        names.len(),
        vrt_xfeat::postprocess::XFEAT_DESC_DIM
    );

    // One reusable output buffer for the whole run: `alloc_result` allocates `top_k × 66` device
    // floats, and this board has 7.4 GB shared with whatever consumes the map.
    let mut out = xfeat.alloc_result()?;
    let mut done = 0usize;
    for p in &names {
        let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let src = match read_image_any_rgb8(p) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  skip {}: {e}", p.display());
                continue;
            }
        };
        // No resize here. `XFeat::submit` fits each frame to its own floor-of-32 size internally
        // and `kpts_to_host` scales the keypoints back, so keypoints are already in SOURCE pixels.
        // The tool this was ported from letterboxed every frame into a fixed 1280×736 and inverted
        // that by hand; doing so now would rescale twice and put every coordinate in a frame the
        // consumer has no knowledge of.
        let dev = src.to_cuda(&stream)?;
        xfeat.submit(&dev, &mut out)?;
        // Sync per frame: `count()` reads a pinned scalar that is only valid after it, and both
        // downloads below are sized from it.
        stream.synchronize()?;

        let kpts = out.kpts_to_host()?;
        let descs = out.descs_to_host()?;
        let dim = out.desc_dim();
        // Unreachable by the library's contract — the two downloads are both sized from the same
        // `count()`. Kept as an ERROR rather than a skip because if it ever fires the contract has
        // changed underneath this tool, and `write_vrtk` would then slice out of bounds and panic
        // mid-directory, leaving a half-written feature set behind.
        if descs.len() < (kpts.len() / 2) * dim {
            return Err(format!(
                "{}: extractor returned {} descriptor floats for {} keypoints at dim {dim} — \
                 the library's buffer contract has changed",
                p.display(),
                descs.len(),
                kpts.len() / 2
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
    eprintln!("xfeat_batch: wrote {done} of {} files", names.len());
    // A silent zero-file success is read by the consumer as "every frame has no features", and it
    // builds a map from empty feature sets. Exit non-zero so the caller's `status.success()` check
    // is enough — it cannot see stderr.
    if names.is_empty() || done != names.len() {
        return Err(format!("wrote {done} of {} files", names.len()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("vrt-xfeat-batch-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn u32_at(raw: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(raw[off..off + 4].try_into().unwrap())
    }
    fn f32_at(raw: &[u8], off: usize) -> f32 {
        f32::from_le_bytes(raw[off..off + 4].try_into().unwrap())
    }

    /// Fixes the header + row geometry against the offsets the consumer decodes with, rather than
    /// against this file's own writer: `n` at 4, `dim` at 8, `fp` at 12, row `k` at
    /// `16 + k*(2+dim)*4`. Two keypoints with distinct coordinates and distinct descriptor rows, so
    /// a transposed row or a swapped x/y is visible rather than symmetric.
    #[test]
    fn vrtk_layout_matches_the_consumer_decode() {
        let dim = 4usize;
        let kpts = [10.0f32, 20.0, 30.0, 40.0];
        let descs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = tmp_dir("layout").join("a.vrtk");
        write_vrtk(&p, &kpts, &descs, dim, 0xdead_beef).unwrap();
        let raw = std::fs::read(&p).unwrap();

        assert_eq!(&raw[0..4], b"VRTK");
        assert_eq!(
            u32_at(&raw, 4),
            2,
            "n is the keypoint count, not the f32 count"
        );
        assert_eq!(u32_at(&raw, 8), dim as u32);
        assert_eq!(u32_at(&raw, 12), 0xdead_beef);
        assert_eq!(raw.len(), 16 + 2 * (2 + dim) * 4);

        for k in 0..2 {
            let base = 16 + k * (2 + dim) * 4;
            assert_eq!(f32_at(&raw, base), kpts[k * 2], "x of row {k}");
            assert_eq!(f32_at(&raw, base + 4), kpts[k * 2 + 1], "y of row {k}");
            for d in 0..dim {
                assert_eq!(
                    f32_at(&raw, base + 8 + d * 4),
                    descs[k * dim + d],
                    "desc[{d}] of row {k}"
                );
            }
        }
    }

    /// Word 12 must carry THIS engine's fingerprint. The control is a second engine file: asserting
    /// only "word 12 is non-zero" would pass on a hardcoded constant, and asserting only
    /// "word 12 == engine_fingerprint(path)" would pass on the old `0` if the fingerprint of that
    /// path happened to be 0 — which is exactly the "unknown" value. So: non-zero, equal to its own
    /// engine's fingerprint, and different from the other engine's.
    #[test]
    fn word_12_identifies_the_engine_that_produced_the_file() {
        let d = tmp_dir("fp");
        let (e1, e2) = (d.join("k2048.engine"), d.join("k4096.engine"));
        std::fs::write(&e1, vec![0xa5u8; 9000]).unwrap();
        std::fs::write(&e2, vec![0x5au8; 9000]).unwrap();
        let (fp1, fp2) = (engine_fingerprint(&e1), engine_fingerprint(&e2));
        assert_ne!(fp1, 0, "a readable engine must not fingerprint as unknown");
        assert_ne!(
            fp1, fp2,
            "two distinct engines must not share a fingerprint"
        );

        let out = d.join("kf0000.vrtk");
        write_vrtk(&out, &[1.0, 2.0], &[0.5], 1, fp1).unwrap();
        let raw = std::fs::read(&out).unwrap();
        assert_eq!(u32_at(&raw, 12), fp1);
        assert_ne!(u32_at(&raw, 12), fp2);
    }

    /// The extension filter is case-insensitive and still rejects non-frames. `.JPG` is the case
    /// that used to be dropped silently; it is COUNTED here so the run fails loudly instead (the
    /// image reader still refuses it — see [`FRAME_EXTS`]).
    #[test]
    fn frame_filter_is_case_insensitive_and_still_rejects_non_frames() {
        for ok in ["a.jpg", "a.JPG", "a.Jpeg", "a.PNG", "a.png"] {
            assert!(is_frame_file(Path::new(ok)), "{ok} should be a frame");
        }
        for no in ["a.txt", "a.vrtk", "a.jpg.bak", "a", "jpg"] {
            assert!(!is_frame_file(Path::new(no)), "{no} should not be a frame");
        }
    }

    /// `write_vrtk` must not reorder: the consumer subsamples with an even stride and relies on the
    /// file preserving the extractor's order. Descending x, which any sort-by-x would flip.
    #[test]
    fn rows_keep_the_order_they_were_given() {
        let kpts = [9.0f32, 0.0, 5.0, 0.0, 1.0, 0.0];
        let descs = [3.0f32, 2.0, 1.0];
        let p = tmp_dir("order").join("o.vrtk");
        write_vrtk(&p, &kpts, &descs, 1, 1).unwrap();
        let raw = std::fs::read(&p).unwrap();
        let xs: Vec<f32> = (0..3).map(|k| f32_at(&raw, 16 + k * 3 * 4)).collect();
        assert_eq!(xs, vec![9.0, 5.0, 1.0]);
    }
}
