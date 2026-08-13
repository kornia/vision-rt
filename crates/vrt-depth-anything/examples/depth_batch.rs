//! Batch metric depth: run Depth Anything V2 over a directory of images and write one raw f32
//! depth raster per input, as `.vrtd`.
//!
//! The depth leg of the out-of-process bridge, alongside `aliked_batch` / `lightglue_batch` /
//! `dino_batch`. It exists for the same reason they do: the consumer (flux-map's densification
//! and depth-prior passes) pins a different kornia than this crate, so the two cannot be linked
//! and meet on files instead.
//!
//! Usage:
//!   depth_batch <depth.engine> <img_dir> <out_dir>
//!
//! Reads `<img_dir>/<stem>.{jpg,jpeg,png}` and writes `<out_dir>/<stem>.vrtd`, preserving the
//! caller's names — the stem IS the frame identity on the other side, so a gap must stay a gap
//! rather than shifting everything after it.
//!
//! ## `.vrtd` layout (little-endian throughout)
//!
//! ```text
//!   0  "VRTD"
//!   4  u32  width     depth-map columns
//!   8  u32  height    depth-map rows
//!  12  u32  reserved  0
//!  16  width * height f32   row-major metric METRES
//! ```
//!
//! Deliberately primitive so the consumer needs no codec and no image crate: it mmaps or reads
//! the file and indexes it. The raster is at the MODEL's grid, stretched from the source aspect —
//! the consumer samples it with normalized coordinates, so no letterbox bookkeeping crosses the
//! boundary and the source resolution never has to be agreed on.
//!
//! Depth is trusted STATISTICALLY downstream (medians, RANSAC consensus, multi-view votes); it
//! hallucinates at occlusion boundaries and on mirrors. Nothing here filters it — a tool that
//! quietly dropped or smoothed values would remove the evidence the consumer's own robust
//! statistics are built to weigh.

use std::io::Write;
use std::path::{Path, PathBuf};

use kornia_io::functional::read_image_any_rgb8;
use vrt_depth_anything::DepthAnything;

/// Frame extensions accepted from the input directory, matched case-INSENSITIVELY.
///
/// Mirrors `vrt_raco_aliked::FRAME_EXTS`, deliberately duplicated rather than depended on: this
/// crate has no reason to link an extractor, and the list is one line. A case-SENSITIVE match
/// here is a measured failure mode — a directory of `.JPG` matched nothing, and the tool then
/// reported success over zero frames.
const FRAME_EXTS: [&str; 3] = ["jpg", "jpeg", "png"];

fn is_frame(p: &Path) -> bool {
    p.extension()
        .and_then(|x| x.to_str())
        .map(|x| x.to_ascii_lowercase())
        .is_some_and(|x| FRAME_EXTS.contains(&x.as_str()))
}

/// Sorted, extension-filtered frame paths.
fn frame_paths(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_frame(p))
        .collect();
    v.sort();
    Ok(v)
}

/// Serialise one depth raster into the `.vrtd` byte layout above.
///
/// The length check is the whole reason this is a function rather than five `write_all` calls: a
/// header declaring `w*h` over a shorter buffer produces a file the consumer reads past the end
/// of, or — worse, since the reader bounds-checks — silently rejects as truncated, one frame at a
/// time, in a pass that treats a missing raster as merely degraded input.
fn encode_vrtd(w: usize, h: usize, depth: &[f32]) -> Result<Vec<u8>, String> {
    if depth.len() != w * h {
        return Err(format!(
            "depth map has {} values but its grid is {w}x{h} ({}) — the library's buffer contract \
             has changed",
            depth.len(),
            w * h
        ));
    }
    let mut buf: Vec<u8> = Vec::with_capacity(16 + depth.len() * 4);
    buf.extend_from_slice(b"VRTD");
    buf.extend_from_slice(&(w as u32).to_le_bytes());
    buf.extend_from_slice(&(h as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    for v in depth {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    Ok(buf)
}

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: depth_batch <depth.engine> <img_dir> <out_dir>");
        std::process::exit(1);
    }
    let (engine, img_dir, out_dir) = (&args[1], Path::new(&args[2]), Path::new(&args[3]));
    std::fs::create_dir_all(out_dir)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut depth = DepthAnything::from_engine_file(engine, stream.clone())?;

    let names = frame_paths(img_dir)?;
    let (mw, mh) = depth.map_size();
    eprintln!(
        "depth_batch: {} frames from {}, {mw}x{mh} metric rasters",
        names.len(),
        img_dir.display()
    );
    if names.is_empty() {
        return Err(format!(
            "no frames in {} matching {FRAME_EXTS:?} (case-insensitive)",
            img_dir.display()
        )
        .into());
    }

    // One frame at a time, synced per frame. The API is async and several frames could be in
    // flight, but each result owns a full-grid device buffer and this runs on a 7.4 GB board that
    // is usually also holding a reconstruction — throughput here is not what the caller waits on,
    // and an OOM in the depth pass would be.
    let mut z = depth.alloc_result()?;
    let mut done = 0usize;
    for p in &names {
        let stem = match p.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => {
                eprintln!("  skip {}: filename is not valid UTF-8", p.display());
                continue;
            }
        };
        let src = match read_image_any_rgb8(p) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  skip {}: {e}", p.display());
                continue;
            }
        };
        let dev = src.to_cuda(&stream)?;
        depth.submit(&dev, &mut z)?; // enqueue, no sync
        stream.synchronize()?; // the one sync
        let dmap = z.depth_host()?;

        // The RESULT's own grid, not the model's. They are the same today, but the header must
        // describe the buffer it is wrapping — a header taken from a different source is how a
        // raster comes to declare a size it does not have.
        let (w, h) = z.map_size();
        let bytes = encode_vrtd(w, h, dmap.as_slice())?;
        std::fs::File::create(out_dir.join(format!("{stem}.vrtd")))?.write_all(&bytes)?;
        done += 1;
        if done.is_multiple_of(50) {
            eprintln!("  {done}/{}", names.len());
        }
    }

    eprintln!("depth_batch: wrote {done} of {} rasters", names.len());
    // A partial run is read by the consumer as "those frames have no depth", which it treats as
    // degraded-but-fine and reconstructs up-to-scale from. Exit non-zero so the caller's
    // `status.success()` check is enough — it cannot see stderr.
    if done != names.len() {
        return Err(format!("wrote {done} of {} rasters", names.len()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le_u32(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    /// The measured failure: a directory of `.JPG` matched nothing under a case-sensitive filter,
    /// and the tool then reported success over zero frames. Asserted against that predicate, not
    /// merely against "the filter accepts something" — a no-op change would pass the latter.
    #[test]
    fn frame_filter_is_case_insensitive_where_the_old_one_was_not() {
        let case_sensitive = |p: &Path| {
            p.extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| FRAME_EXTS.contains(&x))
        };
        let mixed = [
            Path::new("a/kf0000.JPG"),
            Path::new("a/kf0001.Jpeg"),
            Path::new("a/kf0002.PNG"),
        ];
        assert_eq!(mixed.iter().filter(|p| case_sensitive(p)).count(), 0);
        assert_eq!(mixed.iter().filter(|p| is_frame(p)).count(), 3);
        for p in [
            "a/kf0000.vrtd",
            "a/notes.txt",
            "a/kf0000",
            "a/kf0000.jpg.bak",
        ] {
            assert!(!is_frame(Path::new(p)), "{p}");
        }
    }

    /// Header and payload describe the same raster, and the payload starts at 16 — the offsets
    /// the consumer's reader hard-codes.
    #[test]
    fn header_describes_the_raster_it_wraps() {
        let d: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let b = encode_vrtd(3, 2, &d).unwrap();
        assert_eq!(&b[0..4], b"VRTD");
        assert_eq!(le_u32(&b, 4), 3);
        assert_eq!(le_u32(&b, 8), 2);
        assert_eq!(le_u32(&b, 12), 0);
        assert_eq!(b.len(), 16 + 6 * 4);
        let vals: Vec<f32> = b[16..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, d);
    }

    /// A raster whose length disagrees with its declared grid must not be written. Both
    /// directions: short (the reader rejects it as truncated) and long (the tail is silently
    /// dropped, so the file looks fine and is wrong).
    #[test]
    fn grid_mismatch_is_an_error_in_both_directions() {
        let short: Vec<f32> = vec![0.0; 5];
        let long: Vec<f32> = vec![0.0; 7];
        assert!(encode_vrtd(3, 2, &short).is_err());
        assert!(encode_vrtd(3, 2, &long).is_err());
        // Control: the exact-length buffer between them succeeds, so the test is not passing
        // because `encode_vrtd` rejects everything.
        assert!(encode_vrtd(3, 2, &vec![0.0; 6]).is_ok());
    }

    /// Metres are written verbatim, NaN included. The consumer's robust statistics are what
    /// decide whether a value is usable; a tool that scrubbed them here would be making that
    /// call on evidence it cannot see.
    #[test]
    fn non_finite_depths_survive_the_round_trip() {
        let d = vec![f32::NAN, 0.0, f32::INFINITY, -1.5];
        let b = encode_vrtd(4, 1, &d).unwrap();
        let vals: Vec<f32> = b[16..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!(vals[0].is_nan());
        assert_eq!(vals[1], 0.0);
        assert!(vals[2].is_infinite() && vals[2] > 0.0);
        assert_eq!(vals[3], -1.5);
    }
}
