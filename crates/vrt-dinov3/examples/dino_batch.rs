//! Batch DINOv3 global descriptors: one L2-normed CLS vector per image in a directory, written
//! as a single flat file for an out-of-process retrieval stage.
//!
//! The third leg of the out-of-process bridge, alongside `aliked_batch` and `lightglue_batch`.
//! It exists for the same reason they do: the consumer (flux-map's retrieval pass) pins a
//! different kornia than this crate, so the two cannot be linked and meet on files instead.
//!
//! Usage:
//!   dino_batch <dinov3.engine> <img_dir> <out.vrtb>
//!
//! ## `.vrtb` layout (little-endian throughout)
//!
//! ```text
//!   0  "VRTB"
//!   4  u32  n          descriptors in this file
//!   8  u32  dim        descriptor dimension (384 for ViT-S/16)
//!  12  u32  names_len  bytes of the trailing name table (0 = absent)
//!  16  n * dim f32     row-major, one descriptor per frame, in sorted input order
//!      names_len bytes n NUL-terminated UTF-8 file stems, in the same row order
//! ```
//!
//! ## Why a name table
//!
//! Word 12 was a zero reserved field and the rows were addressed by position alone: row `i` was
//! whatever the `i`-th sorted filename happened to be. That is only the caller's frame `i` when
//! the caller's inputs are gapless, and the consumer does not guarantee that — flux-map writes
//! `kfNNNN.jpg` for each keyframe *it has a thumbnail for* and then reads row `i` as keyframe
//! `i`, so one missing thumbnail shifts every descriptor after it onto the wrong keyframe. The
//! file has no way to say so, and the resulting map is wrong in a way that looks like poor
//! retrieval rather than like a bug.
//!
//! `aliked_batch` avoids this by writing one file per input, named by stem, so a gap stays a gap.
//! That is the wrong shape here — the consumer wants every descriptor at once to build a bank,
//! and a 459-frame run would be 459 files of 1.5 kB — so the identity travels inside the file
//! instead. The descriptor block stays at offset 16 with the same layout, so a reader that only
//! wants descriptors is unaffected; a reader that checked word 12 for zero was checking nothing.
//!
//! ## Ordering
//!
//! Sorted input order, nothing dropped and nothing reordered. A frame that fails to decode is an
//! ERROR, not a skipped row: silently emitting `n-1` rows is exactly the misalignment above.

use std::io::Write;
use std::path::{Path, PathBuf};

use kornia_io::functional::read_image_any_rgb8;
use vrt_dinov3::DinoV3;

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

/// Serialise the descriptor rows and their frame stems into the `.vrtb` byte layout above.
///
/// Errors rather than truncating on a ragged row or a stem containing a NUL. Both are unreachable
/// by the library's contract (`descriptor_host` yields exactly `dim` floats, and a path component
/// cannot contain NUL on the platforms this runs on) — but if either ever fires, writing anyway
/// produces a well-formed file whose rows no longer line up with its names, which is the failure
/// this format exists to prevent.
fn encode_vrtb(rows: &[Vec<f32>], stems: &[String], dim: usize) -> Result<Vec<u8>, String> {
    if rows.len() != stems.len() {
        return Err(format!(
            "{} descriptor rows for {} frame names",
            rows.len(),
            stems.len()
        ));
    }
    let mut names: Vec<u8> = Vec::new();
    for s in stems {
        if s.as_bytes().contains(&0) {
            return Err(format!("frame name {s:?} contains a NUL byte"));
        }
        names.extend_from_slice(s.as_bytes());
        names.push(0);
    }

    let mut buf: Vec<u8> = Vec::with_capacity(16 + rows.len() * dim * 4 + names.len());
    buf.extend_from_slice(b"VRTB");
    buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for (row, stem) in rows.iter().zip(stems) {
        if row.len() != dim {
            return Err(format!(
                "{stem}: descriptor has {} values, expected dim {dim} — the library's buffer \
                 contract has changed",
                row.len()
            ));
        }
        for v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf.extend_from_slice(&names);
    Ok(buf)
}

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: dino_batch <dinov3.engine> <img_dir> <out.vrtb>");
        std::process::exit(1);
    }
    let (engine, img_dir, out_p) = (&args[1], Path::new(&args[2]), Path::new(&args[3]));

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut dino = DinoV3::from_engine_file(engine, stream.clone())?;
    let dim = dino.dim();

    let names = frame_paths(img_dir)?;
    eprintln!(
        "dino_batch: {} frames from {}, dim={dim}",
        names.len(),
        img_dir.display()
    );
    // A silent zero-frame success writes a valid 16-byte header and exits 0, which the consumer
    // reads as "retrieval found nothing" and proceeds without any loop closure at all. The caller
    // checks the exit status and cannot see stderr, so this has to be the status.
    if names.is_empty() {
        return Err(format!(
            "no frames in {} matching {FRAME_EXTS:?} (case-insensitive)",
            img_dir.display()
        )
        .into());
    }

    // One reused result buffer: each frame is read back to host before the next submit, so a
    // second buffer would only let frames overlap, and on a 7.4 GB board that is usually also
    // holding a reconstruction the throughput is not what the caller is waiting on.
    let mut r = dino.alloc_result()?;
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(names.len());
    let mut stems: Vec<String> = Vec::with_capacity(names.len());
    for (i, p) in names.iter().enumerate() {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("{}: filename is not valid UTF-8", p.display()))?;
        let src = read_image_any_rgb8(p)?;
        let dev = src.to_cuda(&stream)?;
        dino.submit(&dev, &mut r)?; // enqueue, no sync
        stream.synchronize()?; // the one sync
        rows.push(r.descriptor_host()?);
        stems.push(stem.to_string());
        if i.is_multiple_of(50) {
            eprintln!("  {}/{}", i + 1, names.len());
        }
    }

    let bytes = encode_vrtb(&rows, &stems, dim)?;
    std::fs::File::create(out_p)?.write_all(&bytes)?;
    eprintln!(
        "dino_batch: wrote {} descriptors of dim {dim} -> {}",
        rows.len(),
        out_p.display()
    );
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
        // Still exact about what it accepts: no silent inclusion of neighbouring files.
        for p in [
            "a/kf0000.vrtb",
            "a/notes.txt",
            "a/kf0000",
            "a/kf0000.jpg.bak",
        ] {
            assert!(!is_frame(Path::new(p)), "{p}");
        }
    }

    /// The reason the name table exists. Two rows written from `kf0000` and `kf0002` — the gap a
    /// caller produces by skipping a frame it has no thumbnail for — and the assertion is that
    /// row 1 is `kf0002`, NOT keyframe 1. Reading positionally, as the format previously forced,
    /// gets the wrong frame here and cannot tell.
    #[test]
    fn name_table_distinguishes_a_gap_from_a_shift() {
        let rows = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let stems = vec!["kf0000".to_string(), "kf0002".to_string()];
        let b = encode_vrtb(&rows, &stems, 2).unwrap();

        let n = le_u32(&b, 4) as usize;
        let dim = le_u32(&b, 8) as usize;
        let names_len = le_u32(&b, 12) as usize;
        assert_eq!((n, dim), (2, 2));
        assert_eq!(b.len(), 16 + n * dim * 4 + names_len);

        let table = &b[16 + n * dim * 4..];
        let decoded: Vec<&str> = table
            .split(|&c| c == 0)
            .filter(|s| !s.is_empty())
            .map(|s| std::str::from_utf8(s).unwrap())
            .collect();
        assert_eq!(decoded, ["kf0000", "kf0002"]);
        // The whole point: the second row's identity is 2, while its position is 1.
        assert_ne!(decoded[1], "kf0001");
    }

    /// The descriptor block did not move. A reader written against the previous layout — magic,
    /// `n`, `dim`, then `n*dim` f32 at offset 16 — still gets exactly its rows back.
    #[test]
    fn descriptor_block_is_byte_compatible_with_the_previous_layout() {
        let rows = vec![vec![0.5f32, -0.25, 2.0], vec![1.0f32, 0.0, -1.0]];
        let stems = vec!["a".to_string(), "b".to_string()];
        let b = encode_vrtb(&rows, &stems, 3).unwrap();
        assert_eq!(&b[0..4], b"VRTB");
        let flat: Vec<f32> = b[16..16 + 2 * 3 * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(flat, [0.5, -0.25, 2.0, 1.0, 0.0, -1.0]);
    }

    /// A ragged row must not be written. Padding or truncating produces a file whose declared
    /// `dim` no longer slices its own rows, which is undetectable downstream.
    #[test]
    fn ragged_row_is_an_error_not_a_truncation() {
        let rows = vec![vec![1.0f32, 2.0], vec![3.0f32]];
        let stems = vec!["a".to_string(), "b".to_string()];
        assert!(encode_vrtb(&rows, &stems, 2).is_err());
        // Control: the same call with a well-formed second row succeeds, so the test is not
        // passing for some unrelated reason.
        assert!(encode_vrtb(&rows[..1].to_vec(), &stems[..1].to_vec(), 2).is_ok());
    }

    #[test]
    fn row_and_name_counts_must_agree() {
        let rows = vec![vec![1.0f32], vec![2.0f32]];
        assert!(encode_vrtb(&rows, &["only-one".to_string()], 1).is_err());
    }

    /// A NUL in a stem would inject a phantom entry and shift every name after it — the same
    /// misalignment the table is here to prevent, arriving through the table itself.
    #[test]
    fn nul_in_a_frame_name_is_refused() {
        let rows = vec![vec![1.0f32]];
        assert!(encode_vrtb(&rows, &["kf\0abc".to_string()], 1).is_err());
        // Control: the same shape without the NUL is accepted.
        assert!(encode_vrtb(&rows, &["kfabc".to_string()], 1).is_ok());
    }
}
