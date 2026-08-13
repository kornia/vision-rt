//! A cheap fingerprint of the extractor engine, so files can say which engine produced them.
//!
//! The bridge's whole correctness rests on one unwritten assumption: that the `.vrtk` keypoints and
//! the `.vrtm` indices addressing them came from the SAME extractor. Nothing in either file records
//! that today, so running the two tools with different engines — k1024 for one, k3072 for the other
//! — succeeds, produces indices up to 3071 into files holding 1024 keypoints, and is detectable
//! only by noticing the map came out wrong. The consumer already documents having paid for exactly
//! this once: 4,062,208 correspondences fed, 62 inliers per surviving pair, a map with a quarter of
//! the expected points, caught only by the geometric check.
//!
//! A fingerprint in both files turns that from a silent corruption into a startup error.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// FNV-1a over a few sampled regions of the engine file.
///
/// NOT a cryptographic hash and not a full-file digest: engine blobs run to hundreds of megabytes
/// and rehashing one on every invocation would cost more than the work it guards. The threat here
/// is "someone pointed the two tools at different engines", not a forgery, and size plus the head
/// and tail bytes separates any two distinct TensorRT builds — they differ in the serialised header
/// immediately and in the weight tail comprehensively.
///
/// Returns 0 only for an unreadable file, which the caller should treat as "unknown" rather than as
/// a mismatch, so a permissions quirk cannot turn into a spurious hard failure.
pub fn engine_fingerprint(path: impl AsRef<Path>) -> u32 {
    const SAMPLE: usize = 4096;
    let Ok(mut f) = std::fs::File::open(path.as_ref()) else {
        return 0;
    };
    let Ok(len) = f.metadata().map(|m| m.len()) else {
        return 0;
    };

    let mut h: u32 = 0x811c_9dc5;
    let mut mix = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
    };
    mix(&len.to_le_bytes());

    let mut buf = vec![0u8; SAMPLE];
    for from in [
        SeekFrom::Start(0),
        SeekFrom::End(-(SAMPLE.min(len as usize) as i64)),
    ] {
        if f.seek(from).is_err() {
            return 0;
        }
        match f.read(&mut buf) {
            Ok(n) => mix(&buf[..n]),
            Err(_) => return 0,
        }
    }
    // 0 is reserved for "unknown", so a real digest must never collide with it.
    if h == 0 {
        1
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("vrt-fp-{name}"));
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        p
    }

    #[test]
    fn distinguishes_engines_and_is_stable() {
        let a = tmp("a", &vec![7u8; 9000]);
        let b = tmp("b", &vec![9u8; 9000]);
        // Same length, different content — the tail sample has to carry this.
        assert_ne!(engine_fingerprint(&a), engine_fingerprint(&b));
        assert_eq!(engine_fingerprint(&a), engine_fingerprint(&a));
        // Different length alone is enough.
        let c = tmp("c", &vec![7u8; 9001]);
        assert_ne!(engine_fingerprint(&a), engine_fingerprint(&c));
        assert_ne!(engine_fingerprint(&a), 0);
    }

    /// A file shorter than one sample window must still fingerprint rather than fail.
    #[test]
    fn handles_tiny_files() {
        let p = tmp("tiny", b"xy");
        assert_ne!(engine_fingerprint(&p), 0);
    }

    #[test]
    fn missing_file_is_unknown_not_a_panic() {
        assert_eq!(engine_fingerprint("/nonexistent/engine.plan"), 0);
    }
}
