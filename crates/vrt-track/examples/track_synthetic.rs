//! Synthetic BoT-SORT demo — no camera, no model.
//!
//! Scripts two targets crossing the frame, with measurement noise, one occlusion
//! gap, and a burst of low-confidence detections, then prints the track ids per
//! frame so you can watch them stay stable.
//!
//! Run: `cargo run -p vrt-track --example track_synthetic`

use vrt_track::{CameraIntrinsics, Detection, Tracker, TrackerConfig};

/// Tiny deterministic LCG so the demo needs no `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~[-1, 1)
    }
}

fn main() {
    let intr = CameraIntrinsics::from_hfov(1280.0, 720.0, 70.0);
    let mut tracker = Tracker::new(TrackerConfig::default(), intr).expect("valid config");
    let mut rng = Lcg(0x1234_5678);

    println!("frame | detections in            | tracks out (id:class @ cx,cy [d=depth])");
    println!("------+--------------------------+-----------------------------------------");

    for f in 0..24i32 {
        // Two targets moving in opposite directions (object A also has depth).
        let ax = 20.0 + f as f32 * 9.0;
        let ay = 100.0;
        let bx = 300.0 - f as f32 * 9.0;
        let by = 130.0;
        let depth_a = 3.0 + f as f32 * 0.1; // A recedes from the camera

        let mut dets = Vec::new();

        // Target A: occluded on frames 10..13 (no detection at all).
        if !(10..13).contains(&f) {
            let jitter = 2.0 * rng.next_f32();
            dets.push(
                Detection::new(
                    [
                        ax + jitter,
                        ay + jitter,
                        ax + 34.0 + jitter,
                        ay + 70.0 + jitter,
                    ],
                    0.90,
                    0,
                )
                .with_depth(depth_a),
            );
        }

        // Target B: a low-confidence detection on frames 6..9 (stage-2 recovery).
        let score_b = if (6..9).contains(&f) { 0.30 } else { 0.88 };
        let jitter = 2.0 * rng.next_f32();
        dets.push(Detection::new(
            [
                bx + jitter,
                by + jitter,
                bx + 30.0 + jitter,
                by + 64.0 + jitter,
            ],
            score_b,
            1,
        ));

        let tracks = tracker.update(&dets);

        let in_str = dets
            .iter()
            .map(|d| {
                format!(
                    "c{}@{:.0},{:.0}({:.2})",
                    d.class_id, d.bbox[0], d.bbox[1], d.score
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let out_str = tracks
            .iter()
            .map(|t| {
                let cx = (t.bbox[0] + t.bbox[2]) * 0.5;
                let cy = (t.bbox[1] + t.bbox[3]) * 0.5;
                format!(
                    "{}:{}@{:.0},{:.0}[d={:.1}]",
                    t.id, t.class_id, cx, cy, t.position_3d[2]
                )
            })
            .collect::<Vec<_>>()
            .join("  ");

        println!("{f:>5} | {in_str:<24} | {out_str}");
    }

    println!("\nNote how ids stay stable across the occlusion gap (A, frames 10-12)");
    println!("and the low-confidence burst (B, frames 6-8), and how A's depth (d)");
    println!("tracks toward its measured value while B has none (image-plane fallback).");
}
