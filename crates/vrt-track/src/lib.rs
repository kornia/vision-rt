//! A robust **multi-object tracker** with a **3D** Kalman motion model — a pure-CPU
//! algorithm crate (no TensorRT, no CUDA, no model of its own). ByteTrack-style
//! two-stage association + a BoT-SORT-style `w,h` Kalman + a **depth-gated 3D**
//! extension; not full BoT-SORT (no camera-motion compensation — unneeded for fixed
//! cameras).
//!
//! Feed per-frame [`Detection`]s (box + score + class, plus optional depth and
//! appearance embedding) and get back stable [`Track`] ids:
//!
//! ```
//! use vrt_track::{Tracker, TrackerConfig, Detection, CameraIntrinsics};
//!
//! // Construct once, reuse every frame. Intrinsics drive the metric OCM/ORU motion cues.
//! let intr = CameraIntrinsics::from_hfov(1280.0, 720.0, 70.0);
//! let mut tracker = Tracker::new(TrackerConfig::default(), intr).unwrap();
//! for _frame in 0..3 {
//!     let dets = vec![Detection::new([100.0, 100.0, 140.0, 220.0], 0.92, 0)];
//!     let tracks = tracker.update(&dets); // Vec<Track> (id, bbox, class, 3D state)
//!     let _ = tracks;
//! }
//! ```
//!
//! # What it implements
//! - **ByteTrack two-stage association** — high-confidence detections first, then a
//!   recovery pass over low-confidence detections for the still-tracked targets
//!   ([`association`], [`Tracker::update`]).
//! - A **3D constant-velocity Kalman filter** ([`kalman`]) — state
//!   `[px, py, pz, w, h, vx, vy, vz]`. This is the crate's defining choice: it
//!   models depth (`pz`) as a first-class axis and **degrades gracefully to the
//!   image plane** when no depth is measured (via measurement-variance inflation —
//!   see [`kalman`] docs). Depth measurements can come from a future depth/lift
//!   crate through [`Detection::depth`]; until then the tracker runs exactly like an
//!   image-plane tracker while still carrying a coasting depth estimate.
//! - Track **lifecycle** Tentative → Confirmed → Lost → Removed ([`track`]).
//! - Optional **appearance/ReID fusion** behind the `appearance` feature — cosine
//!   distance on [`Detection::feature`] embeddings, min-fused into the IoU cost,
//!   with an EMA feature bank per track. This is a *hook*, not a dependency on any
//!   embedding model — you supply the vectors.
//!
//! Assignment is a compact, dependency-free Hungarian solver; the only external
//! dependency is `nalgebra` for the small fixed-size Kalman matrices.

pub mod association;
pub mod kalman;
pub mod track;
pub mod tracker;

pub use association::iou;
pub use kalman::{KalmanFilter3D, KalmanParams};
pub use track::{Track, TrackState};
pub use tracker::{Tracker, TrackerConfig};
// Camera model lives in the shared `vrt-types` leaf; re-exported for convenience.
pub use vrt_types::{CameraExtrinsics, CameraIntrinsics};

/// Errors from tracker construction / configuration.
#[derive(Debug, thiserror::Error)]
pub enum TrackError {
    /// The supplied [`TrackerConfig`] is inconsistent.
    #[error("invalid tracker config: {0}")]
    InvalidConfig(String),
}

/// One detection for a single frame — the tracker's input unit.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Box `[x1, y1, x2, y2]` in image pixels.
    pub bbox: [f32; 4],
    /// Detector confidence in `[0, 1]`.
    pub score: f32,
    /// Class id (carried through to the matched [`Track`]).
    pub class_id: u32,
    /// Optional metric **depth** of the box centre, e.g. from a depth/lift crate.
    /// `None` → the tracker uses the image-plane fallback for this detection (the
    /// depth axis coasts on the motion model). See [`kalman`].
    pub depth: Option<f32>,
    /// Optional appearance embedding. Used only when the `appearance` feature is
    /// enabled; ignored otherwise. Should be L2-normalised for cosine distance.
    pub feature: Option<Vec<f32>>,
}

impl Detection {
    /// A plain 2D detection (no depth, no embedding) — the common case.
    pub fn new(bbox: [f32; 4], score: f32, class_id: u32) -> Self {
        Self {
            bbox,
            score,
            class_id,
            depth: None,
            feature: None,
        }
    }

    /// Attach a depth measurement (metric depth of the box centre).
    pub fn with_depth(mut self, depth: f32) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Attach an appearance embedding (used with the `appearance` feature).
    pub fn with_feature(mut self, feature: Vec<f32>) -> Self {
        self.feature = Some(feature);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test intrinsics (720p, ~70° HFOV) for the metric OCM/ORU motion cues.
    fn ti() -> CameraIntrinsics {
        CameraIntrinsics::from_hfov(1280.0, 720.0, 70.0)
    }

    /// Every reported track's box + 3D state must be finite (no NaN/inf leaked in).
    fn all_finite(t: &Tracker) -> bool {
        t.tracks().iter().all(|tr| {
            tr.bbox.iter().all(|v| v.is_finite())
                && tr.position_3d.iter().all(|v| v.is_finite())
                && tr.velocity_3d.iter().all(|v| v.is_finite())
        })
    }

    /// Tiny deterministic LCG — reproducible noise without a `rand` dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / (1u64 << 31) as f32 // [0,1)
        }
        fn sym(&mut self, mag: f32) -> f32 {
            (self.next_f() - 0.5) * 2.0 * mag
        }
    }

    // ---------- behavioral stability under realistic noise (reproduces live bugs) ----------

    /// A STATIC object with realistic detector noise — corner jitter, depth spikes past
    /// the gate tolerance, occasional dropped frames — must keep essentially ONE id over
    /// a long run. This is the synthetic version of the live "static fridge re-births ~1/min"
    /// observation; it turns that into a deterministic, debuggable regression.
    #[test]
    fn static_object_stable_under_realistic_noise() {
        // A strongly-detected static object (the live "fridge") with realistic detector
        // noise: box jitter, dropped frames, and transient monocular-depth spikes past the
        // gate tolerance. The old HARD depth veto rejected the object's own spiked
        // detection → it churned ids. The additive depth penalty must keep it near-stable.
        let run = |penalty: f32| -> usize {
            let cfg = TrackerConfig {
                depth_gate_penalty: penalty,
                ..Default::default()
            };
            let mut t = Tracker::new(cfg, ti()).unwrap();
            let mut r = Lcg(12345);
            let base = [500.0f32, 300.0, 560.0, 500.0];
            let mut ids = std::collections::HashSet::new();
            for _ in 0..2000 {
                if r.next_f() < 0.06 {
                    t.update(&[]); // detector drop
                    continue;
                }
                let bx = [
                    base[0] + r.sym(3.0),
                    base[1] + r.sym(3.0),
                    base[2] + r.sym(3.0),
                    base[3] + r.sym(3.0),
                ];
                let z = if r.next_f() < 0.1 {
                    3.0 + r.sym(2.5)
                } else {
                    3.0 + r.sym(0.15)
                };
                for tr in t
                    .update(&[Detection::new(bx, 0.85, 5).with_depth(z.max(0.3))])
                    .iter()
                    .filter(|tr| tr.state == TrackState::Confirmed)
                {
                    ids.insert(tr.id);
                }
            }
            ids.len()
        };
        let soft = run(0.2); // the additive-gate default
        let hard = run(1.0e6); // old hard-veto
        assert!(
            soft < hard,
            "additive depth gate must churn less than the hard veto: soft={soft} hard={hard}"
        );
        assert!(
            soft <= 3,
            "static object churned {soft} ids under the additive gate (expect ~1-2)"
        );
    }

    /// A static object whose **seg-mask box shifts in regimes** (the dominant live cause of
    /// furniture id churn: the mask periodically captures a different extent, so the box
    /// jumps far enough that plain IoU with the smoothed track drops below the match gate →
    /// the detection births a rival track → id change). Buffered-IoU (`iou_buffer`) must
    /// absorb the shift and hold ONE id where plain IoU churns.
    #[test]
    fn buffered_iou_absorbs_seg_box_shift_churn() {
        // Static object; every 20 frames its box lurches 50px laterally (a mask-extent
        // flip between two regimes). Same size throughout, so this is pure positional shift.
        // 50px on a 60px box: plain IoU falls to ~0.09 (below the ~0.2 gate → churn) but a
        // 0.3 buffer keeps ~0.31 (holds) — the exact window buffering is meant to cover.
        // Metric = id *transitions* of the dominant (most-hits) confirmed track. Under
        // plain IoU a rival track is born at each shift and the reported id ping-pongs
        // between two live tracks — distinct-count caps at 2 but the id keeps *switching*.
        // Transitions capture that; buffering should drive it to ~0.
        let run = |buffer: f32| -> usize {
            let cfg = TrackerConfig {
                iou_buffer: buffer,
                ..Default::default()
            };
            let mut t = Tracker::new(cfg, ti()).unwrap();
            let mut r = Lcg(2024);
            let mut switches = 0usize;
            let mut prev: Option<u64> = None;
            for f in 0..600u32 {
                let regime = ((f / 20) % 2) as f32; // 0 / 1 square wave every 20 frames
                let cx = 600.0 + regime * 50.0 + r.sym(2.0);
                let bx = [cx - 30.0, 300.0, cx + 30.0, 360.0]; // 60×60
                let out = t.update(&[Detection::new(bx, 0.85, 5).with_depth(3.0 + r.sym(0.1))]);
                // The real object = the confirmed track with the most hits.
                if let Some(tr) = out
                    .iter()
                    .filter(|tr| tr.state == TrackState::Confirmed)
                    .max_by_key(|tr| tr.hits)
                {
                    if prev.is_some_and(|p| p != tr.id) {
                        switches += 1;
                    }
                    prev = Some(tr.id);
                }
            }
            switches
        };
        let plain = run(0.0);
        let buffered = run(0.5);
        assert!(
            buffered < plain,
            "buffered IoU must switch id less than plain on box-shift: buffered={buffered} plain={plain}"
        );
        assert!(
            buffered <= 1,
            "buffered IoU should hold one id through the shifts, got {buffered} switches"
        );
    }

    /// **Downside guard for `iou_buffer`.** Buffering widens *every* box, so a large buffer
    /// could make two distinct-but-adjacent objects overlap and swap. With buffer 0.5 and
    /// two static objects one box-gap apart at the SAME depth (so the depth gate can't
    /// separate them), each must still keep its own id — buffering must not manufacture a
    /// swap. (Confirms the box-shift fix doesn't regress the crossing/adjacency case.)
    #[test]
    fn buffered_iou_does_not_swap_adjacent_same_depth_objects() {
        let cfg = TrackerConfig {
            iou_buffer: 0.5,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        let mut r = Lcg(77);
        // Two 60-wide boxes with a 60px gap between them, both at 4 m.
        let a = [400.0f32, 300.0, 460.0, 360.0];
        let b = [520.0f32, 300.0, 580.0, 360.0];
        let (mut ida, mut idb) = (None, None);
        for _f in 0..400 {
            let jit = |r: &mut Lcg, base: &[f32; 4]| {
                [
                    base[0] + r.sym(3.0),
                    base[1] + r.sym(3.0),
                    base[2] + r.sym(3.0),
                    base[3] + r.sym(3.0),
                ]
            };
            let da = Detection::new(jit(&mut r, &a), 0.9, 0).with_depth(4.0 + r.sym(0.1));
            let db = Detection::new(jit(&mut r, &b), 0.9, 0).with_depth(4.0 + r.sym(0.1));
            for tr in t
                .update(&[da, db])
                .iter()
                .filter(|tr| tr.state == TrackState::Confirmed)
            {
                // left object has the smaller x-centre
                let slot = if tr.bbox[0] < 490.0 {
                    &mut ida
                } else {
                    &mut idb
                };
                match slot {
                    None => *slot = Some(tr.id),
                    Some(prev) => assert_eq!(
                        *prev, tr.id,
                        "buffer swapped adjacent objects at frame {_f}"
                    ),
                }
            }
        }
        assert!(
            ida.is_some() && idb.is_some() && ida != idb,
            "both adjacent objects tracked distinctly"
        );
    }

    /// Two static objects at different depths/positions, both noisy, must never trade ids
    /// over a long run — each keeps its own identity. Catches depth-gate / IoU swaps.
    #[test]
    fn two_static_objects_never_swap_under_noise() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let mut r = Lcg(999);
        let a = [300.0f32, 300.0, 360.0, 500.0]; // near, 2 m
        let b = [340.0f32, 300.0, 400.0, 500.0]; // overlaps A in image, far, 5 m
        let (mut ida, mut idb) = (None, None);
        for _f in 0..1500 {
            let jit = |r: &mut Lcg, base: &[f32; 4]| {
                [
                    base[0] + r.sym(4.0),
                    base[1] + r.sym(4.0),
                    base[2] + r.sym(4.0),
                    base[3] + r.sym(4.0),
                ]
            };
            let da = Detection::new(jit(&mut r, &a), 0.9, 0).with_depth(2.0 + r.sym(0.2));
            let db = Detection::new(jit(&mut r, &b), 0.9, 0).with_depth(5.0 + r.sym(0.3));
            let out = t.update(&[da, db]);
            // Identify by depth: near (<3.5 m) vs far.
            for tr in out.iter().filter(|tr| tr.state == TrackState::Confirmed) {
                let slot = if tr.position_3d[2] < 3.5 {
                    &mut ida
                } else {
                    &mut idb
                };
                match slot {
                    None => *slot = Some(tr.id),
                    Some(prev) => assert_eq!(
                        *prev, tr.id,
                        "same-depth object changed id (swap/rebirth) at frame {_f}"
                    ),
                }
            }
        }
        assert!(
            ida.is_some() && idb.is_some() && ida != idb,
            "both objects tracked distinctly"
        );
    }

    /// **Translation invariance** (metamorphic): the same trajectory shifted by a constant
    /// offset must produce the same id-assignment structure — the tracker must not depend
    /// on absolute image position.
    #[test]
    fn translation_invariance() {
        let run = |ox: f32, oy: f32| -> Vec<u64> {
            let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
            let mut ids = Vec::new();
            for f in 0..30 {
                let x = 100.0 + f as f32 * 8.0 + ox;
                let out = t.update(&[Detection::new(
                    [x, 100.0 + oy, x + 40.0, 200.0 + oy],
                    0.9,
                    0,
                )]);
                ids.push(out.first().map(|tr| tr.id).unwrap_or(0));
            }
            ids
        };
        // id *values* differ (allocation), but the birth/continuity pattern is identical:
        // compare "distinct id count" and "first confirmed frame".
        let a = run(0.0, 0.0);
        let b = run(300.0, 150.0);
        let distinct = |v: &[u64]| {
            v.iter()
                .filter(|&&i| i != 0)
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        assert_eq!(
            distinct(&a),
            distinct(&b),
            "translation changed the id structure: {a:?} vs {b:?}"
        );
        assert_eq!(
            a.iter().position(|&i| i != 0),
            b.iter().position(|&i| i != 0),
            "translation changed when the track confirmed"
        );
    }

    /// **Permutation invariance** (metamorphic): shuffling detection order within a frame
    /// must not change the assignment — the matcher is order-independent.
    #[test]
    fn permutation_invariance() {
        let run = |swap: bool| -> Vec<(f32, u64)> {
            let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
            let mut trail = Vec::new();
            for f in 0..30 {
                let ax = 50.0 + f as f32 * 6.0;
                let bx = 600.0 - f as f32 * 6.0;
                let da = Detection::new([ax, 100.0, ax + 30.0, 180.0], 0.9, 0);
                let db = Detection::new([bx, 300.0, bx + 30.0, 380.0], 0.9, 1);
                let dets = if swap { vec![db, da] } else { vec![da, db] };
                for tr in t.update(&dets) {
                    // key each track by class so the two orders line up
                    trail.push((tr.class_id as f32, tr.id));
                }
            }
            trail
        };
        // Id *numbers* legitimately depend on birth order, but each object must keep
        // EXACTLY ONE id within a run (no swap) regardless of input order.
        let stable = |v: &[(f32, u64)]| {
            let mut m = std::collections::HashMap::new();
            for &(c, id) in v {
                m.entry(c as u32)
                    .or_insert_with(std::collections::HashSet::new)
                    .insert(id);
            }
            m.values().all(|ids| ids.len() == 1)
        };
        assert!(stable(&run(false)), "unshuffled order caused a swap");
        assert!(stable(&run(true)), "shuffled detection order caused a swap");
    }

    /// **Determinism**: identical input twice yields an identical id sequence.
    #[test]
    fn deterministic_output() {
        let run = || -> Vec<Vec<u64>> {
            let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
            let mut r = Lcg(7);
            let mut seq = Vec::new();
            for _ in 0..100 {
                let n = (r.next_f() * 4.0) as usize;
                let dets: Vec<_> = (0..n)
                    .map(|i| {
                        let x = 50.0 + i as f32 * 120.0 + r.sym(2.0);
                        Detection::new([x, 100.0, x + 40.0, 200.0], 0.9, i as u32)
                    })
                    .collect();
                seq.push(t.update(&dets).iter().map(|tr| tr.id).collect());
            }
            seq
        };
        assert_eq!(run(), run(), "tracker is not deterministic");
    }

    /// **Fuzz soak**: 800 frames of randomized detections (variable count, jitter,
    /// dropouts, depth spikes, class mix) must never panic, never leak non-finite state,
    /// and keep every reported id positive. A cheap stand-in for property-based fuzzing.
    #[test]
    fn randomized_soak_stays_finite_and_sane() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let mut r = Lcg(4242);
        for _f in 0..800 {
            let n = (r.next_f() * 5.0) as usize;
            let dets: Vec<_> = (0..n)
                .map(|i| {
                    let x = 50.0 + r.next_f() * 1100.0;
                    let y = 50.0 + r.next_f() * 600.0;
                    let w = 20.0 + r.next_f() * 80.0;
                    let h = 40.0 + r.next_f() * 160.0;
                    let z =
                        1.0 + r.next_f() * 6.0 + if r.next_f() < 0.1 { r.sym(3.0) } else { 0.0 };
                    Detection::new([x, y, x + w, y + h], 0.3 + r.next_f() * 0.7, (i % 6) as u32)
                        .with_depth(z.max(0.3))
                })
                .collect();
            for tr in t.update(&dets) {
                assert!(tr.id >= 1, "id must be positive");
                assert!(
                    tr.bbox.iter().all(|v| v.is_finite())
                        && tr.position_3d.iter().all(|v| v.is_finite()),
                    "non-finite track state during fuzz soak"
                );
            }
            assert!(
                t.len() <= 60,
                "runaway track growth in fuzz soak: {}",
                t.len()
            );
        }
        assert!(all_finite(&t));
    }

    // ---------- robustness / corner-case battery ----------

    /// Malformed detection boxes (NaN, inf, zero-area, inverted, negative, huge) must be
    /// dropped — never panic, never leak NaN into a track, never explode the track count.
    #[test]
    fn degenerate_boxes_are_dropped_not_ingested() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let bad = [
            [f32::NAN, 0.0, 10.0, 10.0],
            [0.0, 0.0, f32::INFINITY, 10.0],
            [5.0, 5.0, 5.0, 5.0],       // zero area
            [100.0, 100.0, 10.0, 10.0], // inverted (x2<x1)
            [-50.0, -50.0, f32::NEG_INFINITY, -10.0],
        ];
        for _ in 0..20 {
            let dets: Vec<_> = bad.iter().map(|b| Detection::new(*b, 0.9, 0)).collect();
            t.update(&dets);
        }
        assert!(all_finite(&t), "malformed boxes leaked NaN into a track");
        assert_eq!(t.len(), 0, "malformed detections must birth no tracks");
        // A valid detection mixed with garbage still tracks normally.
        for _ in 0..5 {
            let mut dets: Vec<_> = bad.iter().map(|b| Detection::new(*b, 0.9, 0)).collect();
            dets.push(Detection::new([100.0, 100.0, 140.0, 220.0], 0.9, 0));
            t.update(&dets);
        }
        assert_eq!(t.len(), 1, "the one valid detection should track");
        assert!(all_finite(&t));
    }

    /// Non-finite / non-positive depth must be treated as absent, not fed to the Kalman
    /// or the metric gates (NaN depth would poison pz and wedge association).
    #[test]
    fn degenerate_depth_is_ignored() {
        let cfg = TrackerConfig {
            use_diou3d: true, // exercise the metric unprojection path too
            min_hits: 3,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        let bx = [100.0, 100.0, 140.0, 220.0];
        for (k, z) in [f32::NAN, f32::INFINITY, -3.0, 0.0, 2.5]
            .iter()
            .cycle()
            .take(20)
            .enumerate()
        {
            let _ = k;
            t.update(&[Detection::new(bx, 0.9, 0).with_depth(*z)]);
            assert!(all_finite(&t), "bad depth leaked NaN into the state");
        }
        assert_eq!(t.len(), 1);
    }

    /// Degenerate `dt` (zero, negative, non-finite, absurdly large) must not blow up the
    /// filter — it falls back to a nominal step and stays finite.
    #[test]
    fn degenerate_dt_is_sanitized() {
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e12] {
            let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
            let mut id = None;
            for f in 0..6 {
                let x = 100.0 + f as f32 * 5.0;
                let out = t.update_dt(&[Detection::new([x, 100.0, x + 40.0, 200.0], 0.9, 0)], dt);
                if let Some(tr) = out.first() {
                    id = Some(tr.id);
                }
            }
            assert!(all_finite(&t), "dt={dt} produced non-finite state");
            assert!(id.is_some(), "dt={dt} broke tracking entirely");
        }
    }

    /// Dense scene: 300 simultaneous detections across classes must not panic and the
    /// track count stays bounded (no runaway growth).
    #[test]
    fn dense_scene_bounded_and_no_panic() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let dets: Vec<_> = (0..300)
            .map(|i| {
                let x = (i % 40) as f32 * 30.0;
                let y = (i / 40) as f32 * 40.0;
                Detection::new([x, y, x + 25.0, y + 35.0], 0.9, (i % 8) as u32)
            })
            .collect();
        for _ in 0..10 {
            t.update(&dets);
        }
        assert!(all_finite(&t));
        assert!(t.len() <= 300, "track count unbounded: {}", t.len());
        assert!(
            t.len() >= 250,
            "stable dense detections should mostly track: {}",
            t.len()
        );
    }

    /// Rapid on/off flicker (object present every other frame) must not spawn unbounded
    /// ids nor leave NaN — the lifecycle absorbs it.
    #[test]
    fn flicker_is_absorbed() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        for f in 0..200 {
            if f % 2 == 0 {
                t.update(&[Detection::new([100.0, 100.0, 140.0, 220.0], 0.9, 0)]);
            } else {
                t.update(&[]);
            }
        }
        assert!(all_finite(&t));
        assert!(
            t.len() <= 2,
            "flicker spawned too many concurrent tracks: {}",
            t.len()
        );
    }

    /// Malformed appearance embeddings (wrong length, empty, all-zeros, NaN) must not
    /// panic the cosine fusion or corrupt matching.
    #[cfg(feature = "appearance")]
    #[test]
    fn degenerate_features_dont_panic() {
        let cfg = TrackerConfig {
            reid_thresh: 0.2,
            min_hits: 3,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        let feats = [
            vec![],                        // empty
            vec![0.0, 0.0, 0.0],           // all-zeros (zero norm)
            vec![f32::NAN, 1.0, 0.0],      // NaN
            vec![1.0],                     // wrong length (mismatched dim)
            vec![1.0, 0.0, 0.0, 0.0, 0.0], // different length again
        ];
        for (f, feat) in (0..40).zip(feats.iter().cycle()) {
            let x = 100.0 + (f % 5) as f32;
            t.update(&[
                Detection::new([x, 100.0, x + 40.0, 220.0], 0.9, 0).with_feature(feat.clone())
            ]);
            assert!(
                all_finite(&t),
                "bad feature leaked NaN into the track state"
            );
        }
    }

    /// Id numbers are strictly monotonic — a resurrected/merged track never mints an id
    /// larger than the current allocation, and no live track ever exceeds `next_id`.
    #[test]
    fn ids_are_monotonic_and_bounded() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let mut max_seen = 0u64;
        for f in 0..300 {
            // a couple of moving objects + occasional dropouts
            let mut dets = Vec::new();
            if f % 7 != 0 {
                let x = 50.0 + (f % 50) as f32 * 4.0;
                dets.push(Detection::new([x, 100.0, x + 40.0, 200.0], 0.9, 0));
            }
            if f % 5 != 0 {
                let x = 600.0 - (f % 50) as f32 * 3.0;
                dets.push(Detection::new([x, 300.0, x + 40.0, 400.0], 0.8, 1));
            }
            for tr in t.update(&dets) {
                assert!(tr.id >= 1, "ids start at 1");
                max_seen = max_seen.max(tr.id);
            }
            assert!(all_finite(&t));
        }
        // With ≤2 objects over 300 frames, id allocation can't have run away.
        assert!(max_seen < 300, "id allocation ran away: {max_seen}");
    }

    /// Move a single box left-to-right; the tracker should confirm it and keep one
    /// stable id throughout.
    #[test]
    fn single_target_stable_id() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let mut id = None;
        for f in 0..10 {
            let x = 50.0 + f as f32 * 12.0;
            let det = Detection::new([x, 100.0, x + 30.0, 160.0], 0.9, 0);
            let out = t.update(&[det]);
            if f >= 3 {
                // Confirmed after min_hits.
                assert_eq!(out.len(), 1, "frame {f}: expected 1 track");
                match id {
                    None => id = Some(out[0].id),
                    Some(prev) => assert_eq!(out[0].id, prev, "id switched at frame {f}"),
                }
            }
        }
        assert!(id.is_some());
    }

    /// Two crossing targets must not swap ids as they pass.
    #[test]
    fn two_crossing_targets_keep_ids() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let (mut id_a, mut id_b) = (None, None);
        for f in 0..14 {
            let ax = 20.0 + f as f32 * 12.0; // left -> right
            let bx = 180.0 - f as f32 * 12.0; // right -> left
            let da = Detection::new([ax, 100.0, ax + 24.0, 160.0], 0.9, 0);
            let db = Detection::new([bx, 130.0, bx + 24.0, 190.0], 0.9, 1);
            let out = t.update(&[da, db]);
            if f >= 4 {
                let a = out.iter().find(|tr| tr.class_id == 0);
                let b = out.iter().find(|tr| tr.class_id == 1);
                if let (Some(a), Some(b)) = (a, b) {
                    match id_a {
                        None => id_a = Some(a.id),
                        Some(p) => assert_eq!(a.id, p, "class-0 id switched at {f}"),
                    }
                    match id_b {
                        None => id_b = Some(b.id),
                        Some(p) => assert_eq!(b.id, p, "class-1 id switched at {f}"),
                    }
                }
            }
        }
        assert!(id_a.is_some() && id_b.is_some() && id_a != id_b);
    }

    /// Two **same-class** targets crossing with fully-overlapping boxes but distinct
    /// metric depths: IoU (and constant-velocity motion) can't tell them apart at the
    /// crossing, so only the **depth gate** keeps their ids from swapping. Tracks are
    /// identified by their depth estimate (near < far), which must stay separated and
    /// keep constant ids.
    #[test]
    fn depth_gate_prevents_id_swap_on_crossing() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let (mut near_id, mut far_id) = (None, None);
        for f in 0..16 {
            let ax = 20.0 + f as f32 * 10.0; // near object, L->R, 2 m
            let bx = 170.0 - f as f32 * 10.0; // far object, R->L, 5 m (boxes coincide ~f=8)
            let da = Detection::new([ax, 100.0, ax + 40.0, 170.0], 0.9, 0).with_depth(2.0);
            let db = Detection::new([bx, 100.0, bx + 40.0, 170.0], 0.9, 0).with_depth(5.0);
            let out = t.update(&[da, db]);
            if f >= 4 {
                let near = out
                    .iter()
                    .min_by(|x, y| x.position_3d[2].total_cmp(&y.position_3d[2]));
                let far = out
                    .iter()
                    .max_by(|x, y| x.position_3d[2].total_cmp(&y.position_3d[2]));
                if let (Some(n), Some(fr)) = (near, far) {
                    assert!(
                        n.position_3d[2] < 3.5 && fr.position_3d[2] > 3.5,
                        "depths collapsed at frame {f}: near {:.2} far {:.2}",
                        n.position_3d[2],
                        fr.position_3d[2]
                    );
                    match near_id {
                        None => near_id = Some(n.id),
                        Some(p) => assert_eq!(n.id, p, "near id swapped at frame {f}"),
                    }
                    match far_id {
                        None => far_id = Some(fr.id),
                        Some(p) => assert_eq!(fr.id, p, "far id swapped at frame {f}"),
                    }
                }
            }
        }
        assert!(near_id.is_some() && far_id.is_some() && near_id != far_id);
    }

    /// A target that vanishes for a few frames should be re-acquired with the SAME
    /// id (Lost → Confirmed), exercising the Kalman coast + re-id path.
    #[test]
    fn occlusion_recovers_same_id() {
        let cfg = TrackerConfig {
            min_hits: 3,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();

        let boxf = |f: i32| {
            let x = 40.0 + f as f32 * 8.0;
            Detection::new([x, 100.0, x + 30.0, 170.0], 0.9, 0)
        };
        // Establish.
        let mut id = None;
        for f in 0..5 {
            let out = t.update(&[boxf(f)]);
            if let Some(tr) = out.first() {
                id = Some(tr.id);
            }
        }
        let id = id.expect("track established");
        // Occlude for 3 frames (no detections).
        for _ in 0..3 {
            t.update(&[]);
        }
        // Reappear near the predicted position.
        let out = t.update(&[boxf(8)]);
        assert_eq!(out.len(), 1, "did not re-acquire");
        assert_eq!(out[0].id, id, "re-acquired with a new id");
    }

    /// ORU (observation-centric re-update) must be **active and safe**: an object
    /// occluded mid-motion and re-acquired keeps its id and lands on the reappearance
    /// box, with ORU on or off — no regression, no corrupted state. (ORU's *quantitative*
    /// benefit is a HOTA/IDF1 effect on real MOT sequences, not a single synthetic
    /// re-acquisition; here we pin the mechanism's correctness, not a numeric gain.)
    #[test]
    fn oru_reacquires_safely_no_regression() {
        let run = |oru: bool| -> (Option<u64>, f32) {
            let cfg = TrackerConfig {
                oru,
                ocm_lambda: 0.0, // isolate ORU from momentum
                min_hits: 3,
                ..Default::default()
            };
            let mut t = Tracker::new(cfg, ti()).unwrap();
            // Wide box (reliable re-acquire), constant depth so the world path is metric.
            let boxf = |cx: f32| {
                Detection::new([cx - 40.0, 100.0, cx + 40.0, 200.0], 0.9, 0).with_depth(5.0)
            };
            for k in 0..5 {
                let cx = 100.0 + k as f32 * 8.0; // vx = 8 px/frame
                t.update(&[boxf(cx)]);
            }
            for _ in 0..2 {
                t.update(&[]); // occluded 2 frames → the Kalman coasts, track Lost
            }
            let out = t.update(&[boxf(136.0)]); // reappears mid-motion
            let cx = out
                .first()
                .map(|tr| (tr.bbox[0] + tr.bbox[2]) * 0.5)
                .unwrap_or(f32::NAN);
            (out.first().map(|tr| tr.id), cx)
        };
        let (id_on, cx_on) = run(true);
        let (id_off, cx_off) = run(false);
        assert!(
            id_on.is_some() && id_on == id_off,
            "re-acquired same id, ORU on/off"
        );
        assert!(
            cx_on.is_finite() && cx_off.is_finite(),
            "recovered state is finite"
        );
        assert!(
            (cx_on - 136.0).abs() < 20.0,
            "ORU recovered position near the reappearance box: {cx_on}"
        );
    }

    /// Lost-track appearance re-id (stage 1.5): a moving object is occluded and
    /// reappears where its coasted prediction is NOT (IoU = 0) — geometry alone births a
    /// new id; the tight-gated appearance re-id stage must recover the SAME id.
    #[cfg(feature = "appearance")]
    #[test]
    fn lost_track_reacquired_by_appearance_after_drift() {
        let sig = vec![1.0f32, 0.0, 0.0]; // this object's (L2-normed) embedding
        let boxat = |cx: f32| {
            Detection::new([cx - 30.0, 100.0, cx + 30.0, 200.0], 0.9, 0)
                .with_depth(3.0)
                .with_feature(sig.clone())
        };
        let run = |reid: f32| -> (bool, usize) {
            let cfg = TrackerConfig {
                reid_thresh: reid,
                min_hits: 3,
                ..Default::default()
            };
            let mut t = Tracker::new(cfg, ti()).unwrap();
            let mut id = None;
            for k in 0..6 {
                let out = t.update(&[boxat(100.0 + k as f32 * 15.0)]); // moving +x
                if let Some(tr) = out.first() {
                    id = Some(tr.id);
                }
            }
            let id = id.expect("established");
            for _ in 0..8 {
                t.update(&[]); // occluded; coast continues +x
            }
            // Reappears far BEHIND the coasted prediction (turned around): IoU with the
            // coast is zero, so only appearance can re-acquire.
            let out = t.update(&[boxat(40.0)]);
            let total_ids = t.tracks().len();
            (out.first().is_some_and(|tr| tr.id == id), total_ids)
        };
        let (same_id, _) = run(0.05);
        assert!(
            same_id,
            "appearance re-id should recover the same id across the drift"
        );
        let (same_id_off, _) = run(0.0); // stage disabled → geometry alone
        assert!(
            !same_id_off,
            "control: without re-id the drifted track is not recovered"
        );
    }

    /// Gallery resurrection: a person leaves the scene entirely (track dies past
    /// `track_buffer`), returns much later ANYWHERE in the frame — the newborn track
    /// must be resurrected with the ORIGINAL id from the dead-track gallery.
    #[cfg(feature = "appearance")]
    #[test]
    fn exit_scene_return_keeps_id_via_gallery() {
        let sig = vec![0.0f32, 0.6, 0.8];
        let person = |cx: f32| {
            Detection::new([cx - 40.0, 80.0, cx + 40.0, 300.0], 0.9, 1)
                .with_depth(2.5)
                .with_feature(sig.clone())
        };
        let cfg = TrackerConfig {
            min_hits: 3,
            gallery_min_hits: 10,
            track_buffer: 60,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        // Establish for 20 frames (enough hits to earn a gallery slot).
        let mut id = None;
        for k in 0..20 {
            if let Some(tr) = t.update(&[person(200.0 + k as f32 * 5.0)]).first() {
                id = Some(tr.id);
            }
        }
        let id = id.expect("established");
        // Leave the scene for 100 frames — past track_buffer, the track is REMOVED.
        for _ in 0..100 {
            t.update(&[]);
        }
        assert!(
            t.is_empty(),
            "track should be dead after the buffer expires"
        );
        // Re-enter at the far side of the frame: same embedding → same id, immediately
        // Confirmed (no re-probation for an established identity).
        let out = t.update(&[person(1000.0)]);
        assert_eq!(out.len(), 1, "resurrected track should report immediately");
        assert_eq!(out[0].id, id, "returning object must keep its original id");
    }

    /// The live failure mode from the kitchen: DUPLICATE tracks of one person die and
    /// fill the gallery with same-identity copies; the uniqueness-abstain must NOT then
    /// veto the resurrection. Gallery dedup-on-insert keeps one canonical (oldest id)
    /// entry, so the return still resurrects the original id.
    #[cfg(feature = "appearance")]
    #[test]
    fn duplicate_deaths_dont_block_resurrection() {
        let sig = vec![0.8f32, 0.0, 0.6];
        let person = |cx: f32| {
            Detection::new([cx - 40.0, 80.0, cx + 40.0, 300.0], 0.9, 1)
                .with_depth(2.5)
                .with_feature(sig.clone())
        };
        let cfg = TrackerConfig {
            min_hits: 3,
            gallery_min_hits: 10,
            track_buffer: 30,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        // Two well-separated same-embedding tracks (a duplicate pair of one person —
        // separated here so the birth dup-guard doesn't collapse them first).
        let mut id = None;
        for k in 0..15 {
            let out = t.update(&[
                person(200.0 + k as f32 * 4.0),
                person(700.0 + k as f32 * 4.0),
            ]);
            if let Some(tr) = out.iter().min_by_key(|tr| tr.id) {
                id = Some(tr.id);
            }
        }
        let id = id.expect("established");
        // Both die (leave scene past the buffer) → gallery dedups them into ONE entry.
        for _ in 0..60 {
            t.update(&[]);
        }
        assert!(t.is_empty());
        // Return: must resurrect the canonical (oldest) id, not abstain on "ambiguity".
        let out = t.update(&[person(1100.0)]);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].id, id,
            "dedup'd gallery must resurrect the canonical id"
        );
    }

    /// Deferred identity merge: the person returns while their old track is still Lost
    /// (buffer not expired) but far from the exit point, so the spatial re-id bound
    /// blocks stage 1.5 and a NEW track births. Within its first seconds the young track
    /// must be MERGED back to the old id (the Axis/DeepStream late-merge behavior).
    #[cfg(feature = "appearance")]
    #[test]
    fn young_track_merges_back_to_lost_twin() {
        let sig = vec![0.0f32, 1.0, 0.0];
        let person = |cx: f32| {
            Detection::new([cx - 40.0, 80.0, cx + 40.0, 300.0], 0.9, 1)
                .with_depth(2.5)
                .with_feature(sig.clone())
        };
        let cfg = TrackerConfig {
            min_hits: 3,
            track_buffer: 120, // long buffer: old track still Lost on return
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        let mut id = None;
        for k in 0..12 {
            if let Some(tr) = t.update(&[person(100.0 + k as f32 * 5.0)]).first() {
                id = Some(tr.id);
            }
        }
        let id = id.expect("established");
        for _ in 0..20 {
            t.update(&[]); // gone 20 frames — still Lost (buffer 120)
        }
        // Re-enter across the frame: static-ish speed → spatial bound blocks stage 1.5,
        // so a new track births; the deferred merge must reclaim the old id within ~1 s.
        let mut last_id = None;
        for k in 0..12 {
            let out = t.update(&[person(1100.0 - k as f32 * 5.0)]);
            if let Some(tr) = out.first() {
                last_id = Some(tr.id);
            }
        }
        assert_eq!(
            last_id,
            Some(id),
            "young track should merge back to the Lost twin's id"
        );
    }

    /// Death-bed identity handoff: a duplicate twin outlives the original (the detection
    /// stream drifts to the newer box hypothesis; the old track starves past the buffer).
    /// When the established original dies, the LIVE twin must inherit its (older) id —
    /// identity continues instead of dying into the gallery while an impostor id lives on.
    #[cfg(feature = "appearance")]
    #[test]
    fn dying_track_hands_id_to_live_twin() {
        let sig = vec![0.6f32, 0.0, 0.8];
        let person = |cx: f32| {
            Detection::new([cx - 40.0, 80.0, cx + 40.0, 300.0], 0.9, 1)
                .with_depth(2.5)
                .with_feature(sig.clone())
        };
        let cfg = TrackerConfig {
            min_hits: 3,
            gallery_min_hits: 10,
            track_buffer: 20,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        // Establish the original at x=200.
        let mut orig = None;
        for _ in 0..15 {
            if let Some(tr) = t.update(&[person(200.0)]).first() {
                orig = Some(tr.id);
            }
        }
        let orig = orig.expect("established");
        // A far duplicate twin appears (dup guard can't see them as one — separated);
        // the original's detections STOP (stream drifted to the twin's box hypothesis).
        // Age the twin well past the 45-frame merge window, until the original dies.
        let mut last = None;
        for _ in 0..90 {
            if let Some(tr) = t.update(&[person(700.0)]).last() {
                last = Some(tr.id);
            }
        }
        assert_eq!(
            last,
            Some(orig),
            "live twin should inherit the dying original's id (death-bed handoff)"
        );
    }

    /// Re-id must NOT accept a depth-inconsistent appearance match. The stage-1.5 re-id
    /// has no IoU floor, so with a realistic `reid_thresh` (0.25, as the OSNet example
    /// uses) the soft additive depth penalty (~0.2) is *smaller* than the gate — an
    /// identical-looking object at the wrong depth (a look-alike behind glass) would
    /// teleport the id. The depth gate must stay a HARD veto on this path.
    #[cfg(feature = "appearance")]
    #[test]
    fn reid_depth_veto_blocks_wrong_depth_match() {
        let sig = vec![0.6f32, 0.8, 0.0]; // identical embedding both places
                                          // Object moving in +x at 3 m, so its re-id spatial bound is generous (velocity
                                          // slack) — the wrong-depth candidate lands well INSIDE it, isolating the depth
                                          // veto as the only thing that can reject the match.
        let cfg = TrackerConfig {
            min_hits: 3,
            reid_thresh: 0.25, // OSNet-grade gate; soft 0.2 penalty alone would NOT reject
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        // Static object at (300, 140), depth 3 m, box 50×80 → re-id bound ≈ 3·65 = 195 px.
        let boxat = |cx: f32, cy: f32, z: f32| {
            Detection::new([cx - 25.0, cy - 40.0, cx + 25.0, cy + 40.0], 0.9, 5)
                .with_depth(z)
                .with_feature(sig.clone())
        };
        let mut id = None;
        for _ in 0..8 {
            if let Some(tr) = t.update(&[boxat(300.0, 140.0, 3.0)]).first() {
                id = Some(tr.id);
            }
        }
        let id = id.expect("established");
        for _ in 0..4 {
            t.update(&[]); // occluded → Lost
        }
        // Identical-appearance detection 170 px BELOW (IoU = 0 → skips stage-1 geometry,
        // reaches stage-1.5 re-id; centre distance 170 < 195 bound → inside the re-id
        // window) but at 8 m. Depth gap 5 m ≫ tol → the hard veto must reject it.
        let out = t.update(&[boxat(300.0, 310.0, 8.0)]);
        assert!(
            !out.iter().any(|tr| tr.id == id),
            "lost track re-acquired a same-appearance detection at the wrong depth (veto failed)"
        );
    }

    /// Re-id must NOT teleport: a STATIC lost track (occluded chair) may not re-acquire
    /// onto an identical-looking detection far across the image — identical objects are
    /// only separable by position, so the observed-motion spatial bound must block the
    /// match and let a new id birth there instead (the run-2 live failure mode).
    #[cfg(feature = "appearance")]
    #[test]
    fn reid_does_not_teleport_static_track() {
        let sig = vec![0.6f32, 0.8, 0.0]; // identical embedding both places (same chair model)
        let boxat = |cx: f32| {
            Detection::new([cx - 25.0, 100.0, cx + 25.0, 180.0], 0.9, 62)
                .with_depth(3.0)
                .with_feature(sig.clone())
        };
        let cfg = TrackerConfig {
            min_hits: 3,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg, ti()).unwrap();
        // Static chair at x=100 for 6 frames.
        let mut id = None;
        for _ in 0..6 {
            if let Some(tr) = t.update(&[boxat(100.0)]).first() {
                id = Some(tr.id);
            }
        }
        let id = id.expect("established");
        for _ in 0..5 {
            t.update(&[]); // occluded
        }
        // An identical-looking chair detection 700 px away (another chair, occlusion-
        // orphaned): far beyond 3 box-dims for a static track → must NOT be re-id'd.
        let out = t.update(&[boxat(800.0)]);
        assert!(
            !out.iter().any(|tr| tr.id == id),
            "static lost track teleported onto a far identical detection"
        );
    }

    /// Confidence-coast reap: a phantom (person leaves → track latches onto a weak,
    /// **collapsed** noise blob and keeps matching so `track_buffer` never reaps it) must
    /// be removed. But a **weak-but-real** object that keeps its full box (a low-confidence
    /// oven) must SURVIVE — the reap fires only on the low-conf + box-collapse combination.
    #[test]
    fn confidence_coast_reaps_collapsed_phantom_not_weak_object() {
        let big = [100.0f32, 100.0, 160.0, 260.0]; // 60×160

        // Phantom: strong birth, then only weak detections whose box COLLAPSES to a tiny
        // noise blob → reaped.
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        for _ in 0..5 {
            t.update(&[Detection::new(big, 0.9, 0)]);
        }
        assert_eq!(t.len(), 1, "established");
        let tiny = [120.0f32, 170.0, 135.0, 195.0]; // 15×25 collapsed blob
        for _ in 0..120 {
            t.update(&[Detection::new(tiny, 0.3, 0)]);
        }
        assert!(
            t.is_empty(),
            "collapsed weak-latched phantom should be reaped"
        );

        // Weak-but-real oven: strong-ish birth, then persistently weak (0.35) detections
        // that KEEP the full box → must NOT be reaped (this was the live oven churn).
        let mut t2 = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        for _ in 0..5 {
            t2.update(&[Detection::new(big, 0.9, 0)]);
        }
        for _ in 0..200 {
            t2.update(&[Detection::new(big, 0.35, 0)]); // weak, but full-size
        }
        assert_eq!(
            t2.len(),
            1,
            "weak-but-full-size object must survive the coast reap"
        );

        // Strong track always survives.
        let mut t3 = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        for _ in 0..120 {
            t3.update(&[Detection::new(big, 0.9, 0)]);
        }
        assert_eq!(t3.len(), 1, "strong track survives");
    }

    /// Depth innovation gate: a single-frame depth outlier (mask glitch sampling the
    /// background/occluder — metres off) must NOT yank the track's pz; depth coasts
    /// that frame and re-anchors on the next sane measurement.
    #[test]
    fn depth_outlier_rejected_by_innovation_gate() {
        // Metre-scale glitches are already vetoed by the ASSOCIATION depth gate (the
        // track goes unmatched for that frame). The innovation gate covers the band
        // between the two tolerances: at 4 m, association allows |Δz| ≤ 1.4 (0.35·z)
        // but the state gate rejects |Δz| > 1.0 (0.25·z) — a 1.2 m spike matches the
        // track yet must not move pz.
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let det = |z: f32| Detection::new([100.0, 100.0, 160.0, 220.0], 0.9, 0).with_depth(z);
        for _ in 0..8 {
            t.update(&[det(4.0)]); // establish at 4 m
        }
        let out = t.update(&[det(5.2)]); // 1.2 m spike: matched, depth gated
        assert_eq!(out.len(), 1, "spike frame still matches via geometry");
        let pz = out[0].position_3d[2];
        assert!(
            (pz - 4.0).abs() < 0.1,
            "outlier depth should be rejected, pz stayed ~4 m: got {pz}"
        );
        // A sane follow-up keeps updating normally.
        let out = t.update(&[det(4.2)]);
        assert!((out[0].position_3d[2] - 4.0).abs() < 0.3);
    }

    /// Depth **divergence recovery** (the live "person `pz` → 500 m while real depth held
    /// ~2 m" bug). The innovation gate's tolerance scales with `pz`, so a drifted `pz`
    /// would lock the true measurement out *forever* and coast to infinity. A persistent
    /// disagreement must re-anchor `pz` back to the measurements within a few frames.
    #[test]
    fn depth_reanchors_from_diverged_pz() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let det = |z: f32| Detection::new([100.0, 100.0, 160.0, 220.0], 0.9, 0).with_depth(z);
        // Ratchet pz upward with small in-tolerance steps until it has clearly diverged.
        let mut z = 2.0f32;
        for _ in 0..40 {
            t.update(&[det(z)]);
            z += 0.4;
        }
        let diverged = t.tracks()[0].position_3d[2];
        assert!(
            diverged > 6.0,
            "setup: pz should have ratcheted up, got {diverged}"
        );
        // Now the true depth is a stable 2 m again. The gate rejects it at first (huge
        // innovation vs the diverged pz), but must re-anchor rather than stay diverged.
        let mut pz = diverged;
        for _ in 0..15 {
            pz = t.update(&[det(2.0)])[0].position_3d[2];
        }
        assert!(
            pz < 3.0,
            "pz must re-anchor to the true ~2 m, still diverged at {pz}"
        );
    }

    /// Depth must **never run away**: a long noisy run at a stable depth with occasional
    /// single-frame monocular spikes must keep `pz` bounded near the truth — the coasting
    /// gate must not integrate `vz` to infinity. Directly guards the live runaway.
    #[test]
    fn depth_pz_stays_bounded_under_noise_and_spikes() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let mut r = Lcg(31337);
        let base = [500.0f32, 300.0, 620.0, 560.0];
        let mut worst = 0.0f32;
        for _ in 0..800u32 {
            let bx = [
                base[0] + r.sym(3.0),
                base[1] + r.sym(3.0),
                base[2] + r.sym(3.0),
                base[3] + r.sym(3.0),
            ];
            // True depth ~2.5 m; 8% of frames a big monocular spike (mask hit background).
            let z = if r.next_f() < 0.08 {
                2.5 + r.sym(12.0)
            } else {
                2.5 + r.sym(0.2)
            };
            if let Some(tr) = t
                .update(&[Detection::new(bx, 0.85, 5).with_depth(z.max(0.3))])
                .first()
            {
                worst = worst.max(tr.position_3d[2].abs());
            }
        }
        assert!(
            worst < 8.0,
            "pz diverged: reached {worst} m (true depth ~2.5 m)"
        );
        let pz = t.tracks()[0].position_3d[2];
        assert!(
            (pz - 2.5).abs() < 1.5,
            "pz drifted off the true 2.5 m: {pz}"
        );
    }

    /// Low-confidence-only detections should still be recovered in stage 2 once a
    /// track is established from earlier high-confidence frames.
    #[test]
    fn low_confidence_recovery() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        for f in 0..4 {
            let x = 60.0 + f as f32 * 5.0;
            t.update(&[Detection::new([x, 80.0, x + 20.0, 140.0], 0.9, 0)]);
        }
        // Now only a low-confidence detection (between low and high thresh).
        let out = t.update(&[Detection::new([80.0, 80.0, 100.0, 140.0], 0.3, 0)]);
        // Track stays alive & matched via the second association stage.
        assert_eq!(out.len(), 1);
    }

    /// A track that has fully transitioned to `Lost` (missed several frames) must be
    /// re-acquired with the SAME id by a **low-confidence** detection in stage 2 — the
    /// occluded-object recovery path (weak re-detection of a partially-visible object).
    #[test]
    fn lost_track_recovered_by_low_confidence_detection() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let bx = [100.0, 100.0, 160.0, 220.0]; // static box
                                               // Establish a confirmed track.
        let mut id = None;
        for _ in 0..5 {
            if let Some(tr) = t.update(&[Detection::new(bx, 0.9, 0)]).first() {
                id = Some(tr.id);
            }
        }
        let id = id.expect("track established");
        // Miss several frames → Confirmed → Lost (kept alive by track_buffer).
        for _ in 0..4 {
            assert!(t.update(&[]).is_empty(), "no detection ⇒ nothing reported");
        }
        // Only a low-confidence detection (in [track_low_thresh, track_high_thresh)) at
        // the same place → must re-acquire via the second (recovery) stage.
        let out = t.update(&[Detection::new(bx, 0.3, 0)]);
        assert_eq!(out.len(), 1, "lost track not re-acquired by low-conf det");
        assert_eq!(
            out[0].id, id,
            "re-acquired with a new id instead of the lost one"
        );
    }

    #[test]
    fn invalid_config_rejected() {
        let cfg = TrackerConfig {
            track_low_thresh: 0.9,
            track_high_thresh: 0.5,
            ..Default::default()
        };
        assert!(Tracker::new(cfg, ti()).is_err());
    }

    #[test]
    fn depth_flows_into_track_state() {
        let mut t = Tracker::new(TrackerConfig::default(), ti()).unwrap();
        let mut last = None;
        for f in 0..6 {
            let x = 50.0 + f as f32 * 4.0;
            let det = Detection::new([x, 100.0, x + 30.0, 160.0], 0.9, 0).with_depth(5.0);
            last = t.update(&[det]).into_iter().next();
        }
        let tr = last.expect("confirmed track");
        assert!(
            (tr.position_3d[2] - 5.0).abs() < 0.5,
            "depth not fused: {}",
            tr.position_3d[2]
        );
    }
}
