//! **BoT-SORT** multi-object tracker with a **3D** Kalman motion model — a pure-CPU
//! algorithm crate (no TensorRT, no CUDA, no model of its own).
//!
//! Feed per-frame [`Detection`]s (box + score + class, plus optional depth and
//! appearance embedding) and get back stable [`Track`] ids:
//!
//! ```
//! use vrt_track::{BotSort, BotSortConfig, Detection};
//!
//! // Construct once, reuse every frame.
//! let mut tracker = BotSort::new(BotSortConfig::default()).unwrap();
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
//!   ([`association`], [`BotSort::update`]).
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
//! - A **camera-motion compensation** hook ([`gmc`]) — the [`gmc::CameraMotion`]
//!   trait plus an identity stub; plug a real estimator into
//!   [`BotSort::update_with_motion`].
//!
//! Assignment is a compact, dependency-free Hungarian solver; the only external
//! dependency is `nalgebra` for the small fixed-size Kalman matrices.

pub mod association;
pub mod botsort;
pub mod gmc;
pub mod kalman;
pub mod track;

pub use association::iou;
pub use botsort::{BotSort, BotSortConfig};
pub use kalman::{KalmanFilter3D, KalmanParams};
pub use track::{Track, TrackState};
// Camera intrinsics live in the shared `vrt-types` leaf; re-exported for convenience.
pub use vrt_types::CameraIntrinsics;

/// Errors from tracker construction / configuration.
#[derive(Debug, thiserror::Error)]
pub enum TrackError {
    /// The supplied [`BotSortConfig`] is inconsistent.
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

    /// Move a single box left-to-right; the tracker should confirm it and keep one
    /// stable id throughout.
    #[test]
    fn single_target_stable_id() {
        let mut t = BotSort::new(BotSortConfig::default()).unwrap();
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
        let mut t = BotSort::new(BotSortConfig::default()).unwrap();
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
        let mut t = BotSort::new(BotSortConfig::default()).unwrap();
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
        let cfg = BotSortConfig {
            min_hits: 3,
            ..Default::default()
        };
        let mut t = BotSort::new(cfg).unwrap();

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

    /// Low-confidence-only detections should still be recovered in stage 2 once a
    /// track is established from earlier high-confidence frames.
    #[test]
    fn low_confidence_recovery() {
        let mut t = BotSort::new(BotSortConfig::default()).unwrap();
        for f in 0..4 {
            let x = 60.0 + f as f32 * 5.0;
            t.update(&[Detection::new([x, 80.0, x + 20.0, 140.0], 0.9, 0)]);
        }
        // Now only a low-confidence detection (between low and high thresh).
        let out = t.update(&[Detection::new([80.0, 80.0, 100.0, 140.0], 0.3, 0)]);
        // Track stays alive & matched via the second association stage.
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn invalid_config_rejected() {
        let cfg = BotSortConfig {
            track_low_thresh: 0.9,
            track_high_thresh: 0.5,
            ..Default::default()
        };
        assert!(BotSort::new(cfg).is_err());
    }

    #[test]
    fn depth_flows_into_track_state() {
        let mut t = BotSort::new(BotSortConfig::default()).unwrap();
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
