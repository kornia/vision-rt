//! The [`Tracker`]: ByteTrack two-stage association over the 3D Kalman motion model
//! (NSA measurement noise), with a centre-proximity rescue for occlusion-shrunk
//! boxes, a depth-gated 3D association, and an optional appearance-fusion hook.

use crate::association::{
    diou3d_cost_matrix, fuse_center, gate_depth, iou_cost_matrix, linear_assignment,
};
use crate::kalman::KalmanParams;
use crate::track::{Track, TrackState, Tracklet};
use crate::{Detection, TrackError};
use vrt_types::CameraIntrinsics;

/// Image-plane speed (px per nominal frame) below which a track counts as
/// **near-static** and is eligible for the stage-2 centre-proximity rescue. Above it,
/// a coasting track's centre has drifted too far for centre-only matching to be safe.
const STATIC_SPEED_PX: f64 = 8.0;

/// Minimum detection score for updating a track's EMA appearance bank: a low-score
/// detection is usually partially occluded, and its mask-pooled embedding mixes in the
/// occluder — polluting the signature exactly when identity matters most.
#[cfg(feature = "appearance")]
const BANK_MIN_SCORE: f32 = 0.6;

/// Depth-gate penalty used as a **true veto** on the appearance-decides paths (stage-1.5
/// re-id): those stages match on cosine distance alone (∈ `[0, 0.5]`) with no IoU floor,
/// so the soft additive `depth_gate_penalty` (≈ 0.2) is smaller than `reid_thresh` and
/// would let a look-alike at the wrong depth teleport an id. A penalty this large pushes
/// any depth-inconsistent pair past every assignment gate, restoring the hard veto the
/// re-id stage's comments assume — without touching the soft penalty the geometric
/// stages rely on.
#[cfg(feature = "appearance")]
const DEPTH_VETO: f64 = 1.0e6;

/// Configuration for [`Tracker`]. [`Default`] gives sensible defaults tuned for
/// pixel-space boxes.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Detections at/above this score enter the **first** (high-confidence)
    /// association stage.
    pub track_high_thresh: f32,
    /// Detections in `[track_low_thresh, track_high_thresh)` enter the **second**
    /// (low-confidence recovery) stage. Below `track_low_thresh` are discarded.
    pub track_low_thresh: f32,
    /// A leftover high-confidence detection births a new track only if its score is
    /// at least this.
    pub new_track_thresh: f32,
    /// First-stage gate: accept a match when its cost (`1 − IoU`, optionally fused
    /// with appearance) is `≤ match_thresh`.
    pub match_thresh: f32,
    /// Second-stage (low-confidence) IoU gate.
    pub match_thresh_second: f32,
    /// Frames a `Lost` track is kept for re-identification before removal.
    pub track_buffer: u32,
    /// **Confidence-coast reap** (frames): a `Confirmed`/`Lost` track that hasn't matched
    /// a **high-confidence** detection (≥ `track_high_thresh`) for this many frames is
    /// removed. Kills tracks that latch onto a persistent weak (recovery-tier) false
    /// positive — the box collapses onto background noise but keeps "matching", so it
    /// never goes stale via `track_buffer`. A real object re-hits high confidence far
    /// sooner. `0` disables. Default 90 ≈ 6 s at 15 fps.
    pub max_conf_coast: u32,
    /// Matches required for a `Tentative` track to become `Confirmed`.
    pub min_hits: u32,
    /// Kalman noise / init parameters.
    pub kalman: KalmanParams,
    /// Appearance cosine-distance gate (`appearance` feature).
    #[cfg(feature = "appearance")]
    pub appearance_thresh: f32,
    /// IoU-cost proximity gate above which appearance is ignored (`appearance`).
    #[cfg(feature = "appearance")]
    pub proximity_thresh: f32,
    /// EMA momentum for the per-track feature bank (`appearance`).
    #[cfg(feature = "appearance")]
    pub feature_momentum: f32,
    /// **Inactive-track gallery TTL** in frames (`appearance`): when an established
    /// track dies (`Lost` beyond `track_buffer` — e.g. the person left the scene), its
    /// EMA appearance embedding is kept in a gallery for this long. A newborn track
    /// whose embedding matches a gallery entry (class-gated, `reid_thresh` cosine,
    /// uniqueness-abstain) is **resurrected with the old id** — the DeepStream-style
    /// re-association that keeps one identity across scene exits. `0` disables.
    /// Default 1800 ≈ 2 min at 15 fps.
    #[cfg(feature = "appearance")]
    pub gallery_ttl: u32,
    /// Minimum lifetime `hits` for a dying track to enter the gallery (`appearance`) —
    /// flickery short-lived tracks don't deserve resurrection.
    #[cfg(feature = "appearance")]
    pub gallery_min_hits: u32,
    /// **Identity-decision gate** (`appearance`): max appearance cosine distance
    /// (`(1−cos)/2` ∈ [0,1]) for the three appearance-decides mechanisms — the Lost
    /// re-id stage (no IoU requirement), gallery resurrection, and the deferred merge.
    /// Class equality is required, spatial bounds and the depth gate still apply, and
    /// look-alike ambiguity abstains. **Calibrate to the embedder** feeding
    /// [`Detection::feature`](crate::Detection::feature) for the [`reid_classes`]:
    /// a metric-learned re-id net (e.g. OSNet persons: same ≈ 0.05–0.15, different
    /// ≈ 0.25+) works around `0.18`; detector-backbone tokens have NO usable identity
    /// margin under occlusion (different objects at 0.004–0.05, measured live) — do
    /// not let them decide. `0` disables all identity decisions (tie-breaker remains).
    ///
    /// [`reid_classes`]: Self::reid_classes
    #[cfg(feature = "appearance")]
    pub reid_thresh: f32,
    /// Classes whose embeddings are trusted for identity **decisions** (`appearance`):
    /// the Lost re-id stage, gallery resurrection, and deferred merge apply only to
    /// these class ids. Empty = no restriction. Rationale: identity decisions need a
    /// metric-learned embedder (e.g. OSNet for persons, class 1); weaker features
    /// (detector-backbone tokens) collapse under occlusion and must stay tie-breaker
    /// only — set this to exactly the classes with a trained embedding source.
    #[cfg(feature = "appearance")]
    pub reid_classes: Vec<u32>,
    /// Enable **depth-gated association**: reject a track↔detection match whose
    /// metric depth disagrees beyond the tolerance below, when both sides carry a
    /// depth (e.g. a depth crate feeding [`Detection::depth`]). No effect on
    /// depth-less detections. See [`crate::association::gate_depth`].
    ///
    /// [`Detection::depth`]: crate::Detection::depth
    pub depth_gate: bool,
    /// Relative depth tolerance for the gate: a pair is rejected when
    /// `|z_track − z_det| > max(depth_gate_abs, depth_gate_rel · z_track)`. Kept
    /// **loose** (`0.35` = 35 %) so ordinary monocular-depth noise on a static object
    /// doesn't false-reject a valid match (which would churn its ID); genuine
    /// foreground/background separation is far larger and still vetoes a swap.
    pub depth_gate_rel: f32,
    /// Absolute floor (metres) for the depth tolerance, so nearby objects aren't
    /// gated too aggressively when `rel · z` is tiny.
    pub depth_gate_abs: f32,
    /// **Additive** cost penalty for a depth-mismatched pair (not a hard veto). A
    /// strongly-overlapping self-match (low base cost) survives it — so an object's own
    /// detection with a transient monocular-depth spike is NOT rejected (the cause of
    /// static objects churning ids). A genuine crossing is still resolved because the
    /// correct same-depth detection is cheaper. Set very high (≥ `match_thresh`) to
    /// recover the old hard-veto behaviour. Default 0.2.
    pub depth_gate_penalty: f32,
    /// **OC-SORT observation-centric momentum** weight (`0` disables). Adds an angular
    /// penalty to stage-1 association for detections moving *against* a track's observed
    /// velocity direction — stops crossing same-class objects from swapping ids when IoU
    /// ties. Read from raw observations, so it is robust to Kalman velocity noise and a
    /// no-op for stationary tracks. See [`crate::association::fuse_momentum`].
    pub ocm_lambda: f32,
    /// **OC-SORT observation-centric re-update**: when a `Lost` track is re-acquired
    /// after a coasting gap, replay a virtual trajectory from the pre-gap observation to
    /// the re-acquisition box to correct the drifted Kalman state (better than trusting
    /// the stale coasted velocity). Improves id stability through occlusion.
    pub oru: bool,
    /// Use **Distance-IoU** ([`crate::association::diou`]) instead of plain IoU for the
    /// association cost. Discriminates among equal-IoU candidates by centre proximity —
    /// helps when an occlusion-shrunk detection overlaps two tracks equally but sits on
    /// one's centre. Off by default (plain IoU is the validated baseline); the
    /// centre-proximity rescue already covers the common near-static occlusion case.
    pub use_diou: bool,
    /// Use **metric-3D DIoU** ([`crate::association::diou3d`]) for association: 2D pixel
    /// IoU overlap with a **metric** 3D centre penalty (unprojected via the tracker's
    /// intrinsics + per-object depth). Folds depth separation smoothly into the cost —
    /// two image-overlapping boxes at different depths stop matching. Takes precedence
    /// over `use_diou`. Off by default; the depth **gate** already vetoes cross-depth
    /// matches, so this experiments with replacing that hard veto with a soft cost (at
    /// the price of inheriting monocular-depth noise). Needs per-detection depth to have
    /// a 3D effect (falls back to 2D DIoU without it).
    pub use_diou3d: bool,
    /// **Buffered-IoU** margin ([`crate::association::biou`], C-BIoU): expand both boxes by
    /// this fraction of their `(w, h)` before the plain-IoU cost, so a detection whose
    /// seg-mask box *shifted* between frames (a common seg instability on furniture/clutter)
    /// still overlaps its track and re-associates instead of dying → re-birthing under a new
    /// id. `0.0` = plain IoU (default). Only applies on the plain-IoU path (ignored when
    /// `use_diou`/`use_diou3d` are set). Rescues positional shift, **not** wild size
    /// mismatch (buffering scales with box size). Live data: ~1/4 of static-furniture id
    /// churn is box-shift the gate rejects; a moderate `~0.3` recovers it.
    pub iou_buffer: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            track_high_thresh: 0.5,
            track_low_thresh: 0.1,
            new_track_thresh: 0.6,
            match_thresh: 0.8,
            match_thresh_second: 0.5,
            track_buffer: 60, // ~4 s at 15 fps — survive intermittent/occluded detection
            max_conf_coast: 90, // ~6 s — reap tracks latched on weak false positives
            min_hits: 3,
            kalman: KalmanParams::default(),
            #[cfg(feature = "appearance")]
            appearance_thresh: 0.25,
            #[cfg(feature = "appearance")]
            proximity_thresh: 0.5,
            #[cfg(feature = "appearance")]
            feature_momentum: 0.9,
            #[cfg(feature = "appearance")]
            reid_thresh: 0.05,
            #[cfg(feature = "appearance")]
            reid_classes: Vec::new(),
            #[cfg(feature = "appearance")]
            gallery_ttl: 1800,
            #[cfg(feature = "appearance")]
            gallery_min_hits: 15,
            depth_gate: true,
            depth_gate_rel: 0.35,
            depth_gate_abs: 0.7,
            depth_gate_penalty: 0.2,
            ocm_lambda: 0.2,
            oru: true,
            use_diou: false,
            use_diou3d: false,
            iou_buffer: 0.0,
        }
    }
}

impl TrackerConfig {
    fn validate(&self) -> Result<(), TrackError> {
        let bad = |m: &str| Err(TrackError::InvalidConfig(m.to_string()));
        if !(0.0..=1.0).contains(&self.track_high_thresh)
            || !(0.0..=1.0).contains(&self.track_low_thresh)
            || !(0.0..=1.0).contains(&self.new_track_thresh)
        {
            return bad("score thresholds must be in [0, 1]");
        }
        if self.track_low_thresh > self.track_high_thresh {
            return bad("track_low_thresh must not exceed track_high_thresh");
        }
        if self.min_hits == 0 {
            return bad("min_hits must be >= 1");
        }
        if self.depth_gate && (self.depth_gate_rel < 0.0 || self.depth_gate_abs < 0.0) {
            return bad("depth_gate_rel / depth_gate_abs must be >= 0");
        }
        Ok(())
    }
}

/// Robust multi-object tracker. Construct once with [`Tracker::new`], then call
/// [`update`](Self::update) every frame with that frame's detections.
///
/// ```
/// use vrt_track::{Tracker, TrackerConfig, Detection, CameraIntrinsics};
///
/// let intr = CameraIntrinsics::from_hfov(1280.0, 720.0, 70.0);
/// let mut tracker = Tracker::new(TrackerConfig::default(), intr).unwrap();
/// let dets = vec![Detection::new([10.0, 10.0, 40.0, 80.0], 0.9, 0)];
/// let tracks = tracker.update(&dets); // Vec<Track> with stable ids
/// let _ = tracks;
/// ```
/// One identity decision — diagnostics for tuning `reid_thresh` and the spatial
/// bounds. Fired by stage-1.5 re-id (`jump_px` = image jump from the coast), gallery
/// resurrection (`jump_px = -1`), and deferred merge (`jump_px = -2`). Read via
/// [`Tracker::reid_events`].
#[cfg(feature = "appearance")]
#[derive(Debug, Clone)]
pub struct ReidEvent {
    /// Re-acquired track id.
    pub track_id: u64,
    /// Appearance cosine distance of the match (`(1−cos)/2`).
    pub cos_dist: f32,
    /// Image-plane jump from the coasted prediction to the matched detection (px).
    pub jump_px: f32,
    /// Frames the track had been unmatched when re-acquired.
    pub lost_frames: u32,
}

/// A detection is usable only if its box is finite, non-inverted, and has positive
/// area, and its score is finite. Malformed detections (NaN/inf from a broken decode,
/// zero-area or inverted boxes) are dropped before they can poison IoU/Kalman/assignment.
fn valid_det(d: &Detection) -> bool {
    let b = &d.bbox;
    d.score.is_finite() && b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]
}

/// Accept a depth only if it is finite and positive (a monocular estimate must be a
/// real metre value in front of the camera); a NaN/inf/≤0 depth is treated as absent so
/// it never reaches the gate, the 3D DIoU unprojection, or the Kalman.
fn finite_depth(z: Option<f32>) -> Option<f32> {
    z.filter(|v| v.is_finite() && *v > 0.0)
}

/// A dead established track's appearance signature, kept for resurrection.
#[cfg(feature = "appearance")]
struct GalleryEntry {
    id: u64,
    class_id: u32,
    feat: Vec<f32>,
    removed_frame: u64,
}

pub struct Tracker {
    config: TrackerConfig,
    intr: CameraIntrinsics,
    tracks: Vec<Tracklet>,
    next_id: u64,
    /// Frames processed (drives gallery TTL).
    frame: u64,
    #[cfg(feature = "appearance")]
    reid_events: Vec<ReidEvent>,
    #[cfg(feature = "appearance")]
    gallery: Vec<GalleryEntry>,
}

impl Tracker {
    /// Build a tracker. `intr` is the camera intrinsics the metric OC-SORT
    /// momentum/re-update (OCM/ORU) work in — pass the same intrinsics the rest of the
    /// pipeline uses (e.g. [`CameraIntrinsics::from_hfov`]). Returns
    /// [`TrackError::InvalidConfig`] on a nonsensical configuration.
    pub fn new(config: TrackerConfig, intr: CameraIntrinsics) -> Result<Self, TrackError> {
        config.validate()?;
        Ok(Self {
            config,
            intr,
            tracks: Vec::new(),
            next_id: 1,
            frame: 0,
            #[cfg(feature = "appearance")]
            reid_events: Vec::new(),
            #[cfg(feature = "appearance")]
            gallery: Vec::new(),
        })
    }

    /// Stage-1.5 re-id matches from the **most recent** `update` call (cleared each
    /// frame) — diagnostics for tuning `reid_thresh` / watching for teleports.
    #[cfg(feature = "appearance")]
    pub fn reid_events(&self) -> &[ReidEvent] {
        &self.reid_events
    }

    /// Drop all tracks and reset ids (and the resurrection gallery).
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
        self.frame = 0;
        #[cfg(feature = "appearance")]
        self.gallery.clear();
    }

    /// Read-only view of every live (`Confirmed`/`Lost`/`Tentative`) track.
    pub fn tracks(&self) -> Vec<Track> {
        self.tracks
            .iter()
            .filter(|t| t.state != TrackState::Removed)
            .map(Tracklet::to_track)
            .collect()
    }

    /// Number of live tracks.
    pub fn len(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.state != TrackState::Removed)
            .count()
    }

    /// Whether there are no live tracks.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Advance one frame (fixed cadence, `dt = 1`).
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        self.step(detections, 1.0)
    }

    /// Advance by a real inter-frame interval `dt`. Pass the elapsed time since the
    /// previous frame in the same units the [`KalmanParams`] are tuned for (nominal
    /// frames — e.g. `actual_interval / nominal_interval`), so the constant-velocity
    /// prediction stays consistent under variable fps and dropped frames. `dt = 1.0`
    /// is identical to [`update`](Self::update).
    pub fn update_dt(&mut self, detections: &[Detection], dt: f64) -> Vec<Track> {
        self.step(detections, dt)
    }

    /// Depth-gate one stage's cost matrix in place (no-op when `depth_gate` is off):
    /// build the track (measured pz) + detection depth vectors from the two index
    /// slices, then hard-reject depth-mismatched pairs. The single 3D-association
    /// mechanism, shared by all three stages. The tolerance is deliberately loose
    /// (`depth_gate_rel`/`abs`) so ordinary monocular-depth noise on a static object
    /// doesn't false-reject a valid match, while genuine cross-depth separation still
    /// vetoes an ID swap.
    fn gate_stage(
        &self,
        cost: &mut [Vec<f64>],
        track_idx: &[usize],
        det_idx: &[usize],
        detections: &[Detection],
        penalty: f64,
    ) {
        let cfg = &self.config;
        if !cfg.depth_gate {
            return;
        }
        let track_d: Vec<Option<f32>> = track_idx
            .iter()
            .map(|&ti| self.tracks[ti].measured_depth())
            .collect();
        let det_d: Vec<Option<f32>> = det_idx
            .iter()
            .map(|&i| finite_depth(detections[i].depth))
            .collect();
        gate_depth(
            cost,
            &track_d,
            &det_d,
            cfg.depth_gate_rel,
            cfg.depth_gate_abs,
            penalty,
        );
    }

    /// Build a stage's association cost matrix, honoring the IoU variant config:
    /// metric-3D DIoU (`use_diou3d`, needs depths + intrinsics) > 2D DIoU (`use_diou`) >
    /// plain/buffered IoU (`iou_buffer`). `track_idx`/`det_idx` index into
    /// `self.tracks`/`detections` to fetch per-object depth for the 3D variant.
    fn assoc_cost(
        &self,
        cfg: &TrackerConfig,
        track_idx: &[usize],
        track_boxes: &[[f32; 4]],
        det_idx: &[usize],
        det_boxes: &[[f32; 4]],
        detections: &[Detection],
    ) -> Vec<Vec<f64>> {
        if cfg.use_diou3d {
            let tz: Vec<Option<f32>> = track_idx
                .iter()
                .map(|&ti| self.tracks[ti].measured_depth())
                .collect();
            let dz: Vec<Option<f32>> = det_idx
                .iter()
                .map(|&i| finite_depth(detections[i].depth))
                .collect();
            diou3d_cost_matrix(track_boxes, det_boxes, &tz, &dz, &self.intr)
        } else {
            iou_cost_matrix(track_boxes, det_boxes, cfg.use_diou, cfg.iou_buffer)
        }
    }

    /// Whether identity DECISIONS (lost re-id / gallery / merge) may act on `class`
    /// (`reid_classes` empty = no restriction).
    #[cfg(feature = "appearance")]
    fn reid_class_ok(cfg: &TrackerConfig, class: u32) -> bool {
        cfg.reid_classes.is_empty() || cfg.reid_classes.contains(&class)
    }

    /// The core update — [`update`](Self::update) / [`update_dt`](Self::update_dt)
    /// delegate here. `dt` scales the Kalman predict; the lifecycle counters still
    /// advance per frame.
    fn step(&mut self, detections: &[Detection], dt: f64) -> Vec<Track> {
        let cfg = self.config.clone();
        let intr = self.intr; // Copy — frees `self.tracks` for mutable borrows below
        #[cfg(feature = "appearance")]
        self.reid_events.clear();

        // Sanitize dt: a non-finite / non-positive step would poison the Kalman F/Q; an
        // absurd one would explode the covariance. Fall back to one nominal frame and cap.
        let dt = if dt.is_finite() && dt > 0.0 {
            dt.min(1000.0)
        } else {
            1.0
        };

        // Partition detections by confidence — **skipping malformed detections** (a
        // non-finite or degenerate box would propagate NaN through IoU/Kalman and wedge
        // the assignment). Invalid detections enter no stage and birth nothing.
        let mut high_det = Vec::new();
        let mut low_det = Vec::new();
        for (i, d) in detections.iter().enumerate() {
            if !valid_det(d) {
                continue;
            }
            if d.score >= cfg.track_high_thresh {
                high_det.push(i);
            } else if d.score >= cfg.track_low_thresh {
                low_det.push(i);
            }
        }

        // Predict every live track forward by `dt`.
        for t in &mut self.tracks {
            if t.state != TrackState::Removed {
                t.predict(dt);
            }
        }

        // Pool = confirmed + lost (re-identifiable); unconfirmed = tentative.
        let pool: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| {
                matches!(
                    self.tracks[i].state,
                    TrackState::Confirmed | TrackState::Lost
                )
            })
            .collect();
        let unconfirmed: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| self.tracks[i].state == TrackState::Tentative)
            .collect();

        // ---- Stage 1: pool vs high-confidence detections (IoU + appearance) ----
        let high_boxes: Vec<[f32; 4]> = high_det.iter().map(|&i| detections[i].bbox).collect();
        let pool_boxes: Vec<[f32; 4]> = pool.iter().map(|&ti| self.tracks[ti].bbox()).collect();
        // Stage 1 uses strict IoU (+ appearance): these tracks matched a strong
        // detection recently, so their boxes are fresh and IoU is precise. A
        // centre-proximity rescue here would let adjacent same-depth objects swap;
        // it is confined to the stage-2 recovery pass below.
        #[cfg_attr(not(feature = "appearance"), allow(unused_mut))]
        let mut cost1 =
            self.assoc_cost(&cfg, &pool, &pool_boxes, &high_det, &high_boxes, detections);
        #[cfg(feature = "appearance")]
        {
            let track_feats: Vec<Option<Vec<f32>>> = pool
                .iter()
                .map(|&ti| self.tracks[ti].smooth_feat.clone())
                .collect();
            let det_feats: Vec<Option<&[f32]>> = high_det
                .iter()
                .map(|&i| detections[i].feature.as_deref())
                .collect();
            crate::association::fuse_appearance(
                &mut cost1,
                &track_feats,
                &det_feats,
                cfg.appearance_thresh,
                cfg.proximity_thresh,
            );
        }
        // OC-SORT momentum (metric): penalize matches whose world direction disagrees
        // with the track's observed 3D velocity, before the gate, so two crossing
        // same-class objects don't swap when IoU/appearance tie. Depth-aware — sees
        // motion toward/away from the camera, not just lateral image motion.
        if cfg.ocm_lambda > 0.0 {
            let track_dir: Vec<Option<[f64; 3]>> = pool
                .iter()
                .map(|&ti| self.tracks[ti].obs_direction())
                .collect();
            let track_center: Vec<[f64; 3]> = pool
                .iter()
                .map(|&ti| self.tracks[ti].last_world())
                .collect();
            let det_center: Vec<Option<[f64; 3]>> = high_det
                .iter()
                .map(|&i| {
                    let b = &detections[i].bbox;
                    let (cx, cy) = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5);
                    // Launder depth: a NaN/inf/≤0 would unproject to a NaN world point,
                    // and fuse_momentum's `n < 1e-3` guard is false for NaN → NaN leaks
                    // into the cost matrix and poisons the Hungarian assignment.
                    finite_depth(detections[i].depth).map(|z| {
                        let p = intr.unproject(cx, cy, z);
                        [p[0] as f64, p[1] as f64, p[2] as f64]
                    })
                })
                .collect();
            crate::association::fuse_momentum(
                &mut cost1,
                &track_dir,
                &track_center,
                &det_center,
                cfg.ocm_lambda,
            );
        }
        // Depth gate last, so it overrides an appearance rescue on a depth-mismatched
        // pair (two similar-looking objects at different distances).
        self.gate_stage(
            &mut cost1,
            &pool,
            &high_det,
            detections,
            cfg.depth_gate_penalty as f64,
        );
        let (m1, u_pool, u_high) =
            linear_assignment(&cost1, pool.len(), high_det.len(), cfg.match_thresh as f64);
        for (pi, di) in m1 {
            let ti = pool[pi];
            let det = &detections[high_det[di]];
            self.tracks[ti].update(det, cfg.min_hits, cfg.oru, &intr);
            if det.score >= cfg.track_high_thresh {
                self.tracks[ti].note_high_conf();
            }
            #[cfg(feature = "appearance")]
            if det.score >= BANK_MIN_SCORE {
                if let Some(f) = det.feature.as_deref() {
                    self.tracks[ti].smooth_feature(f, cfg.feature_momentum);
                }
            }
        }

        // ---- Stage 1.5 (`appearance`): Lost-track re-id, appearance-first ----
        // A Lost track's coasted prediction has drifted (the object turned, or was
        // occluded while moving), so IoU with its reappearance box is ~0 and stage 1
        // can never re-acquire it — the *defining* job of appearance ReID. Match still-
        // unmatched **Lost** tracks against leftover strong detections purely by
        // embedding cosine — no IoU — gated by class equality, a TIGHT cosine gate
        // (`reid_thresh`; backbone embeddings are compressed, see config docs), and the
        // depth hard veto. Matched tracks go through the normal update, so ORU replays
        // the occlusion gap.
        #[cfg(feature = "appearance")]
        let (u_pool, u_high) = {
            let mut u_pool = u_pool;
            let mut u_high = u_high;
            if cfg.reid_thresh > 0.0 {
                let lost: Vec<usize> = u_pool
                    .iter()
                    .copied()
                    .filter(|&pi| {
                        let t = &self.tracks[pool[pi]];
                        t.state == TrackState::Lost
                            && t.smooth_feat.is_some()
                            && Self::reid_class_ok(&cfg, t.class_id)
                    })
                    .collect();
                let cands: Vec<usize> = u_high
                    .iter()
                    .copied()
                    .filter(|&hi| detections[high_det[hi]].feature.is_some())
                    .collect();
                if !lost.is_empty() && !cands.is_empty() {
                    // Large finite sentinel (not f64::MAX — Hungarian row/col reductions
                    // subtract costs and infinities would breed NaNs).
                    let mut cost = vec![vec![1.0e6_f64; cands.len()]; lost.len()];
                    for (r, &pi) in lost.iter().enumerate() {
                        let t = &self.tracks[pool[pi]];
                        let tf = t.smooth_feat.as_deref().expect("filtered above");
                        // Observed-motion spatial bound, measured from the LAST
                        // OBSERVATION (the coasted prediction is exactly what's wrong
                        // during a turn-around): an object seen static may only re-id
                        // near where it was last seen (~3 box dims — identical objects
                        // are only separable by position); one seen moving at v px/frame
                        // gets `2·v·lost` of slack. Blocks cross-room teleports between
                        // look-alike objects while allowing the person-turned re-acquire.
                        let (vx, vy, _) = {
                            let v = t.kf.velocity_3d();
                            (v[0], v[1], v[2])
                        };
                        let speed = (vx * vx + vy * vy).sqrt();
                        let bound =
                            (3.0 * t.last_box_dim()).max(2.0 * speed * t.time_since_update as f64);
                        let (lx, ly) = t.last_center_px();
                        for (c, &hi) in cands.iter().enumerate() {
                            let det = &detections[high_det[hi]];
                            if det.class_id != t.class_id {
                                continue; // re-id never crosses classes
                            }
                            let (cx, cy) = (
                                ((det.bbox[0] + det.bbox[2]) * 0.5) as f64,
                                ((det.bbox[1] + det.bbox[3]) * 0.5) as f64,
                            );
                            let (jx, jy) = (cx - lx, cy - ly);
                            if (jx * jx + jy * jy).sqrt() > bound {
                                continue;
                            }
                            let df = det.feature.as_deref().expect("filtered above");
                            cost[r][c] = crate::association::cosine_distance(tf, df) as f64 / 2.0;
                        }
                        // Uniqueness margin: when the best and runner-up candidates are
                        // within half the cosine gate of each other, appearance has no
                        // real opinion (identical same-class objects) — ABSTAIN rather
                        // than guess, and let the track re-acquire geometrically or die.
                        let mut best = f64::MAX;
                        let mut second = f64::MAX;
                        for &v in cost[r].iter() {
                            if v < best {
                                second = best;
                                best = v;
                            } else if v < second {
                                second = v;
                            }
                        }
                        if second < 1.0e6 && (second - best) < (cfg.reid_thresh as f64) * 0.5 {
                            for v in cost[r].iter_mut() {
                                *v = 1.0e6;
                            }
                        }
                    }
                    // Depth stays a HARD veto even on a perfect appearance match: this
                    // stage has no IoU floor, so it needs the veto magnitude (`DEPTH_VETO`),
                    // not the soft additive penalty the geometric stages use.
                    let track_idx: Vec<usize> = lost.iter().map(|&pi| pool[pi]).collect();
                    let det_idx: Vec<usize> = cands.iter().map(|&hi| high_det[hi]).collect();
                    self.gate_stage(&mut cost, &track_idx, &det_idx, detections, DEPTH_VETO);
                    let (mr, _, _) =
                        linear_assignment(&cost, lost.len(), cands.len(), cfg.reid_thresh as f64);
                    for (r, c) in mr {
                        let ti = pool[lost[r]];
                        let det = &detections[det_idx[c]];
                        // Diagnostics before the update overwrites the coasted state.
                        {
                            let t = &self.tracks[ti];
                            let tb = t.bbox();
                            let (tx, ty) = ((tb[0] + tb[2]) * 0.5, (tb[1] + tb[3]) * 0.5);
                            let (dx, dy) = (
                                (det.bbox[0] + det.bbox[2]) * 0.5 - tx,
                                (det.bbox[1] + det.bbox[3]) * 0.5 - ty,
                            );
                            self.reid_events.push(ReidEvent {
                                track_id: t.id,
                                cos_dist: cost[r][c] as f32,
                                jump_px: (dx * dx + dy * dy).sqrt(),
                                lost_frames: t.time_since_update,
                            });
                        }
                        self.tracks[ti].update(det, cfg.min_hits, cfg.oru, &intr);
                        if det.score >= cfg.track_high_thresh {
                            self.tracks[ti].note_high_conf();
                        }
                        if det.score >= BANK_MIN_SCORE {
                            if let Some(f) = det.feature.as_deref() {
                                self.tracks[ti].smooth_feature(f, cfg.feature_momentum);
                            }
                        }
                        u_pool.retain(|&pi| pi != lost[r]);
                        u_high.retain(|&hi| hi != cands[c]);
                    }
                }
            }
            (u_pool, u_high)
        };

        // ---- Stage 2: unmatched pool (Confirmed + Lost) tracks vs low dets ----
        // Every still-unmatched pool track chases low-confidence boxes, so an object
        // the detector only fires on weakly while partially occluded (a half-hidden
        // chair) re-acquires its id instead of churning. The centre-proximity fuse
        // below is what makes this land: the occlusion-shrunk box has poor IoU but its
        // centre still sits on the track's coasting position. The depth gate stays the
        // hard veto, so a weak box can't steal an id across a depth gap. (`pool` is
        // already exactly Confirmed|Lost, so no further state filter is needed here.)
        let r_tracked: Vec<usize> = u_pool.iter().map(|&pi| pool[pi]).collect();
        let low_boxes: Vec<[f32; 4]> = low_det.iter().map(|&i| detections[i].bbox).collect();
        let r_boxes: Vec<[f32; 4]> = r_tracked.iter().map(|&ti| self.tracks[ti].bbox()).collect();
        // Restrict the centre-proximity rescue to near-static tracks: a fast-coasting
        // track's predicted centre has drifted, so centre-only matching could capture a
        // different object that drifted into range. A slow/stationary track (the
        // occluded-chair case) is safe to rescue.
        let r_static: Vec<bool> = r_tracked
            .iter()
            .map(|&ti| {
                let v = self.tracks[ti].kf.velocity_3d();
                (v[0] * v[0] + v[1] * v[1]).sqrt() < STATIC_SPEED_PX
            })
            .collect();
        let mut cost2 =
            self.assoc_cost(&cfg, &r_tracked, &r_boxes, &low_det, &low_boxes, detections);
        fuse_center(&mut cost2, &r_boxes, &low_boxes, &r_static);
        self.gate_stage(
            &mut cost2,
            &r_tracked,
            &low_det,
            detections,
            cfg.depth_gate_penalty as f64,
        );
        let (m2, _u_r, _u_low) = linear_assignment(
            &cost2,
            r_tracked.len(),
            low_det.len(),
            cfg.match_thresh_second as f64,
        );
        for (ri, di) in m2 {
            let ti = r_tracked[ri];
            self.tracks[ti].update(&detections[low_det[di]], cfg.min_hits, cfg.oru, &intr);
        }

        // Pool tracks unmatched after both stages -> missed (Confirmed -> Lost).
        for &ti in &pool {
            if !self.tracks[ti].matched_this_frame {
                self.tracks[ti].mark_missed();
            }
        }

        // ---- Stage 3: unconfirmed (tentative) vs leftover high dets ----
        let remaining_high: Vec<usize> = u_high.iter().map(|&hi| high_det[hi]).collect();
        let rem_boxes: Vec<[f32; 4]> = remaining_high.iter().map(|&i| detections[i].bbox).collect();
        let unconf_boxes: Vec<[f32; 4]> = unconfirmed
            .iter()
            .map(|&ti| self.tracks[ti].bbox())
            .collect();
        #[cfg_attr(not(feature = "appearance"), allow(unused_mut))]
        let mut cost3 = self.assoc_cost(
            &cfg,
            &unconfirmed,
            &unconf_boxes,
            &remaining_high,
            &rem_boxes,
            detections,
        );
        #[cfg(feature = "appearance")]
        {
            let track_feats: Vec<Option<Vec<f32>>> = unconfirmed
                .iter()
                .map(|&ti| self.tracks[ti].smooth_feat.clone())
                .collect();
            let det_feats: Vec<Option<&[f32]>> = remaining_high
                .iter()
                .map(|&i| detections[i].feature.as_deref())
                .collect();
            crate::association::fuse_appearance(
                &mut cost3,
                &track_feats,
                &det_feats,
                cfg.appearance_thresh,
                cfg.proximity_thresh,
            );
        }
        self.gate_stage(
            &mut cost3,
            &unconfirmed,
            &remaining_high,
            detections,
            cfg.depth_gate_penalty as f64,
        );
        let (m3, u_unconf, u_rem) = linear_assignment(
            &cost3,
            unconfirmed.len(),
            remaining_high.len(),
            cfg.match_thresh as f64,
        );
        for (ui, di) in m3 {
            let ti = unconfirmed[ui];
            let det = &detections[remaining_high[di]];
            self.tracks[ti].update(det, cfg.min_hits, cfg.oru, &intr);
            if det.score >= cfg.track_high_thresh {
                self.tracks[ti].note_high_conf();
            }
            #[cfg(feature = "appearance")]
            if det.score >= BANK_MIN_SCORE {
                if let Some(f) = det.feature.as_deref() {
                    self.tracks[ti].smooth_feature(f, cfg.feature_momentum);
                }
            }
        }
        // Unmatched tentative tracks die immediately.
        for &ui in &u_unconf {
            self.tracks[unconfirmed[ui]].mark_missed();
        }

        // ---- Deferred identity merge (`appearance`) ----
        // Matching only at birth is a single-instant gamble (the first det may be a
        // half-body at the frame edge, or the old track may still be Lost). Production
        // trackers match over a WINDOW and rewrite the id (DeepStream re-association,
        // Axis merge). Here: during a young track's first ~3 s, each clean match tries to
        // claim an OLDER identity — from the dead-track gallery or a long-Lost same-class
        // twin — with the usual tight-cosine + uniqueness-abstain gates. Ids only move
        // toward older values, so duplicate identities converge to one canonical id.
        #[cfg(feature = "appearance")]
        if cfg.reid_thresh > 0.0 {
            const MERGE_WINDOW: u32 = 45; // frames — a young track's first ~3 s
            const TWIN_MIN_LOST: u32 = 8; // twin must be genuinely gone, not a flicker
            let thresh = cfg.reid_thresh as f64;
            for yi in 0..self.tracks.len() {
                let y = &self.tracks[yi];
                if y.age > MERGE_WINDOW
                    || !y.matched_this_frame
                    || y.state != TrackState::Confirmed
                    || y.score < BANK_MIN_SCORE
                    || y.claimed
                    || !Self::reid_class_ok(&cfg, y.class_id)
                {
                    continue;
                }
                let Some(yf) = y.smooth_feat.clone() else {
                    continue;
                };
                let (yid, yclass) = (y.id, y.class_id);
                // Candidates: gallery entries and long-Lost twins with OLDER ids.
                enum Cand {
                    Gal(usize),
                    Twin(usize),
                }
                let (mut best, mut second) = (f64::MAX, f64::MAX);
                let mut best_c: Option<Cand> = None;
                for (gi, g) in self.gallery.iter().enumerate() {
                    if g.class_id != yclass || g.id >= yid {
                        continue;
                    }
                    let d = crate::association::cosine_distance(&g.feat, &yf) as f64 / 2.0;
                    if d < best {
                        second = best;
                        best = d;
                        best_c = Some(Cand::Gal(gi));
                    } else if d < second {
                        second = d;
                    }
                }
                for (ti2, t2) in self.tracks.iter().enumerate() {
                    if ti2 == yi
                        || t2.class_id != yclass
                        || t2.id >= yid
                        || t2.state != TrackState::Lost
                        || t2.time_since_update < TWIN_MIN_LOST
                    {
                        continue;
                    }
                    let Some(tf) = t2.smooth_feat.as_deref() else {
                        continue;
                    };
                    let d = crate::association::cosine_distance(tf, &yf) as f64 / 2.0;
                    if d < best {
                        second = best;
                        best = d;
                        best_c = Some(Cand::Twin(ti2));
                    } else if d < second {
                        second = d;
                    }
                }
                if best > thresh || (second < f64::MAX && (second - best) < thresh * 0.5) {
                    continue; // no match, or ambiguous look-alikes → keep the young id
                }
                let old_id = match best_c {
                    Some(Cand::Gal(gi)) => {
                        let e = self.gallery.swap_remove(gi);
                        e.id
                    }
                    Some(Cand::Twin(ti2)) => {
                        // Absorb the twin: the young track IS the old identity returned.
                        let (tid, thits) = (self.tracks[ti2].id, self.tracks[ti2].hits);
                        self.tracks[ti2].state = TrackState::Removed;
                        self.tracks[ti2].hits = 0; // don't let the husk enter the gallery
                        self.tracks[yi].hits += thits;
                        tid
                    }
                    None => continue,
                };
                self.tracks[yi].id = old_id;
                self.tracks[yi].claimed = true; // one identity claim per track, ever
                self.reid_events.push(ReidEvent {
                    track_id: old_id,
                    cos_dist: best as f32,
                    jump_px: -2.0, // sentinel: deferred identity merge
                    lost_frames: self.tracks[yi].age,
                });
            }
        }

        // ---- Birth new tracks from the still-unmatched high detections ----
        for &ri in &u_rem {
            let di = remaining_high[ri];
            if detections[di].score >= cfg.new_track_thresh {
                // Duplicate-birth guard: an NMS-free DETR run at a low decode threshold
                // leaks duplicate boxes on one object; the best one matched its track in
                // stage 1, so a leftover det sitting ON a live same-class track is a
                // duplicate, not a new object — birthing it creates a doppelgänger track
                // that later poisons the re-id gallery with copies of the same identity.
                let dup = self.tracks.iter().any(|t| {
                    t.state != TrackState::Removed
                        && t.class_id == detections[di].class_id
                        && crate::association::iou(&t.bbox(), &detections[di].bbox) > 0.6
                });
                if dup {
                    continue;
                }
                // Resurrection first (`appearance`): if this newborn's embedding matches
                // a dead established track in the gallery, it IS that object returning
                // (the person came back into the scene) — revive the old id, instantly
                // re-Confirmed, instead of minting a fresh one.
                #[cfg(feature = "appearance")]
                if let Some(old_id) = self.gallery_match(&detections[di], &cfg) {
                    let mut t = Tracklet::new(old_id, &detections[di], cfg.kalman, &intr);
                    t.state = TrackState::Confirmed; // identity is established, not new
                    t.hits = cfg.min_hits;
                    t.claimed = true; // no chaining into further identity claims
                    self.tracks.push(t);
                    continue;
                }
                let id = self.next_id;
                self.next_id += 1;
                self.tracks
                    .push(Tracklet::new(id, &detections[di], cfg.kalman, &intr));
            }
        }

        // ---- Reap dead / expired tracks (established ones retire INTO the gallery) ----
        let buffer = cfg.track_buffer;
        let mut i = 0;
        while i < self.tracks.len() {
            let t = &self.tracks[i];
            // Confidence-coast reap fires only for a track that (a) hasn't seen a
            // high-confidence detection for a long time AND (b) whose box has COLLAPSED
            // far below its established size — the phantom (person-left → noise blob)
            // signature. A weak-but-real object (low-confidence oven) keeps its size and
            // survives, so it isn't churned.
            let stale = cfg.max_conf_coast > 0
                && matches!(t.state, TrackState::Confirmed | TrackState::Lost)
                && t.low_conf_age > cfg.max_conf_coast
                && t.area() < 0.35 * t.max_area;
            let dead = t.state == TrackState::Removed
                || (t.state == TrackState::Lost && t.time_since_update > buffer)
                || stale;
            if dead {
                #[cfg(feature = "appearance")]
                {
                    let t = self.tracks.swap_remove(i);
                    // Death-bed merge: a duplicate twin often OUTLIVES the original (the
                    // detection stream drifts to the newer box hypothesis and the old
                    // track starves). If a LIVE same-class track is an embedding twin of
                    // the dying established track, the live one inherits the OLDER id —
                    // identity continues instead of going to the grave and re-birthing.
                    if cfg.gallery_ttl > 0
                        && t.hits >= cfg.gallery_min_hits
                        && Self::reid_class_ok(&cfg, t.class_id)
                    {
                        if let Some(tf) = t.smooth_feat.as_deref() {
                            let thresh = cfg.reid_thresh as f64;
                            let mut best = f64::MAX;
                            let mut best_j = usize::MAX;
                            for (j, l) in self.tracks.iter().enumerate() {
                                if l.class_id != t.class_id
                                    || l.id <= t.id
                                    || l.state == TrackState::Removed
                                {
                                    continue;
                                }
                                let Some(lf) = l.smooth_feat.as_deref() else {
                                    continue;
                                };
                                let d = crate::association::cosine_distance(tf, lf) as f64 / 2.0;
                                if d < best {
                                    best = d;
                                    best_j = j;
                                }
                            }
                            if thresh > 0.0 && best <= thresh && best_j != usize::MAX {
                                let heir = &mut self.tracks[best_j];
                                heir.id = t.id;
                                heir.hits += t.hits;
                                heir.claimed = true;
                                self.reid_events.push(ReidEvent {
                                    track_id: t.id,
                                    cos_dist: best as f32,
                                    jump_px: -3.0, // sentinel: death-bed identity handoff
                                    lost_frames: t.time_since_update,
                                });
                                continue; // identity lives on — nothing enters the gallery
                            }
                        }
                    }
                    if cfg.gallery_ttl > 0 && t.hits >= cfg.gallery_min_hits {
                        if let Some(feat) = t.smooth_feat {
                            // Dedup on insert: if an existing same-class entry already IS
                            // this identity (tight cosine), merge — keep the OLDER id as
                            // the canonical one and refresh the embedding + TTL. Without
                            // this, duplicate tracks of one object fill the gallery with
                            // same-identity copies and the uniqueness-abstain check then
                            // vetoes every legitimate resurrection as "ambiguous".
                            let twin = self.gallery.iter_mut().find(|g| {
                                g.class_id == t.class_id
                                    && crate::association::cosine_distance(&g.feat, &feat) / 2.0
                                        <= cfg.reid_thresh
                            });
                            match twin {
                                Some(g) => {
                                    g.id = g.id.min(t.id); // oldest id = canonical identity
                                    g.feat = feat;
                                    g.removed_frame = self.frame;
                                }
                                None => self.gallery.push(GalleryEntry {
                                    id: t.id,
                                    class_id: t.class_id,
                                    feat,
                                    removed_frame: self.frame,
                                }),
                            }
                        }
                    }
                }
                #[cfg(not(feature = "appearance"))]
                {
                    self.tracks.swap_remove(i);
                }
            } else {
                i += 1;
            }
        }
        // Expire stale gallery entries; cap the size (oldest out).
        #[cfg(feature = "appearance")]
        {
            let ttl = cfg.gallery_ttl as u64;
            let now = self.frame;
            self.gallery
                .retain(|g| now.saturating_sub(g.removed_frame) <= ttl);
            const GALLERY_CAP: usize = 64;
            if self.gallery.len() > GALLERY_CAP {
                let excess = self.gallery.len() - GALLERY_CAP;
                self.gallery.drain(0..excess);
            }
        }
        self.frame += 1;

        // Output: confirmed tracks that matched a detection this frame.
        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed && t.matched_this_frame)
            .map(Tracklet::to_track)
            .collect()
    }

    /// Match a birthing detection against the dead-track gallery: class-gated, tight
    /// cosine (`reid_thresh`), and the same uniqueness-abstain as live re-id (two
    /// look-alike gallery entries → no opinion → fresh id). On a match the entry is
    /// consumed and its id returned for resurrection.
    #[cfg(feature = "appearance")]
    fn gallery_match(&mut self, det: &Detection, cfg: &TrackerConfig) -> Option<u64> {
        if cfg.gallery_ttl == 0 || cfg.reid_thresh <= 0.0 {
            return None;
        }
        if !Self::reid_class_ok(cfg, det.class_id) {
            return None;
        }
        let df = det.feature.as_deref()?;
        let (mut best, mut second, mut best_i) = (f64::MAX, f64::MAX, usize::MAX);
        for (i, g) in self.gallery.iter().enumerate() {
            if g.class_id != det.class_id {
                continue;
            }
            let d = crate::association::cosine_distance(&g.feat, df) as f64 / 2.0;
            if d < best {
                second = best;
                best = d;
                best_i = i;
            } else if d < second {
                second = d;
            }
        }
        let thresh = cfg.reid_thresh as f64;
        if best > thresh {
            return None;
        }
        if second < f64::MAX && (second - best) < thresh * 0.5 {
            return None; // ambiguous look-alikes → abstain
        }
        let entry = self.gallery.swap_remove(best_i);
        self.reid_events.push(ReidEvent {
            track_id: entry.id,
            cos_dist: best as f32,
            jump_px: -1.0, // sentinel: gallery resurrection (no meaningful jump)
            lost_frames: (self.frame - entry.removed_frame) as u32,
        });
        Some(entry.id)
    }
}
