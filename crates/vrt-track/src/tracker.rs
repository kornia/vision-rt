//! The [`Tracker`]: ByteTrack two-stage association over the 3D Kalman motion model
//! (NSA measurement noise), with a depth-gated 3D association and an optional
//! appearance-fusion hook.

use crate::association::{gate_depth, iou_cost_matrix, linear_assignment};
use crate::kalman::KalmanParams;
use crate::track::{Track, TrackState, Tracklet};
use crate::{Detection, TrackError};

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
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            track_high_thresh: 0.5,
            track_low_thresh: 0.1,
            new_track_thresh: 0.6,
            match_thresh: 0.8,
            match_thresh_second: 0.5,
            track_buffer: 30,
            min_hits: 3,
            kalman: KalmanParams::default(),
            #[cfg(feature = "appearance")]
            appearance_thresh: 0.25,
            #[cfg(feature = "appearance")]
            proximity_thresh: 0.5,
            #[cfg(feature = "appearance")]
            feature_momentum: 0.9,
            depth_gate: true,
            depth_gate_rel: 0.35,
            depth_gate_abs: 0.7,
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
/// use vrt_track::{Tracker, TrackerConfig, Detection};
///
/// let mut tracker = Tracker::new(TrackerConfig::default()).unwrap();
/// let dets = vec![Detection::new([10.0, 10.0, 40.0, 80.0], 0.9, 0)];
/// let tracks = tracker.update(&dets); // Vec<Track> with stable ids
/// let _ = tracks;
/// ```
pub struct Tracker {
    config: TrackerConfig,
    tracks: Vec<Tracklet>,
    next_id: u64,
}

impl Tracker {
    /// Build a tracker. Returns [`TrackError::InvalidConfig`] on a nonsensical
    /// configuration.
    pub fn new(config: TrackerConfig) -> Result<Self, TrackError> {
        config.validate()?;
        Ok(Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        })
    }

    /// Drop all tracks and reset ids.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
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
    ) {
        let cfg = &self.config;
        if !cfg.depth_gate {
            return;
        }
        let track_d: Vec<Option<f32>> = track_idx
            .iter()
            .map(|&ti| self.tracks[ti].measured_depth())
            .collect();
        let det_d: Vec<Option<f32>> = det_idx.iter().map(|&i| detections[i].depth).collect();
        gate_depth(
            cost,
            &track_d,
            &det_d,
            cfg.depth_gate_rel,
            cfg.depth_gate_abs,
        );
    }

    /// The core update — [`update`](Self::update) / [`update_dt`](Self::update_dt)
    /// delegate here. `dt` scales the Kalman predict; the lifecycle counters still
    /// advance per frame.
    fn step(&mut self, detections: &[Detection], dt: f64) -> Vec<Track> {
        let cfg = self.config.clone();

        // Partition detections by confidence.
        let mut high_det = Vec::new();
        let mut low_det = Vec::new();
        for (i, d) in detections.iter().enumerate() {
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
        #[cfg_attr(not(feature = "appearance"), allow(unused_mut))]
        let mut cost1 = iou_cost_matrix(&pool_boxes, &high_boxes);
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
        // Depth gate last, so it overrides an appearance rescue on a depth-mismatched
        // pair (two similar-looking objects at different distances).
        self.gate_stage(&mut cost1, &pool, &high_det, detections);
        let (m1, u_pool, u_high) =
            linear_assignment(&cost1, pool.len(), high_det.len(), cfg.match_thresh as f64);
        for (pi, di) in m1 {
            let ti = pool[pi];
            let det = &detections[high_det[di]];
            self.tracks[ti].update(det, cfg.min_hits);
            #[cfg(feature = "appearance")]
            if let Some(f) = det.feature.as_deref() {
                self.tracks[ti].smooth_feature(f, cfg.feature_momentum);
            }
        }

        // ---- Stage 2: still-tracked (was Confirmed) pool tracks vs low dets ----
        // Only previously-tracked tracks chase low-confidence boxes; genuinely
        // lost tracks are not re-found on weak evidence (ByteTrack rule).
        let r_tracked: Vec<usize> = u_pool
            .iter()
            .map(|&pi| pool[pi])
            .filter(|&ti| self.tracks[ti].state == TrackState::Confirmed)
            .collect();
        let low_boxes: Vec<[f32; 4]> = low_det.iter().map(|&i| detections[i].bbox).collect();
        let r_boxes: Vec<[f32; 4]> = r_tracked.iter().map(|&ti| self.tracks[ti].bbox()).collect();
        let mut cost2 = iou_cost_matrix(&r_boxes, &low_boxes);
        self.gate_stage(&mut cost2, &r_tracked, &low_det, detections);
        let (m2, _u_r, _u_low) = linear_assignment(
            &cost2,
            r_tracked.len(),
            low_det.len(),
            cfg.match_thresh_second as f64,
        );
        for (ri, di) in m2 {
            let ti = r_tracked[ri];
            self.tracks[ti].update(&detections[low_det[di]], cfg.min_hits);
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
        let mut cost3 = iou_cost_matrix(&unconf_boxes, &rem_boxes);
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
        self.gate_stage(&mut cost3, &unconfirmed, &remaining_high, detections);
        let (m3, u_unconf, u_rem) = linear_assignment(
            &cost3,
            unconfirmed.len(),
            remaining_high.len(),
            cfg.match_thresh as f64,
        );
        for (ui, di) in m3 {
            let ti = unconfirmed[ui];
            let det = &detections[remaining_high[di]];
            self.tracks[ti].update(det, cfg.min_hits);
            #[cfg(feature = "appearance")]
            if let Some(f) = det.feature.as_deref() {
                self.tracks[ti].smooth_feature(f, cfg.feature_momentum);
            }
        }
        // Unmatched tentative tracks die immediately.
        for &ui in &u_unconf {
            self.tracks[unconfirmed[ui]].mark_missed();
        }

        // ---- Birth new tracks from the still-unmatched high detections ----
        for &ri in &u_rem {
            let di = remaining_high[ri];
            if detections[di].score >= cfg.new_track_thresh {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks
                    .push(Tracklet::new(id, &detections[di], cfg.kalman));
            }
        }

        // ---- Reap dead / expired tracks ----
        let buffer = cfg.track_buffer;
        self.tracks.retain(|t| {
            t.state != TrackState::Removed
                && !(t.state == TrackState::Lost && t.time_since_update > buffer)
        });

        // Output: confirmed tracks that matched a detection this frame.
        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed && t.matched_this_frame)
            .map(Tracklet::to_track)
            .collect()
    }
}
