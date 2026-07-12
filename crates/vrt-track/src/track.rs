//! Track lifecycle: the internal [`Tracklet`] (Kalman + bookkeeping) and the
//! public [`Track`] snapshot returned to callers.

use crate::kalman::{KalmanFilter3D, KalmanParams};
use crate::Detection;
use vrt_types::{CameraExtrinsics, CameraIntrinsics};

/// NSA measurement-noise inflation strength: `R ← R·(1 + α·(1−score))`. At `score=1`
/// the noise is the tuned base; a 0.4-score detection gets ~2.2× the noise.
const NSA_ALPHA: f64 = 2.0;

/// Lifecycle stage of a track.
///
/// ```text
///   new det ──► Tentative ──(min_hits matches)──► Confirmed
///                   │                                 │ miss
///                   │ miss                            ▼
///                   ▼                               Lost ──(match)──► Confirmed
///                Removed ◄───────(> track_buffer frames lost)───────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    /// Just born; not yet reported until it survives `min_hits` frames.
    Tentative,
    /// Established, reported to the caller.
    Confirmed,
    /// Was confirmed, currently unmatched — kept alive for re-identification.
    Lost,
    /// Dead; pending removal from the pool.
    Removed,
}

/// Public per-frame snapshot of a track.
#[derive(Debug, Clone)]
pub struct Track {
    /// Stable track identity.
    pub id: u64,
    /// Current box `[x1, y1, x2, y2]` in image pixels (from the filter state).
    pub bbox: [f32; 4],
    /// Class id carried from the matched detection.
    pub class_id: u32,
    /// Score of the most recent matched detection.
    pub score: f32,
    /// Lifecycle stage.
    pub state: TrackState,
    /// 3D centre estimate `[px, py, pz]` (`pz` is depth; coasts when unmeasured).
    pub position_3d: [f32; 3],
    /// 3D velocity estimate `[vx, vy, vz]`.
    pub velocity_3d: [f32; 3],
    /// Frames since birth.
    pub age: u32,
    /// Total number of matched detections.
    pub hits: u32,
    /// Frames since the last match (0 on the frame it was updated).
    pub time_since_update: u32,
}

impl Track {
    /// Metric **3D position** (metres, camera frame) — back-project the filtered
    /// image-plane centre + depth through `k`. `pz` is meaningful only once the track
    /// has had a real depth measurement (otherwise it is the coasting nominal).
    pub fn metric_position(&self, k: &CameraIntrinsics) -> [f32; 3] {
        let [px, py, pz] = self.position_3d;
        k.unproject(px, py, pz)
    }

    /// Metric **3D position in a shared WORLD frame** — back-project through `k`, then
    /// apply the camera pose `e`. With [`CameraExtrinsics::IDENTITY`] this equals
    /// [`metric_position`](Self::metric_position) (single-camera). Supplying each
    /// camera's real pose puts every camera's tracks in one coordinate system — the
    /// basis for a shared multi-camera BEV / cross-camera association.
    pub fn world_position(&self, k: &CameraIntrinsics, e: &CameraExtrinsics) -> [f32; 3] {
        e.to_world(self.metric_position(k))
    }

    /// Metric **velocity per nominal frame** (metres/frame, camera frame): the
    /// change in [`metric_position`](Self::metric_position) over one state-velocity
    /// step. Divide by the real seconds-per-frame to get m/s. Uses a finite
    /// difference through `unproject` so depth motion (`vz`) and the depth-scaled
    /// image motion both contribute.
    pub fn metric_velocity(&self, k: &CameraIntrinsics) -> [f32; 3] {
        let [px, py, pz] = self.position_3d;
        let [vx, vy, vz] = self.velocity_3d;
        let a = k.unproject(px, py, pz);
        let b = k.unproject(px + vx, py + vy, pz + vz);
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    }
}

/// Internal mutable track: owns the Kalman filter and lifecycle counters.
pub(crate) struct Tracklet {
    pub id: u64,
    pub kf: KalmanFilter3D,
    pub class_id: u32,
    pub score: f32,
    pub state: TrackState,
    pub age: u32,
    pub hits: u32,
    pub time_since_update: u32,
    /// Whether this track was matched on the current frame (drives second-stage
    /// pool selection and output filtering).
    pub matched_this_frame: bool,
    /// Whether a **real** depth measurement has ever corrected this track. Until it
    /// has, `pz` is the birth nominal (coasting) and must not gate association.
    pub depth_measured: bool,
    #[cfg(feature = "appearance")]
    pub smooth_feat: Option<Vec<f32>>,
}

fn xyxy_to_cxcywh(b: &[f32; 4]) -> (f64, f64, f64, f64) {
    let w = (b[2] - b[0]) as f64;
    let h = (b[3] - b[1]) as f64;
    (b[0] as f64 + w * 0.5, b[1] as f64 + h * 0.5, w, h)
}

impl Tracklet {
    /// Birth a tentative track from a detection.
    pub fn new(id: u64, det: &Detection, params: KalmanParams) -> Self {
        let (cx, cy, w, h) = xyxy_to_cxcywh(&det.bbox);
        let depth = det.depth.map(|z| z as f64);
        Self {
            id,
            kf: KalmanFilter3D::new(cx, cy, w, h, depth, params),
            class_id: det.class_id,
            score: det.score,
            state: TrackState::Tentative,
            age: 0,
            hits: 1,
            time_since_update: 0,
            matched_this_frame: true,
            depth_measured: det.depth.is_some(),
            #[cfg(feature = "appearance")]
            smooth_feat: det.feature.clone(),
        }
    }

    /// Kalman predict over `dt` time units + advance the "missed" counters. Called
    /// once per frame before association; [`update`](Self::update) resets the counters
    /// on a match. The lifecycle counters advance per *frame* (they gate re-id/removal
    /// in frames), independent of `dt` which only scales the motion prediction.
    pub fn predict(&mut self, dt: f64) {
        self.kf.predict(dt);
        self.age += 1;
        self.time_since_update += 1;
        self.matched_this_frame = false;
    }

    /// Correct with a matched detection and advance the lifecycle. **NSA Kalman**
    /// (StrongSORT): the measurement noise is inflated for low-confidence detections,
    /// `R ← R · (1 + α·(1−score))`, so a shaky box nudges the state gently while a
    /// crisp one updates firmly.
    pub fn update(&mut self, det: &Detection, min_hits: u32) {
        let (cx, cy, w, h) = xyxy_to_cxcywh(&det.bbox);
        let meas_scale = 1.0 + NSA_ALPHA * (1.0 - det.score.clamp(0.0, 1.0) as f64);
        self.kf
            .update(cx, cy, w, h, det.depth.map(|z| z as f64), meas_scale);
        self.class_id = det.class_id;
        self.score = det.score;
        self.hits += 1;
        self.time_since_update = 0;
        self.matched_this_frame = true;
        self.depth_measured |= det.depth.is_some(); // once measured, pz stays anchored

        // Tentative -> Confirmed after enough support; a re-found Lost track is
        // immediately re-confirmed.
        match self.state {
            TrackState::Tentative if self.hits >= min_hits => self.state = TrackState::Confirmed,
            TrackState::Lost => self.state = TrackState::Confirmed,
            _ => {}
        }
    }

    /// Mark unmatched: a confirmed track becomes `Lost`; a tentative one that never
    /// established dies immediately.
    pub fn mark_missed(&mut self) {
        match self.state {
            TrackState::Confirmed => self.state = TrackState::Lost,
            TrackState::Tentative => self.state = TrackState::Removed,
            _ => {}
        }
    }

    /// EMA-smooth the appearance feature bank toward a new embedding
    /// (`smooth ← momentum·smooth + (1−momentum)·new`, L2-renormalised).
    #[cfg(feature = "appearance")]
    pub fn smooth_feature(&mut self, feat: &[f32], momentum: f32) {
        match &mut self.smooth_feat {
            Some(s) if s.len() == feat.len() => {
                let mut norm = 0.0f32;
                for (a, b) in s.iter_mut().zip(feat.iter()) {
                    *a = momentum * *a + (1.0 - momentum) * b;
                    norm += *a * *a;
                }
                if norm > 0.0 {
                    let inv = 1.0 / norm.sqrt();
                    for a in s.iter_mut() {
                        *a *= inv;
                    }
                }
            }
            _ => self.smooth_feat = Some(feat.to_vec()),
        }
    }

    /// Current box as `[x1, y1, x2, y2]` pixels from the filter state.
    pub fn bbox(&self) -> [f32; 4] {
        let (cx, cy) = self.kf.center();
        let (w, h) = self.kf.size();
        [
            (cx - w * 0.5) as f32,
            (cy - h * 0.5) as f32,
            (cx + w * 0.5) as f32,
            (cy + h * 0.5) as f32,
        ]
    }

    /// Current metric depth (`pz`) **iff a real depth measurement has ever corrected
    /// this track** — else `None` (its `pz` is the coasting birth nominal and must
    /// not gate association). Consumed by [`crate::association::gate_depth`].
    pub fn measured_depth(&self) -> Option<f32> {
        self.depth_measured.then(|| self.kf.position_3d()[2] as f32)
    }

    /// Build the public snapshot.
    pub fn to_track(&self) -> Track {
        let p = self.kf.position_3d();
        let v = self.kf.velocity_3d();
        Track {
            id: self.id,
            bbox: self.bbox(),
            class_id: self.class_id,
            score: self.score,
            state: self.state,
            position_3d: [p[0] as f32, p[1] as f32, p[2] as f32],
            velocity_3d: [v[0] as f32, v[1] as f32, v[2] as f32],
            age: self.age,
            hits: self.hits,
            time_since_update: self.time_since_update,
        }
    }
}
