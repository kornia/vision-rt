//! Track lifecycle: the internal [`Tracklet`] (Kalman + bookkeeping) and the
//! public [`Track`] snapshot returned to callers.

use std::collections::VecDeque;

use crate::kalman::{KalmanFilter3D, KalmanParams};
use crate::Detection;
use vrt_types::{CameraExtrinsics, CameraIntrinsics};

/// NSA measurement-noise inflation strength: `R ← R·(1 + α·(1−score))`. At `score=1`
/// the noise is the tuned base; a 0.4-score detection gets ~2.2× the noise.
const NSA_ALPHA: f64 = 2.0;

/// OC-SORT observation-centric momentum lookback (frames): the velocity direction is
/// measured over the last `OCM_DELTA` observations, not the noisy Kalman velocity.
const OCM_DELTA: usize = 3;

/// Depth **innovation gate** for the state update: a matched detection whose depth
/// disagrees with the track's estimate by more than `max(ABS, REL·pz)` has its depth
/// treated as *missing* for that frame (pz coasts) instead of being swallowed —
/// mask-sampled monocular depth throws single-frame outliers of metres when the mask
/// momentarily covers a limb, the background, or an occluder. Same shape as the
/// association depth gate, but protecting the Kalman state rather than the matching.
const DEPTH_INNOV_REL: f64 = 0.25;
const DEPTH_INNOV_ABS: f64 = 0.7;
/// After this many **consecutive** gate-rejected (but individually valid) depth
/// measurements, the disagreement is persistent — the coasted `pz` has drifted, not the
/// measurement — so force-accept it and re-anchor. Bounds divergence to a few frames of
/// `vz` drift instead of an unbounded runaway. Small enough that a genuine 1–2 frame
/// monocular spike is still coasted, large enough to absorb it.
const DEPTH_REJECT_MAX: u32 = 3;

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
    /// Consecutive frames a valid depth measurement was **rejected** by the innovation
    /// gate. The gate coasts `pz` through transient monocular-depth spikes, but its
    /// tolerance scales with `pz`, so a *diverging* `pz` would otherwise lock the true
    /// measurement out forever and run to infinity (observed live: `pz` → 500 m while the
    /// real depth held ~2 m). After [`DEPTH_REJECT_MAX`] consecutive rejections the
    /// disagreement is persistent, not a glitch — force-accept the measurement to
    /// re-anchor `pz`. Reset whenever a measurement is accepted.
    depth_reject_streak: u32,
    /// Frames since this track last matched a **high-confidence** detection. A track
    /// that only ever matches weak (recovery-tier) boxes for a long stretch has latched
    /// onto a persistent false positive — the box collapses onto background noise and,
    /// because it keeps "matching", it never goes `Lost` and `track_buffer` never reaps
    /// it. This counter drives the confidence-coast reap. Reset on a high-conf match.
    pub low_conf_age: u32,
    /// Largest box area (px²) this track has ever observed. The confidence-coast reap
    /// fires only when the current box has **collapsed** far below this — the phantom
    /// signature (a person-left track latching onto a tiny noise blob). A weak-but-real
    /// object (e.g. a low-confidence oven) keeps its full size and is NOT reaped.
    pub max_area: f32,
    /// Recent **metric world** observations `[X, Y, Z]` (metres, camera frame; newest at
    /// back, capped at `OCM_DELTA + 1`) — the raw-observation history OC-SORT momentum
    /// reads its velocity direction from. Metric, not pixel, so the direction cue is
    /// perspective-correct and sees motion in depth (`Z`), not just lateral image motion.
    obs_hist: VecDeque<[f64; 3]>,
    /// Last observed box `(cx, cy, w, h)` in **pixels** — ORU interpolates the box *size*
    /// (a pixel quantity) from here and reprojects the interpolated world position back.
    last_obs: [f64; 4],
    /// Last observed **world** position `[X, Y, Z]` (metres): ORU interpolates the
    /// trajectory in world space from here, and OCM measures candidate directions from it.
    last_world: [f64; 3],
    /// Kalman state snapshot taken at the last real observation. ORU rolls back to this
    /// before replaying a virtual trajectory across a re-acquisition gap, so the filter
    /// isn't stuck with the stale velocity it coasted on while the track was Lost.
    kf_snapshot: KalmanFilter3D,
    /// Whether this track already claimed an established identity (born by gallery
    /// resurrection or rewritten by a deferred merge) — it must not chain into ANOTHER
    /// identity claim, or occlusion storms turn the merge machinery into id musical
    /// chairs.
    #[cfg(feature = "appearance")]
    pub claimed: bool,
    #[cfg(feature = "appearance")]
    pub smooth_feat: Option<Vec<f32>>,
}

fn xyxy_to_cxcywh(b: &[f32; 4]) -> (f64, f64, f64, f64) {
    let w = (b[2] - b[0]) as f64;
    let h = (b[3] - b[1]) as f64;
    (b[0] as f64 + w * 0.5, b[1] as f64 + h * 0.5, w, h)
}

/// f64 back-projection of a pixel `(u, v)` + depth `z` to a camera-frame metric point
/// (the tracker works in f64; [`CameraIntrinsics`] is f32).
fn unproject_f64(k: &CameraIntrinsics, u: f64, v: f64, z: f64) -> [f64; 3] {
    let p = k.unproject(u as f32, v as f32, z as f32);
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

/// f64 projection of a camera-frame metric point back to a pixel `(u, v)`.
fn project_f64(k: &CameraIntrinsics, p: [f64; 3]) -> (f64, f64) {
    let uv = k.project([p[0] as f32, p[1] as f32, p[2] as f32]);
    (uv[0] as f64, uv[1] as f64)
}

impl Tracklet {
    /// Birth a tentative track from a detection. `intr` back-projects the observed box
    /// centre to a metric world point for the OCM/ORU history.
    pub fn new(id: u64, det: &Detection, params: KalmanParams, intr: &CameraIntrinsics) -> Self {
        let (cx, cy, w, h) = xyxy_to_cxcywh(&det.bbox);
        // Guard the birth depth: a non-finite / ≤0 depth would seed pz with NaN before
        // `update`'s gate ever runs. Treat it as absent (image-plane birth).
        let depth = det
            .depth
            .map(|z| z as f64)
            .filter(|z| z.is_finite() && *z > 0.0);
        let kf = KalmanFilter3D::new(cx, cy, w, h, depth, params);
        let world = unproject_f64(intr, cx, cy, depth.unwrap_or(params.nominal_depth));
        let mut obs_hist = VecDeque::with_capacity(OCM_DELTA + 1);
        obs_hist.push_back(world);
        Self {
            id,
            kf_snapshot: kf.clone(),
            kf,
            class_id: det.class_id,
            score: det.score,
            state: TrackState::Tentative,
            age: 0,
            hits: 1,
            time_since_update: 0,
            matched_this_frame: true,
            depth_measured: det.depth.is_some(),
            depth_reject_streak: 0,
            low_conf_age: 0,
            max_area: (w * h) as f32,
            obs_hist,
            last_obs: [cx, cy, w, h],
            last_world: world,
            #[cfg(feature = "appearance")]
            claimed: false,
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
        self.low_conf_age += 1;
        self.matched_this_frame = false;
    }

    /// Reset the confidence-coast counter — call on a match whose detection score meets
    /// the high-confidence tier. A track that never gets this reset has latched onto a
    /// weak false positive and will be reaped by [`Tracker`]'s confidence-coast rule.
    pub fn note_high_conf(&mut self) {
        self.low_conf_age = 0;
    }

    /// Correct with a matched detection and advance the lifecycle. **NSA Kalman**
    /// (StrongSORT): the measurement noise is inflated for low-confidence detections,
    /// `R ← R · (1 + α·(1−score))`, so a shaky box nudges the state gently while a
    /// crisp one updates firmly.
    pub fn update(&mut self, det: &Detection, min_hits: u32, oru: bool, intr: &CameraIntrinsics) {
        let (cx, cy, w, h) = xyxy_to_cxcywh(&det.bbox);
        // Depth innovation gate: drop a non-finite/≤0 depth outright, and once this track
        // has a real depth history, drop an outlier measurement (mask glitch) too — let
        // depth coast rather than yanking pz.
        let depth = match det
            .depth
            .map(|z| z as f64)
            .filter(|&z| z.is_finite() && z > 0.0)
        {
            None => None, // no valid depth this frame → coast (streak unchanged: no disagreement)
            Some(z) if !self.depth_measured => {
                self.depth_reject_streak = 0;
                Some(z) // first real depth anchors pz
            }
            Some(z) => {
                let pz = self.kf.position_3d()[2];
                if (z - pz).abs() <= DEPTH_INNOV_ABS.max(DEPTH_INNOV_REL * pz) {
                    self.depth_reject_streak = 0;
                    Some(z)
                } else {
                    // Outlier vs the current estimate. Coast through a transient spike, but
                    // if the measurement *persistently* disagrees, pz has diverged — accept
                    // it to re-anchor rather than let pz run to infinity.
                    self.depth_reject_streak += 1;
                    if self.depth_reject_streak >= DEPTH_REJECT_MAX {
                        self.depth_reject_streak = 0;
                        Some(z)
                    } else {
                        None
                    }
                }
            }
        };
        // Metric world position of this observation (gated depth — an outlier must not
        // distort the OCM direction either); no depth → the filter's coasting pz.
        let pz = depth.unwrap_or(self.kf.position_3d()[2]);
        let curr_world = unproject_f64(intr, cx, cy, pz);

        // ORU (OC-SORT): re-acquired after a coasting gap. The Kalman drifted on the
        // stale velocity it held while lost; rebuild the trajectory from a straight
        // virtual path — in **world** space — between the pre-gap observation and this
        // one before the real correction, so its velocity/position reflect the true
        // endpoints (perspective-correct: with constant depth this equals a pixel path).
        if oru && self.time_since_update >= 2 {
            self.reupdate_virtual(curr_world, (w, h), self.time_since_update, intr);
        }

        let meas_scale = 1.0 + NSA_ALPHA * (1.0 - det.score.clamp(0.0, 1.0) as f64);
        self.kf.update(cx, cy, w, h, depth, meas_scale);
        // Class is mutable only while Tentative: once an identity is established, a
        // detector flicker (person box on an occluded chair) must not re-label the
        // track — a class flip is an id-corruption and breaks class-gated re-id.
        if self.state == TrackState::Tentative {
            self.class_id = det.class_id;
        }
        self.score = det.score;
        self.hits += 1;
        self.time_since_update = 0;
        self.matched_this_frame = true;
        self.depth_measured |= depth.is_some(); // once measured (gate-accepted), pz stays anchored

        // Track the largest observed box area (the confidence-coast reap's collapse ref).
        self.max_area = self.max_area.max((w * h) as f32);

        // Record this observation for the next frame's OCM direction / ORU anchor, and
        // snapshot the corrected state as the next ORU roll-back point.
        self.last_obs = [cx, cy, w, h];
        self.last_world = curr_world;
        self.obs_hist.push_back(curr_world);
        while self.obs_hist.len() > OCM_DELTA + 1 {
            self.obs_hist.pop_front();
        }
        self.kf_snapshot = self.kf.clone();

        // Tentative -> Confirmed after enough support; a re-found Lost track is
        // immediately re-confirmed.
        match self.state {
            TrackState::Tentative if self.hits >= min_hits => self.state = TrackState::Confirmed,
            TrackState::Lost => self.state = TrackState::Confirmed,
            _ => {}
        }
    }

    /// ORU virtual re-update (OC-SORT §3.2), in **metric world space**: roll the filter
    /// back to the last-observed state, then replay a straight-line trajectory in metres
    /// from the pre-gap world observation (`last_world`) to the re-acquisition point
    /// (`curr_world`) over `gap` frames — reprojecting each virtual world point to a
    /// pixel `(u, v)` + its depth `Z` for the Kalman update, with the box size lerped in
    /// pixels. The final predict lands on the current frame; the caller's real `update`
    /// then applies the true measurement. World-linear (not pixel-linear) so an object
    /// crossing depth during the occlusion is reconstructed with correct perspective.
    fn reupdate_virtual(
        &mut self,
        curr_world: [f64; 3],
        curr_wh: (f64, f64),
        gap: u32,
        intr: &CameraIntrinsics,
    ) {
        let prev_world = self.last_world;
        let (prev_w, prev_h) = (self.last_obs[2], self.last_obs[3]);
        let g = gap as f64;
        self.kf = self.kf_snapshot.clone(); // roll back to the last real observation
        for i in 1..gap {
            let a = i as f64 / g;
            let lerp = |p: f64, c: f64| p * (1.0 - a) + c * a;
            let vw = [
                lerp(prev_world[0], curr_world[0]),
                lerp(prev_world[1], curr_world[1]),
                lerp(prev_world[2], curr_world[2]),
            ];
            let (u, v) = project_f64(intr, vw);
            self.kf.predict(1.0);
            // Correct depth only if this track has ever had a real depth; else the world
            // Z is the nominal pz and reprojection round-trips to the pixel path (None).
            let depth = self.depth_measured.then_some(vw[2]);
            self.kf.update(
                u,
                v,
                lerp(prev_w, curr_wh.0),
                lerp(prev_h, curr_wh.1),
                depth,
                1.0,
            );
        }
        self.kf.predict(1.0); // final step onto this frame; real update applies the obs
    }

    /// OC-SORT momentum direction in **metric world space**: the unit velocity direction
    /// over the observation history (newest − oldest, up to `OCM_DELTA` frames), in
    /// metres. `None` when the track has too little history or the displacement is below
    /// a **metric significance floor** — box-jitter on a *static* object unprojects to
    /// ~5–10 mm of phantom motion (±2 px at 3 m), and a noise direction would make OCM
    /// randomly penalize the correct re-match (churn on occluded static objects). 6 cm
    /// over the ≤3-frame window (≈2 cm/frame) means genuine motion.
    pub fn obs_direction(&self) -> Option<[f64; 3]> {
        const MIN_DISPLACEMENT_M: f64 = 0.06;
        if self.obs_hist.len() < 2 {
            return None;
        }
        let (a, b) = (self.obs_hist.front()?, self.obs_hist.back()?);
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        (n >= MIN_DISPLACEMENT_M).then(|| [d[0] / n, d[1] / n, d[2] / n])
    }

    /// Last observed **world** position `[X, Y, Z]` (metres) — OCM measures candidate
    /// directions from here.
    pub fn last_world(&self) -> [f64; 3] {
        self.last_world
    }

    /// Last **observed** box centre in pixels — the re-id spatial bound measures from
    /// here (the coasted prediction is exactly what's wrong during a turn-around).
    #[cfg(feature = "appearance")]
    pub fn last_center_px(&self) -> (f64, f64) {
        (self.last_obs[0], self.last_obs[1])
    }

    /// Last observed mean box dimension (pixels) — scales the re-id spatial bound.
    #[cfg(feature = "appearance")]
    pub fn last_box_dim(&self) -> f64 {
        (self.last_obs[2] + self.last_obs[3]) * 0.5
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

    /// Current box area (px²) from the filter state — compared against
    /// [`max_area`](Self::max_area) for the confidence-coast collapse check.
    pub fn area(&self) -> f32 {
        let (w, h) = self.kf.size();
        (w * h) as f32
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
