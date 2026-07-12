//! 3D constant-velocity Kalman filter for BoT-SORT tracks.
//!
//! # State
//! The filter keeps an **8-dimensional** state that models the target's centre in
//! **3D** plus its image-plane box size:
//!
//! ```text
//! x = [ px, py, pz,  w,  h,  vx, vy, vz ]ᵀ
//!        └──────┘   └──┘   └────────┘
//!       3D centre  box wh  3D velocity
//! ```
//!
//! `px, py` are the box-centre pixel coordinates in the image plane, `pz` is
//! **depth** (metric, or any monotone depth proxy). `w, h` are box width/height in
//! pixels. Motion is **constant-velocity in 3D** — `px, py, pz` each integrate their
//! velocity every frame; `w, h` follow a random walk (no size "momentum", which is
//! how box aspect actually behaves, and avoids the runaway growth of a size-velocity
//! model during occlusion).
//!
//! This deliberately is **not** the classic 8-dim image-plane `xywh + ẋẏẇḣ` SORT
//! Kalman: we carry a real depth axis so that when a depth/lift source becomes
//! available the tracker fuses 3D position directly.
//!
//! # Measurement & graceful degradation to the image plane
//! The measurement is **always** 5-dimensional:
//!
//! ```text
//! z = [ px, py, pz, w, h ]ᵀ,   H = [ I₅ | 0₅ₓ₃ ]
//! ```
//!
//! Depth (`pz`) frequently is **not** measured — a plain 2D detector gives only
//! `px, py, w, h`. Rather than switch matrices per frame, we keep one fixed-size
//! update and **degrade through the measurement covariance**: when a detection has
//! no depth we set the depth measurement variance `R[2,2]` to a huge value
//! ([`KalmanParams::meas_depth_missing`]). The Kalman gain for the depth row then
//! collapses to ≈ 0, so `pz`/`vz` are left to **coast on the motion model** and are
//! never corrupted by a fake measurement — the filter behaves exactly like an
//! image-plane tracker on the observed axes while still maintaining a (growing-
//! uncertainty) depth estimate. Feed a real `pz` (with the small
//! [`KalmanParams::meas_depth`] variance) the moment a depth crate provides one and
//! the 3D estimate sharpens automatically. This single-code-path trick is why the
//! measurement stays a fixed `SVector<_, 5>`.

use nalgebra::{SMatrix, SVector};

/// State dimension: `[px, py, pz, w, h, vx, vy, vz]`.
pub const NX: usize = 8;
/// Measurement dimension: `[px, py, pz, w, h]`.
pub const NZ: usize = 5;

type State = SVector<f64, NX>;
type StateCov = SMatrix<f64, NX, NX>;
type Meas = SVector<f64, NZ>;
type MeasCov = SMatrix<f64, NZ, NZ>;
type ObsMat = SMatrix<f64, NZ, NX>;

/// Noise / initialisation parameters for [`KalmanFilter3D`].
///
/// Standard deviations are in the natural units of each axis (pixels for
/// `px,py,w,h`, depth-units for `pz`). [`Default`] gives sane values for a
/// pixel-space tracker with unit-scaled depth.
#[derive(Debug, Clone, Copy)]
pub struct KalmanParams {
    /// Process-noise std for image-plane position (`px, py`), per frame.
    pub std_position: f64,
    /// Process-noise std for depth (`pz`), per frame.
    pub std_depth: f64,
    /// Process-noise std for box size (`w, h`), per frame.
    pub std_size: f64,
    /// Process-noise std for image-plane velocity (`vx, vy`).
    pub std_velocity: f64,
    /// Process-noise std for depth velocity (`vz`).
    pub std_velocity_depth: f64,
    /// Measurement std for image-plane position (`px, py`).
    pub meas_position: f64,
    /// Measurement std for box size (`w, h`).
    pub meas_size: f64,
    /// Measurement std for depth **when a depth measurement is present** (small).
    pub meas_depth: f64,
    /// Measurement std for depth **when absent** — huge, so the gain ≈ 0 and depth
    /// coasts on the motion model (image-plane fallback).
    pub meas_depth_missing: f64,
    /// Depth value assigned at track birth when the first detection has no depth.
    pub nominal_depth: f64,
    /// Initial std for image-plane position / size at birth.
    pub init_position: f64,
    /// Initial std for depth at birth (large — depth is barely known up front).
    pub init_depth: f64,
    /// Initial std for image-plane / size velocity at birth.
    pub init_velocity: f64,
    /// Initial std for depth velocity at birth.
    pub init_velocity_depth: f64,
}

impl Default for KalmanParams {
    fn default() -> Self {
        Self {
            std_position: 1.5,
            std_depth: 0.2,
            std_size: 1.5,
            std_velocity: 1.0,
            std_velocity_depth: 0.1,
            meas_position: 1.0,
            meas_size: 1.0,
            meas_depth: 0.1,
            meas_depth_missing: 1.0e6,
            nominal_depth: 1.0,
            init_position: 4.0,
            init_depth: 100.0,
            init_velocity: 10.0,
            init_velocity_depth: 1.0,
        }
    }
}

/// A single track's 3D constant-velocity Kalman filter (see module docs).
#[derive(Debug, Clone)]
pub struct KalmanFilter3D {
    x: State,
    p: StateCov,
    params: KalmanParams,
}

impl KalmanFilter3D {
    /// Birth the filter from a first measurement.
    ///
    /// `depth` is the optional metric depth of the box centre; `None` seeds `pz`
    /// with [`KalmanParams::nominal_depth`] and a large initial covariance.
    pub fn new(cx: f64, cy: f64, w: f64, h: f64, depth: Option<f64>, params: KalmanParams) -> Self {
        let pz = depth.unwrap_or(params.nominal_depth);
        let x = State::from_column_slice(&[cx, cy, pz, w, h, 0.0, 0.0, 0.0]);

        // Diagonal birth covariance; depth (+ its velocity) starts very uncertain
        // when unmeasured so early real depths dominate the estimate.
        let init_depth = if depth.is_some() {
            params.meas_depth
        } else {
            params.init_depth
        };
        let diag = [
            params.init_position.powi(2),
            params.init_position.powi(2),
            init_depth.powi(2),
            params.init_position.powi(2),
            params.init_position.powi(2),
            params.init_velocity.powi(2),
            params.init_velocity.powi(2),
            params.init_velocity_depth.powi(2),
        ];
        let p = StateCov::from_diagonal(&SVector::<f64, NX>::from_column_slice(&diag));

        Self { x, p, params }
    }

    /// State-transition matrix `F` for a step of `dt` time units (constant
    /// velocity: `p ← p + v·dt`). `dt = 1` is one nominal frame.
    fn transition(dt: f64) -> StateCov {
        let mut f = StateCov::identity();
        f[(0, 5)] = dt; // px += vx·dt
        f[(1, 6)] = dt; // py += vy·dt
        f[(2, 7)] = dt; // pz += vz·dt
        f
    }

    /// Observation matrix `H = [I₅ | 0]` mapping state → `[px, py, pz, w, h]`.
    fn observation() -> ObsMat {
        let mut h = ObsMat::zeros();
        for i in 0..NZ {
            h[(i, i)] = 1.0;
        }
        h
    }

    /// Diagonal process-noise covariance `Q` for a `dt`-length step. The per-axis
    /// variances accumulate linearly with `dt` (random-walk), so `dt = 1` reproduces
    /// the per-frame `Q` exactly and a longer interval injects proportionally more
    /// uncertainty (a dropped frame → a wider predict).
    fn process_noise(&self, dt: f64) -> StateCov {
        let p = &self.params;
        let diag = [
            p.std_position.powi(2),
            p.std_position.powi(2),
            p.std_depth.powi(2),
            p.std_size.powi(2),
            p.std_size.powi(2),
            p.std_velocity.powi(2),
            p.std_velocity.powi(2),
            p.std_velocity_depth.powi(2),
        ];
        StateCov::from_diagonal(&(SVector::<f64, NX>::from_column_slice(&diag) * dt))
    }

    /// Measurement covariance `R`. `depth_present` picks the small vs. huge depth
    /// variance that drives the image-plane fallback (see module docs).
    fn measurement_noise(&self, depth_present: bool) -> MeasCov {
        let p = &self.params;
        let rz = if depth_present {
            p.meas_depth
        } else {
            p.meas_depth_missing
        };
        let diag = [
            p.meas_position.powi(2),
            p.meas_position.powi(2),
            rz.powi(2),
            p.meas_size.powi(2),
            p.meas_size.powi(2),
        ];
        MeasCov::from_diagonal(&SVector::<f64, NZ>::from_column_slice(&diag))
    }

    /// Predict `dt` time units forward: `x ← Fx`, `P ← F P Fᵀ + Q`, both built for
    /// `dt`. Pass `dt = 1.0` for the fixed-cadence case; pass the real inter-frame
    /// interval (in the same units the [`KalmanParams`] were tuned for, e.g. nominal
    /// frames) to stay consistent under variable fps / dropped frames.
    pub fn predict(&mut self, dt: f64) {
        let f = Self::transition(dt);
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + self.process_noise(dt);
    }

    /// Correct with a measurement `[cx, cy, w, h]` plus optional depth.
    ///
    /// When `depth` is `None` the depth axis is updated with a huge variance, i.e.
    /// left essentially unchanged (image-plane fallback). Returns `false` if the
    /// innovation covariance is singular (update skipped) — should not happen with
    /// positive `R`, but the filter degrades safely rather than panicking.
    pub fn update(&mut self, cx: f64, cy: f64, w: f64, h: f64, depth: Option<f64>) -> bool {
        let z = Meas::from_column_slice(&[cx, cy, depth.unwrap_or(self.x[2]), w, h]);
        let h_mat = Self::observation();
        let r = self.measurement_noise(depth.is_some());

        let y = z - h_mat * self.x; // innovation
        let s = h_mat * self.p * h_mat.transpose() + r; // innovation covariance
        let Some(s_inv) = s.try_inverse() else {
            return false;
        };
        let k = self.p * h_mat.transpose() * s_inv; // Kalman gain

        self.x += k * y;

        // Joseph form keeps P symmetric positive-definite under finite precision.
        let i = StateCov::identity();
        let ikh = i - k * h_mat;
        self.p = ikh * self.p * ikh.transpose() + k * r * k.transpose();
        true
    }

    /// Squared Mahalanobis distance of a measurement to the predicted measurement,
    /// in the image-plane subspace `[px, py, w, h]` (depth excluded so the gate is
    /// meaningful with or without depth). Useful as an association gate.
    pub fn gating_distance(&self, cx: f64, cy: f64, w: f64, h: f64) -> f64 {
        // Project onto the 4 observed image-plane rows (indices 0,1,3,4).
        let idx = [0usize, 1, 3, 4];
        let mut proj_x = [0.0f64; 4];
        let mut proj_s = SMatrix::<f64, 4, 4>::zeros();
        let full_s = {
            let h_mat = Self::observation();
            let r = self.measurement_noise(false);
            h_mat * self.p * h_mat.transpose() + r
        };
        let z = [cx, cy, w, h];
        for (a, &ia) in idx.iter().enumerate() {
            proj_x[a] = z[a] - self.x[ia];
            for (b, &ib) in idx.iter().enumerate() {
                proj_s[(a, b)] = full_s[(ia, ib)];
            }
        }
        let d = SVector::<f64, 4>::from_column_slice(&proj_x);
        match proj_s.try_inverse() {
            Some(inv) => (d.transpose() * inv * d)[(0, 0)],
            None => f64::INFINITY,
        }
    }

    /// Current box centre `(cx, cy)` in the image plane.
    pub fn center(&self) -> (f64, f64) {
        (self.x[0], self.x[1])
    }

    /// Current box size `(w, h)`.
    pub fn size(&self) -> (f64, f64) {
        (self.x[3], self.x[4])
    }

    /// Current 3D position `[px, py, pz]`.
    pub fn position_3d(&self) -> [f64; 3] {
        [self.x[0], self.x[1], self.x[2]]
    }

    /// Current 3D velocity `[vx, vy, vz]`.
    pub fn velocity_3d(&self) -> [f64; 3] {
        [self.x[5], self.x[6], self.x[7]]
    }

    /// Apply a 2×3 affine warp (row-major) to the image-plane centre & velocity —
    /// the hook camera-motion compensation ([`crate::gmc`]) uses to re-anchor a
    /// track after global background motion between frames.
    pub fn apply_affine(&mut self, m: &[[f32; 3]; 2]) {
        let (m00, m01, m02) = (m[0][0] as f64, m[0][1] as f64, m[0][2] as f64);
        let (m10, m11, m12) = (m[1][0] as f64, m[1][1] as f64, m[1][2] as f64);
        let (x, y) = (self.x[0], self.x[1]);
        self.x[0] = m00 * x + m01 * y + m02;
        self.x[1] = m10 * x + m11 * y + m12;
        // Velocity transforms by the linear (rotation/scale) part only.
        let (vx, vy) = (self.x[5], self.x[6]);
        self.x[5] = m00 * vx + m01 * vy;
        self.x[6] = m10 * vx + m11 * vy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> KalmanParams {
        KalmanParams::default()
    }

    #[test]
    fn predict_advances_by_velocity() {
        let mut kf = KalmanFilter3D::new(100.0, 50.0, 20.0, 40.0, None, params());
        // Inject a known velocity by updating with a shifted measurement a few times.
        for i in 1..=5 {
            let cx = 100.0 + 10.0 * i as f64;
            kf.update(cx, 50.0, 20.0, 40.0, None);
            kf.predict(1.0);
        }
        // After learning ~+10 px/frame, one more predict should move right, not left.
        let (cx0, _) = kf.center();
        kf.predict(1.0);
        let (cx1, _) = kf.center();
        assert!(cx1 > cx0, "velocity not tracked: {cx0} -> {cx1}");
        assert!(
            (cx1 - cx0 - 10.0).abs() < 4.0,
            "velocity off: {}",
            cx1 - cx0
        );
    }

    #[test]
    fn dt_scales_prediction_and_matches_unit_steps() {
        // A single predict(2.0) advances position by 2× the velocity — same position
        // as two predict(1.0) steps (constant velocity), so a dropped frame is
        // predicted correctly instead of lagging.
        let mut a = KalmanFilter3D::new(0.0, 0.0, 20.0, 20.0, Some(4.0), params());
        for i in 1..=6 {
            a.update(10.0 * i as f64, 0.0, 20.0, 20.0, Some(4.0));
            a.predict(1.0);
        }
        let mut b = a.clone();
        let (x0, _) = a.center();
        a.predict(1.0);
        a.predict(1.0);
        let (x_two, _) = a.center();
        b.predict(2.0);
        let (x_dt2, _) = b.center();
        assert!(
            (x_two - x_dt2).abs() < 1e-9,
            "predict(2) != two predict(1): {x_two} vs {x_dt2}"
        );
        assert!(x_dt2 > x0, "dt=2 should advance forward");
    }

    #[test]
    fn update_pulls_state_toward_measurement() {
        let mut kf = KalmanFilter3D::new(100.0, 100.0, 20.0, 20.0, None, params());
        let (before, _) = kf.center();
        kf.update(140.0, 100.0, 20.0, 20.0, None);
        let (after, _) = kf.center();
        assert!(
            after > before && after < 140.0,
            "no partial correction: {after}"
        );
    }

    #[test]
    fn depth_coasts_without_measurement() {
        // No depth measurement ever => pz must stay at the nominal seed (motion
        // model has zero depth velocity), proving the huge-R fallback works.
        let mut kf = KalmanFilter3D::new(0.0, 0.0, 10.0, 10.0, None, params());
        let z0 = kf.position_3d()[2];
        for _ in 0..20 {
            kf.predict(1.0);
            kf.update(5.0, 5.0, 10.0, 10.0, None);
        }
        let z1 = kf.position_3d()[2];
        assert!(
            (z0 - z1).abs() < 1e-3,
            "depth drifted without measurement: {z0} -> {z1}"
        );
    }

    #[test]
    fn depth_measurement_sharpens_estimate() {
        // With real depth measurements the estimate converges to the measured value.
        let mut kf = KalmanFilter3D::new(0.0, 0.0, 10.0, 10.0, None, params());
        for _ in 0..30 {
            kf.predict(1.0);
            kf.update(0.0, 0.0, 10.0, 10.0, Some(7.5));
        }
        let z = kf.position_3d()[2];
        assert!((z - 7.5).abs() < 0.2, "depth did not converge: {z}");
    }

    #[test]
    fn covariance_stays_symmetric() {
        let mut kf = KalmanFilter3D::new(0.0, 0.0, 10.0, 10.0, Some(3.0), params());
        for _ in 0..10 {
            kf.predict(1.0);
            kf.update(1.0, 1.0, 10.0, 10.0, Some(3.0));
        }
        let p = &kf.p;
        for r in 0..NX {
            for c in 0..NX {
                assert!(
                    (p[(r, c)] - p[(c, r)]).abs() < 1e-6,
                    "asymmetric P at {r},{c}"
                );
            }
        }
    }

    #[test]
    fn affine_identity_is_noop() {
        let mut kf = KalmanFilter3D::new(10.0, 20.0, 5.0, 5.0, None, params());
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        kf.apply_affine(&id);
        let (x, y) = kf.center();
        assert!((x - 10.0).abs() < 1e-9 && (y - 20.0).abs() < 1e-9);
    }
}
