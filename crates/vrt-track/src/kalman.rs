//! `Box2DModel` — the per-track 2D Kalman state (constant-velocity).
//!
//! This is the concrete [`TrackModel`](crate::TrackModel) for 2D box tracking;
//! a future `Box3DModel` implements the same trait and drops into the generic
//! `Tracker` unchanged. State is `[cx, cy, w, h, vcx, vcy, vw, vh]` (BoT-SORT
//! parameterization — width/height directly rather than SORT's area/aspect).

use nalgebra::{SMatrix, SVector};

use crate::{Box3D, Observation, TrackModel};

type Vec8 = SVector<f32, 8>;
type Mat8 = SMatrix<f32, 8, 8>;
type Mat4x8 = SMatrix<f32, 4, 8>;
type Mat4 = SMatrix<f32, 4, 4>;

/// Constant-velocity Kalman filter over a 2D box.
#[derive(Debug, Clone)]
pub struct Box2DModel {
    x: Vec8, // state
    p: Mat8, // covariance
}

impl Box2DModel {
    fn measurement_matrix() -> Mat4x8 {
        let mut h = Mat4x8::zeros();
        for i in 0..4 {
            h[(i, i)] = 1.0;
        }
        h
    }
}

impl TrackModel for Box2DModel {
    type Box = [f32; 4];
    type Obs = crate::Obs2D;

    fn new(obs: &Self::Obs) -> Self {
        let z = bbox_to_z(&obs.bbox());
        let mut x = Vec8::zeros();
        for i in 0..4 {
            x[i] = z[i];
        } // velocities start at 0
          // Low uncertainty on the observed box, high on the unobserved velocities.
        let mut p = Mat8::zeros();
        for i in 0..4 {
            p[(i, i)] = 10.0;
        }
        for i in 4..8 {
            p[(i, i)] = 1.0e4;
        }
        Self { x, p }
    }

    fn predict(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        let mut f = Mat8::identity();
        for i in 0..4 {
            f[(i, i + 4)] = dt;
        }
        // Process noise: small on the box, smaller on velocity.
        let mut q = Mat8::zeros();
        for i in 0..4 {
            q[(i, i)] = 1.0;
        }
        for i in 4..8 {
            q[(i, i)] = 1.0e-2;
        }

        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;
    }

    fn update(&mut self, obs: &Self::Obs) {
        let z = SVector::<f32, 4>::from(bbox_to_z(&obs.bbox()));
        let h = Self::measurement_matrix();
        let mut r = Mat4::zeros();
        for i in 0..4 {
            r[(i, i)] = 1.0;
        }

        let y = z - h * self.x; // innovation
        let s = h * self.p * h.transpose() + r; // innovation covariance
        let Some(s_inv) = s.try_inverse() else {
            return;
        };
        let k = self.p * h.transpose() * s_inv; // Kalman gain
        self.x += k * y;
        let i = Mat8::identity();
        self.p = (i - k * h) * self.p;
    }

    fn bbox(&self) -> [f32; 4] {
        z_to_bbox(&[self.x[0], self.x[1], self.x[2], self.x[3]])
    }
}

/// `[x1,y1,x2,y2]` → `[cx,cy,w,h]`.
fn bbox_to_z(b: &[f32; 4]) -> [f32; 4] {
    let w = b[2] - b[0];
    let h = b[3] - b[1];
    [b[0] + w * 0.5, b[1] + h * 0.5, w, h]
}

/// `[cx,cy,w,h]` → `[x1,y1,x2,y2]`.
fn z_to_bbox(z: &[f32; 4]) -> [f32; 4] {
    let (w, h) = (z[2].max(0.0), z[3].max(0.0));
    [
        z[0] - w * 0.5,
        z[1] - h * 0.5,
        z[0] + w * 0.5,
        z[1] + h * 0.5,
    ]
}

// ── Box3DModel ───────────────────────────────────────────────────────────────

type Vec10 = SVector<f32, 10>;
type Mat10 = SMatrix<f32, 10, 10>;
type Mat7x10 = SMatrix<f32, 7, 10>;
type Mat7 = SMatrix<f32, 7, 7>;

/// Constant-velocity Kalman over an oriented 3D box (AB3DMOT parameterization):
/// constant velocity on the center (x,y,z); yaw and size are random-walk. State
/// `[x, y, z, yaw, l, w, h, vx, vy, vz]`; measurement `[x, y, z, yaw, l, w, h]`.
///
/// This is the 3D counterpart of [`Box2DModel`] — same generic [`TrackModel`]
/// trait, so it drops into the [`Tracker`](crate::Tracker) unchanged, with
/// [`Box3D`] BEV-IoU as the association metric. (A full 6-DOF variant would swap
/// the Euclidean yaw for an SE(3) state via kornia-algebra's Lie groups; not
/// needed for ground-plane tracking.)
#[derive(Debug, Clone)]
pub struct Box3DModel {
    x: Vec10,
    p: Mat10,
}

impl Box3DModel {
    fn measurement_matrix() -> Mat7x10 {
        let mut h = Mat7x10::zeros();
        for i in 0..7 {
            h[(i, i)] = 1.0;
        }
        h
    }
}

impl TrackModel for Box3DModel {
    type Box = Box3D;
    type Obs = crate::Obs3D;

    fn new(obs: &Self::Obs) -> Self {
        let z = box3d_to_z(&obs.bbox());
        let mut x = Vec10::zeros();
        for i in 0..7 {
            x[i] = z[i];
        } // velocities start at 0
        let mut p = Mat10::zeros();
        for i in 0..7 {
            p[(i, i)] = 1.0;
        } // moderate initial position/size uncertainty
        for i in 7..10 {
            p[(i, i)] = 4.0;
        } // velocity ≈ 2 m/s std (indoor people), NOT "unknown"
          // (was 1e3) — a huge prior let one noisy detection slam
          // in a giant velocity that the CV predict flung off
        Self { x, p }
    }

    fn predict(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        let mut f = Mat10::identity();
        f[(0, 7)] = dt;
        f[(1, 8)] = dt;
        f[(2, 9)] = dt; // center += velocity·dt
                        // Process noise, scaled by the time step. Small on position/size — the state evolves
                        // smoothly, so the old Q=1.0 (≈1 m of unmodeled motion per frame) just made the filter
                        // copy the noisy measurement instead of smoothing it. Modest on velocity so it adapts
                        // without exploding.
        let mut q = Mat10::zeros();
        let (q_pos, q_yaw, q_size, q_vel) = (0.02 * dt, 0.01 * dt, 0.01 * dt, 0.05 * dt);
        q[(0, 0)] = q_pos;
        q[(1, 1)] = q_pos;
        q[(2, 2)] = q_pos;
        q[(3, 3)] = q_yaw;
        q[(4, 4)] = q_size;
        q[(5, 5)] = q_size;
        q[(6, 6)] = q_size;
        q[(7, 7)] = q_vel;
        q[(8, 8)] = q_vel;
        q[(9, 9)] = q_vel;

        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;
    }

    fn update(&mut self, obs: &Self::Obs) {
        let mut z = SVector::<f32, 7>::from(box3d_to_z(&obs.bbox()));
        // Yaw: take the short way around the ±π seam, and undo a heading flip
        // (detection ~180° opposite the track) so the filter doesn't spin.
        let mut dyaw = wrap_pi(z[3] - self.x[3]);
        if dyaw.abs() > std::f32::consts::FRAC_PI_2 {
            z[3] += std::f32::consts::PI;
            dyaw = wrap_pi(z[3] - self.x[3]);
        }
        z[3] = self.x[3] + dyaw; // express the measurement continuously vs. state

        let h = Self::measurement_matrix();
        // Range-dependent measurement noise: stereo/ToF depth error grows ~quadratically with range
        // and the back-projected lateral/size error ~linearly, so a far detection is heavily
        // discounted while a near one is trusted. The old constant R=1.0 trusted noisy far depth
        // equally → jitter + velocity spikes. `zr` is the *measured* range.
        let (s_xy, s_z) = pos_meas_var(z[2]);
        let s_size = (0.05 + 0.05 * z[2].max(0.3)).powi(2);
        let mut r = Mat7::zeros();
        r[(0, 0)] = s_xy;
        r[(1, 1)] = s_xy;
        r[(2, 2)] = s_z;
        r[(3, 3)] = 0.2; // yaw is ~unobservable from a single-view lift → don't react to it
        r[(4, 4)] = s_size;
        r[(5, 5)] = s_size;
        r[(6, 6)] = s_size;

        let y = z - h * self.x;
        let s = h * self.p * h.transpose() + r;
        let Some(s_inv) = s.try_inverse() else {
            return;
        };
        let k = self.p * h.transpose() * s_inv;
        self.x += k * y;
        // Velocity clamp: an indoor object can't exceed ~5 m/s, so a spurious velocity from a noisy
        // depth spike is capped instead of flinging the predicted box out of the gate next frame —
        // which is exactly how a single bad detection used to "lose" the track.
        const MAX_V: f32 = 5.0;
        let sp = (self.x[7] * self.x[7] + self.x[8] * self.x[8] + self.x[9] * self.x[9]).sqrt();
        if sp > MAX_V {
            let scale = MAX_V / sp;
            self.x[7] *= scale;
            self.x[8] *= scale;
            self.x[9] *= scale;
        }
        let i = Mat10::identity();
        self.p = (i - k * h) * self.p;
    }

    fn bbox(&self) -> Box3D {
        Box3D {
            center: [self.x[0], self.x[1], self.x[2]],
            size: [self.x[4].max(0.0), self.x[5].max(0.0), self.x[6].max(0.0)],
            yaw: wrap_pi(self.x[3]),
        }
    }

    /// 3D association affinity = **metric position** × **size consistency**, both in
    /// `[0,1]`. Position is a Mahalanobis-distance Gaussian using the Kalman center
    /// covariance (a coasting/uncertain track widens its own gate, a confident one
    /// tightens it) — distance in *metres*, not box overlap, so small depth-noisy
    /// boxes that never IoU-overlap still associate. Size consistency lets the real
    /// 3D extent discriminate identity and veto class mislabels.
    fn affinity(&self, obs: &Self::Obs) -> f32 {
        let db = &obs.bbox;
        // Gate on the **innovation covariance** S = P + R(range): the predicted-state uncertainty
        // PLUS the range-dependent measurement noise. So the gate widens for far/noisy detections —
        // plausible depth jitter ASSOCIATES (and is then smoothed by R) instead of being rejected,
        // which is what made a noisy frame drop the track. Floored so a near gate isn't razor-thin.
        let (s_xy, s_z) = pos_meas_var(db.center[2]);
        let var = [
            (self.p[(0, 0)] + s_xy).max(0.0225),
            (self.p[(1, 1)] + s_xy).max(0.0225),
            (self.p[(2, 2)] + s_z).max(0.0225),
        ];
        let d = [
            self.x[0] - db.center[0],
            self.x[1] - db.center[1],
            self.x[2] - db.center[2],
        ];
        let m2 = d[0] * d[0] / var[0] + d[1] * d[1] / var[1] + d[2] * d[2] / var[2];
        let pos = (-0.5 * m2).exp();
        let tsize = [self.x[4].max(0.0), self.x[5].max(0.0), self.x[6].max(0.0)];
        pos * size_affinity(&tsize, &db.size)
    }

    /// Allow a *near + size-consistent* cross-class detection to bind (so a momentary
    /// mislabel is absorbed, not duplicated); the sticky class vote keeps the label.
    fn class_mismatch_factor() -> f32 {
        0.4
    }
}

/// Agreement of two oriented-box extents `[l,w,h]` in `[0,1]` — the geometric mean of
/// the per-axis min/max ratios (1 = identical, →0 = very different). This is what lets
/// 3D size disambiguate identity: a phone-labeled box that is actually person-sized
/// scores ~1 against the person track (so it's absorbed) and ~0.1 against a real phone.
/// Range-dependent position **measurement** variance `(lateral, depth)` for a measured depth `zr`.
/// Stereo/ToF depth error grows ~quadratically with range; the back-projected lateral error
/// ~linearly. Shared by the Kalman `R` (how much to trust a measurement) and the association gate
/// (how far to look) so both scale consistently with the sensor physics.
fn pos_meas_var(zr: f32) -> (f32, f32) {
    let zr = zr.max(0.3);
    (
        (0.015 + 0.010 * zr).powi(2),
        (0.020 + 0.010 * zr * zr).powi(2),
    )
}

fn size_affinity(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let mut acc = 1.0f32;
    for i in 0..3 {
        let lo = a[i].min(b[i]).max(1.0e-3);
        let hi = a[i].max(b[i]).max(1.0e-3);
        acc *= lo / hi;
    }
    acc.cbrt()
}

fn box3d_to_z(b: &Box3D) -> [f32; 7] {
    [
        b.center[0],
        b.center[1],
        b.center[2],
        b.yaw,
        b.size[0],
        b.size[1],
        b.size[2],
    ]
}

/// Wrap an angle to `[-π, π]`.
fn wrap_pi(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut r = (a + PI).rem_euclid(TAU) - PI;
    if r <= -PI {
        r += TAU;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Obs2D, Obs3D};

    fn o3(center: [f32; 3], yaw: f32) -> Obs3D {
        Obs3D {
            bbox: Box3D {
                center,
                size: [4.0, 2.0, 1.5],
                yaw,
            },
            score: 1.0,
            class_id: 0,
        }
    }

    #[test]
    fn box3d_constant_velocity_prediction() {
        // Object moving +1 m/frame in x; predict() should extrapolate the step.
        let mut m = Box3DModel::new(&o3([0.0, 0.0, 0.0], 0.0));
        m.predict(1.0);
        m.update(&o3([1.0, 0.0, 0.0], 0.0));
        m.predict(1.0);
        m.update(&o3([2.0, 0.0, 0.0], 0.0));
        m.predict(1.0);
        let x = m.bbox().center[0];
        assert!(x > 2.5 && x < 3.5, "x={x}"); // learned ≈ +1/frame → ~3
    }

    #[test]
    fn box3d_yaw_wrap() {
        // True heading sits near the ±π seam; measuring just across it must not
        // make the filter spin most of a turn.
        let mut m = Box3DModel::new(&o3([0.0, 0.0, 0.0], 3.10));
        m.predict(1.0);
        m.update(&o3([0.0, 0.0, 0.0], -3.10)); // ≈ +0.08 rad across the seam
        let yaw = m.bbox().yaw;
        assert!(yaw.abs() > 3.0, "yaw should stay near π, got {yaw}");
    }

    #[test]
    fn box3d_yaw_flip() {
        // Detection heading flipped 180° (common box-orientation ambiguity):
        // the track must not rotate ~π.
        let mut m = Box3DModel::new(&o3([0.0, 0.0, 0.0], 0.2));
        m.predict(1.0);
        m.update(&o3([0.0, 0.0, 0.0], 0.2 + std::f32::consts::PI));
        let yaw = m.bbox().yaw;
        assert!(yaw.abs() < 0.6, "flip should be undone, got {yaw}");
    }

    #[test]
    fn constant_velocity_prediction() {
        // A box moving +10px/frame in x; after update on two frames, predict()
        // should extrapolate roughly the same step.
        let mut m = Box2DModel::new(&Obs2D {
            bbox: [0.0, 0.0, 10.0, 10.0],
            score: 1.0,
            class_id: 0,
        });
        m.predict(1.0);
        m.update(&Obs2D {
            bbox: [10.0, 0.0, 20.0, 10.0],
            score: 1.0,
            class_id: 0,
        });
        m.predict(1.0);
        m.update(&Obs2D {
            bbox: [20.0, 0.0, 30.0, 10.0],
            score: 1.0,
            class_id: 0,
        });
        m.predict(1.0);
        let b = m.bbox();
        let cx = (b[0] + b[2]) * 0.5;
        // velocity learned ≈ +10/frame → next cx should be near 35 (was 25).
        assert!(cx > 30.0 && cx < 40.0, "cx={cx}");
    }
}
