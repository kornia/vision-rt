//! Minimal pinhole camera model for lifting a track's image-plane + depth state to
//! a **metric 3D** position/velocity in the camera frame.
//!
//! The tracker itself stays native **2.5D** — the Kalman runs in a well-conditioned
//! `[px, py (pixels), pz (metres)]` space (tracking metric `X, Y` directly would
//! couple the horizontal estimate with depth noise and need re-tuning, and
//! association is image-plane IoU regardless). Metric 3D is therefore a **derived
//! readout**: back-project the filtered `(px, py, pz)` through the intrinsics.
//!
//! [`CameraIntrinsics`] mirrors kornia-rs' `kornia_imgproc::calibration::CameraIntrinsic`
//! (`fx, fy, cx, cy`) field-for-field, so it is a trivial swap when this lift moves
//! upstream — we keep a local copy only to spare the lean, pure-CPU tracker the whole
//! `kornia-imgproc` (GPU) dependency. No lens distortion is modelled: it is
//! negligible next to monocular-depth error for this readout.

/// Pinhole camera intrinsics (no distortion). `fx, fy` are focal lengths in pixels,
/// `cx, cy` the principal point in pixels — for the **resolution the tracker runs
/// at** (i.e. after any resize).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl CameraIntrinsics {
    /// Explicit intrinsics.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// **Approximate** intrinsics from the horizontal field-of-view and image size:
    /// `fx = (width / 2) / tan(hfov / 2)`, square pixels (`fy = fx`), principal point
    /// at the image centre, no distortion. Good to ~±10–15 % — adequate next to
    /// monocular-depth's own error, but replace with a checkerboard calibration for
    /// accuracy. `width`/`height` are the tracker's working resolution.
    pub fn from_hfov(width: f32, height: f32, hfov_deg: f32) -> Self {
        let fx = (width * 0.5) / (hfov_deg.to_radians() * 0.5).tan();
        Self {
            fx,
            fy: fx,
            cx: width * 0.5,
            cy: height * 0.5,
        }
    }

    /// Back-project a pixel `(u, v)` at metric depth `z` (metres) to a 3D point in the
    /// camera frame (metres): `X = (u − cx)/fx · z`, `Y = (v − cy)/fy · z`, `Z = z`.
    /// Camera looks down `+Z`, `x` right, `y` down (image convention).
    pub fn unproject(&self, u: f32, v: f32, z: f32) -> [f32; 3] {
        [(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unproject_principal_ray_is_on_axis() {
        let k = CameraIntrinsics::new(800.0, 800.0, 640.0, 360.0);
        // A pixel at the principal point unprojects onto the optical axis (X=Y=0).
        assert_eq!(k.unproject(640.0, 360.0, 3.0), [0.0, 0.0, 3.0]);
        // One focal length off-centre at Z = fx metres ⇒ X = 1 m.
        let p = k.unproject(640.0 + 800.0, 360.0, 800.0 / 800.0);
        assert!((p[0] - 1.0).abs() < 1e-5 && p[2] == 1.0);
    }

    #[test]
    fn from_hfov_focal_and_centre() {
        // 90° HFOV over 1280 px ⇒ fx = 640 / tan(45°) = 640.
        let k = CameraIntrinsics::from_hfov(1280.0, 720.0, 90.0);
        assert!((k.fx - 640.0).abs() < 1e-3);
        assert_eq!((k.cx, k.cy), (640.0, 360.0));
        assert_eq!(k.fx, k.fy); // square pixels
    }
}
