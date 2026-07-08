//! Pinhole camera model — the geometry that ties a [`VrtDepthMap`](crate::VrtDepthMap)
//! to metric 3D. Sensor-agnostic: any depth source (OAK stereo, RealSense, a mono
//! depth model) supplies its intrinsics and gets the same un/projection.

/// Pinhole intrinsics (focal lengths + principal point, in pixels) of an image,
/// at the resolution the pixels are sampled at.
#[derive(Debug, Clone, Copy)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl Intrinsics {
    /// Back-project a pixel `(u,v)` + metric depth to a 3D point in the camera
    /// frame (x right, y down, z forward), in the same units as `depth`.
    pub fn unproject(&self, u: f32, v: f32, depth: f32) -> [f32; 3] {
        [
            (u - self.cx) / self.fx * depth,
            (v - self.cy) / self.fy * depth,
            depth,
        ]
    }

    /// Project a camera-frame point to pixels. `None` if behind the camera
    /// (`z <= ~0`) or projecting to an absurd coordinate.
    pub fn project(&self, p: [f32; 3]) -> Option<(f32, f32)> {
        if p[2] <= 0.05 {
            return None;
        }
        let u = self.fx * p[0] / p[2] + self.cx;
        let v = self.fy * p[1] / p[2] + self.cy;
        (u.abs() <= 1.0e4 && v.abs() <= 1.0e4).then_some((u, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intr() -> Intrinsics {
        // Realistic 640×360 intrinsics.
        Intrinsics {
            fx: 516.0,
            fy: 516.0,
            cx: 320.0,
            cy: 180.0,
        }
    }

    /// `unproject` then `project` round-trips a pixel + depth back to the same pixel.
    #[test]
    fn unproject_project_roundtrips() {
        let k = intr();
        let (u, v, z) = (412.0f32, 95.0f32, 2.7f32);
        let p = k.unproject(u, v, z);
        let (u2, v2) = k.project(p).expect("point in front of camera must project");
        assert!((u2 - u).abs() < 1e-3, "u round-trip {u2} != {u}");
        assert!((v2 - v).abs() < 1e-3, "v round-trip {v2} != {v}");
    }

    /// The principal point (cx,cy) unprojects to (0,0,z): it lies on the optical axis.
    #[test]
    fn center_pixel_unprojects_to_axis() {
        let k = intr();
        let p = k.unproject(k.cx, k.cy, 1.5);
        assert!(
            p[0].abs() < 1e-6,
            "x must be 0 on the optical axis, got {}",
            p[0]
        );
        assert!(
            p[1].abs() < 1e-6,
            "y must be 0 on the optical axis, got {}",
            p[1]
        );
        assert!(
            (p[2] - 1.5).abs() < 1e-6,
            "z must equal depth, got {}",
            p[2]
        );
    }

    /// A known intrinsics + point gives the expected pixel and 3D coords.
    #[test]
    fn known_point_gives_expected_coords() {
        let k = intr();
        // One focal-length to the right at z=1 → +1 px·... actually offset = fx*(x/z).
        // Take x = 1.0, z = 2.0 → u = fx*0.5 + cx = 258 + 320 = 578.
        let (u, v) = k.project([1.0, 0.0, 2.0]).unwrap();
        assert!((u - 578.0).abs() < 1e-4, "u = fx*x/z + cx = 578, got {u}");
        assert!((v - 180.0).abs() < 1e-4, "v = cy = 180, got {v}");
        // And unproject of (578,180,2) returns (1,0,2).
        let p = k.unproject(578.0, 180.0, 2.0);
        assert!((p[0] - 1.0).abs() < 1e-4, "x = 1.0, got {}", p[0]);
        assert!(p[1].abs() < 1e-4, "y = 0.0, got {}", p[1]);
        assert!((p[2] - 2.0).abs() < 1e-6, "z = 2.0, got {}", p[2]);
    }

    /// `project` rejects points at or behind the near plane (z <= 0.05).
    #[test]
    fn project_rejects_non_positive_z() {
        let k = intr();
        assert_eq!(k.project([0.0, 0.0, 0.0]), None, "z = 0 must be rejected");
        assert_eq!(k.project([0.0, 0.0, -1.0]), None, "z < 0 must be rejected");
        assert_eq!(
            k.project([0.0, 0.0, 0.05]),
            None,
            "z = 0.05 (boundary) must be rejected"
        );
        assert!(
            k.project([0.0, 0.0, 0.051]).is_some(),
            "z just past 0.05 must project"
        );
    }

    /// `project` rejects absurd image coordinates (|u| or |v| > 1e4).
    #[test]
    fn project_rejects_absurd_coords() {
        let k = intr();
        // Large lateral offset at a tiny positive z drives u way past 1e4.
        assert_eq!(
            k.project([1000.0, 0.0, 0.06]),
            None,
            "absurd u must be rejected"
        );
        assert_eq!(
            k.project([0.0, 1000.0, 0.06]),
            None,
            "absurd v must be rejected"
        );
    }
}
