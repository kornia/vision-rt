//! Sensor-agnostic **depth → 3D bridge**: turn a 2D detection (box + class) plus
//! an aligned [`VrtDepthMap`] and camera [`Intrinsics`] into a metric [`Obs3D`]
//! the [`Box3DTracker`](vrt_track::Box3DTracker) can consume.
//!
//! This is intentionally independent of any specific camera crate (OAK, RealSense,
//! a mono-depth model) — it depends only on the core `vrt` geometry/depth types
//! and the `vrt-track` observation type. Any source that produces a depth map +
//! intrinsics reuses the same lifter.

use std::collections::HashMap;

use vrt::{Intrinsics, VrtDepthMap};
use vrt_track::{Box3D, Obs3D};

/// Physical size priors `[l, w, h]` in metres, indexed by detector class id. The
/// 3D *position* of a track comes from depth; its box *dimensions* are a per-class
/// prior (refined later if a sized detector is added). COCO-ish defaults; override
/// per class via [`SizePriors::set`].
#[derive(Debug, Clone)]
pub struct SizePriors {
    default: [f32; 3],
    by_class: HashMap<u32, [f32; 3]>,
}

impl Default for SizePriors {
    fn default() -> Self {
        let mut by_class = HashMap::new();
        by_class.insert(1, [0.6, 0.6, 1.7]); // person
        by_class.insert(2, [1.8, 0.6, 1.2]); // bicycle
        by_class.insert(3, [4.2, 1.8, 1.5]); // car
        Self {
            default: [0.5, 0.5, 0.5],
            by_class,
        }
    }
}

impl SizePriors {
    /// Size prior for a class id (the default if unset).
    pub fn get(&self, class_id: u32) -> [f32; 3] {
        *self.by_class.get(&class_id).unwrap_or(&self.default)
    }
    /// Override the size prior for a class id.
    pub fn set(&mut self, class_id: u32, size: [f32; 3]) {
        self.by_class.insert(class_id, size);
    }
}

/// Lifts 2D detections to 3D observations in the camera frame, using aligned
/// depth + intrinsics + per-class size priors. The reusable depth→3D bridge
/// between any 2D detector and the `Box3DTracker`.
#[derive(Debug, Clone)]
pub struct Lifter {
    pub intr: Intrinsics,
    pub priors: SizePriors,
    /// Fraction of each box (centered) to sample depth over. Default 0.6.
    pub core_frac: f32,
}

impl Lifter {
    pub fn new(intr: Intrinsics) -> Self {
        Self {
            intr,
            priors: SizePriors::default(),
            core_frac: 0.6,
        }
    }

    /// Lift one 2D detection to an `Obs3D`. Returns `None` when the box has no
    /// valid depth (the tracker then coasts on prediction for that object).
    pub fn lift(
        &self,
        depth: &VrtDepthMap,
        bbox: &[f32; 4],
        score: f32,
        class_id: u32,
    ) -> Option<Obs3D> {
        let z = depth.sample_box(bbox, self.core_frac)?;
        let u = (bbox[0] + bbox[2]) * 0.5;
        let v = (bbox[1] + bbox[3]) * 0.5;
        let center = self.intr.unproject(u, v, z);
        // Metric extent from the pixel box + depth: a detection's *real* apparent size in the world
        // (`px · z / focal`), so a near object reads bigger than a far one and a phone stays smaller
        // than a person — instead of a flat per-class guess. The depth-extent (`length`, the forward
        // dimension) is unobservable from a single view, so keep the class prior for that axis only.
        let prior = self.priors.get(class_id);
        let w_m = ((bbox[2] - bbox[0]).max(1.0) * z / self.intr.fx).clamp(0.05, 4.0);
        let h_m = ((bbox[3] - bbox[1]).max(1.0) * z / self.intr.fy).clamp(0.05, 4.0);
        // COORDINATE FRAME (important): `center` and the whole `Box3D` below are in the CAMERA
        // frame — x = right, y = DOWN, z = forward/depth — as produced by `Intrinsics::unproject`.
        // This is NOT the z-up / ground-plane world frame that `vrt-track`'s `Box3D` (and its
        // BEV-IoU) nominally documents. We intentionally feed camera-frame coordinates: the lift →
        // tracker → viewer pipeline is internally self-consistent (the tracker's tuning and the
        // viewer's axis mapping all assume this same camera frame), and rotating to z-up would
        // ripple through both. `size` is `[length_prior, width_metric, height_metric]`: width/height
        // are the metric back-projection of the pixel box (`px·z/focal`), length is the per-class
        // prior since the forward extent is unobservable from one view.
        // CONSUMER WARNING: any downstream that trusts `Box3D`'s documented z-up/ground-plane
        // semantics (e.g. a world-map or furniture-anchoring step) must account for this — the
        // points here are camera-frame (y is down, not up), not world-frame.
        Some(Obs3D {
            bbox: Box3D {
                center,
                size: [prior[0], w_m, h_m],
                yaw: 0.0,
            },
            score,
            class_id,
        })
    }

    /// Lift a single 2D point `(u,v)` (e.g. a pose keypoint) to a 3D point in the camera frame,
    /// using a robust small-window depth median. Returns `None` when no valid depth is near the
    /// point — the caller drops that joint (and simply doesn't draw its bones).
    pub fn lift_point(&self, depth: &VrtDepthMap, u: f32, v: f32) -> Option<[f32; 3]> {
        let z = depth.sample_point(u, v, 2)?; // 5×5 window
        Some(self.intr.unproject(u, v, z))
    }

    /// Lift a pose keypoint `(u,v)` **anchored to a reference body depth** `z_ref` (e.g. the person's
    /// torso/box depth). The joint uses its own depth only when it's valid AND within `tol` metres of
    /// `z_ref`; otherwise (hole, spike, or an implausible deviation from the body) it falls back to
    /// `z_ref`. This keeps the whole skeleton at a coherent distance — a single noisy depth sample
    /// can no longer fling a joint off in z (and, since `x,y ∝ z`, off laterally too). The 2D pose
    /// (which the detector predicts reliably) drives the skeleton shape; depth only adds bounded
    /// relief. Always returns a point (the joint is placed even where depth is missing).
    pub fn lift_point_anchored(
        &self,
        depth: &VrtDepthMap,
        u: f32,
        v: f32,
        z_ref: f32,
        tol: f32,
    ) -> [f32; 3] {
        let z = match depth.sample_point(u, v, 2) {
            Some(z) if (z - z_ref).abs() <= tol => z, // trustworthy, body-consistent depth → use it
            _ => z_ref, // missing / spike / outlier → anchor to the body
        };
        self.intr.unproject(u, v, z)
    }

    /// Depth for a LIMB joint (arm/leg) — its OWN foreground depth, so a reach toward/away from the
    /// camera (e.g. an arm raised forward, "roman salute") projects correctly instead of being pinned
    /// to the body plane. A thin limb occupies few pixels, so a plain median picks the BACKGROUND
    /// behind it; [`VrtDepthMap::sample_point_foreground`] takes the closer (foreground) cluster to
    /// recover the limb. The result is clamped to a plausible reach (`LIMB_REACH` m) around `z_ref` and
    /// falls back to `z_ref` when there's no valid/plausible depth, so noise can't fling a joint off.
    /// TORSO joints must NOT use this — keep them at `z_ref`, or (since `x,y ∝ z`) the skeleton shears.
    pub fn limb_depth(&self, depth: &VrtDepthMap, u: f32, v: f32, z_ref: f32) -> f32 {
        const LIMB_REACH: f32 = 0.9; // m — max a limb plausibly extends in z from the body
        match depth.sample_point_foreground(u, v, 3) {
            Some(z) if (z - z_ref).abs() <= LIMB_REACH => z,
            _ => z_ref,
        }
    }

    /// **Bone-length kinematic constraint** (Stage 2 of the 2.5D lift) — treats each limb bone as
    /// RIGID. Refines a lifted 17-joint COCO skeleton (camera-frame `[x,y,z,conf]` per joint) in place
    /// so each limb bone has a CONSTANT 3D length frame-to-frame. A real bone doesn't change length —
    /// only its orientation does — so depth must not be allowed to set the length, only to pick the
    /// joint's front/back placement. For each limb joint we slide it ALONG ITS OWN 2D RAY (the reliable
    /// signal) to the depth at which the bone to its already-placed parent equals the target length,
    /// choosing the branch nearest the measured depth. The 2D pose shape and the reach (an arm extended
    /// toward the camera foreshortens in 2D and reads a nearer depth → the joint lands on the front of
    /// the sphere) are both preserved; the limb can no longer stretch/shrink as a weak depth flickers.
    ///
    /// `target[t]` is the length to ENFORCE for bone-type `t` (0=upper arm, 1=forearm, 2=thigh,
    /// 3=shank), e.g. a per-track self-calibrated length ramped in from the default; `None` falls back
    /// to the anthropometric length (estimated stature × standard fraction). Returns, per type, the mean
    /// MEASURED length of the bones whose raw measurement was already PLAUSIBLE this frame (within a
    /// confidence-widened band of the target) — the clean observations the caller folds into its
    /// calibration. Implausible (garbage-depth) bones are not returned, so the calibration never learns
    /// from a bone that was itself forced onto the prior.
    ///
    /// Chains are walked shoulder→elbow→wrist and hip→knee→ankle, anchored at the torso joints
    /// (5,6,11,12) which Stage 1 pins to the stable root depth. A bone is skipped when neither a
    /// caller-supplied nor an anthropometric length is available (no torso scale). Head joints (0–4) are
    /// left to Stage 1 + smoothing — they form no clean chain.
    ///
    /// `prev_depth` carries each joint's last enforced depth across frames (per-track state owned by the
    /// caller, all-zero initially) for temporal branch hysteresis — it's read to hold a near-coplanar
    /// joint on its previous front/back side and written with this frame's enforced depth (reset to 0
    /// when the joint isn't placed, so a re-appearing joint starts fresh).
    pub fn refine_bones(
        &self,
        sk: &mut [[f32; 4]],
        target: &[Option<f32>; 4],
        prev_depth: &mut [f32; 17],
    ) -> [Option<f32>; 4] {
        let mut observed: [Option<f32>; 4] = [None; 4];
        if sk.len() < 17 {
            return observed;
        }
        let stature = estimate_stature(sk);
        let mut acc = [(0.0f32, 0u32); 4]; // (sum, count) of plausible raw bone lengths per type
        for (p, c, ty) in LIMB_BONES {
            // Resolve the length to enforce: caller's value, else anthropometric (stature × fraction);
            // skip the bone if neither is known.
            let Some(len) = target[ty].or_else(|| stature.map(|h| h * BONE_FRACTION[ty])) else {
                continue;
            };
            if sk[p][3] <= 0.0 || sk[c][3] <= 0.0 {
                prev_depth[c] = 0.0; // joint not placed → forget its side (re-appears fresh)
                continue;
            }
            let parent = [sk[p][0], sk[p][1], sk[p][2]];
            let zc = sk[c][2];
            if zc <= 0.05 {
                prev_depth[c] = 0.0;
                continue;
            }
            let (ax, ay) = (sk[c][0] / zc, sk[c][1] / zc); // the child's fixed view ray (x,y per unit z)
            let measured = dist3(parent, [sk[c][0], sk[c][1], sk[c][2]]);
            // Learn ONLY from a raw measurement that's already plausible (confidence-widened band) — a
            // garbage-depth bone shouldn't pollute the calibration. `slack` ∈ ~[0.5,0.8] over conf.
            let slack = 0.35 + 0.45 * sk[c][3];
            let (lo, hi) = ((1.0 - slack).max(0.2), 1.0 + slack);
            if measured >= lo * len && measured <= hi * len {
                acc[ty].0 += measured;
                acc[ty].1 += 1;
            }
            // ENFORCE the rigid length every frame: place the child on its ray at distance `len` from
            // the parent, side chosen by the measured depth with temporal hysteresis (held via
            // `prev_depth[c]`). When the measurement is already on the sphere this barely moves it;
            // when it isn't, the length is corrected without leaving the 2D ray. Constant length
            // frame-to-frame is the point.
            let t = depth_for_bone_length(parent, ax, ay, len, zc, prev_depth[c]);
            sk[c] = [ax * t, ay * t, t, sk[c][3]];
            prev_depth[c] = t;
        }
        for ty in 0..4 {
            if acc[ty].1 > 0 {
                observed[ty] = Some(acc[ty].0 / acc[ty].1 as f32);
            }
        }
        observed
    }
}

/// COCO limb bones as `(parent, child, bone_type)`, walked proximal→distal so each parent is placed
/// before its child. Left and right share a bone-TYPE (0=upper arm, 1=forearm, 2=thigh, 3=shank), so a
/// caller calibrating per type pools both sides — symmetry for free, and twice the samples per type.
pub const LIMB_BONES: [(usize, usize, usize); 8] = [
    (5, 7, 0),
    (7, 9, 1), // left  arm: upper arm, forearm
    (6, 8, 0),
    (8, 10, 1), // right arm
    (11, 13, 2),
    (13, 15, 3), // left  leg: thigh, shank
    (12, 14, 2),
    (14, 16, 3), // right leg
];

/// Anthropometric bone length as a fraction of stature `H`, indexed by bone-type (matches
/// [`LIMB_BONES`]). Standard proportions: upper arm 0.186·H, forearm 0.146·H, thigh 0.245·H, shank
/// 0.246·H. The fallback when the caller has no calibrated length yet.
pub const BONE_FRACTION: [f32; 4] = [0.186, 0.146, 0.245, 0.246];

/// Default per-bone-type lengths (metres) for a skeleton, from its estimated stature × [`BONE_FRACTION`].
/// `None` when the torso scale can't be measured. A convenience for callers that don't self-calibrate.
pub fn default_bone_lengths(sk: &[[f32; 4]]) -> Option<[f32; 4]> {
    let h = estimate_stature(sk)?;
    Some([
        h * BONE_FRACTION[0],
        h * BONE_FRACTION[1],
        h * BONE_FRACTION[2],
        h * BONE_FRACTION[3],
    ])
}

/// Euclidean distance between two camera-frame points.
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Estimate the person's stature `H` (metres) from the most reliable torso measurement available, used
/// to scale anatomical bone lengths. Torso joints sit at the stable root depth, so their 3D span is
/// trustworthy. Prefers shoulder width, then hip width, then shoulder-to-hip torso height; `None` if
/// none is measurable. Fractions are standard anthropometry (biacromial ≈ 0.259·H, bi-iliac ≈ 0.191·H,
/// acromion-to-hip ≈ 0.30·H).
fn estimate_stature(sk: &[[f32; 4]]) -> Option<f32> {
    let pt = |i: usize| (sk[i][3] > 0.0).then(|| [sk[i][0], sk[i][1], sk[i][2]]);
    let mid = |a: usize, b: usize| match (pt(a), pt(b)) {
        (Some(p), Some(q)) => Some([
            (p[0] + q[0]) * 0.5,
            (p[1] + q[1]) * 0.5,
            (p[2] + q[2]) * 0.5,
        ]),
        (Some(p), None) | (None, Some(p)) => Some(p),
        (None, None) => None,
    };
    if let (Some(l), Some(r)) = (pt(5), pt(6)) {
        return Some(dist3(l, r) / 0.259); // shoulder (biacromial) width
    }
    if let (Some(l), Some(r)) = (pt(11), pt(12)) {
        return Some(dist3(l, r) / 0.191); // hip (bi-iliac) width
    }
    if let (Some(sh), Some(hip)) = (mid(5, 6), mid(11, 12)) {
        return Some(dist3(sh, hip) / 0.30); // acromion-to-hip torso height
    }
    None
}

/// Depth `t` along the ray `(ax·t, ay·t, t)` (a pixel's view ray, `ax = (u-cx)/fx`) at which the point
/// is exactly `len` from `parent`. Solving `|ray(t) − parent|² = len²` is a quadratic in `t`; with two
/// real roots the ray pierces the sphere of radius `len` — a FRONT (nearer) and a BACK (farther)
/// placement of the joint, split by the closest-approach depth `t*`. We follow the measured depth
/// `meas` to choose the side, EXCEPT — for **temporal branch hysteresis** — while `meas` sits in a
/// deadband around `t*` (the limb near-coplanar with the view, where the choice is ambiguous and noise
/// would flicker it) we HOLD the side the joint was on last frame (`prev`, the previously enforced
/// depth; `≤0.05` means no history). A decisive crossing past the deadband still flips it. If the ray
/// misses the sphere, take the closest-approach depth (the best achievable). Always returns a small
/// positive depth.
fn depth_for_bone_length(
    parent: [f32; 3],
    ax: f32,
    ay: f32,
    len: f32,
    meas: f32,
    prev: f32,
) -> f32 {
    const HYST: f32 = 0.05; // m — deadband half-width around the front/back crossing
    let a = ax * ax + ay * ay + 1.0;
    let b = -2.0 * (ax * parent[0] + ay * parent[1] + parent[2]);
    let c = parent[0].powi(2) + parent[1].powi(2) + parent[2].powi(2) - len * len;
    let disc = b * b - 4.0 * a * c;
    let t_star = -b / (2.0 * a); // closest-approach depth = midpoint of the two roots
    if disc <= 0.0 {
        return t_star.max(0.05); // ray misses the sphere → one (closest) option, no branch to pick
    }
    let sd = disc.sqrt();
    let t_front = (-b - sd) / (2.0 * a); // nearer root
    let t_back = (-b + sd) / (2.0 * a); // farther root
                                        // Side choice with hysteresis: hold the previous side inside the deadband, else follow the measurement.
    let pick_front = if prev > 0.05 && (meas - t_star).abs() < HYST {
        prev < t_star // ambiguous → keep the side the joint was on
    } else {
        meas < t_star // decisive (or no history) → nearest the measured depth
    };
    let t = if pick_front { t_front } else { t_back };
    // guard: the chosen branch must be in front of the camera; else fall back to the other / closest.
    if t > 0.05 {
        t
    } else if t_front > 0.05 {
        t_front
    } else if t_back > 0.05 {
        t_back
    } else {
        t_star.max(0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vrt::VrtDepthMap;

    #[test]
    fn lift_size_matches_back_projection() {
        // Flat wall at 3 m; a 100×290 px person box centered at the principal point of a 640×360
        // image. The metric extent must be the exact pixel-to-metre back-projection: width = px·z/fx,
        // height = px·z/fy — verifying the 3D sizes are physically correct (a person ≈ 0.58 × 1.69 m).
        let (w, h) = (640u32, 360u32);
        let depth = VrtDepthMap::owned(vec![3000u16; (w * h) as usize], w, h); // 3000 mm everywhere
        let intr = Intrinsics {
            fx: 516.0,
            fy: 515.0,
            cx: 320.0,
            cy: 180.0,
        };
        let lifter = Lifter::new(intr);
        let bbox = [270.0, 35.0, 370.0, 325.0]; // 100 px wide, 290 px tall, centered
        let o = lifter.lift(&depth, &bbox, 0.9, 1).unwrap();
        assert!(
            (o.bbox.center[2] - 3.0).abs() < 0.01,
            "range {} != 3 m",
            o.bbox.center[2]
        );
        assert!(
            (o.bbox.size[1] - 100.0 * 3.0 / 516.0).abs() < 0.01,
            "width {} != ~0.58 m",
            o.bbox.size[1]
        );
        assert!(
            (o.bbox.size[2] - 290.0 * 3.0 / 515.0).abs() < 0.01,
            "height {} != ~1.69 m",
            o.bbox.size[2]
        );
    }

    #[test]
    fn lift_size_shrinks_with_distance() {
        // Same pixel box read at 6 m vs 3 m must be twice the metric size — confirms size tracks
        // range (a far object reads bigger in metres for the same pixel box, smaller for a fixed
        // real object whose pixel box shrinks). Sanity that size is genuinely metric, not pixels.
        let (w, h) = (640u32, 360u32);
        let intr = Intrinsics {
            fx: 516.0,
            fy: 515.0,
            cx: 320.0,
            cy: 180.0,
        };
        let lifter = Lifter::new(intr);
        let bbox = [300.0, 150.0, 340.0, 210.0];
        let near = lifter
            .lift(
                &VrtDepthMap::owned(vec![3000u16; (w * h) as usize], w, h),
                &bbox,
                0.9,
                1,
            )
            .unwrap();
        let far = lifter
            .lift(
                &VrtDepthMap::owned(vec![6000u16; (w * h) as usize], w, h),
                &bbox,
                0.9,
                1,
            )
            .unwrap();
        assert!(
            (far.bbox.size[2] - 2.0 * near.bbox.size[2]).abs() < 0.02,
            "size must scale with range"
        );
    }

    fn intr() -> Intrinsics {
        Intrinsics {
            fx: 516.0,
            fy: 516.0,
            cx: 320.0,
            cy: 180.0,
        }
    }

    /// `lift_point_anchored` uses the joint's OWN depth when it is valid and within `tol`
    /// of the reference body depth.
    #[test]
    fn anchored_uses_joint_depth_when_within_tol() {
        // Uniform 3.0 m wall; reference body depth 3.0 m, tol 0.5 m → joint depth (3.0)
        // is within tol, so the lifted z must be the joint's 3.0 m.
        let (w, h) = (640u32, 360u32);
        let depth = VrtDepthMap::owned(vec![3000u16; (w * h) as usize], w, h);
        let lifter = Lifter::new(intr());
        let p = lifter.lift_point_anchored(&depth, 320.0, 180.0, 3.0, 0.5);
        assert!(
            (p[2] - 3.0).abs() < 1e-6,
            "z must use joint depth 3.0 m, got {}",
            p[2]
        );
    }

    /// When the joint's depth deviates from `z_ref` by more than `tol`, it falls back to
    /// `z_ref` (a spike can't fling the joint off in depth).
    #[test]
    fn anchored_falls_back_to_zref_when_outside_tol() {
        // Wall at 8.0 m everywhere, but the body reference is 3.0 m with a tight tol 0.5 m.
        // The 8.0 m joint sample is implausible → fall back to z_ref = 3.0 m.
        let (w, h) = (640u32, 360u32);
        let depth = VrtDepthMap::owned(vec![8000u16; (w * h) as usize], w, h);
        let lifter = Lifter::new(intr());
        let p = lifter.lift_point_anchored(&depth, 320.0, 180.0, 3.0, 0.5);
        assert!(
            (p[2] - 3.0).abs() < 1e-6,
            "z must fall back to z_ref 3.0 m, got {}",
            p[2]
        );
    }

    /// With NO valid depth anywhere, `lift_point_anchored` still returns a point anchored
    /// at `z_ref` (the joint is always placed).
    #[test]
    fn anchored_all_invalid_returns_zref() {
        let (w, h) = (640u32, 360u32);
        let depth = VrtDepthMap::owned(vec![0u16; (w * h) as usize], w, h); // all holes
        let lifter = Lifter::new(intr());
        let z_ref = 2.5;
        // At the principal point x,y must be 0; z must be z_ref.
        let p = lifter.lift_point_anchored(&depth, 320.0, 180.0, z_ref, 0.5);
        assert!(
            (p[2] - z_ref).abs() < 1e-6,
            "z must anchor to z_ref, got {}",
            p[2]
        );
        assert!(
            p[0].abs() < 1e-6 && p[1].abs() < 1e-6,
            "principal point must lift to (0,0,z_ref)"
        );
    }

    /// `lift_point` returns `None` when no valid depth is near the point (all holes).
    #[test]
    fn lift_point_none_when_no_depth() {
        let (w, h) = (640u32, 360u32);
        let depth = VrtDepthMap::owned(vec![0u16; (w * h) as usize], w, h);
        let lifter = Lifter::new(intr());
        assert!(
            lifter.lift_point(&depth, 320.0, 180.0).is_none(),
            "all-hole window → None"
        );
    }

    // ---- refine_bones (Stage 2: bone-length constraint) ----

    /// Build a skeleton stub: shoulders at the root depth (defining stature), and the joints we name.
    /// `j` entries are `(index, [x,y,z], conf)`.
    fn skel(joints: &[(usize, [f32; 3], f32)]) -> Vec<[f32; 4]> {
        let mut sk = vec![[0.0f32; 4]; 17];
        for &(i, p, c) in joints {
            sk[i] = [p[0], p[1], p[2], c];
        }
        sk
    }

    fn d3(a: [f32; 4], b: [f32; 4]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// A correctly-proportioned forearm is left untouched (its measured depth is trusted).
    #[test]
    fn refine_keeps_plausible_bone() {
        let lifter = Lifter::new(intr());
        // 0.30 m shoulder width → H = 0.30/0.259 ≈ 1.158 m; forearm ≈ 0.146·H ≈ 0.169 m.
        let h = 0.30 / 0.259;
        let forearm = 0.146 * h;
        let upper = 0.186 * h;
        let mut sk = skel(&[
            (5, [-0.15, 0.0, 2.0], 0.9),
            (6, [0.15, 0.0, 2.0], 0.9),
            (7, [-0.15, upper, 2.0], 0.9), // elbow at exact upper-arm length below shoulder
            (9, [-0.15, upper + forearm, 2.0], 0.9), // wrist at exact forearm length below elbow
        ]);
        let before = sk[9];
        let obs = lifter.refine_bones(&mut sk, &[None; 4], &mut [0.0; 17]);
        assert!(
            d3(sk[9], before) < 1e-4,
            "a plausible forearm must be left unchanged, moved by {}",
            d3(sk[9], before)
        );
        // a kept (plausible) bone is reported back for calibration; forearm is type 1.
        assert!(
            obs[1].is_some_and(|l| (l - forearm).abs() < 1e-3),
            "kept forearm length must be observed, got {:?}",
            obs[1]
        );
    }

    /// A wrist with a garbage (far) depth — forearm absurdly long — is slid back along its OWN view ray
    /// to the expected forearm length: the 2D ray (x/z, y/z) is preserved, only the depth changes.
    #[test]
    fn refine_snaps_overlong_bone_onto_ray() {
        let lifter = Lifter::new(intr());
        let h = 0.30 / 0.259;
        let upper = 0.186 * h;
        let forearm = 0.146 * h;
        // Garbage wrist: way out at z=4 (forearm ≈ 2 m, ~12× expected). Ray dirs ax,ay from its coords.
        let (wx, wy, wz) = (-0.30, 0.40, 4.0);
        let (ax, ay) = (wx / wz, wy / wz);
        let mut sk = skel(&[
            (5, [-0.15, 0.0, 2.0], 0.9),
            (6, [0.15, 0.0, 2.0], 0.9),
            (7, [-0.15, upper, 2.0], 0.9),
            (9, [wx, wy, wz], 0.9),
        ]);
        let obs = lifter.refine_bones(&mut sk, &[None; 4], &mut [0.0; 17]);
        // bone length is now ≈ the anatomical forearm
        let elbow = sk[7];
        let blen = d3(sk[9], elbow);
        assert!(
            (blen - forearm).abs() < 0.02,
            "snapped forearm length {blen} != expected {forearm}"
        );
        // and the wrist stayed on its original ray (x/z and y/z unchanged)
        assert!(
            (sk[9][0] / sk[9][2] - ax).abs() < 1e-4,
            "wrist left its ray in x"
        );
        assert!(
            (sk[9][1] / sk[9][2] - ay).abs() < 1e-4,
            "wrist left its ray in y"
        );
        // a SNAPPED bone is NOT reported as an observation (learning from the prior would be circular).
        assert!(
            obs[1].is_none(),
            "a snapped forearm must not be returned as a clean observation"
        );
    }

    /// A caller-supplied (calibrated) length overrides the anthropometric default. With the wrist on the
    /// elbow's OWN view ray at a garbage depth (so its measured bone is hugely overlong → snapped, and
    /// any target length is geometrically reachable along the ray), the snap lands exactly on whichever
    /// length the caller supplies: the anthropometric default vs a different calibrated value.
    #[test]
    fn refine_uses_caller_calibrated_length() {
        let lifter = Lifter::new(intr());
        let h = 0.30 / 0.259;
        let upper = 0.186 * h; // ≈0.215; default forearm ≈0.169
        let elbow = [-0.15, upper, 2.0];
        // Wrist on the elbow's view ray (passes through the elbow) but at depth 4 → measured forearm
        // ≈2 m (snapped), and the ray pierces any radius sphere → the snap hits the target exactly.
        let (ax, ay) = (elbow[0] / elbow[2], elbow[1] / elbow[2]);
        let wrist_in = [ax * 4.0, ay * 4.0, 4.0];
        let base = [
            (5, [-0.15, 0.0, 2.0], 0.9),
            (6, [0.15, 0.0, 2.0], 0.9),
            (7, elbow, 0.9),
            (9, wrist_in, 0.9),
        ];
        // default (anthropometric) → forearm snaps to ≈0.169
        let mut sk_def = skel(&base);
        lifter.refine_bones(&mut sk_def, &[None; 4], &mut [0.0; 17]);
        assert!(
            (d3(sk_def[9], sk_def[7]) - 0.169).abs() < 0.01,
            "default forearm ≈0.169, got {}",
            d3(sk_def[9], sk_def[7])
        );
        // calibrated 0.12 overrides → forearm snaps to ≈0.12 instead
        let mut sk = skel(&base);
        lifter.refine_bones(&mut sk, &[None, Some(0.12), None, None], &mut [0.0; 17]);
        let blen = d3(sk[9], sk[7]);
        assert!(
            (blen - 0.12).abs() < 0.01,
            "must snap to the CALIBRATED 0.12, got {blen}"
        );
    }

    /// Temporal branch hysteresis: in the ambiguous deadband around the front/back crossing the joint
    /// holds its previous side; a decisive measurement still flips it.
    #[test]
    fn depth_hysteresis_holds_side_in_deadband_but_flips_decisively() {
        // parent at (0,0,2); ray ax=0.1 → two roots ≈1.757 (front) / 2.204 (back), crossing t*≈1.980.
        let parent = [0.0, 0.0, 2.0];
        let (ax, ay, len) = (0.1, 0.0, 0.3);
        // No history + measurement just into the back side → nearest-measured picks the back root.
        let no_hist = depth_for_bone_length(parent, ax, ay, len, 1.99, 0.0);
        assert!(
            no_hist > 2.0,
            "no history → nearest measured picks back, got {no_hist}"
        );
        // Previously on the FRONT, measurement only just into the deadband → side is HELD (no flip).
        let held = depth_for_bone_length(parent, ax, ay, len, 1.99, 1.757);
        assert!(
            held < 1.85,
            "deadband must hold the previous front side, got {held}"
        );
        // A decisive back measurement flips even though prev was front.
        let flipped = depth_for_bone_length(parent, ax, ay, len, 2.30, 1.757);
        assert!(
            flipped > 2.0,
            "a decisive crossing must still flip the side, got {flipped}"
        );
    }

    /// With no torso joints visible, stature can't be estimated → the skeleton is returned untouched.
    #[test]
    fn refine_noop_without_torso_scale() {
        let lifter = Lifter::new(intr());
        // Only an arm, no shoulders/hips → no scale.
        let mut sk = skel(&[(7, [0.2, 0.3, 2.0], 0.9), (9, [0.2, 0.3, 5.0], 0.9)]);
        let before = sk.clone();
        let obs = lifter.refine_bones(&mut sk, &[None; 4], &mut [0.0; 17]);
        assert_eq!(sk, before, "without a torso scale, refine must be a no-op");
        assert_eq!(
            obs, [None; 4],
            "no scale → nothing constrained → nothing observed"
        );
        assert!(
            default_bone_lengths(&sk).is_none(),
            "no torso scale → no default lengths"
        );
    }

    /// `lift_point` back-projects through the same pinhole math: a known pixel + uniform
    /// depth lands at the expected camera-frame coords.
    #[test]
    fn lift_point_matches_unproject() {
        let (w, h) = (640u32, 360u32);
        let depth = VrtDepthMap::owned(vec![2000u16; (w * h) as usize], w, h); // 2.0 m
        let k = intr();
        let lifter = Lifter::new(k);
        // Pixel 100 px right of cx at 2 m (in-bounds for 640×360): x = (u-cx)/fx * z.
        let du = 100.0_f32;
        let expected_x = (du / k.fx) * 2.0;
        let p = lifter.lift_point(&depth, k.cx + du, k.cy).unwrap();
        assert!((p[2] - 2.0).abs() < 1e-6, "z = 2.0 m, got {}", p[2]);
        assert!(
            (p[0] - expected_x).abs() < 1e-4,
            "x = (u-cx)/fx*z = {expected_x}, got {}",
            p[0]
        );
        assert!(p[1].abs() < 1e-4, "y = 0.0 at cy, got {}", p[1]);
    }
}
