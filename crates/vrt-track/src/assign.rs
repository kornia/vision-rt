//! Data-association primitives: IoU and a minimum-cost assignment (Hungarian).
//!
//! Association is small (N tracks × M detections, both ≲ a few hundred) and runs
//! on the CPU — the expensive parts of tracking (detection, ReID) are the GPU
//! models, not this.

/// IoU of two `[x1,y1,x2,y2]` boxes.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let xx1 = a[0].max(b[0]);
    let yy1 = a[1].max(b[1]);
    let xx2 = a[2].min(b[2]);
    let yy2 = a[3].min(b[3]);
    let inter = (xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

// ── 3D (BEV) IoU ────────────────────────────────────────────────────────────
//
// The standard 3D-MOT association metric (AB3DMOT): intersect the two boxes'
// ground-plane footprints (rotated rectangles, via Sutherland–Hodgman convex
// clipping) and multiply by their height overlap → 3D intersection volume, then
// IoU over the union of volumes. z is the vertical axis; the BEV plane is (x,y).

use crate::Box3D;

type P = [f32; 2];

/// The four ground-plane corners of a [`Box3D`], wound CCW (`l` along heading
/// `yaw`, `w` perpendicular). A proper rotation preserves the CCW winding the
/// clip below relies on.
fn corners_bev(b: &Box3D) -> [P; 4] {
    let (hl, hw) = (b.size[0] * 0.5, b.size[1] * 0.5);
    let (c, s) = (b.yaw.cos(), b.yaw.sin());
    let (cx, cy) = (b.center[0], b.center[1]);
    let local = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]; // CCW
    let mut out = [[0.0f32; 2]; 4];
    for (i, &(dx, dy)) in local.iter().enumerate() {
        out[i] = [cx + dx * c - dy * s, cy + dx * s + dy * c];
    }
    out
}

/// Absolute polygon area (shoelace).
fn poly_area(p: &[P]) -> f32 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += p[i][0] * p[j][1] - p[j][0] * p[i][1];
    }
    a.abs() * 0.5
}

/// Intersection of segment `p→q` with the infinite line `a→b`.
fn seg_isect(p: P, q: P, a: P, b: P) -> P {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let d1 = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0]);
    let d2 = ab[0] * (q[1] - a[1]) - ab[1] * (q[0] - a[0]);
    let denom = d1 - d2;
    let t = if denom.abs() < 1e-12 { 0.0 } else { d1 / denom };
    [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
}

/// Sutherland–Hodgman: clip the (convex) `subject` polygon by the convex,
/// CCW-wound `clip` polygon. Returns the intersection polygon's vertices.
fn poly_clip(subject: &[P], clip: &[P]) -> Vec<P> {
    let mut output: Vec<P> = subject.to_vec();
    let m = clip.len();
    for e in 0..m {
        if output.is_empty() {
            break;
        }
        let (a, b) = (clip[e], clip[(e + 1) % m]);
        // CCW: a point is inside the half-plane when it's left of edge a→b.
        let inside = |p: P| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0;
        let input = std::mem::take(&mut output);
        let k = input.len();
        for i in 0..k {
            let cur = input[i];
            let prev = input[(i + k - 1) % k];
            let (cur_in, prev_in) = (inside(cur), inside(prev));
            if cur_in {
                if !prev_in {
                    output.push(seg_isect(prev, cur, a, b));
                }
                output.push(cur);
            } else if prev_in {
                output.push(seg_isect(prev, cur, a, b));
            }
        }
    }
    output
}

/// Height-aware 3D IoU of two oriented boxes (BEV footprint × vertical overlap).
pub fn iou3d(a: &Box3D, b: &Box3D) -> f32 {
    let inter_bev = poly_area(&poly_clip(&corners_bev(a), &corners_bev(b)));
    if inter_bev <= 0.0 {
        return 0.0;
    }
    let (az0, az1) = (a.center[2] - a.size[2] * 0.5, a.center[2] + a.size[2] * 0.5);
    let (bz0, bz1) = (b.center[2] - b.size[2] * 0.5, b.center[2] + b.size[2] * 0.5);
    let oh = (az1.min(bz1) - az0.max(bz0)).max(0.0);
    let inter = inter_bev * oh;
    if inter <= 0.0 {
        return 0.0;
    }
    let va = a.size[0] * a.size[1] * a.size[2];
    let vb = b.size[0] * b.size[1] * b.size[2];
    let union = va + vb - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

const PAD: f32 = 1.0e6;

/// Minimum-cost assignment on a rectangular cost matrix `cost[row][col]`.
///
/// Returns the matched `(row, col)` pairs of an optimal one-to-one assignment
/// (Kuhn–Munkres, O(n³) on the padded square). Callers gate the result by
/// dropping pairs whose cost exceeds a threshold.
pub fn min_cost_assign(cost: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let r = cost.len();
    if r == 0 {
        return Vec::new();
    }
    let c = cost[0].len();
    if c == 0 {
        return Vec::new();
    }

    let n = r.max(c);
    let mut sq = vec![vec![PAD; n]; n];
    for (i, row) in cost.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            sq[i][j] = v;
        }
    }

    let assign = hungarian_square(&sq);
    let mut out = Vec::new();
    for (i, &j) in assign.iter().enumerate().take(r) {
        if j < c {
            out.push((i, j));
        }
    }
    out
}

/// Hungarian algorithm for a square matrix (minimization). `assign[i] = j`.
/// Classic O(n³) potentials method (1-indexed internally).
fn hungarian_square(a: &[Vec<f32>]) -> Vec<usize> {
    let n = a.len();
    let inf = f32::INFINITY;
    let mut u = vec![0.0f32; n + 1];
    let mut v = vec![0.0f32; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row matched to col j (1-indexed)
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assign = vec![usize::MAX; n];
    for j in 1..=n {
        if p[j] != 0 {
            assign[p[j] - 1] = j - 1;
        }
    }
    assign
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_basics() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
        assert_eq!(iou(&a, &[20.0, 20.0, 30.0, 30.0]), 0.0); // disjoint
                                                             // half overlap: [0,0,10,10] vs [5,0,15,10] → inter 50, union 150 → 1/3
        assert!((iou(&a, &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn hungarian_known_optimum() {
        // Optimal assignment of this 3×3 is (0->1, 1->0, 2->2), cost 1+2+3? No —
        // optimum is the min-sum permutation. Pick the diagonal-beating one.
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let m = min_cost_assign(&cost);
        let total: f32 = m.iter().map(|&(i, j)| cost[i][j]).sum();
        // best: 0->1(1) + 1->0(2) + 2->2(2) = 5
        assert_eq!(m.len(), 3);
        assert!((total - 5.0).abs() < 1e-5, "got {total}");
    }

    #[test]
    fn rectangular_more_cols() {
        // 2 rows, 3 cols — every row matched to its cheapest distinct col.
        let cost = vec![vec![1.0, 9.0, 9.0], vec![9.0, 1.0, 9.0]];
        let mut m = min_cost_assign(&cost);
        m.sort();
        assert_eq!(m, vec![(0, 0), (1, 1)]);
    }

    fn box3(center: [f32; 3], size: [f32; 3], yaw: f32) -> Box3D {
        Box3D { center, size, yaw }
    }

    #[test]
    fn bev_iou_identical() {
        let a = box3([1.0, 2.0, 0.0], [2.0, 1.0, 1.5], 0.7);
        assert!((iou3d(&a, &a) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn bev_iou_disjoint() {
        let a = box3([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 0.0);
        let b = box3([10.0, 0.0, 0.0], [2.0, 2.0, 2.0], 0.0);
        assert_eq!(iou3d(&a, &b), 0.0);
    }

    #[test]
    fn bev_iou_no_height_overlap() {
        let a = box3([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], 0.0);
        let b = box3([0.0, 0.0, 5.0], [2.0, 2.0, 1.0], 0.0); // same footprint, far in z
        assert_eq!(iou3d(&a, &b), 0.0);
    }

    #[test]
    fn bev_iou_axis_aligned_hand_computed() {
        // A x∈[-1,1] y∈[-1,1], B x∈[0,2] y∈[-1,1], same z & height.
        // BEV inter = 1*2 = 2, height overlap 2 → inter vol 4; vols 8+8.
        // IoU = 4 / (16-4) = 1/3.
        let a = box3([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 0.0);
        let b = box3([1.0, 0.0, 0.0], [2.0, 2.0, 2.0], 0.0);
        assert!(
            (iou3d(&a, &b) - 1.0 / 3.0).abs() < 1e-4,
            "got {}",
            iou3d(&a, &b)
        );
    }

    #[test]
    fn bev_iou_rotated_45deg() {
        // Two unit boxes, same center, one rotated 45°. Footprint intersection is
        // a regular octagon of area 2(√2−1); IoU = that/(2−that) = 1/√2 ≈ 0.7071.
        // Equal height cancels, so 3D IoU == BEV IoU here.
        let a = box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        let b = box3(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            std::f32::consts::FRAC_PI_4,
        );
        let got = iou3d(&a, &b);
        assert!(
            (got - std::f32::consts::FRAC_1_SQRT_2).abs() < 2e-3,
            "got {got}"
        );
    }
}
