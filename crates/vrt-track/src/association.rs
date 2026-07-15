//! Cost matrices and optimal linear assignment for track ↔ detection matching.
//!
//! The primary cost is **IoU distance** (`1 − IoU`) in the image plane, min-fused
//! with a size-normalised **centre-proximity** rescue ([`fuse_center`]) so an
//! occlusion-shrunk box still matches its coasting track. When the `appearance`
//! feature is enabled, per-track appearance embeddings are fused in via
//! BoT-SORT-style gated **cosine** distance (see [`fuse_appearance`]).
//!
//! Assignment uses a compact, dependency-free **Hungarian** (Kuhn–Munkres,
//! O(n³)) solver — optimal, and for MOT-scale problems (tens of tracks/dets) its
//! cost is negligible while giving strictly better matches than the greedy
//! alternative on crossing/overlapping targets. Rectangular problems are padded to
//! square with a large sentinel cost that the gate rejects.

use vrt_types::CameraIntrinsics;

/// A large finite cost used to pad rectangular assignment problems to square.
const PAD_COST: f64 = 1.0e6;

/// Penalize track↔detection pairs whose **metric depth** disagrees, in place.
///
/// For every pair where *both* the track and the detection carry a known metric depth,
/// if `|z_track − z_det|` exceeds a relative tolerance `max(abs_floor, rel · z_track)`
/// the pair's cost is **increased by `penalty`** (additive, not a hard veto). This kills
/// the ID swap between two objects that overlap in the image but sit at different depths
/// (the correct same-depth match keeps its low cost and wins), while a strongly-
/// overlapping self-match survives a transient monocular-depth spike instead of being
/// rejected (which made static objects churn ids). A pair with an unknown depth on
/// either side is untouched — graceful degradation to pure IoU. Apply this **after**
/// [`fuse_appearance`] so a depth-inconsistent pair is penalized even when appearance
/// rescued it. `rel`/`abs_floor`/`penalty` are
/// [`TrackerConfig::depth_gate_rel`]/[`depth_gate_abs`]/[`depth_gate_penalty`]; a very
/// large `penalty` recovers the old hard-veto behaviour.
///
/// [`TrackerConfig::depth_gate_penalty`]: crate::TrackerConfig::depth_gate_penalty
///
/// [`TrackerConfig::depth_gate_rel`]: crate::TrackerConfig::depth_gate_rel
/// [`depth_gate_abs`]: crate::TrackerConfig::depth_gate_abs
pub fn gate_depth(
    cost: &mut [Vec<f64>],
    track_depths: &[Option<f32>],
    det_depths: &[Option<f32>],
    rel: f32,
    abs_floor: f32,
    penalty: f64,
) {
    for (t, row) in cost.iter_mut().enumerate() {
        let Some(zt) = track_depths.get(t).copied().flatten() else {
            continue;
        };
        let tol = (rel * zt).max(abs_floor);
        for (d, c) in row.iter_mut().enumerate() {
            let Some(zd) = det_depths.get(d).copied().flatten() else {
                continue;
            };
            if (zt - zd).abs() > tol {
                // **Additive** penalty, not a hard veto. A strongly-overlapping (high-IoU,
                // low-cost) pair survives it and still matches — so an object's OWN
                // detection with a transient monocular-depth spike is not rejected (which
                // caused static objects to churn ids). But a genuine crossing is still
                // resolved: the correct same-depth detection has ~0 base cost and no
                // penalty, so the Hungarian prefers it over the penalized cross pair. A
                // low-overlap cross-depth pair (already near the match gate) is pushed
                // over it and dropped, preserving the swap veto where it matters.
                *c += penalty;
            }
        }
    }
}

/// Intersection-over-union of two `[x1, y1, x2, y2]` boxes.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let xx1 = a[0].max(b[0]);
    let yy1 = a[1].max(b[1]);
    let xx2 = a[2].min(b[2]);
    let yy2 = a[3].min(b[3]);
    let iw = (xx2 - xx1).max(0.0);
    let ih = (yy2 - yy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// IoU-distance cost matrix `cost[t][d] = 1 − IoU(track_t, det_d)`, in `[0, 1]`.
/// **Distance-IoU** (Zheng et al., AAAI 2020): `IoU − ρ²(centres)/c²`, where `ρ` is the
/// distance between box centres and `c` the diagonal of the smallest box enclosing both.
/// Range `[-1, 1]`. Unlike plain IoU it discriminates among **equal-IoU** candidates by
/// centre proximity — which only differs from IoU when the boxes have **different sizes**
/// (for equal-size axis-aligned boxes IoU is already monotonic in centre distance). The
/// case it helps: an occlusion-shrunk detection overlaps two tracks equally in IoU but
/// sits on one's centre — DIoU picks that one, stabilizing the association.
pub fn diou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let i = iou(a, b);
    let (acx, acy) = ((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5);
    let (bcx, bcy) = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5);
    let d2 = (acx - bcx).powi(2) + (acy - bcy).powi(2);
    let (ex1, ey1) = (a[0].min(b[0]), a[1].min(b[1]));
    let (ex2, ey2) = (a[2].max(b[2]), a[3].max(b[3]));
    let c2 = (ex2 - ex1).powi(2) + (ey2 - ey1).powi(2);
    if c2 <= 0.0 {
        i
    } else {
        i - d2 / c2
    }
}

/// **DIoU with a metric-3D centre penalty.** Keeps the crisp 2D pixel IoU overlap but
/// makes the centre-distance penalty `ρ²/c²` metric: unproject each box's centre and
/// corners to camera-frame metres at its own depth, so `ρ` is the real 3D centre
/// distance (including depth) and `c` the diagonal of the enclosing metric box (its
/// Z-extent is the depth gap between the two flat box-planes). Two boxes overlapping in
/// the image but at **different depths** get a large `ρ` → DIoU-3D collapses → no match:
/// depth separation folded smoothly into the cost. For equal depth it reduces to 2D
/// [`diou`]. Falls back to 2D `diou` when either depth is absent (can't lift to metres).
pub fn diou3d(
    a: &[f32; 4],
    b: &[f32; 4],
    az: Option<f32>,
    bz: Option<f32>,
    intr: &CameraIntrinsics,
) -> f32 {
    let (Some(za), Some(zb)) = (az, bz) else {
        return diou(a, b);
    };
    let i = iou(a, b);
    let ca = intr.unproject((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5, za);
    let cb = intr.unproject((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5, zb);
    let rho2 = (ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2) + (ca[2] - cb[2]).powi(2);
    // Metric corners at each box's depth → X/Y extent of the enclosing box; the boxes
    // are flat planes so the enclosing Z-extent is just the depth gap |za − zb|.
    let (atl, abr) = (
        intr.unproject(a[0], a[1], za),
        intr.unproject(a[2], a[3], za),
    );
    let (btl, bbr) = (
        intr.unproject(b[0], b[1], zb),
        intr.unproject(b[2], b[3], zb),
    );
    let xs = [atl[0], abr[0], btl[0], bbr[0]];
    let ys = [atl[1], abr[1], btl[1], bbr[1]];
    let ex =
        xs.iter().copied().fold(f32::MIN, f32::max) - xs.iter().copied().fold(f32::MAX, f32::min);
    let ey =
        ys.iter().copied().fold(f32::MIN, f32::max) - ys.iter().copied().fold(f32::MAX, f32::min);
    let c2 = ex * ex + ey * ey + (za - zb).powi(2);
    if c2 <= 0.0 {
        i
    } else {
        i - rho2 / c2
    }
}

/// **Buffered IoU** (C-BIoU, Yang et al. WACV 2023): IoU after expanding both boxes by
/// `buffer × (w, h)` on each side. Tolerates a shifted/jittered box — a detection whose
/// seg-mask box moved (a common instability) still overlaps its track's buffered box, so
/// the match survives instead of the track dying and re-birthing. `buffer = 0` is plain
/// IoU. Note: buffering scales with box size, so it rescues *positional* shift, not a
/// wild size mismatch (a tiny box vs a huge one stays non-overlapping).
pub fn biou(a: &[f32; 4], b: &[f32; 4], buffer: f32) -> f32 {
    if buffer <= 0.0 {
        return iou(a, b);
    }
    let ex = |x: &[f32; 4]| {
        let (w, h) = (x[2] - x[0], x[3] - x[1]);
        [
            x[0] - buffer * w,
            x[1] - buffer * h,
            x[2] + buffer * w,
            x[3] + buffer * h,
        ]
    };
    iou(&ex(a), &ex(b))
}

/// Association cost matrix `1 − similarity`. `use_diou` swaps plain IoU for [`diou`]
/// (distance-aware); otherwise plain IoU with an optional `iou_buffer` (Buffered IoU,
/// [`biou`]) for seg-box-shift tolerance. Both off by default.
pub fn iou_cost_matrix(
    tracks: &[[f32; 4]],
    dets: &[[f32; 4]],
    use_diou: bool,
    iou_buffer: f32,
) -> Vec<Vec<f64>> {
    tracks
        .iter()
        .map(|t| {
            dets.iter()
                .map(|d| {
                    let s = if use_diou {
                        diou(t, d)
                    } else {
                        biou(t, d, iou_buffer)
                    };
                    1.0 - s as f64
                })
                .collect()
        })
        .collect()
}

/// Cost matrix using [`diou3d`] — per-object depths + intrinsics lift the centre
/// penalty to metres. `track_z[i]` / `det_z[j]` are the boxes' metric depths (`None` →
/// that pair falls back to 2D `diou`). Enable via [`TrackerConfig::use_diou3d`].
pub fn diou3d_cost_matrix(
    tracks: &[[f32; 4]],
    dets: &[[f32; 4]],
    track_z: &[Option<f32>],
    det_z: &[Option<f32>],
    intr: &CameraIntrinsics,
) -> Vec<Vec<f64>> {
    tracks
        .iter()
        .enumerate()
        .map(|(ti, t)| {
            dets.iter()
                .enumerate()
                .map(|(di, d)| 1.0 - diou3d(t, d, track_z[ti], det_z[di], intr) as f64)
                .collect()
        })
        .collect()
}

/// Track half-extents at which the centre-proximity cost saturates to 1 (no
/// rescue). At 1.5 a detection whose centre sits within ~0.75 box half-extents of a
/// coasting track yields a cost ≤ 0.5 (the second-stage gate), so an occlusion-shrunk
/// box is re-acquired; a detection a full box-width away contributes nothing.
const CENTER_SPAN: f64 = 1.5;

/// Size-normalised centre-distance cost in `[0, 1]` between a track box and a
/// detection box. Normalising the offset by the **track** half-extents makes it
/// scale-invariant: a partially-occluded detection (same object centre, shrunken
/// area) scores ~0 where IoU — which counts the missing area against the pair —
/// would wrongly reject it. Saturates to 1 beyond [`CENTER_SPAN`] half-extents.
fn center_cost(track: &[f32; 4], det: &[f32; 4]) -> f64 {
    let hw = ((track[2] - track[0]) * 0.5).max(1.0) as f64;
    let hh = ((track[3] - track[1]) * 0.5).max(1.0) as f64;
    let nx = ((track[0] + track[2] - det[0] - det[2]) as f64 * 0.5) / hw;
    let ny = ((track[1] + track[3] - det[1] - det[3]) as f64 * 0.5) / hh;
    ((nx * nx + ny * ny).sqrt() / CENTER_SPAN).min(1.0)
}

/// Min-fuse the size-normalised centre-proximity cost into an IoU cost matrix:
/// `cost ← min(cost, center_cost)`, but only for track rows flagged in `rescue`.
/// This **rescues** coasting/occluded tracks whose box shrank under occlusion (so IoU
/// is poor) but whose centre the constant-velocity motion model still tracks — the
/// dominant ID-churn cause on partially-visible objects. Like [`fuse_appearance`] it
/// only ever *lowers* a cost, so it can add a match but never block one; apply
/// [`gate_depth`] afterwards so a cross-depth pair is still hard-rejected.
///
/// `rescue[t]` gates the fuse to **near-static** tracks: a fast-coasting track's
/// predicted centre drifts far from where it was last seen, so centre-only matching
/// (which discards box shape) would let it capture a *different* object that drifted
/// into range. Restricting the rescue to slow/stationary tracks keeps the occluded
/// re-acquisition while removing that steal. A row with `rescue[t] == false` (or beyond
/// the slice) keeps its plain IoU cost.
pub fn fuse_center(cost: &mut [Vec<f64>], tracks: &[[f32; 4]], dets: &[[f32; 4]], rescue: &[bool]) {
    for (t, row) in cost.iter_mut().enumerate() {
        if !rescue.get(t).copied().unwrap_or(false) {
            continue;
        }
        for (d, c) in row.iter_mut().enumerate() {
            *c = c.min(center_cost(&tracks[t], &dets[d]));
        }
    }
}

/// Cosine distance `1 − cos(a, b)` in `[0, 2]` (0 = identical direction).
#[cfg(feature = "appearance")]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 1.0;
    }
    (1.0 - dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 2.0)
}

/// Fuse appearance cosine distance into an IoU cost matrix, BoT-SORT style.
///
/// For each pair the appearance distance is gated: it is ignored (set to 1) when it
/// exceeds `appearance_thresh` or when the boxes are too far apart in IoU
/// (`iou_cost > proximity_thresh`). The fused cost is the **minimum** of the IoU
/// cost and the gated appearance cost — so a strong appearance match can rescue a
/// pair with weak IoU (e.g. through a brief occlusion), but a bad embedding never
/// blocks a clean IoU match. `track_feats[t]` / `det_feats[d]` are `None` when an
/// embedding is unavailable, in which case only IoU is used for that pair.
#[cfg(feature = "appearance")]
#[allow(clippy::too_many_arguments)]
pub fn fuse_appearance(
    iou_cost: &mut [Vec<f64>],
    track_feats: &[Option<Vec<f32>>],
    det_feats: &[Option<&[f32]>],
    appearance_thresh: f32,
    proximity_thresh: f32,
) {
    for (t, row) in iou_cost.iter_mut().enumerate() {
        let Some(tf) = track_feats.get(t).and_then(|f| f.as_deref()) else {
            continue;
        };
        for (d, cost) in row.iter_mut().enumerate() {
            let Some(df) = det_feats.get(d).and_then(|f| *f) else {
                continue;
            };
            let mut emb = cosine_distance(tf, df) as f64 / 2.0; // -> [0,1]
            if emb > appearance_thresh as f64 || *cost > proximity_thresh as f64 {
                emb = 1.0;
            }
            *cost = cost.min(emb);
        }
    }
}

/// Result of [`linear_assignment`]: `(matches, unmatched_rows, unmatched_cols)`.
pub type Assignment = (Vec<(usize, usize)>, Vec<usize>, Vec<usize>);

/// Optimal minimum-cost assignment, keeping only matches with `cost ≤ thresh`.
///
/// Rows are tracks, columns are detections. Pairs whose optimal cost exceeds
/// `thresh` (or that fall on padding) are returned as unmatched instead.
pub fn linear_assignment(
    cost: &[Vec<f64>],
    n_rows: usize,
    n_cols: usize,
    thresh: f64,
) -> Assignment {
    if n_rows == 0 || n_cols == 0 {
        return (Vec::new(), (0..n_rows).collect(), (0..n_cols).collect());
    }

    let n = n_rows.max(n_cols);
    let mut square = vec![vec![PAD_COST; n]; n];
    for (r, row) in square.iter_mut().enumerate().take(n_rows) {
        for (c, cell) in row.iter_mut().enumerate().take(n_cols) {
            *cell = cost[r][c];
        }
    }

    let assign = hungarian(&square);

    let mut matches = Vec::new();
    let mut unmatched_rows = Vec::new();
    let mut matched_cols = vec![false; n_cols];
    for (r, &c) in assign.iter().enumerate().take(n_rows) {
        if c < n_cols && cost[r][c] <= thresh {
            matches.push((r, c));
            matched_cols[c] = true;
        } else {
            unmatched_rows.push(r);
        }
    }
    let unmatched_cols = (0..n_cols).filter(|&c| !matched_cols[c]).collect();
    (matches, unmatched_rows, unmatched_cols)
}

/// Kuhn–Munkres on a **square** cost matrix (minimisation). Returns `assign` where
/// `assign[row] = col`. O(n³), potentials-based (Jonker–Volgenant augmentation).
fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    let inf = f64::INFINITY;
    // 1-indexed potentials/matching, index 0 is the augmentation sentinel.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[col] = row matched to col
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
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
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
        // Augment along the found path.
        while j0 != 0 {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
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

/// Observation-Centric Momentum (OC-SORT), in **metric world space**: penalize a match
/// whose direction — from the track's last world observation to the candidate detection's
/// world position — disagrees with the track's **observed** world-velocity direction.
/// Adds `lambda · (Δθ / π)` to the IoU cost (Δθ ∈ `[0, π]`), so a detection moving
/// *against* the track's established 3D motion is pushed away even when IoU (or an
/// appearance rescue) would otherwise tie — the crossing-swap case. Working in metres
/// (not pixels) makes the cue perspective-correct and sensitive to motion in **depth**,
/// which a pixel-plane direction can't see.
///
/// `track_dir[t]` is the unit observed world-velocity direction (`None` = skip the
/// track), `track_center[t]` the track's last world position, `det_center[d]` each
/// detection's world position (`None` = no depth → skip). Applied after the IoU/appearance
/// fusion and **before** the depth gate, so the gate stays the hard veto.
pub fn fuse_momentum(
    cost: &mut [Vec<f64>],
    track_dir: &[Option<[f64; 3]>],
    track_center: &[[f64; 3]],
    det_center: &[Option<[f64; 3]>],
    lambda: f32,
) {
    if lambda <= 0.0 {
        return;
    }
    let lambda = lambda as f64;
    for (t, row) in cost.iter_mut().enumerate() {
        let Some(dir) = track_dir.get(t).and_then(|d| *d) else {
            continue;
        };
        let tc = track_center[t];
        for (d, c) in row.iter_mut().enumerate() {
            let Some(dc) = det_center.get(d).and_then(|p| *p) else {
                continue; // detection has no world position (no depth) → no cue
            };
            let delta = [dc[0] - tc[0], dc[1] - tc[1], dc[2] - tc[2]];
            let n = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            if n < 1e-3 {
                continue; // detection ≈ track centre → no direction to compare
            }
            let cos = (dir[0] * delta[0] + dir[1] * delta[1] + dir[2] * delta[2]) / n;
            let dtheta = cos.clamp(-1.0, 1.0).acos(); // [0, π]
            *c += lambda * (dtheta / std::f64::consts::PI);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_basic() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
        let b = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(iou(&a, &b), 0.0);
        let c = [5.0, 0.0, 15.0, 10.0]; // half overlap => inter 50, union 150
        assert!((iou(&a, &c) - (50.0 / 150.0)).abs() < 1e-6);
    }

    #[test]
    fn assignment_prefers_low_cost_diagonal() {
        // Identity-ish cost => each row matches its own column.
        let cost = vec![
            vec![0.0, 0.9, 0.9],
            vec![0.9, 0.0, 0.9],
            vec![0.9, 0.9, 0.0],
        ];
        let (m, ur, uc) = linear_assignment(&cost, 3, 3, 0.5);
        assert_eq!(m.len(), 3);
        assert!(ur.is_empty() && uc.is_empty());
        for (r, c) in m {
            assert_eq!(r, c);
        }
    }

    #[test]
    fn assignment_crossing_is_optimal_not_greedy() {
        // Greedy on the (0,0) cell would take 0.10 then be forced into 0.90,
        // total 1.00. The optimal assignment is the anti-diagonal, total 0.30.
        let cost = vec![vec![0.10, 0.15], vec![0.15, 0.90]];
        let (m, _, _) = linear_assignment(&cost, 2, 2, 1.0);
        let total: f64 = m.iter().map(|&(r, c)| cost[r][c]).sum();
        assert!((total - 0.30).abs() < 1e-9, "not optimal: {total}");
    }

    #[test]
    fn assignment_rectangular_and_threshold() {
        // 3 tracks, 2 dets; track 2 has no good match.
        let cost = vec![vec![0.1, 0.8], vec![0.8, 0.1], vec![0.7, 0.7]];
        let (m, ur, uc) = linear_assignment(&cost, 3, 2, 0.5);
        assert_eq!(m.len(), 2);
        assert_eq!(ur, vec![2]);
        assert!(uc.is_empty());
    }

    #[test]
    fn depth_gate_rejects_mismatch_only() {
        // track0 @ 2 m, track1 depth unknown; det0 ~2 m (consistent), det1 @ 5 m.
        let mut cost = vec![vec![0.1, 0.1], vec![0.1, 0.1]];
        let track_d = [Some(2.0f32), None];
        let det_d = [Some(2.1f32), Some(5.0f32)];
        gate_depth(&mut cost, &track_d, &det_d, 0.25, 0.5, 0.2);
        assert!(
            (cost[0][0] - 0.1).abs() < 1e-6,
            "consistent depth pair unpenalized"
        );
        assert!(
            (cost[0][1] - 0.3).abs() < 1e-6,
            "3 m mismatch penalized (+0.2)"
        );
        // Unknown track depth → whole row untouched (graceful fallback to IoU).
        assert!(cost[1][0] < 1.0 && cost[1][1] < 1.0);
        // A large penalty recovers the old hard-veto (mismatch pushed past the gate).
        let mut hard = vec![vec![0.1, 0.1]];
        gate_depth(
            &mut hard,
            &[Some(2.0)],
            &[Some(2.1), Some(5.0)],
            0.25,
            0.5,
            1.0e6,
        );
        assert!(hard[0][1] > 1.0, "large penalty = hard veto");
    }

    #[test]
    fn depth_gate_tolerance_scales_with_distance() {
        // At 10 m the 25% relative tol is 2.5 m, so a 2 m gap is allowed; the same
        // 2 m gap at 2 m distance (tol = max(0.5, 0.5) = 0.5 m) is rejected.
        let mut far = vec![vec![0.2]];
        gate_depth(&mut far, &[Some(10.0)], &[Some(12.0)], 0.25, 0.5, 1.0e6);
        assert!(
            far[0][0] < 1.0,
            "2 m gap within 2.5 m tol at 10 m → unpenalized"
        );
        let mut near = vec![vec![0.2]];
        gate_depth(&mut near, &[Some(2.0)], &[Some(4.0)], 0.25, 0.5, 1.0e6);
        assert!(
            near[0][0] > 1.0,
            "2 m gap exceeds 0.5 m tol at 2 m → penalized"
        );
    }

    #[test]
    fn center_fuse_rescues_occluded_box() {
        // 100×100 track box; the detection is the object half-occluded — a shrunken
        // box at (almost) the same centre. IoU alone rejects it; the centre fuse,
        // normalised by the track size, brings the cost below the second-stage gate.
        let track = [50.0, 50.0, 150.0, 150.0];
        let frag = [60.0, 100.0, 140.0, 150.0]; // bottom slice, centre offset 0.5·hh
        let base = 1.0 - iou(&track, &frag) as f64;
        assert!(
            base > 0.5,
            "IoU alone would reject the fragment (cost {base:.2})"
        );
        let mut cost = vec![vec![base]];
        fuse_center(&mut cost, &[track], &[frag], &[true]);
        assert!(
            cost[0][0] < 0.5,
            "centre fuse should rescue it (cost {:.2})",
            cost[0][0]
        );
        // A detection a full box-width away is not rescued.
        let far = [400.0, 400.0, 480.0, 450.0];
        let mut cost_far = vec![vec![1.0]];
        fuse_center(&mut cost_far, &[track], &[far], &[true]);
        assert!(cost_far[0][0] >= 1.0, "distant box must not be rescued");
        // A non-static track (rescue = false) keeps its plain IoU cost — no rescue.
        let mut cost_moving = vec![vec![base]];
        fuse_center(&mut cost_moving, &[track], &[frag], &[false]);
        assert!(
            (cost_moving[0][0] - base).abs() < 1e-9,
            "rescue=false must leave the IoU cost untouched"
        );
    }

    #[test]
    fn assignment_empty_inputs() {
        let (m, ur, uc) = linear_assignment(&[], 0, 3, 0.5);
        assert!(m.is_empty() && ur.is_empty());
        assert_eq!(uc, vec![0, 1, 2]);
    }

    #[cfg(feature = "appearance")]
    #[test]
    fn cosine_distance_bounds() {
        let a = [1.0, 0.0];
        assert!(cosine_distance(&a, &a) < 1e-6);
        let b = [0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
        let c = [-1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn diou3d_penalizes_depth_separated_overlap() {
        let intr = vrt_types::CameraIntrinsics::from_hfov(1280.0, 720.0, 70.0);
        // Two boxes overlapping heavily in the image.
        let a = [100.0, 100.0, 200.0, 300.0];
        let b = [110.0, 100.0, 210.0, 300.0];
        // Same depth → behaves like 2D DIoU (high similarity, boxes overlap).
        let same = diou3d(&a, &b, Some(3.0), Some(3.0), &intr);
        // Different depth (2 m vs 6 m) → large metric ρ → similarity collapses.
        let diff = diou3d(&a, &b, Some(2.0), Some(6.0), &intr);
        assert!(
            same > diff + 0.2,
            "depth separation must lower DIoU-3D: same={same} diff={diff}"
        );
        // No depth on one side → 2D fallback, exactly plain diou.
        let fb = diou3d(&a, &b, None, Some(3.0), &intr);
        assert!(
            (fb - diou(&a, &b)).abs() < 1e-6,
            "no depth → 2D DIoU fallback"
        );
        // Same-depth DIoU-3D ≈ 2D DIoU (centre penalty ~scale-consistent).
        assert!(
            same > 0.5,
            "same-depth overlap should stay a strong match: {same}"
        );
    }

    #[test]
    fn diou_breaks_equal_iou_tie_by_centre() {
        // A big track box; two SMALLER detections both fully inside it → identical IoU,
        // but D0 sits on the track's centre and D1 is offset (the occlusion-shrunk
        // crossing case). Plain IoU can't choose; DIoU prefers the centre-aligned D0.
        let t = [0.0, 0.0, 20.0, 20.0]; // centre (10,10)
        let d0 = [5.0, 5.0, 15.0, 15.0]; // centre (10,10) — on the track centre
        let d1 = [10.0, 5.0, 20.0, 15.0]; // centre (15,10) — offset, SAME iou
        assert!(
            (iou(&t, &d0) - iou(&t, &d1)).abs() < 1e-6,
            "IoU is a genuine tie"
        );
        assert!(
            diou(&t, &d0) > diou(&t, &d1),
            "DIoU ranks the centre-aligned box higher"
        );
        // Plain-IoU cost matrix ties; DIoU cost breaks it → assignment picks D0.
        let plain = iou_cost_matrix(&[t], &[d0, d1], false, 0.0);
        assert!(
            (plain[0][0] - plain[0][1]).abs() < 1e-6,
            "plain IoU cost is a tie"
        );
        let dcost = iou_cost_matrix(&[t], &[d0, d1], true, 0.0);
        assert!(
            dcost[0][0] < dcost[0][1],
            "DIoU cost prefers the centre-aligned det"
        );
        let (m, _, _) = linear_assignment(&dcost, 1, 2, 0.9);
        assert!(
            m.contains(&(0, 0)),
            "DIoU assigns the centre-aligned detection: {m:?}"
        );
    }

    #[test]
    fn biou_rescues_shifted_box_below_iou_gate() {
        // A seg-mask box that SHIFTED between frames: the track sits at its smoothed
        // position; the new detection is the same-size box translated far enough that plain
        // IoU falls below the match gate (real live case: id11 chair, IoU 0.17). Buffering
        // both boxes restores enough overlap to keep the match — without letting a genuinely
        // distant box match.
        let track = [700.0, 380.0, 760.0, 440.0]; // 60×60
        let shifted = [745.0, 380.0, 805.0, 440.0]; // same size, +45px in x
        let plain = iou(&track, &shifted);
        assert!(
            plain < 0.2,
            "setup: shifted box must be below the ~0.2 gate, got {plain}"
        );
        let buffered = biou(&track, &shifted, 0.3);
        assert!(
            buffered > plain,
            "buffering must raise overlap: {buffered} vs {plain}"
        );
        assert!(
            buffered > 0.2,
            "buffered IoU must clear the gate: {buffered}"
        );
        // buffer=0 is exactly plain IoU (no behavior change when disabled).
        assert!(
            (biou(&track, &shifted, 0.0) - plain).abs() < 1e-6,
            "buffer=0 must equal plain IoU"
        );
        // A truly far box (2× a box-width away) must NOT be rescued — buffering tolerates
        // jitter, not teleport.
        let far = [900.0, 380.0, 960.0, 440.0]; // +200px
        assert_eq!(
            biou(&track, &far, 0.3),
            0.0,
            "buffering must not match a distant box"
        );
    }

    #[test]
    fn momentum_breaks_iou_tie_on_crossing() {
        // Two objects that just crossed: IoU slightly favors the swapped assignment.
        // But each detection only continues its track's OBSERVED direction on the
        // diagonal — the swap reverses direction. OC-SORT momentum penalizes the
        // reversal and restores the correct pairing. (No depth/appearance needed.)
        let mut cost = vec![vec![0.30, 0.28], vec![0.28, 0.30]];
        let (m, _, _) = linear_assignment(&cost, 2, 2, 0.9);
        assert!(
            m.contains(&(0, 1)) && m.contains(&(1, 0)),
            "geometry-only should swap: {m:?}"
        );

        // World space (metres), constant depth Z=5. T0 moving +X from (4,0,5); T1 moving
        // −X from (6,0,5). After the crossing D0 sits at (7,0,5) (continues +X), D1 at
        // (3,0,5) (continues −X).
        let dir = [Some([1.0, 0.0, 0.0]), Some([-1.0, 0.0, 0.0])];
        let tc = [[4.0, 0.0, 5.0], [6.0, 0.0, 5.0]];
        let dc = [Some([7.0, 0.0, 5.0]), Some([3.0, 0.0, 5.0])];
        fuse_momentum(&mut cost, &dir, &tc, &dc, 0.5);
        let (m2, _, _) = linear_assignment(&cost, 2, 2, 0.9);
        assert!(
            m2.contains(&(0, 0)) && m2.contains(&(1, 1)),
            "momentum should keep correct ids through the crossing: {m2:?}"
        );
    }

    #[test]
    fn momentum_sees_depth_motion() {
        // Pure motion in DEPTH (Z), zero lateral — invisible to a pixel-plane direction,
        // caught by the metric version. T0 receding (+Z), T1 approaching (−Z). D0 is
        // further (Z=7), D1 nearer (Z=3); the swap reverses depth direction.
        let mut cost = vec![vec![0.30, 0.28], vec![0.28, 0.30]];
        let dir = [Some([0.0, 0.0, 1.0]), Some([0.0, 0.0, -1.0])];
        let tc = [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]];
        let dc = [Some([0.0, 0.0, 7.0]), Some([0.0, 0.0, 3.0])];
        fuse_momentum(&mut cost, &dir, &tc, &dc, 0.5);
        let (m, _, _) = linear_assignment(&cost, 2, 2, 0.9);
        assert!(
            m.contains(&(0, 0)) && m.contains(&(1, 1)),
            "depth-motion OCM: {m:?}"
        );
    }

    #[test]
    fn momentum_ignores_stationary_and_depthless() {
        // No direction (stationary track) or a detection with no world position (no
        // depth) → no penalty added (OCM is a no-op there).
        let mut cost = vec![vec![0.2, 0.2]];
        let before = cost.clone();
        fuse_momentum(
            &mut cost,
            &[None],
            &[[5.0, 5.0, 5.0]],
            &[Some([9.0, 9.0, 5.0]), None],
            0.5,
        );
        assert_eq!(cost, before, "no track direction → untouched");
        let mut cost2 = vec![vec![0.2]];
        fuse_momentum(
            &mut cost2,
            &[Some([1.0, 0.0, 0.0])],
            &[[5.0, 5.0, 5.0]],
            &[None],
            0.5,
        );
        assert_eq!(
            cost2,
            vec![vec![0.2]],
            "depthless detection → no world pos, untouched"
        );
    }

    #[cfg(feature = "appearance")]
    #[test]
    fn appearance_breaks_iou_tie_on_same_class_crossing() {
        // Two same-class objects crossing so their boxes overlap: the IoU cost slightly
        // FAVORS the swapped (anti-diagonal) assignment, so geometry alone flips their
        // ids — the exact ID-switch failure the ReID tie-breaker exists to prevent. The
        // depth gate can't help here (same class, assume same range). Distinct appearance
        // embeddings (the RF-DETR blk-11 tokens this PR pools) break the tie correctly.
        let swap_cost = vec![vec![0.30, 0.28], vec![0.28, 0.30]];

        // Geometry-only: anti-diagonal (0.28 + 0.28) beats the diagonal (0.30 + 0.30),
        // so T0↔D1, T1↔D0 — the ids SWAP.
        let (m, _, _) = linear_assignment(&swap_cost, 2, 2, 0.9);
        assert!(
            m.contains(&(0, 1)) && m.contains(&(1, 0)),
            "geometry-only should swap on this crossing: {m:?}"
        );

        // With appearance: D0 matches T0, D1 matches T1 (orthogonal embeddings). Matching
        // pairs fuse the cost to ~0; the mismatched (swapped) pairs are appearance-gated
        // and stay at their IoU cost → the correct diagonal is now cheapest.
        let mut cost = swap_cost.clone();
        let tf = [Some(vec![1.0f32, 0.0]), Some(vec![0.0f32, 1.0])];
        let (d0, d1) = ([1.0f32, 0.0], [0.0f32, 1.0]);
        let df = [Some(&d0[..]), Some(&d1[..])];
        fuse_appearance(&mut cost, &tf, &df, 0.25, 0.5);
        let (m2, _, _) = linear_assignment(&cost, 2, 2, 0.9);
        assert!(
            m2.contains(&(0, 0)) && m2.contains(&(1, 1)),
            "appearance should keep correct ids through the crossing: {m2:?}"
        );
    }
}
