//! Cost matrices and optimal linear assignment for track ↔ detection matching.
//!
//! The primary cost is **IoU distance** (`1 − IoU`) in the image plane. When the
//! `appearance` feature is enabled, per-track appearance embeddings are fused in
//! via BoT-SORT-style gated **cosine** distance (see [`fuse_appearance`]).
//!
//! Assignment uses a compact, dependency-free **Hungarian** (Kuhn–Munkres,
//! O(n³)) solver — optimal, and for MOT-scale problems (tens of tracks/dets) its
//! cost is negligible while giving strictly better matches than the greedy
//! alternative on crossing/overlapping targets. Rectangular problems are padded to
//! square with a large sentinel cost that the gate rejects.

/// A large finite cost used to pad rectangular assignment problems to square.
const PAD_COST: f64 = 1.0e6;

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
pub fn iou_cost_matrix(tracks: &[[f32; 4]], dets: &[[f32; 4]]) -> Vec<Vec<f64>> {
    tracks
        .iter()
        .map(|t| dets.iter().map(|d| 1.0 - iou(t, d) as f64).collect())
        .collect()
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
}
