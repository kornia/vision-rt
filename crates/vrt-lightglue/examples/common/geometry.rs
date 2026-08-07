//! Row-major 3x3 helpers shared by the evaluation examples.
//!
//! Small enough to inline in each example, and that is exactly why they live here
//! instead: `inv3` shipped once with the adjugate transpose missing, which is invisible
//! (no NaN, no panic, correct determinant, correct for symmetric input) and wrong for
//! every intrinsics matrix. One implementation with one test is the fix.

#![allow(dead_code)]

pub type M3 = [f64; 9];

pub fn mul3(a: &M3, b: &M3) -> M3 {
    let mut o = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            o[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum();
        }
    }
    o
}

pub fn transpose3(a: &M3) -> M3 {
    let mut o = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            o[c * 3 + r] = a[r * 3 + c];
        }
    }
    o
}

pub fn matvec3(a: &M3, v: &[f64; 3]) -> [f64; 3] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

pub fn scale3(sx: f64, sy: f64) -> M3 {
    [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0]
}

pub fn skew3(t: &[f64; 3]) -> M3 {
    [
        0.0, -t[2], t[1], //
        t[2], 0.0, -t[0], //
        -t[1], t[0], 0.0,
    ]
}

pub fn inv3(m: &M3) -> Option<M3> {
    // Cofactor matrix. The inverse is the *adjugate* over the determinant, and the
    // adjugate is this TRANSPOSED. Dropping the transpose yields the inverse-transpose,
    // which has the right determinant, never NaNs, and is correct for symmetric input --
    // so it survives any test that does not use an asymmetric matrix.
    let c = [
        m[4] * m[8] - m[5] * m[7],
        m[5] * m[6] - m[3] * m[8],
        m[3] * m[7] - m[4] * m[6],
        m[2] * m[7] - m[1] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[1] * m[5] - m[2] * m[4],
        m[2] * m[3] - m[0] * m[5],
        m[0] * m[4] - m[1] * m[3],
    ];
    let det = m[0] * c[0] + m[1] * c[1] + m[2] * c[2];
    if det.abs() < 1e-12 {
        return None;
    }
    Some(transpose3(&c).map(|v| v / det))
}

/// Apply a homography to a pixel, returning a point at infinity as a coordinate that can
/// never be counted as an inlier -- rather than a NaN, which silently compares false.
pub fn warp(h: &M3, x: f32, y: f32) -> (f32, f32) {
    let p = matvec3(h, &[x as f64, y as f64, 1.0]);
    if p[2].abs() < 1e-12 {
        return (f32::MAX, f32::MAX);
    }
    ((p[0] / p[2]) as f32, (p[1] / p[2]) as f32)
}

/// Symmetric epipolar distance in pixels: the sum of each point's distance to the
/// epipolar line induced by the other.
pub fn sym_epipolar(f: &M3, p1: (f32, f32), p2: (f32, f32)) -> f64 {
    let x1 = [p1.0 as f64, p1.1 as f64, 1.0];
    let x2 = [p2.0 as f64, p2.1 as f64, 1.0];
    let fx1 = matvec3(f, &x1);
    let ftx2 = matvec3(&transpose3(f), &x2);
    let num = (x2[0] * fx1[0] + x2[1] * fx1[1] + x2[2] * fx1[2]).abs();
    let (d1, d2) = (fx1[0].hypot(fx1[1]), ftx2[0].hypot(ftx2[1]));
    if d1 < 1e-12 || d2 < 1e-12 {
        return f64::MAX;
    }
    num * (1.0 / d1 + 1.0 / d2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An asymmetric matrix is the only kind that catches a missing adjugate transpose,
    /// so this uses a real (upper-triangular) intrinsics matrix.
    #[test]
    fn inv3_inverts_an_asymmetric_matrix() {
        let k: M3 = [799.419, 0.0, 524.0, 0.0, 799.419, 289.0, 0.0, 0.0, 1.0];
        let i = mul3(&k, &inv3(&k).expect("K is invertible"));
        for (n, (got, want)) in i
            .iter()
            .zip([1., 0., 0., 0., 1., 0., 0., 0., 1.])
            .enumerate()
        {
            assert!((got - want).abs() < 1e-9, "K·K⁻¹[{n}] = {got}, want {want}");
        }
    }

    #[test]
    fn inv3_reports_singular() {
        assert!(inv3(&[1., 2., 3., 2., 4., 6., 0., 0., 1.]).is_none());
    }

    /// A point on the epipolar line is an inlier; one far off it is not.
    #[test]
    fn sym_epipolar_is_zero_on_the_line() {
        // Pure sideways translation: F maps every point to its own horizontal line.
        let k: M3 = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0];
        let ki = inv3(&k).unwrap();
        let e = mul3(
            &skew3(&[1.0, 0.0, 0.0]),
            &[1., 0., 0., 0., 1., 0., 0., 0., 1.],
        );
        let f = mul3(&mul3(&transpose3(&ki), &e), &ki);
        assert!(sym_epipolar(&f, (100.0, 200.0), (140.0, 200.0)) < 1e-6);
        assert!(sym_epipolar(&f, (100.0, 200.0), (140.0, 230.0)) > 25.0);
    }
}
