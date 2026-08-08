//! Geometry tests for the shared benchmark-harness module.
//!
//! They live here rather than inside `examples/common/mod.rs` because six examples
//! `#[path]`-include that module, so a `#[cfg(test)]` block in it compiles into six
//! binaries and runs the same assertions 26 times per `cargo test`. One integration
//! target compiles and runs them once.

#[path = "../examples/common/mod.rs"]
mod common;
use common::*;
use kornia_algebra::Vec3F64;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;

mod tests {
    use super::*;

    /// Row-major in, column-major out. A symmetric fixture cannot catch a dropped
    /// transpose, so this uses an intrinsics matrix — the shape that actually flows
    /// through these harnesses, and the one whose silent transposition would rewrite
    /// every published number without failing anything.
    #[test]
    fn mat3_from_row_major_transposes() {
        let m = mat3_from_row_major(&[799.4, 0.0, 524.0, 0.0, 799.4, 289.0, 0.0, 0.0, 1.0])
            .expect("9 floats is the valid length");
        // Column-major storage: `x_axis` is the first COLUMN of the logical matrix.
        assert_eq!(m.x_axis.x, 799.4);
        assert_eq!(m.x_axis.y, 0.0);
        assert_eq!(
            m.z_axis.x, 524.0,
            "cx must land in the top-right, not the bottom-left"
        );
        assert_eq!(m.z_axis.y, 289.0);
        // A wrong length is an error, not a panic.
        assert!(mat3_from_row_major(&[1.0, 2.0, 3.0]).is_err());
        // And it must act like K on a point.
        let p = m * Vec3F64::new(1.0, 2.0, 1.0);
        assert!((p.x - (799.4 + 524.0)).abs() < 1e-9);
        assert!((p.y - (2.0 * 799.4 + 289.0)).abs() < 1e-9);
    }

    /// The half-pixel term is exactly what a bare `diag(s, s, 1)` gets wrong.
    /// `resize_to_fit` must report the scale it *applied*, not the one it asked for.
    ///
    /// This calls `resize_to_fit`. An earlier version re-derived the formula inline and
    /// asserted on its own copy, so reverting the function to a single uniform scale left
    /// it green — it could not fail for the bug it was written to catch.
    #[test]
    fn resize_to_fit_reports_the_applied_scale_per_axis() {
        // Oxford bark: 765x512 -> 640x428, cropped to 640x416.
        let src = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 765,
                height: 512,
            },
            0,
        )
        .unwrap();
        let out = resize_to_fit(&src, 640, InterpolationMode::Bilinear).unwrap();

        assert_eq!(out.image.cols(), 640);
        assert_eq!(out.image.rows(), 416, "cropped to the 32px grid");

        let requested = 640.0 / 765.0;
        assert!((out.scale_x - requested).abs() < 1e-12, "x is exact here");
        assert!(
            (out.scale_y - requested).abs() > 1e-6,
            "y must differ from the requested scale — that is the whole bug"
        );
        assert!(
            (out.scale_y - 428.0 / 512.0).abs() < 1e-12,
            "y is the applied scale"
        );
        // ~0.28 px at the bottom of the cropped image, against a ~2.5 px threshold.
        assert!(((out.scale_y - requested) * 416.0).abs() > 0.2);
    }

    /// A square image scales exactly on both axes — the case that must NOT report a
    /// spurious difference, so the test above is measuring something real.
    #[test]
    fn resize_to_fit_is_exact_when_the_scale_divides() {
        let src = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 1280,
                height: 640,
            },
            0,
        )
        .unwrap();
        let out = resize_to_fit(&src, 640, InterpolationMode::Bilinear).unwrap();
        assert_eq!((out.scale_x, out.scale_y), (0.5, 0.5));
    }

    #[test]
    fn resize_matrix_carries_the_half_pixel_offset() {
        let m = resize_matrix(0.5, 0.5);
        let p = m * Vec3F64::new(0.0, 0.0, 1.0);
        assert!((p.x - (-0.25)).abs() < 1e-12, "got {}", p.x);
        assert!((p.y - (-0.25)).abs() < 1e-12);
        // Identity scale must be exactly the identity, offset included.
        let i = resize_matrix(1.0, 1.0);
        assert_eq!(i.z_axis.x, 0.0);
        assert_eq!(i.z_axis.y, 0.0);
    }
}
