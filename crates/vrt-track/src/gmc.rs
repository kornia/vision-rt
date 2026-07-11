//! Global (camera) Motion Compensation — the GMC hook of BoT-SORT.
//!
//! On a moving camera the whole background shifts between frames, so a track's
//! image-plane prediction drifts even when the *object* is static. BoT-SORT
//! corrects this by estimating a 2D affine warp of the background (from optical
//! flow / feature matching) and re-anchoring every track's predicted centre through
//! it before association.
//!
//! This crate ships the **interface** plus a no-op stub. A real estimator (ECC,
//! sparse-flow, ORB) lives outside this pure-CPU algorithm crate — plug it in by
//! implementing [`CameraMotion`] and passing it to
//! [`BotSort::update_with_motion`](crate::BotSort::update_with_motion).

/// Estimates the inter-frame background affine warp used for camera-motion
/// compensation.
///
/// Implementations return a `2×3` affine matrix (row-major) mapping a point in the
/// **previous** frame's image coordinates to the **current** frame:
///
/// ```text
/// [ x' ]   [ m00 m01 m02 ] [ x ]
/// [ y' ] = [ m10 m11 m12 ] [ y ]
///                          [ 1 ]
/// ```
pub trait CameraMotion {
    /// Affine warp to apply to track centres entering frame `frame_id`.
    fn warp(&mut self, frame_id: u64) -> [[f32; 3]; 2];
}

/// Identity stub: no camera-motion compensation (the default). Every call returns
/// the identity affine, so track predictions are left untouched.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCameraMotion;

/// The identity `2×3` affine.
pub const IDENTITY_AFFINE: [[f32; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

impl CameraMotion for NoCameraMotion {
    #[inline]
    fn warp(&mut self, _frame_id: u64) -> [[f32; 3]; 2] {
        IDENTITY_AFFINE
    }
}

/// A fixed affine applied every frame — handy for tests and for callers that
/// pre-compute a constant pan/zoom.
#[derive(Debug, Clone, Copy)]
pub struct ConstantCameraMotion(pub [[f32; 3]; 2]);

impl CameraMotion for ConstantCameraMotion {
    #[inline]
    fn warp(&mut self, _frame_id: u64) -> [[f32; 3]; 2] {
        self.0
    }
}
