//! Element data type for TensorRT I/O tensors, and the precision an engine is
//! built at.
//!
//! The pipeline's data currency is now kornia's `Tensor`/`Image`; `DType` remains
//! the small dtype tag the engine ↔ output binding needs (kornia tensors are
//! statically typed, but TRT outputs are resolved from the engine at runtime).
//!
//! [`Precision`] is the *build*-side counterpart: which kernel precision the
//! builder was asked for. It lives here rather than in `builder` because that
//! module is feature-gated, while consumers such as `vrt-hub`'s engine registry
//! need to reason about precision unconditionally.

/// Element data type of a TRT input/output tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    /// Brain-float16 — same 2-byte width as `F16` but a different bit layout, so it
    /// must never be confused with it (or with `F32`) when reading a TRT output.
    BF16,
    U8,
    I32,
}

/// Numeric precision a TensorRT engine was built at.
///
/// Part of an engine's identity, not a performance hint: precision changes an
/// engine's *outputs*, so an engine built at one precision is not a drop-in for a
/// caller that asked for another. DINOv3 is the worked example — its fp16 engine
/// emits all-NaN where bf16 is correct, so an fp16 artifact handed to a bf16
/// request would poison every descriptor with no error anywhere.
///
/// `Fp16` and `Bf16` are mutually exclusive by construction: setting both TensorRT
/// flags lets it choose per layer on speed alone, with no knowledge of dynamic
/// range, which is how that NaN comes back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
}
