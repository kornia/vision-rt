//! Element data type for TensorRT I/O tensors.
//!
//! The pipeline's data currency is now kornia's `Tensor`/`Image`; `DType` remains
//! the small dtype tag the engine ↔ output binding needs (kornia tensors are
//! statically typed, but TRT outputs are resolved from the engine at runtime).

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
