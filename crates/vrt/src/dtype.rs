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
    U8,
    I32,
}

impl DType {
    /// Width of one element in bytes.
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
        }
    }
}
