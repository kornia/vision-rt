use std::sync::Arc;
use cudarc::driver::CudaStream;
use crate::buffer::{DeviceBuffer, Stream};
use crate::error::Result;

/// Element data type of a [`TRTensor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType { F32, F16, U8, I32 }

impl DType {
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16             => 2,
            DType::U8              => 1,
        }
    }
}

/// GPU-resident typed tensor — the common data currency between pipeline stages.
///
/// All stages in a [`Pipeline`](crate::Pipeline) share the same CUDA stream
/// (held inside the tensor for self-contained pointer access).
pub struct TRTensor {
    buf:    DeviceBuffer,
    stream: Arc<CudaStream>,
    /// Logical shape (e.g. `[1, 3, H, W]` for a CHW FP32 frame).
    pub shape: Vec<usize>,
    pub dtype: DType,
}

// SAFETY: device pointer is stable; callers must ensure CUDA ordering via the shared stream.
unsafe impl Send for TRTensor {}

impl TRTensor {
    /// Allocate a zero-filled tensor on `cuda_stream`.
    pub fn alloc(
        cuda_stream: &Arc<CudaStream>,
        shape: impl Into<Vec<usize>>,
        dtype: DType,
    ) -> Result<Self> {
        let shape = shape.into();
        let bytes = shape.iter().product::<usize>() * dtype.byte_size();
        let buf = DeviceBuffer::alloc_with_stream(cuda_stream, bytes)?;
        Ok(Self { buf, stream: cuda_stream.clone(), shape, dtype })
    }

    pub fn numel(&self) -> usize { self.shape.iter().product() }

    pub fn cuda_stream(&self) -> &Arc<CudaStream> { &self.stream }

    /// Shape as `i64` slice for TRT `setInputShape`.
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&d| d as i64).collect()
    }

    /// Raw mutable device pointer (write path: preprocessor → input buffer).
    pub fn as_mut_ptr(&self) -> *mut std::ffi::c_void {
        let s = Stream::from_cuda_stream(self.stream.clone());
        self.buf.as_device_ptr(&s)
    }

    /// Raw const device pointer (read path: input buffer → TRT).
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.as_mut_ptr() as *const _
    }
}
