use std::sync::Arc;
use cudarc::driver::CudaStream;
use crate::buffer::{DeviceBuffer, Stream};
use crate::error::{Result, TrtError};

/// Element data type of a [`VrtTensor`].
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

/// Where a tensor's device memory lives.
///
/// Independent of *ownership*: a [`VrtTensor`] borrowing a TensorRT output is
/// [`MemKind::Device`] but does not free it on drop (its `owner` is `None`),
/// while an NVMM camera import is [`MemKind::Imported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemKind {
    /// `cudaMalloc` device memory.
    Device,
    /// CUDA unified/managed memory (CPU+GPU addressable).
    Unified,
    /// Memory owned externally — TRT session output buffers, NVMM DMA-BUF
    /// imports, or sub-views.  Validity is the producer's documented contract.
    Imported,
}

/// GPU-resident typed tensor — the single data currency between pipeline stages.
///
/// Replaces the former `TRTensor` (owned) and `TensorView` (borrowed): one type
/// carries `shape`, element `strides`, `dtype`, and [`MemKind`], and either owns
/// its [`DeviceBuffer`] (freed on drop) or borrows a pointer it does not free.
///
/// All stages in a [`Pipeline`](crate::Pipeline) share one CUDA stream, held
/// inside the tensor for self-contained pointer access.
pub struct VrtTensor {
    ptr:      *mut std::ffi::c_void,
    shape:    Vec<usize>,
    strides:  Vec<usize>,   // element strides (contiguous row-major by default)
    dtype:    DType,
    kind:     MemKind,
    byte_len: usize,
    stream:   Arc<CudaStream>,
    /// `Some` => this tensor owns the allocation and frees it on drop.
    /// `None` => borrowed; the producer guarantees validity.
    /// RAII keep-alive only — never read directly (its `Drop` frees the memory).
    #[allow(dead_code)]
    owner:    Option<DeviceBuffer>,
}

// SAFETY: device pointer is stable; callers enforce CUDA ordering via the shared stream.
unsafe impl Send for VrtTensor {}

impl VrtTensor {
    /// Allocate a zero-filled contiguous device tensor on `cuda_stream`.
    pub fn alloc(
        cuda_stream: &Arc<CudaStream>,
        shape: impl Into<Vec<usize>>,
        dtype: DType,
    ) -> Result<Self> {
        let shape = shape.into();
        let byte_len = shape.iter().product::<usize>() * dtype.byte_size();
        let buf = DeviceBuffer::alloc_with_stream(cuda_stream, byte_len)?;
        let ptr = buf.as_device_ptr(&Stream::from_cuda_stream(cuda_stream.clone()));
        let strides = contiguous_strides(&shape);
        Ok(Self {
            ptr, shape, strides, dtype,
            kind: MemKind::Device, byte_len,
            stream: cuda_stream.clone(), owner: Some(buf),
        })
    }

    /// Wrap a device pointer this tensor does **not** own (TRT output, import).
    ///
    /// # Safety
    /// `ptr` must be a valid device pointer of at least `byte_len` bytes,
    /// remaining valid for as long as this tensor is read (the producer's
    /// contract — e.g. until the owning session's next `run_*` call).
    pub unsafe fn borrowed(
        ptr:      *mut std::ffi::c_void,
        shape:    Vec<usize>,
        dtype:    DType,
        kind:     MemKind,
        byte_len: usize,
        stream:   Arc<CudaStream>,
    ) -> Self {
        let strides = contiguous_strides(&shape);
        Self { ptr, shape, strides, dtype, kind, byte_len, stream, owner: None }
    }

    pub fn numel(&self) -> usize { self.shape.iter().product() }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn kind(&self) -> MemKind { self.kind }
    pub fn byte_len(&self) -> usize { self.byte_len }
    pub fn cuda_stream(&self) -> &Arc<CudaStream> { &self.stream }

    /// `i`-th dimension of the logical shape.
    pub fn dim(&self, i: usize) -> usize { self.shape[i] }

    /// True if the tensor is contiguous row-major (no gaps; views can break this).
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Shape as `i64` for TRT `setInputShape` / `Dims64`.
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&d| d as i64).collect()
    }

    /// Raw mutable device pointer (write path: preprocessor → input buffer).
    pub fn as_mut_ptr(&self) -> *mut std::ffi::c_void { self.ptr }

    /// Raw const device pointer (read path: input buffer → TRT).
    pub fn as_ptr(&self) -> *const std::ffi::c_void { self.ptr as *const _ }

    /// Device pointer as `*const f32`, checked against the tensor dtype.
    pub fn f32_ptr(&self) -> Result<*const f32> {
        if self.dtype != DType::F32 {
            return Err(TrtError::Shape(format!("tensor is {:?}, not F32", self.dtype)));
        }
        Ok(self.ptr as *const f32)
    }
}

/// Contiguous row-major element strides for `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
