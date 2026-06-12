use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DriverError};
use crate::error::{Result, TrtError};
use vrt_sys::btrt_cuda_memcpy_d2h;

fn driver_err(e: DriverError, msg: &'static str) -> TrtError {
    TrtError::Cuda { code: e.0 as i32, msg }
}

/// Owned CUDA device memory buffer backed by cudarc.
///
/// Built on cudarc's primary-context model — same context TRT's Runtime API uses,
/// so no cross-context copies: device pointers from this buffer can be bound
/// directly to TRT execution contexts via `setTensorAddress`.
pub struct DeviceBuffer {
    slice: CudaSlice<u8>,
    pub len_bytes: usize,
}

// SAFETY: CudaSlice wraps a stable device pointer. Moving across threads is safe;
// callers must synchronize GPU ops (enforced by Stream::sync in Session::run).
unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `len_bytes` of zero-initialized CUDA device memory on `stream`.
    /// The memory is freed asynchronously on that stream when dropped.
    pub fn alloc_with_stream(stream: &Arc<CudaStream>, len_bytes: usize) -> Result<Self> {
        let slice = stream
            .alloc_zeros::<u8>(len_bytes)
            .map_err(|e| driver_err(e, "cudaMalloc"))?;
        Ok(Self { slice, len_bytes })
    }

    /// Copy host bytes → device on the given stream (asynchronous).
    pub fn copy_from_host(&mut self, src: &[u8], stream: &Stream) -> Result<()> {
        assert_eq!(src.len(), self.len_bytes, "host/device size mismatch");
        stream
            .inner
            .memcpy_htod(src, &mut self.slice)
            .map_err(|e| driver_err(e, "cudaMemcpyH2D"))
    }

    /// Copy device → host on the given stream (call `stream.sync()` before reading result).
    pub fn copy_to_host(&self, dst: &mut Vec<u8>, stream: &Stream) -> Result<()> {
        *dst = stream
            .inner
            .memcpy_dtov(&self.slice)
            .map_err(|e| driver_err(e, "cudaMemcpyD2H"))?;
        Ok(())
    }

    /// Raw CUDA device pointer for binding to TRT `setTensorAddress`.
    ///
    /// `sys::CUdeviceptr` is `u64`; cast to `usize` then `*mut c_void` for the C ABI.
    /// The `_guard` ensures the stream records a dependency — drop it after TRT has
    /// enqueued work on the same stream.
    pub fn as_device_ptr_guarded<'a>(
        &'a self,
        stream: &'a Stream,
    ) -> (*mut std::ffi::c_void, cudarc::driver::SyncOnDrop<'a>) {
        let (ptr, guard) = self.slice.device_ptr(stream.inner.as_ref());
        (ptr as usize as *mut std::ffi::c_void, guard)
    }

    /// Convenience: raw device pointer (no guard held — suitable when TRT manages sync).
    pub fn as_device_ptr(&self, stream: &Stream) -> *mut std::ffi::c_void {
        let (ptr, _guard) = self.slice.device_ptr(stream.inner.as_ref());
        ptr as usize as *mut std::ffi::c_void
    }
}

/// Owned CUDA stream backed by cudarc.
pub struct Stream {
    pub(crate) inner: Arc<CudaStream>,
}

// SAFETY: CudaStream is Arc-backed; safe to send across threads.
// One thread enqueues at a time (enforced by Session being !Sync).
unsafe impl Send for Stream {}

impl Stream {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let inner = ctx
            .new_stream()
            .map_err(|e| driver_err(e, "cudaStreamCreate"))?;
        Ok(Self { inner })
    }

    /// Raw `cudaStream_t` / `CUstream` cast to `*mut c_void` for the TRT C bridge.
    ///
    /// `CUstream` and `cudaStream_t` are the same opaque pointer type at the ABI level.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.inner.cu_stream() as *mut std::ffi::c_void
    }

    /// Block until all operations enqueued on this stream complete.
    pub fn sync(&self) -> Result<()> {
        self.inner
            .synchronize()
            .map_err(|e| driver_err(e, "cudaStreamSynchronize"))
    }

    pub fn cuda_stream(&self) -> &Arc<CudaStream> {
        &self.inner
    }

    /// Wrap an existing `Arc<CudaStream>` without creating a new CUDA stream.
    ///
    /// Used by [`Session::with_stream`](crate::Session::with_stream) and
    /// [`Pipeline`](crate::Pipeline) to share one stream across all stages.
    pub fn from_cuda_stream(inner: Arc<CudaStream>) -> Self {
        Self { inner }
    }

    /// Create a standalone CUDA stream bound to the primary device context.
    ///
    /// Use this to obtain a shared `Arc<CudaStream>` before constructing pipeline
    /// stages.  All stages that call `Session::with_stream` with the same arc
    /// are guaranteed to share one CUDA stream.
    pub fn new_standalone() -> Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| TrtError::Cuda { code: e.0 as i32, msg: "CudaContext::new" })?;
        Self::new(&ctx)
    }

    /// Enqueue an async device → host copy on this stream.
    ///
    /// The copy completes when the stream is synced.  `dst` must remain alive
    /// until after the sync.
    ///
    /// # Safety
    /// `src` must be a valid CUDA device pointer of at least `bytes` bytes.
    pub unsafe fn memcpy_d2h_raw(
        &self,
        dst:   *mut u8,
        src:   *const std::ffi::c_void,
        bytes: usize,
    ) -> Result<()> {
        let rc = btrt_cuda_memcpy_d2h(dst as *mut _, src as *const _, bytes, self.as_raw());
        if rc != 0 {
            return Err(TrtError::Cuda { code: rc, msg: "cudaMemcpyAsync D2H" });
        }
        Ok(())
    }
}
