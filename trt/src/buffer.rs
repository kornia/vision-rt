use trt_sys::{btrt_cuda_malloc, btrt_cuda_free,
              btrt_cuda_memcpy_h2d, btrt_cuda_memcpy_d2h,
              btrt_cuda_stream_create, btrt_cuda_stream_sync, btrt_cuda_stream_destroy};
use crate::error::{Result, TrtError};

/// Owned CUDA device memory buffer.
///
/// # Concurrency
/// `Send` — device memory addresses are stable and can be passed across threads,
/// but you must ensure no concurrent kernel/copy uses the same region.
/// `!Sync` by default (raw pointer).
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    pub len_bytes: usize,
}

// SAFETY: A device buffer is an allocated region in GPU memory. The pointer
// is an address, not a reference — sending it to another thread is safe as
// long as the caller synchronizes GPU operations (enforced by Stream sync in
// Session::run before outputs are returned).
unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `len_bytes` of CUDA device memory.
    pub fn alloc(len_bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let code = unsafe { btrt_cuda_malloc(&mut ptr, len_bytes) };
        if code != 0 { return Err(TrtError::Cuda { code, msg: "cudaMalloc" }); }
        Ok(Self { ptr, len_bytes })
    }

    /// Copy host slice → device (async on `stream`).
    pub fn copy_from_host(&self, src: &[u8], stream: &Stream) -> Result<()> {
        assert_eq!(src.len(), self.len_bytes, "host/device size mismatch");
        let code = unsafe {
            btrt_cuda_memcpy_h2d(self.ptr, src.as_ptr() as *const _, src.len(), stream.ptr)
        };
        if code != 0 { Err(TrtError::Cuda { code, msg: "cudaMemcpyH2D" }) } else { Ok(()) }
    }

    /// Copy device → host slice (async on `stream`). Call `stream.sync()` before reading.
    pub fn copy_to_host(&self, dst: &mut Vec<u8>, stream: &Stream) -> Result<()> {
        dst.resize(self.len_bytes, 0);
        let code = unsafe {
            btrt_cuda_memcpy_d2h(dst.as_mut_ptr() as *mut _, self.ptr, self.len_bytes, stream.ptr)
        };
        if code != 0 { Err(TrtError::Cuda { code, msg: "cudaMemcpyD2H" }) } else { Ok(()) }
    }

    pub fn as_device_ptr(&self) -> *mut std::ffi::c_void { self.ptr }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { btrt_cuda_free(self.ptr); }
        }
    }
}

/// Owned CUDA stream.
pub struct Stream {
    pub(crate) ptr: *mut std::ffi::c_void,
}

// SAFETY: A CUDA stream is a handle to a serialized command queue. It can be
// sent to another thread safely (cudaStream_t is opaque pointer). Only one
// thread should enqueue to a stream at a time — enforced by Session being !Sync.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let code = unsafe { btrt_cuda_stream_create(&mut ptr) };
        if code != 0 { return Err(TrtError::Cuda { code, msg: "cudaStreamCreate" }); }
        Ok(Self { ptr })
    }

    /// Block until all operations enqueued on this stream complete.
    pub fn sync(&self) -> Result<()> {
        let code = unsafe { btrt_cuda_stream_sync(self.ptr) };
        if code != 0 { Err(TrtError::Cuda { code, msg: "cudaStreamSync" }) } else { Ok(()) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { btrt_cuda_stream_destroy(self.ptr); }
        }
    }
}
