use std::os::raw::c_void;

extern "C" {
    /// **DEPRECATED** — `dataPtr` is documented invalid for `NVBUF_MEM_SURFACE_ARRAY`
    /// (Jetson default).  Use `nvbuf_cuda_import` for all NVMM CUDA access.
    /// Only valid for `NVBUF_MEM_CUDA_UNIFIED` buffers.
    pub fn nvbuf_cuda_ptr(nvbuf_surface: *const c_void) -> *mut c_void;

    /// Width in pixels of the first surface.
    pub fn nvbuf_width(nvbuf_surface: *const c_void) -> u32;

    /// Height in pixels of the first surface.
    pub fn nvbuf_height(nvbuf_surface: *const c_void) -> u32;

    /// Row stride in bytes of the first surface.
    pub fn nvbuf_pitch(nvbuf_surface: *const c_void) -> u32;

    /// DMA-BUF file descriptor (-1 on error).  Valid for SURFACE_ARRAY / HANDLE.
    /// Caller retains ownership; do not close the returned FD.
    pub fn nvbuf_dmabuf_fd(nvbuf_surface: *const c_void) -> i32;

    /// Memory layout: 0 = pitch-linear, 1 = block-linear, -1 = error.
    /// Only pitch-linear (0) surfaces can be imported for linear CUDA access.
    pub fn nvbuf_layout(nvbuf_surface: *const c_void) -> i32;

    /// Total allocated bytes for the first surface.
    pub fn nvbuf_data_size(nvbuf_surface: *const c_void) -> u64;

    /// Import a DMA-BUF FD into CUDA as an external memory object.
    ///
    /// Returns 0 on success.  Fills `*ext_mem_out` and `*dev_ptr_out` with
    /// handles that **must** be released via `nvbuf_cuda_release` after the
    /// consuming CUDA stream is synced.
    ///
    /// Internally `dup()`s `fd` — caller retains ownership of the original.
    /// Must be called from a thread where the CUDA primary context is active.
    pub fn nvbuf_cuda_import(
        fd: i32,
        size: u64,
        ext_mem_out: *mut *mut c_void,
        dev_ptr_out: *mut *mut c_void,
    ) -> i32;

    /// Release handles returned by `nvbuf_cuda_import`.  Call only after sync.
    pub fn nvbuf_cuda_release(ext_mem: *mut c_void, dev_ptr: *mut c_void);
}
