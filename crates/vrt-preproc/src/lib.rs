//! Zero-copy GPU preprocessing: bilinear letterbox + RGBA→CHW normalize.
//!
//! Uses vrt::cuda::Kernels to JIT-compile the kernel for the current device, plus a
//! C helper that creates `cudaTextureObject_t` over pitch-2D RGBA device memory
//! so the kernel uses the TMU's hardware bilinear sampler.
//!
//! ## Copy budget per frame
//! - **NVMM path** (`Stage::enqueue` / `process_device_ptr_tex`): caller provides dev_ptr
//!   from `cudaExternalMemoryGetMappedBuffer`; zero copies — only a texture-object
//!   create/destroy pair (~10 µs) and the kernel itself.
//! - **CPU path** (`process`): one H2D transfer then kernel.

use std::sync::Arc;
use std::ffi::c_void;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use cudarc::driver::sys::CUdeviceptr;

use vrt::{Stage, BoxError, TRTensor, DType};
use vrt::cuda::{Kernels, cfg_2d};

/// Errors from GPU preprocessing.
#[derive(Debug, thiserror::Error)]
pub enum PreprocError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("cudaCreateTextureObject failed (err={0})")]
    TextureCreate(i32),
}

// ── C helpers (preproc_helpers.cpp) ─────────────────────────────────────────

extern "C" {
    fn preproc_create_tex2d(
        dev_ptr:     *mut c_void,
        width:       u32,
        height:      u32,
        pitch:       u32,
        tex_obj_out: *mut u64,
    ) -> i32;

    fn preproc_destroy_tex2d(tex_obj: u64);
}

// ── Texture RAII guard ───────────────────────────────────────────────────────

/// Keeps a `cudaTextureObject_t` alive until the guard is dropped.
///
/// **Drop only after stream sync** — the kernel that reads through this texture
/// must have completed before the texture object is destroyed.
pub struct TextureGuard {
    tex:  u64,
    /// For the CPU path only: the H2D-copied device buffer backing this texture.
    _src: Option<CudaSlice<u8>>,
}

impl Drop for TextureGuard {
    fn drop(&mut self) {
        if self.tex != 0 {
            unsafe { preproc_destroy_tex2d(self.tex); }
        }
    }
}

// ── DeviceFrame ───────────────────────────────────────────────────────────────

/// Primitive stage input: a CUDA device pointer to an RGBA pitch-linear buffer.
///
/// No NVMM or GStreamer dependency — the caller is responsible for keeping the
/// backing allocation alive for the duration of the GPU work.
pub struct DeviceFrame {
    pub dev_ptr: *mut c_void,
    pub pitch:   u32,
}

// Raw pointer represents device memory; safe to send given CUDA stream ordering.
unsafe impl Send for DeviceFrame {}

// ── CUDA kernel source ────────────────────────────────────────────────────────

const KERNEL_SRC: &str = r#"
#include <cuda_texture_types.h>

extern "C" __global__ void letterbox_rgba_to_chw(
    unsigned long long src_tex,
    float* __restrict__ dst,
    float scale, float pad_x, float pad_y,
    int src_w, int src_h,
    int dst_w, int dst_h
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    float sx = ((float)ox - pad_x) / scale;
    float sy = ((float)oy - pad_y) / scale;

    int pixels = dst_w * dst_h;
    int out    = oy * dst_w + ox;

    if (sx < 0.0f || sy < 0.0f || sx >= (float)src_w || sy >= (float)src_h) {
        float g = 114.0f / 255.0f;
        dst[out]            = g;
        dst[pixels + out]   = g;
        dst[2*pixels + out] = g;
        return;
    }

    float4 px = tex2D<float4>((cudaTextureObject_t)src_tex,
                               sx + 0.5f, sy + 0.5f);
    dst[out]            = px.x;
    dst[pixels + out]   = px.y;
    dst[2*pixels + out] = px.z;
}
"#;

// ── Preprocessor ─────────────────────────────────────────────────────────────

/// GPU letterbox preprocessor: [`DeviceFrame`] → [`TRTensor`] (CHW FP32).
///
/// Implements [`Stage`] so it plugs directly into a [`vrt::Pipeline`].
/// The output [`TRTensor`] is pre-allocated in `new` and reused every frame.
/// `_pending` holds the [`TextureGuard`] across enqueue → sync; dropped in
/// [`finalize`](Stage::finalize) after the stream is synced.
pub struct Preprocessor {
    func:   cudarc::driver::CudaFunction,
    stream: Arc<CudaStream>,
    src_w: u32, src_h: u32,
    dst_w: u32, dst_h: u32,
    scale: f32, pad_x: f32, pad_y: f32,
    output:   TRTensor,
    _pending: Option<TextureGuard>,
}

impl Preprocessor {
    /// Compile the kernel (JIT, ~10 ms first time; CUDA caches the PTX),
    /// bind to fixed source/destination dimensions, and pre-allocate the
    /// output tensor on `stream`.
    pub fn new(
        stream: Arc<CudaStream>,
        src_w: u32, src_h: u32,
        dst_w: u32, dst_h: u32,
    ) -> Result<Self, PreprocError> {
        let kernels = Kernels::compile(stream.clone(), KERNEL_SRC)?;
        let func    = kernels.function("letterbox_rgba_to_chw")?;

        let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
        let pad_x = (dst_w as f32 - src_w as f32 * scale) * 0.5;
        let pad_y = (dst_h as f32 - src_h as f32 * scale) * 0.5;

        let output = TRTensor::alloc(
            &stream,
            [1, 3, dst_h as usize, dst_w as usize],
            DType::F32,
        )?;

        Ok(Self { func, stream, src_w, src_h, dst_w, dst_h, scale, pad_x, pad_y, output, _pending: None })
    }

    /// The CUDA stream this preprocessor launches on.
    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }

    /// H2D + kernel: RGBA host bytes → device, then letterbox into `dst_dev_ptr`.
    ///
    /// Returns a [`TextureGuard`] that must be kept alive until the stream is synced.
    pub fn process(
        &self,
        rgba_host: &[u8],
        src_pitch: u32,
        dst_dev_ptr: *mut f32,
    ) -> Result<TextureGuard, PreprocError> {
        let src_dev = self.stream.memcpy_stod(rgba_host)?;
        let raw_ptr: u64 = {
            let (ptr, _guard) = src_dev.device_ptr(self.stream.as_ref());
            ptr
        };

        let mut tex: u64 = 0;
        let rc = unsafe {
            preproc_create_tex2d(
                raw_ptr as usize as *mut c_void,
                self.src_w, self.src_h, src_pitch,
                &mut tex,
            )
        };
        if rc != 0 {
            return Err(PreprocError::TextureCreate(rc));
        }

        self.launch_kernel(tex, dst_dev_ptr)?;
        Ok(TextureGuard { tex, _src: Some(src_dev) })
    }

    /// Zero-copy kernel: `src_dev_ptr` is already a CUDA device pointer.
    ///
    /// # Safety
    /// `src_dev_ptr` must be a valid CUDA device pointer of at least
    /// `src_h × src_pitch` bytes, alive for the kernel's duration.
    pub unsafe fn process_device_ptr_tex(
        &self,
        src_dev_ptr: *mut c_void,
        src_pitch: u32,
        dst_dev_ptr: *mut f32,
    ) -> Result<TextureGuard, PreprocError> {
        let mut tex: u64 = 0;
        let rc = unsafe {
            preproc_create_tex2d(
                src_dev_ptr,
                self.src_w, self.src_h, src_pitch,
                &mut tex,
            )
        };
        if rc != 0 {
            return Err(PreprocError::TextureCreate(rc));
        }

        self.launch_kernel(tex, dst_dev_ptr)?;
        Ok(TextureGuard { tex, _src: None })
    }

    fn launch_kernel(
        &self,
        src_tex: u64,
        dst_dev_ptr: *mut f32,
    ) -> Result<(), PreprocError> {
        let dst_raw: CUdeviceptr = dst_dev_ptr as usize as CUdeviceptr;

        let cfg = cfg_2d(self.dst_w as usize, self.dst_h as usize);

        let dst_w = self.dst_w as i32;
        let dst_h = self.dst_h as i32;
        let src_w = self.src_w as i32;
        let src_h = self.src_h as i32;

        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&src_tex)
                .arg(&dst_raw)
                .arg(&self.scale)
                .arg(&self.pad_x)
                .arg(&self.pad_y)
                .arg(&src_w)
                .arg(&src_h)
                .arg(&dst_w)
                .arg(&dst_h)
                .launch(cfg)?;
        }
        Ok(())
    }
}

// ── Stage impl ────────────────────────────────────────────────────────────────

impl Stage for Preprocessor {
    type Input  = DeviceFrame;
    type Output = TRTensor;

    fn enqueue(&mut self, frame: &DeviceFrame) -> Result<(), BoxError> {
        // A still-pending texture means the previous frame never reached
        // finalize (error path).  Drain the stream before dropping it —
        // the in-flight kernel may still read through the texture object.
        if self._pending.is_some() {
            self.stream.synchronize()?;
            self._pending = None;
        }
        let dst = self.output.as_mut_ptr() as *mut f32;
        let tex = unsafe { self.process_device_ptr_tex(frame.dev_ptr, frame.pitch, dst) }
            ?;
        self._pending = Some(tex);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), BoxError> {
        self._pending = None;
        Ok(())
    }

    fn output(&self) -> &TRTensor { &self.output }
}
