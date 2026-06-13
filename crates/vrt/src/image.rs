//! Pitch-linear image view — the camera-ingest counterpart to [`VrtTensor`].
//!
//! A [`VrtTensor`] models a dense N-D array with element strides; a [`VrtImage`]
//! models a 2-D pixel surface with a **byte pitch** (row stride, typically
//! padded past `width * bytes_per_pixel` for hardware alignment) and a pixel
//! [`Format`].  Both are borrowed views over device memory the producer owns.

use crate::tensor::MemKind;

/// Pixel layout of a [`VrtImage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// 8-bit RGBA, 4 bytes/pixel (the NVMM/`nvvidconv` output format).
    Rgba8,
    /// 8-bit RGB, 3 bytes/pixel.
    Rgb8,
    /// Single 8-bit channel.
    Gray8,
}

impl Format {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Format::Rgba8 => 4,
            Format::Rgb8  => 3,
            Format::Gray8 => 1,
        }
    }
}

/// A borrowed view of a pitch-linear image in device memory.
///
/// Replaces the former `DeviceFrame`.  Carries the dimensions and format that
/// preprocessing previously had to be told out of band, so a preprocessor can
/// validate the surface it's handed instead of trusting fixed constructor args.
///
/// The backing memory is owned elsewhere (an NVMM import, a CUDA allocation) —
/// the producer must keep it alive for the duration of any GPU work reading it.
pub struct VrtImage {
    ptr:    *mut std::ffi::c_void,
    width:  u32,
    height: u32,
    pitch:  u32,
    format: Format,
    kind:   MemKind,
}

// SAFETY: device pointer is stable; callers enforce CUDA ordering via the stream.
unsafe impl Send for VrtImage {}

impl VrtImage {
    /// Wrap a device pointer to a pitch-linear image this view does not own.
    ///
    /// # Safety
    /// `ptr` must point to at least `pitch * height` bytes of valid device
    /// memory in `format`, alive for the duration of any GPU work reading it.
    pub unsafe fn borrowed(
        ptr:    *mut std::ffi::c_void,
        width:  u32,
        height: u32,
        pitch:  u32,
        format: Format,
        kind:   MemKind,
    ) -> Self {
        Self { ptr, width, height, pitch, format, kind }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void { self.ptr }
    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    /// Row stride in bytes (≥ `width * format.bytes_per_pixel()`).
    pub fn pitch(&self) -> u32 { self.pitch }
    pub fn format(&self) -> Format { self.format }
    pub fn kind(&self) -> MemKind { self.kind }
}
