//! Raw FFI bindings to TensorRT 10.x.
//!
//! This crate exposes a flat `extern "C"` API generated from `include/shim.h`
//! via bindgen. All functions are `unsafe`.
//!
//! The C++ shim (`src/shim.cpp`) wraps the TensorRT 10 C++ API and presents a
//! pure-C surface that bindgen can consume without needing TRT or CUDA headers.
//!
//! # Updating for a new TensorRT version
//! See `UPDATING.md` in the repository root.

#![allow(unsafe_code)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// TensorRT major version this crate was compiled against.
/// Update when upgrading TensorRT (also update build.rs and NvInferVersion.h check).
pub const TENSORRT_VERSION_MAJOR: u32 = 10;

/// TensorRT minor version this crate was compiled against.
pub const TENSORRT_VERSION_MINOR: u32 = 3;

/// TensorRT patch version this crate was compiled against.
pub const TENSORRT_VERSION_PATCH: u32 = 0;

/// TensorRT build number this crate was compiled against.
pub const TENSORRT_VERSION_BUILD: u32 = 30;

/// Convenience: returns `true` if `status` indicates success (status == 0).
#[inline]
pub fn is_ok(status: i32) -> bool {
    status == 0
}
