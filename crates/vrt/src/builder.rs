//! In-process ONNX → serialized engine builder (feature = `builder`).
//!
//! Produces the same artifact as `trtexec --onnx=... --saveEngine=...` without
//! the subprocess.  Engines remain machine-locked (TRT version + GPU arch) —
//! build them on the device that will run them, never copy across hosts.
//!
//! ```no_run
//! # use vrt::{Logger, builder::EngineBuilder};
//! # use vrt::logger::Severity;
//! let logger = Logger::new(Severity::Warning)?;
//! let blob = EngineBuilder::from_onnx("xfeat_backbone.onnx")
//!     .fp16(true)
//!     .workspace_mb(2048)
//!     .shape_profile("image", &[1,3,240,320], &[1,3,640,640], &[1,3,1088,1920])
//!     .build_serialized(&logger)?;
//! std::fs::write("xfeat.engine", &blob)?;
//! # Ok::<(), vrt::BoxError>(())
//! ```

use crate::error::{last_trt_error, Result, TrtError};
use crate::logger::Logger;

/// A dynamic-shape optimization profile: `(input_name, min, opt, max)` dims.
pub type ShapeProfile = (String, Vec<i64>, Vec<i64>, Vec<i64>);

/// Builder for serialized TensorRT engines from ONNX files.
pub struct EngineBuilder {
    onnx_path: String,
    fp16: bool,
    workspace_bytes: i64,
    profile: Option<ShapeProfile>,
}

impl EngineBuilder {
    /// Start from an ONNX file.  External-data sidecars (`.onnx.data`) are
    /// resolved relative to the file by the parser.
    pub fn from_onnx(path: impl Into<String>) -> Self {
        Self {
            onnx_path: path.into(),
            fp16: true,
            workspace_bytes: 2048 << 20,
            profile: None,
        }
    }

    /// Enable/disable FP16 kernels (default: enabled).
    pub fn fp16(mut self, on: bool) -> Self {
        self.fp16 = on;
        self
    }

    /// Workspace memory-pool limit in MiB (default 2048 — conservative for
    /// Jetson unified memory).
    pub fn workspace_mb(mut self, mb: i64) -> Self {
        self.workspace_bytes = mb << 20;
        self
    }

    /// Attach a min/opt/max optimization profile to a dynamic-shape input.
    /// Required for dynamic models; omit for static shapes.
    pub fn shape_profile(
        mut self,
        input: impl Into<String>,
        min: &[i64],
        opt: &[i64],
        max: &[i64],
    ) -> Self {
        self.profile = Some((input.into(), min.to_vec(), opt.to_vec(), max.to_vec()));
        self
    }

    /// Build and return the serialized engine bytes.
    ///
    /// Slow (minutes for real models — TRT exhaustively times kernels).
    /// `logger` receives TRT's build diagnostics.
    pub fn build_serialized(&self, logger: &Logger) -> Result<Vec<u8>> {
        let c_path = std::ffi::CString::new(self.onnx_path.as_str())
            .map_err(|_| TrtError::Create("onnx path contains NUL"))?;

        let (c_input, min, opt, max, ndims) = match &self.profile {
            Some((name, min, opt, max)) => {
                if min.len() != opt.len() || opt.len() != max.len() {
                    return Err(TrtError::Shape(
                        "profile min/opt/max must have equal rank".into(),
                    ));
                }
                let c = std::ffi::CString::new(name.as_str())
                    .map_err(|_| TrtError::Create("input name contains NUL"))?;
                (
                    Some(c),
                    min.as_ptr(),
                    opt.as_ptr(),
                    max.as_ptr(),
                    min.len() as i32,
                )
            }
            None => (
                None,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
            ),
        };

        let mut blob: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        let rc = unsafe {
            trt_sys::btrt_build_engine_from_onnx(
                logger.as_ptr(),
                c_path.as_ptr(),
                self.fp16 as i32,
                c_input.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                min,
                opt,
                max,
                ndims,
                self.workspace_bytes,
                &mut blob,
                &mut len,
            )
        };
        if rc != 0 || blob.is_null() {
            return Err(TrtError::Trt(format!(
                "engine build failed (rc={rc}): {}",
                last_trt_error()
            )));
        }

        let bytes = unsafe { std::slice::from_raw_parts(blob, len) }.to_vec();
        unsafe { trt_sys::btrt_blob_free(blob) };
        Ok(bytes)
    }
}
