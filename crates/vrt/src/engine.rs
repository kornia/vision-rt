use crate::{
    error::{last_trt_error, Result, TrtError},
    runtime::Runtime,
};
use std::path::Path;
use std::sync::Arc;
use trt_sys::*;

/// I/O mode of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMode {
    None = 0,
    Input = 1,
    Output = 2,
}

/// Data type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
    Int32 = 3,
    Bool = 4,
    UInt8 = 5,
}

/// Metadata for one engine I/O tensor (discovered via named-tensor API).
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub mode: TensorMode,
    pub dtype: DataType,
    /// Shape dims. `-1` means dynamic (resolved at runtime via `setInputShape`).
    pub dims: Vec<i64>,
}

/// Wraps `nvinfer1::ICudaEngine`. `Send + Sync` — TRT guarantees ICudaEngine
/// is thread-safe for creating execution contexts and read-only queries.
///
/// # Ownership
/// Holds `Arc<Runtime>` (and transitively `Arc<Logger>`) for correct Drop ordering.
pub struct Engine {
    ptr: *mut btrt_engine_t,
    _runtime: Arc<Runtime>,
    pub(crate) specs: Vec<TensorSpec>,
}

// SAFETY: ICudaEngine is documented as thread-safe for concurrent context
// creation and read-only operations. We own the pointer exclusively.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Load a pre-built `.engine` file from disk.
    pub fn from_file(runtime: Arc<Runtime>, path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let bytes = std::fs::read(path)
            .map_err(|e| TrtError::Deserialize(format!("could not read file: {e}")))?;
        Self::deserialize(runtime, &bytes)
    }

    /// Deserialize from raw bytes (e.g. built by `trtexec` or the in-process builder).
    pub fn deserialize(runtime: Arc<Runtime>, bytes: &[u8]) -> Result<Arc<Self>> {
        let ptr = unsafe {
            btrt_engine_deserialize(runtime.as_ptr(), bytes.as_ptr() as *const _, bytes.len())
        };
        if ptr.is_null() {
            let msg = last_trt_error();
            return Err(TrtError::Deserialize(if msg.is_empty() {
                "unknown error (wrong TRT version or architecture?)".into()
            } else {
                msg
            }));
        }
        let specs = discover_specs(ptr)?;
        Ok(Arc::new(Self {
            ptr,
            _runtime: runtime,
            specs,
        }))
    }

    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }
    pub fn inputs(&self) -> impl Iterator<Item = &TensorSpec> {
        self.specs.iter().filter(|s| s.mode == TensorMode::Input)
    }
    /// The single input's static `[N, C, H, W]` dims as `usize`.
    ///
    /// Errors if the engine has no input, the input isn't 4-D, or any dim is
    /// non-positive (dynamic `-1` or zero). The image models call this to size
    /// their reused CHW input tensor instead of re-deriving it each.
    pub fn static_input_nchw(&self) -> Result<[usize; 4]> {
        let d = &self
            .inputs()
            .next()
            .ok_or_else(|| TrtError::Trt("engine has no input".into()))?
            .dims;
        if d.len() != 4 || d.iter().any(|&x| x <= 0) {
            return Err(TrtError::Trt(format!(
                "input must be static [N,3,H,W], got {d:?}"
            )));
        }
        Ok([d[0] as usize, d[1] as usize, d[2] as usize, d[3] as usize])
    }
    pub fn outputs(&self) -> impl Iterator<Item = &TensorSpec> {
        self.specs.iter().filter(|s| s.mode == TensorMode::Output)
    }
    /// Names of the input tensors, in engine order.
    pub fn input_names(&self) -> Vec<String> {
        self.inputs().map(|s| s.name.clone()).collect()
    }
    /// Names of the output tensors, in engine order.
    pub fn output_names(&self) -> Vec<String> {
        self.outputs().map(|s| s.name.clone()).collect()
    }

    pub(crate) fn as_ptr(&self) -> *mut btrt_engine_t {
        self.ptr
    }
}

fn discover_specs(engine: *mut btrt_engine_t) -> Result<Vec<TensorSpec>> {
    let n = unsafe { btrt_engine_num_io_tensors(engine) };
    let mut specs = Vec::with_capacity(n as usize);
    for i in 0..n {
        let name_ptr = unsafe { btrt_engine_io_tensor_name(engine, i) };
        let name = unsafe {
            std::ffi::CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .into_owned()
        };
        let c_name = std::ffi::CString::new(name.as_bytes()).unwrap();
        let mode = match unsafe { btrt_engine_tensor_io_mode(engine, c_name.as_ptr()) } {
            1 => TensorMode::Input,
            2 => TensorMode::Output,
            _ => TensorMode::None,
        };
        let dtype = match unsafe { btrt_engine_tensor_dtype(engine, c_name.as_ptr()) } {
            0 => DataType::Float32,
            1 => DataType::Float16,
            2 => DataType::Int8,
            3 => DataType::Int32,
            4 => DataType::Bool,
            5 => DataType::UInt8,
            _ => DataType::Float32,
        };
        let mut raw_dims = [0i64; 8];
        let mut ndims = 0i32;
        let rc = unsafe {
            btrt_engine_tensor_shape(engine, c_name.as_ptr(), raw_dims.as_mut_ptr(), &mut ndims)
        };
        // A failed query left ndims at 0 → empty dims → silent buffer
        // under-allocation downstream. Surface it, and bound ndims to the buffer.
        if rc != 0 || ndims < 0 || ndims as usize > raw_dims.len() {
            return Err(TrtError::Trt(format!(
                "failed to query shape for tensor '{name}' (rc={rc}, ndims={ndims})"
            )));
        }
        let dims = raw_dims[..ndims as usize].to_vec();
        specs.push(TensorSpec {
            name,
            mode,
            dtype,
            dims,
        });
    }
    Ok(specs)
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: unique owner; all Contexts (which hold Arc<Engine>) have
        // dropped before this fires.
        unsafe {
            btrt_engine_destroy(self.ptr);
        }
    }
}
