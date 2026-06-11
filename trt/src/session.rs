use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use trt_sys::*;
use crate::{
    engine::{Engine, TensorMode, DataType},
    buffer::{DeviceBuffer, Stream},
    error::{Result, TrtError, last_trt_error},
};

/// Raw output tensor: bytes, dtype, shape.
#[derive(Debug)]
pub struct OutputTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub dtype: DataType,
    pub shape: Vec<i64>,
}

impl OutputTensor {
    /// Interpret as f32 slice (panics if dtype != Float32).
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DataType::Float32, "tensor is not f32");
        // SAFETY: u8 → f32 reinterpret; data is aligned to byte boundaries
        // but f32 may require 4-byte alignment — use from_raw_parts only if
        // the pointer is aligned. Use bytemuck-free approach:
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.data.len() / 4,
            )
        }
    }

    /// Convert FP16 data to f32 (requires the `half` crate feature).
    #[cfg(feature = "yolo")]
    pub fn as_f32_from_f16(&self) -> Vec<f32> {
        assert_eq!(self.dtype, DataType::Float16, "tensor is not f16");
        use half::f16;
        self.data.chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()
    }
}

/// Per-tensor device buffer state for one inference session.
struct TensorState {
    buf: DeviceBuffer,
    shape: Vec<i64>,
    dtype: DataType,
}

/// An inference session: one `IExecutionContext` + owned device buffers + stream.
///
/// # Thread safety
/// `Session` is `Send` (can be moved to another thread) but **not `Sync`**
/// (cannot be shared across threads concurrently). This mirrors TRT's contract:
/// `IExecutionContext` is NOT thread-safe. For concurrent inference, create
/// multiple `Session`s from one `Arc<Engine>`.
///
/// # Ownership
/// Holds `Arc<Engine>` (which holds `Arc<Runtime>` → `Arc<Logger>`), ensuring
/// the entire chain stays alive until all sessions are dropped.
pub struct Session {
    ctx: *mut btrt_context_t,
    _engine: Arc<Engine>,     // keeps Engine (and Runtime, Logger) alive
    stream: Stream,
    inputs: HashMap<String, TensorState>,
    outputs: HashMap<String, TensorState>,

    // Makes Session explicitly !Sync at the type level.
    // IExecutionContext is not thread-safe; this PhantomData marker
    // prevents the compiler from ever auto-deriving Sync for Session.
    _not_sync: std::marker::PhantomData<std::cell::UnsafeCell<()>>,
}

// SAFETY: Session can be sent to another thread — the context and device
// buffers are safe to move. Only one thread may use a Session at a time
// (enforced by !Sync). The `_not_sync` PhantomData above ensures this.
unsafe impl Send for Session {}
// NOT Sync — intentionally omitted. IExecutionContext is not thread-safe.

impl Session {
    /// Create a new inference session for the given engine.
    /// Allocates device buffers sized to the engine's static output shapes.
    /// For dynamic-shape engines, call `set_input_shape` before `run`.
    pub fn new(engine: Arc<Engine>) -> Result<Self> {
        let ctx = unsafe { btrt_context_create(engine.as_ptr()) };
        if ctx.is_null() {
            return Err(TrtError::Create("ExecutionContext"));
        }
        let stream = Stream::new()?;

        // Pre-allocate device buffers for all I/O tensors.
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        for spec in engine.specs() {
            let n_elems: i64 = spec.dims.iter()
                .filter(|&&d| d > 0)
                .product::<i64>()
                .max(1);
            let bytes_per_elem = dtype_bytes(spec.dtype);
            let buf = DeviceBuffer::alloc(n_elems as usize * bytes_per_elem)?;
            let state = TensorState { buf, shape: spec.dims.clone(), dtype: spec.dtype };
            if spec.mode == TensorMode::Input {
                inputs.insert(spec.name.clone(), state);
            } else {
                outputs.insert(spec.name.clone(), state);
            }
        }

        Ok(Self { ctx, _engine: engine, stream, inputs, outputs,
                  _not_sync: std::marker::PhantomData })
    }

    /// Set the runtime shape for a dynamic-shape input (call before `run`).
    pub fn set_input_shape(&mut self, name: &str, shape: &[i64]) -> Result<()> {
        let c_name = CString::new(name).map_err(|_| TrtError::UnknownTensor(name.into()))?;
        let code = unsafe {
            btrt_context_set_input_shape(self.ctx, c_name.as_ptr(),
                                          shape.as_ptr(), shape.len() as i32)
        };
        if code != 0 { return Err(TrtError::Trt(last_trt_error())); }
        // Re-size output buffers now that shapes are resolved.
        self.resize_output_buffers()?;
        Ok(())
    }

    /// Run inference.
    /// `inputs`: list of `(tensor_name, f32_slice)` pairs.
    /// Returns a map of output name → `OutputTensor`.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<HashMap<String, OutputTensor>> {
        // 1. Copy inputs H2D and bind.
        for (name, data) in inputs {
            let c_name = CString::new(*name)
                .map_err(|_| TrtError::UnknownTensor((*name).into()))?;
            let state = self.inputs.get(*name)
                .ok_or_else(|| TrtError::UnknownTensor((*name).into()))?;
            let bytes = bytemuck_f32_to_u8(data);
            state.buf.copy_from_host(bytes, &self.stream)?;
            let code = unsafe {
                btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(),
                                                 state.buf.as_device_ptr())
            };
            if code != 0 { return Err(TrtError::Trt(last_trt_error())); }
        }

        // 2. Bind output device buffers.
        for (name, state) in &self.outputs {
            let c_name = CString::new(name.as_str()).unwrap();
            let code = unsafe {
                btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(),
                                                 state.buf.as_device_ptr())
            };
            if code != 0 { return Err(TrtError::Trt(last_trt_error())); }
        }

        // 3. Enqueue inference (asynchronous).
        let code = unsafe { btrt_context_enqueue_v3(self.ctx, self.stream.ptr) };
        if code != 0 { return Err(TrtError::Trt(last_trt_error())); }

        // 4. Copy outputs D2H.
        let mut raw_outputs: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, state) in &self.outputs {
            let mut buf = Vec::new();
            state.buf.copy_to_host(&mut buf, &self.stream)?;
            raw_outputs.insert(name.clone(), buf);
        }

        // 5. Synchronize: MUST happen before caller reads outputs.
        self.stream.sync()?;

        // 6. Build result map.
        let mut result = HashMap::new();
        for (name, data) in raw_outputs {
            let state = &self.outputs[&name];
            let shape = self.resolved_output_shape(&name)?;
            result.insert(name.clone(), OutputTensor { name, data, dtype: state.dtype, shape });
        }
        Ok(result)
    }

    fn resolved_output_shape(&self, name: &str) -> Result<Vec<i64>> {
        let c_name = CString::new(name).unwrap();
        let mut dims = [0i64; 8];
        let mut ndims = 0i32;
        let code = unsafe {
            btrt_context_get_tensor_shape(self.ctx, c_name.as_ptr(),
                                           dims.as_mut_ptr(), &mut ndims)
        };
        if code != 0 { return Err(TrtError::UnknownTensor(name.into())); }
        Ok(dims[..ndims as usize].to_vec())
    }

    fn resize_output_buffers(&mut self) -> Result<()> {
        // Collect names to avoid borrow conflict
        let names: Vec<String> = self.outputs.keys().cloned().collect();
        for name in names {
            let shape = self.resolved_output_shape(&name)?;
            let dtype = self.outputs[&name].dtype;
            let n: i64 = shape.iter().filter(|&&d| d > 0).product::<i64>().max(1);
            let new_len = n as usize * dtype_bytes(dtype);
            if self.outputs[&name].buf.len_bytes != new_len {
                self.outputs.get_mut(&name).unwrap().buf = DeviceBuffer::alloc(new_len)?;
                self.outputs.get_mut(&name).unwrap().shape = shape;
            }
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: unique owner of ctx. Arc<Engine> dropped after, ensuring
        // correct IExecutionContext → ICudaEngine → IRuntime → ILogger order.
        if !self.ctx.is_null() {
            unsafe { btrt_context_destroy(self.ctx); }
        }
    }
}

fn dtype_bytes(dtype: DataType) -> usize {
    match dtype {
        DataType::Float32 | DataType::Int32 => 4,
        DataType::Float16 => 2,
        DataType::Int8 | DataType::UInt8 | DataType::Bool => 1,
    }
}

// Safe f32→u8 reinterpret (no copy on little-endian, which Jetson is).
fn bytemuck_f32_to_u8(s: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
    }
}
