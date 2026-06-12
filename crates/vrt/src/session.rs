use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use vrt_sys::*;
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
    /// Interpret as f32 slice (panics if dtype != Float32 or data is misaligned).
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DataType::Float32, "tensor is not f32");
        let ptr = self.data.as_ptr();
        // from_raw_parts requires f32 alignment; Vec<u8> only guarantees 1.
        // The global allocator aligns these sizes in practice — verify anyway.
        assert!(
            (ptr as usize).is_multiple_of(std::mem::align_of::<f32>()),
            "output buffer misaligned for f32 view"
        );
        unsafe { std::slice::from_raw_parts(ptr as *const f32, self.data.len() / 4) }
    }

    /// Convert FP16 output data to f32.
    pub fn as_f32_from_f16(&self) -> Vec<f32> {
        assert_eq!(self.dtype, DataType::Float16, "tensor is not f16");
        use half::f16;
        self.data.chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()
    }
}

/// Typed view of a device tensor owned by a [`Session`].
///
/// Carries the resolved shape, dtype and byte length alongside the raw device
/// pointer, so downstream stages never re-derive dimensions out of band.
///
/// ## Validity window
/// The pointer aliases Session-owned device memory.  It is valid **until the
/// owning Session's next `run_*` call** (a shape change reallocates the
/// buffer) **or Session drop** — not indefinitely.  Pipeline stages must
/// consume views within the same frame (enqueue → sync → finalize) and never
/// store them across frames.
#[derive(Debug, Clone)]
pub struct TensorView {
    ptr:      *mut std::ffi::c_void,
    shape:    Vec<i64>,
    dtype:    DataType,
    byte_len: usize,
}

// SAFETY: the pointer is a CUDA device address only dereferenced by kernels;
// cross-thread moves are safe as long as the validity window above is honored.
unsafe impl Send for TensorView {}

impl TensorView {
    /// Raw device pointer (see "Validity window").
    pub fn ptr(&self) -> *mut std::ffi::c_void { self.ptr }

    /// Device pointer as `*const f32`, checked against the tensor dtype.
    pub fn f32_ptr(&self) -> Result<*const f32> {
        if self.dtype != DataType::Float32 {
            return Err(TrtError::Shape(format!(
                "tensor is {:?}, not Float32", self.dtype
            )));
        }
        Ok(self.ptr as *const f32)
    }

    /// Resolved shape (after dynamic-shape inference).
    pub fn shape(&self) -> &[i64] { &self.shape }

    /// Dimension `i` of the resolved shape.
    pub fn dim(&self, i: usize) -> i64 { self.shape[i] }

    pub fn dtype(&self) -> DataType { self.dtype }
    pub fn byte_len(&self) -> usize { self.byte_len }
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
/// `Session` is `Send` but **not `Sync`** — `IExecutionContext` is not thread-safe.
/// For concurrent inference create multiple sessions from one `Arc<Engine>`.
pub struct Session {
    ctx: *mut btrt_context_t,
    _engine: Arc<Engine>,
    stream: Stream,
    inputs: HashMap<String, TensorState>,
    outputs: HashMap<String, TensorState>,
    _not_sync: std::marker::PhantomData<std::cell::UnsafeCell<()>>,
}

unsafe impl Send for Session {}

impl Session {
    /// The CUDA stream this session enqueues work on.
    pub fn stream(&self) -> &Stream { &self.stream }

    /// Names of all output tensors (in engine order).
    pub fn output_shape_names(&self) -> Vec<String> {
        self.outputs.keys().cloned().collect()
    }

    /// Resolved shape of a named output tensor (valid after at least one inference).
    pub fn output_shape(&self, name: &str) -> Option<&[i64]> {
        self.outputs.get(name).map(|s| s.shape.as_slice())
    }

    /// Byte length of a named output tensor's device buffer.
    pub fn output_byte_len(&self, name: &str) -> Option<usize> {
        self.outputs.get(name).map(|s| s.buf.len_bytes)
    }

    /// Create a session that shares `cuda_stream` with other pipeline stages.
    ///
    /// All device-buffer allocations and TRT enqueue calls use the provided
    /// stream instead of creating a private one.  The caller is responsible
    /// for syncing the stream (the [`Pipeline`](crate::Pipeline) does this).
    pub fn with_stream(engine: Arc<Engine>, cuda_stream: Arc<CudaStream>) -> Result<Self> {
        Self::init(engine, Stream::from_cuda_stream(cuda_stream))
    }

    /// Create a new inference session for the given engine (private stream).
    pub fn new(engine: Arc<Engine>) -> Result<Self> {
        // Retain the primary CUDA context (same context TRT uses internally).
        let cuda_ctx = CudaContext::new(0)
            .map_err(|e| TrtError::Cuda { code: e.0 as i32, msg: "CudaContext" })?;
        let stream = Stream::new(&cuda_ctx)?;
        Self::init(engine, stream)
    }

    fn init(engine: Arc<Engine>, stream: Stream) -> Result<Self> {
        // Guard the raw context so it is destroyed on every early-exit path
        // (e.g. a buffer allocation failure below).
        struct CtxGuard(*mut btrt_context_t);
        impl Drop for CtxGuard {
            fn drop(&mut self) {
                if !self.0.is_null() { unsafe { btrt_context_destroy(self.0) } }
            }
        }

        let ctx = unsafe { btrt_context_create(engine.as_ptr()) };
        if ctx.is_null() {
            return Err(TrtError::Create("ExecutionContext"));
        }
        let mut guard = CtxGuard(ctx);

        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        for spec in engine.specs() {
            let n_elems: i64 = spec.dims.iter()
                .filter(|&&d| d > 0)
                .product::<i64>()
                .max(1);
            let bytes_per_elem = dtype_bytes(spec.dtype);
            let buf = DeviceBuffer::alloc_with_stream(
                stream.cuda_stream(),
                n_elems as usize * bytes_per_elem,
            )?;
            let state = TensorState { buf, shape: spec.dims.clone(), dtype: spec.dtype };
            if spec.mode == TensorMode::Input {
                inputs.insert(spec.name.clone(), state);
            } else {
                outputs.insert(spec.name.clone(), state);
            }
        }

        guard.0 = std::ptr::null_mut(); // ownership transfers to Session::drop
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
        self.resize_output_buffers()?;
        Ok(())
    }

    /// Run inference with CPU inputs (H2D copy then enqueue).
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<HashMap<String, OutputTensor>> {
        for (name, data) in inputs {
            let c_name = CString::new(*name)
                .map_err(|_| TrtError::UnknownTensor((*name).into()))?;
            let state = self.inputs.get_mut(*name)
                .ok_or_else(|| TrtError::UnknownTensor((*name).into()))?;
            state.buf.copy_from_host(bytemuck_f32_to_u8(data), &self.stream)?;
            let dev_ptr = state.buf.as_device_ptr(&self.stream);
            let code = unsafe {
                btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), dev_ptr)
            };
            if code != 0 { return Err(TrtError::Trt(last_trt_error())); }
        }
        self.enqueue_and_collect()
    }

    /// Run inference with inputs **already in CUDA device memory** — no H2D copy.
    ///
    /// `device_inputs`: `(tensor_name, cuda_device_ptr, shape)`.
    ///
    /// # Safety
    /// Each `*mut c_void` must be a valid CUDA device pointer of the right size,
    /// alive until this function returns (after stream sync).
    pub unsafe fn run_device_inputs(
        &mut self,
        device_inputs: &[(&str, *mut std::ffi::c_void, &[i64])],
    ) -> Result<HashMap<String, OutputTensor>> {
        for (name, dev_ptr, shape) in device_inputs {
            let c_name = CString::new(*name)
                .map_err(|_| TrtError::UnknownTensor((*name).into()))?;
            let rc = btrt_context_set_input_shape(
                self.ctx, c_name.as_ptr(), shape.as_ptr(), shape.len() as i32,
            );
            if rc != 0 { return Err(TrtError::Trt(last_trt_error())); }
            let rc = btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), *dev_ptr);
            if rc != 0 { return Err(TrtError::Trt(last_trt_error())); }
        }
        self.resize_output_buffers()?;
        self.enqueue_and_collect()
    }

    /// Like `run_device_inputs` but leaves outputs in GPU memory.
    ///
    /// Returns a [`TensorView`] per output tensor: device pointer plus the
    /// resolved shape, dtype, and byte length.  The views remain valid until
    /// the next `run_*` call or `Session` drop (see [`TensorView`]).
    ///
    /// **Caller must call `session.stream().sync()` before reading the outputs.**
    ///
    /// # Safety
    /// Same as `run_device_inputs`.  Additionally the returned views alias
    /// Session-owned device memory — do not outlive the Session or hold them
    /// across a subsequent `run_*` call.
    pub unsafe fn run_device_inputs_on_device(
        &mut self,
        device_inputs: &[(&str, *mut std::ffi::c_void, &[i64])],
    ) -> Result<HashMap<String, TensorView>> {
        for (name, dev_ptr, shape) in device_inputs {
            let c_name = CString::new(*name)
                .map_err(|_| TrtError::UnknownTensor((*name).into()))?;
            let rc = btrt_context_set_input_shape(
                self.ctx, c_name.as_ptr(), shape.as_ptr(), shape.len() as i32,
            );
            if rc != 0 { return Err(TrtError::Trt(last_trt_error())); }
            let rc = btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), *dev_ptr);
            if rc != 0 { return Err(TrtError::Trt(last_trt_error())); }
        }
        self.resize_output_buffers()?;
        self.enqueue_outputs_only()
    }

    fn enqueue_outputs_only(&mut self) -> Result<HashMap<String, TensorView>> {
        for (name, state) in &self.outputs {
            let c_name = CString::new(name.as_str()).unwrap();
            let dev_ptr = state.buf.as_device_ptr(&self.stream);
            let code = unsafe {
                btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), dev_ptr)
            };
            if code != 0 { return Err(TrtError::Trt(last_trt_error())); }
        }

        let code = unsafe { btrt_context_enqueue_v3(self.ctx, self.stream.as_raw()) };
        if code != 0 { return Err(TrtError::Trt(last_trt_error())); }

        let mut result = HashMap::new();
        for (name, state) in &self.outputs {
            result.insert(name.clone(), TensorView {
                ptr:      state.buf.as_device_ptr(&self.stream),
                shape:    state.shape.clone(),
                dtype:    state.dtype,
                byte_len: state.buf.len_bytes,
            });
        }
        Ok(result)
    }

    fn enqueue_and_collect(&mut self) -> Result<HashMap<String, OutputTensor>> {
        for (name, state) in &self.outputs {
            let c_name = CString::new(name.as_str()).unwrap();
            let dev_ptr = state.buf.as_device_ptr(&self.stream);
            let code = unsafe {
                btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), dev_ptr)
            };
            if code != 0 { return Err(TrtError::Trt(last_trt_error())); }
        }

        let code = unsafe { btrt_context_enqueue_v3(self.ctx, self.stream.as_raw()) };
        if code != 0 { return Err(TrtError::Trt(last_trt_error())); }

        let mut raw_outputs: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, state) in &self.outputs {
            let mut buf = Vec::new();
            state.buf.copy_to_host(&mut buf, &self.stream)?;
            raw_outputs.insert(name.clone(), buf);
        }

        self.stream.sync()?;

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
        let names: Vec<String> = self.outputs.keys().cloned().collect();
        for name in names {
            let shape = self.resolved_output_shape(&name)?;
            let dtype = self.outputs[&name].dtype;
            let n: i64 = shape.iter().filter(|&&d| d > 0).product::<i64>().max(1);
            let new_len = n as usize * dtype_bytes(dtype);
            if self.outputs[&name].buf.len_bytes != new_len {
                self.outputs.get_mut(&name).unwrap().buf =
                    DeviceBuffer::alloc_with_stream(self.stream.cuda_stream(), new_len)?;
            }
            // Shape can change without the byte length changing (e.g. a
            // transposed dynamic profile) — always record the resolved shape.
            self.outputs.get_mut(&name).unwrap().shape = shape;
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
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

fn bytemuck_f32_to_u8(s: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
    }
}
