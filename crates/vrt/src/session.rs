use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use crate::{
    buffer::{DeviceBuffer, Stream},
    dtype::DType,
    engine::{DataType, Engine, TensorMode},
    error::{last_trt_error, Result, TrtError},
};
use cudarc::driver::{CudaContext, CudaStream};
use std::ffi::c_void;
use trt_sys::*;

/// Map an engine I/O [`DataType`] to a tensor [`DType`].
///
/// Int8/Bool fall back to `U8` (same 1-byte width); our models never emit them,
/// and `VrtTensor::f32_ptr` rejects any mismatched read regardless.
fn dtype_of(d: DataType) -> DType {
    match d {
        DataType::Float32 => DType::F32,
        DataType::Float16 => DType::F16,
        DataType::Int32 => DType::I32,
        DataType::Int8 | DataType::UInt8 | DataType::Bool => DType::U8,
    }
}

/// Borrowed device-side view of a TRT output: device pointer + resolved
/// shape/dtype/byte-length.
///
/// Aliases Session-owned output memory and is valid only until the next `run_*`
/// call or `Session` drop. Decode reads it **on-device** (kornia has no
/// borrowed-device-tensor constructor and the session reuses these buffers, so
/// outputs stay raw device views rather than owned kornia tensors).
pub struct OutputView {
    ptr: *mut c_void,
    shape: Vec<usize>,
    dtype: DType,
}

// SAFETY: a device pointer is a stable address; the holder serializes access via
// the per-frame stream sync, exactly as the old borrowed VrtTensor did.
unsafe impl Send for OutputView {}

impl OutputView {
    /// Shape as `i64` (TRT / decode convention).
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&d| d as i64).collect()
    }
    /// Device pointer as `*const f32`, checked against the output dtype — an
    /// `--fp16`-output engine fails loudly here instead of being misread.
    pub fn f32_ptr(&self) -> Result<*const f32> {
        if self.dtype != DType::F32 {
            return Err(TrtError::Shape(format!(
                "output is {:?}, not F32",
                self.dtype
            )));
        }
        Ok(self.ptr as *const f32)
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
/// `Session` is `Send` but **not `Sync`** — `IExecutionContext` is not thread-safe.
/// For concurrent inference create multiple sessions from one `Arc<Engine>`.
pub struct Session {
    ctx: *mut btrt_context_t,
    _engine: Arc<Engine>,
    stream: Stream,
    // Only OUTPUT buffers are session-owned; inputs are the caller's device
    // pointers, bound per call in `run_device_inputs_on_device`.
    outputs: HashMap<String, TensorState>,
    _not_sync: std::marker::PhantomData<std::cell::UnsafeCell<()>>,
}

unsafe impl Send for Session {}

impl Session {
    /// The CUDA stream this session enqueues work on.
    pub fn stream(&self) -> &Stream {
        &self.stream
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
        let cuda_ctx = CudaContext::new(0).map_err(|e| TrtError::Cuda {
            code: e.0 as i32,
            msg: "CudaContext",
        })?;
        // Single-stream usage — drop cudarc's cross-stream event tracking overhead.
        // SAFETY: all session buffers live on this one stream; none crosses streams.
        unsafe {
            cuda_ctx.disable_event_tracking();
        }
        let stream = Stream::new(&cuda_ctx)?;
        Self::init(engine, stream)
    }

    fn init(engine: Arc<Engine>, stream: Stream) -> Result<Self> {
        // Guard the raw context so it is destroyed on every early-exit path
        // (e.g. a buffer allocation failure below).
        struct CtxGuard(*mut btrt_context_t);
        impl Drop for CtxGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { btrt_context_destroy(self.0) }
                }
            }
        }

        let ctx = unsafe { btrt_context_create(engine.as_ptr()) };
        if ctx.is_null() {
            return Err(TrtError::Create("ExecutionContext"));
        }
        let mut guard = CtxGuard(ctx);

        // Allocate device buffers for OUTPUT tensors only; inputs are supplied by
        // the caller as device pointers at run time.
        let mut outputs = HashMap::new();
        for spec in engine.specs() {
            if spec.mode != TensorMode::Output {
                continue;
            }
            let n_elems: i64 = spec.dims.iter().filter(|&&d| d > 0).product::<i64>().max(1);
            let bytes_per_elem = dtype_bytes(spec.dtype);
            let buf = DeviceBuffer::alloc_with_stream(
                stream.cuda_stream(),
                n_elems as usize * bytes_per_elem,
            )?;
            outputs.insert(
                spec.name.clone(),
                TensorState {
                    buf,
                    shape: spec.dims.clone(),
                    dtype: spec.dtype,
                },
            );
        }

        guard.0 = std::ptr::null_mut(); // ownership transfers to Session::drop
        Ok(Self {
            ctx,
            _engine: engine,
            stream,
            outputs,
            _not_sync: std::marker::PhantomData,
        })
    }

    /// Set the runtime shape for a dynamic-shape input (call before `run`).
    pub fn set_input_shape(&mut self, name: &str, shape: &[i64]) -> Result<()> {
        let c_name = CString::new(name).map_err(|_| TrtError::UnknownTensor(name.into()))?;
        let code = unsafe {
            btrt_context_set_input_shape(
                self.ctx,
                c_name.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            )
        };
        if code != 0 {
            return Err(TrtError::Trt(last_trt_error()));
        }
        self.resize_output_buffers()?;
        Ok(())
    }

    /// Run inference with inputs **already in CUDA device memory**, leaving the
    /// outputs in GPU memory — no H2D/D2H copies, no sync.
    ///
    /// `device_inputs`: `(tensor_name, cuda_device_ptr, shape)`. Returns a
    /// borrowed [`OutputView`] per output tensor: device pointer plus the
    /// resolved shape, dtype, and byte length. The views remain valid until the
    /// next `run_*` call or `Session` drop.
    ///
    /// **Caller must call `session.stream().sync()` before reading the outputs.**
    ///
    /// # Safety
    /// This does NOT sync — it enqueues async work and returns. The GPU reads the
    /// bound input device pointers during the caller's later `stream().sync()`, so
    /// every input buffer must stay valid until that sync (not merely until this
    /// call returns). Additionally the returned views alias Session-owned device
    /// memory — do not outlive the Session or hold them across a subsequent
    /// `run_*` call.
    pub unsafe fn run_device_inputs_on_device(
        &mut self,
        device_inputs: &[(&str, *mut std::ffi::c_void, &[i64])],
    ) -> Result<HashMap<String, OutputView>> {
        for (name, dev_ptr, shape) in device_inputs {
            let c_name =
                CString::new(*name).map_err(|_| TrtError::UnknownTensor((*name).into()))?;
            let rc = btrt_context_set_input_shape(
                self.ctx,
                c_name.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            if rc != 0 {
                return Err(TrtError::Trt(last_trt_error()));
            }
            let rc = btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), *dev_ptr);
            if rc != 0 {
                return Err(TrtError::Trt(last_trt_error()));
            }
        }
        self.resize_output_buffers()?;
        self.enqueue_outputs_only()
    }

    fn enqueue_outputs_only(&mut self) -> Result<HashMap<String, OutputView>> {
        for (name, state) in &self.outputs {
            let c_name = CString::new(name.as_str()).unwrap();
            let dev_ptr = state.buf.as_device_ptr(&self.stream);
            let code =
                unsafe { btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), dev_ptr) };
            if code != 0 {
                return Err(TrtError::Trt(last_trt_error()));
            }
        }

        let code = unsafe { btrt_context_enqueue_v3(self.ctx, self.stream.as_raw()) };
        if code != 0 {
            return Err(TrtError::Trt(last_trt_error()));
        }

        let mut result = HashMap::new();
        for (name, state) in &self.outputs {
            // Borrows a Session-owned output buffer; the validity window
            // (until next run_* / Session drop) is the documented caller contract.
            let view = OutputView {
                ptr: state.buf.as_device_ptr(&self.stream),
                shape: state.shape.iter().map(|&d| d as usize).collect(),
                dtype: dtype_of(state.dtype),
            };
            result.insert(name.clone(), view);
        }
        Ok(result)
    }

    fn resolved_output_shape(&self, name: &str) -> Result<Vec<i64>> {
        let c_name = CString::new(name).unwrap();
        let mut dims = [0i64; 8];
        let mut ndims = 0i32;
        let code = unsafe {
            btrt_context_get_tensor_shape(self.ctx, c_name.as_ptr(), dims.as_mut_ptr(), &mut ndims)
        };
        if code != 0 {
            return Err(TrtError::UnknownTensor(name.into()));
        }
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
            unsafe {
                btrt_context_destroy(self.ctx);
            }
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
