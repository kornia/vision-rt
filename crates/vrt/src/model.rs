//! [`ModelSession`] — the building block for TensorRT-backed pipeline operators.
//!
//! Wraps a [`Session`] with a safe, ergonomic inference call: hand it a
//! [`VrtTensor`] input, get back a typed [`TRTensorMap`] of device outputs.
//! No `unsafe`, no manual `setInputShape`/pointer wrangling, no re-deriving
//! shapes — so a new NN operator is mostly its pre/post-processing.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use vrt::{Engine, ModelSession, Operator, ExecCtx, VrtTensor, BoxError, CudaStream};
//! struct Depth { model: ModelSession }            // a new operator in ~20 lines
//!
//! impl Operator for Depth {
//!     type Input = VrtTensor;
//!     type Pending = VrtTensor;                    // the device depth map
//!     type Output = VrtTensor;
//!     fn enqueue(&mut self, frame: &VrtTensor, _: &ExecCtx) -> Result<VrtTensor, BoxError> {
//!         let out = self.model.run(frame)?;        // safe: no unsafe at the call site
//!         Ok(out.get("depth").ok_or("no 'depth' output")?.view())
//!     }
//!     fn finalize(&mut self, depth: VrtTensor, _: &ExecCtx) -> Result<VrtTensor, BoxError> {
//!         Ok(depth)                                // (real op would post-process here)
//!     }
//! }
//! ```

use std::sync::Arc;
use std::ffi::c_void;

use cudarc::driver::CudaStream;

use crate::engine::Engine;
use crate::runtime::Runtime;
use crate::session::{Session, OutputTensor};
use crate::buffer::Stream;
use crate::tensor::VrtTensor;
use crate::pipeline::TRTensorMap;
use crate::error::{Result, TrtError};

/// A TensorRT model bound to a CUDA stream — inference without the FFI sharp edges.
pub struct ModelSession {
    session: Session,
    inputs:  Vec<String>,
    outputs: Vec<String>,
}

impl ModelSession {
    /// Bind an engine to a shared pipeline stream.
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self> {
        let inputs  = engine.input_names();
        let outputs = engine.output_names();
        let session = Session::with_stream(engine, stream)?;
        Ok(Self { session, inputs, outputs })
    }

    /// Load an engine file and bind it to a fresh private stream.
    pub fn load(runtime: Arc<Runtime>, engine_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let engine  = Engine::from_file(runtime, engine_path)?;
        let inputs  = engine.input_names();
        let outputs = engine.output_names();
        let session = Session::new(engine)?;
        Ok(Self { session, inputs, outputs })
    }

    /// Input tensor names, in engine order.
    pub fn input_names(&self) -> &[String] { &self.inputs }
    /// Output tensor names, in engine order.
    pub fn output_names(&self) -> &[String] { &self.outputs }

    pub fn stream(&self) -> &Stream { self.session.stream() }
    pub fn cuda_stream(&self) -> Arc<CudaStream> { self.session.stream().cuda_stream().clone() }

    /// Run inference on a single device input, returning the device outputs.
    ///
    /// The common case for single-input models (the input name is auto-detected).
    /// Errors if the model has more than one input — use [`run_inputs`] then.
    ///
    /// [`run_inputs`]: ModelSession::run_inputs
    pub fn run(&mut self, input: &VrtTensor) -> Result<TRTensorMap> {
        match self.inputs.as_slice() {
            [name] => {
                let name = name.clone();
                self.run_inputs(&[(name.as_str(), input)])
            }
            names => Err(TrtError::Trt(format!(
                "model has {} inputs {:?}; use run_inputs() to bind by name",
                names.len(), names
            ))),
        }
    }

    /// Run inference binding each named device input → device outputs.
    ///
    /// Leaves outputs in GPU memory; the caller (the pipeline) syncs the stream
    /// once, then reads via the returned [`TRTensorMap`].
    pub fn run_inputs(
        &mut self,
        inputs: &[(&str, &VrtTensor)],
    ) -> Result<TRTensorMap> {
        // Own the shape vecs, then borrow them for the FFI binding slice.
        let owned: Vec<(&str, *mut c_void, Vec<i64>)> = inputs.iter()
            .map(|(n, t)| (*n, t.as_mut_ptr(), t.shape_i64()))
            .collect();
        let binds: Vec<(&str, *mut c_void, &[i64])> = owned.iter()
            .map(|(n, p, s)| (*n, *p, s.as_slice()))
            .collect();
        // SAFETY: every pointer comes from a live VrtTensor borrowed for this
        // call; run_device_inputs_on_device requires they stay valid until it
        // returns, which the borrow guarantees.
        let views = unsafe { self.session.run_device_inputs_on_device(&binds)? };
        Ok(TRTensorMap::new(views))
    }

    /// Synchronous host-input inference (H2D → enqueue → sync → D2H).
    ///
    /// For non-pipeline one-shot use; returns host-resident [`OutputTensor`]s.
    pub fn run_host(
        &mut self,
        inputs: &[(&str, &[f32])],
    ) -> Result<std::collections::HashMap<String, OutputTensor>> {
        self.session.run(inputs)
    }
}
