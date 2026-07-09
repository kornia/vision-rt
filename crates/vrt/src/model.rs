//! [`ModelSession`] — the building block for TensorRT-backed algorithms.
//!
//! Wraps a [`Session`] with a safe, ergonomic inference call: hand it a kornia
//! [`Tensor<f32, 4>`](kornia_tensor::Tensor) input, get back a typed
//! [`TRTensorMap`] of device outputs. No `unsafe`, no manual
//! `setInputShape`/pointer wrangling, no re-deriving shapes — so a new model is
//! mostly its pre/post-processing.
//!
//! ```no_run
//! # use kornia_tensor::Tensor;
//! # use vrt::ModelSession;
//! # fn run(model: &mut ModelSession, frame: &Tensor<f32, 4>) -> Result<Vec<i64>, vrt::BoxError> {
//! let out = model.run(frame)?;                  // safe: no unsafe at the call site
//! Ok(out.get("depth").ok_or("no 'depth' output")?.shape_i64())
//! # }
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_tensor::Tensor;
use std::collections::HashMap;

use crate::engine::Engine;
use crate::error::{Result, TrtError};
use crate::session::{OutputView, Session};

/// Device-side output map from a TRT inference: output tensor name → borrowed
/// [`OutputView`] (device pointer + resolved shape/dtype/byte-length).
///
/// The views are valid until the owning session's next `run` (or drop) —
/// consume them within the same call, never store them across calls.
pub struct TRTensorMap(HashMap<String, OutputView>);

impl TRTensorMap {
    pub fn new(views: HashMap<String, OutputView>) -> Self {
        Self(views)
    }

    /// Borrowed device view for a named output.
    pub fn get(&self, name: &str) -> Option<&OutputView> {
        self.0.get(name)
    }
}

/// A TensorRT model bound to a CUDA stream — inference without the FFI sharp edges.
pub struct ModelSession {
    session: Session,
    inputs: Vec<String>,
}

impl ModelSession {
    /// Bind an engine to a shared pipeline stream.
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self> {
        let inputs = engine.input_names();
        let session = Session::with_stream(engine, stream)?;
        Ok(Self { session, inputs })
    }

    /// Run inference on a single device input, returning the device outputs.
    ///
    /// The common case for single-input models (the input name is auto-detected).
    /// Errors if the model has more than one input — use [`run_inputs`] then.
    ///
    /// [`run_inputs`]: ModelSession::run_inputs
    pub fn run(&mut self, input: &Tensor<f32, 4>) -> Result<TRTensorMap> {
        match self.inputs.as_slice() {
            [name] => {
                let name = name.clone();
                self.run_inputs(&[(name.as_str(), input)])
            }
            names => Err(TrtError::Trt(format!(
                "model has {} inputs {:?}; use run_inputs() to bind by name",
                names.len(),
                names
            ))),
        }
    }

    /// Run inference binding each named device input → device outputs.
    ///
    /// Leaves outputs in GPU memory; the caller (the pipeline) syncs the stream
    /// once, then reads via the returned [`TRTensorMap`].
    pub fn run_inputs(&mut self, inputs: &[(&str, &Tensor<f32, 4>)]) -> Result<TRTensorMap> {
        // Own the shape vecs, then borrow them for the FFI binding slice. The
        // input is a device-resident kornia tensor; `as_ptr()` is its cached
        // device pointer (not host-dereferenceable, bound straight into TRT).
        let owned: Vec<(&str, *mut c_void, Vec<i64>)> = inputs
            .iter()
            .map(|(n, t)| {
                (
                    *n,
                    t.as_ptr() as *mut c_void,
                    t.shape.iter().map(|&d| d as i64).collect(),
                )
            })
            .collect();
        let binds: Vec<(&str, *mut c_void, &[i64])> = owned
            .iter()
            .map(|(n, p, s)| (*n, *p, s.as_slice()))
            .collect();
        // SAFETY: each pointer is the device buffer of an input tensor the CALLER
        // owns. run_device_inputs_on_device enqueues async work without syncing,
        // so the GPU reads these pointers during the caller's *later*
        // `stream().sync()` — the inputs must stay valid until then. Callers
        // (e.g. RfDetr/XFeat holding `self.input`) keep the tensor alive across
        // the sync, so the contract holds.
        let views = unsafe { self.session.run_device_inputs_on_device(&binds)? };
        Ok(TRTensorMap::new(views))
    }
}
