//! Safe, idiomatic Rust wrapper for TensorRT 10.x.
//!
//! # Usage
//! Load an engine, then run inference through [`ModelSession`] with a device
//! tensor on a shared CUDA stream (one async enqueue + one sync per call):
//! ```no_run
//! use vrt::{Logger, Runtime, Engine};
//! use vrt::logger::Severity;
//!
//! let logger  = Logger::new(Severity::Warning)?;
//! let runtime = Runtime::new(logger)?;
//! let engine  = Engine::from_file(runtime, "model.fp16.engine")?;
//! // let mut session = ModelSession::new(engine, stream)?;
//! // let out = session.run(&input_tensor)?;   // Tensor<f32,4> device input
//! # Ok::<(), vrt::error::TrtError>(())
//! ```
//!
//! # Thread safety
//! - `Engine` is `Send + Sync` — safe to share across threads.
//! - `Session` is `Send` but **not `Sync`** — IExecutionContext is not thread-safe.
//!   Create one `Session` per thread from a shared `Arc<Engine>`.

pub mod buffer;
#[cfg(feature = "builder")]
pub mod builder;
pub mod cuda;
pub mod dtype;
pub mod engine;
pub mod error;
pub mod logger;
pub mod model;
pub mod runtime;
pub mod session;

/// Boxed, thread-safe error — the convenient return type for algorithm
/// constructors that aggregate several error kinds.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub use buffer::{DeviceBuffer, PinnedBuffer, Stream};
pub use cudarc;
pub use cudarc::driver::CudaStream;
pub use dtype::DType;
pub use engine::{DataType, Engine, TensorMode, TensorSpec};
pub use error::{Result, TrtError};
pub use logger::Logger;
pub use model::{ModelSession, TRTensorMap};
pub use runtime::Runtime;
pub use session::{OutputView, Session};
pub use trt_sys::TENSORRT_VERSION;
