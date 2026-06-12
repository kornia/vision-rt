//! Safe, idiomatic Rust wrapper for TensorRT 10.x.
//!
//! # Usage
//! ```no_run
//! use std::sync::Arc;
//! use trt::{Logger, Runtime, Engine, Session};
//! use trt::logger::Severity;
//!
//! let logger  = Logger::new(Severity::Warning)?;
//! let runtime = Runtime::new(logger)?;
//! let engine  = Engine::from_file(runtime, "model.fp16.engine")?;
//! let mut session = Session::new(engine)?;
//!
//! let input = vec![0.0f32; 3 * 640 * 640];
//! let outputs = session.run(&[("images", &input)])?;
//! # Ok::<(), trt::error::TrtError>(())
//! ```
//!
//! # Thread safety
//! - `Engine` is `Send + Sync` — safe to share across threads.
//! - `Session` is `Send` but **not `Sync`** — IExecutionContext is not thread-safe.
//!   Create one `Session` per thread from a shared `Arc<Engine>`.

pub mod error;
pub mod logger;
pub mod runtime;
pub mod engine;
pub mod buffer;
pub mod session;
pub mod tensor;
pub mod pipeline;

pub use error::{TrtError, Result};
pub use logger::Logger;
pub use runtime::Runtime;
pub use engine::{Engine, TensorSpec, TensorMode, DataType};
pub use buffer::{DeviceBuffer, Stream};
pub use cudarc::driver::CudaStream;
pub use session::{Session, OutputTensor};
pub use tensor::{TRTensor, DType};
pub use pipeline::{Source, Stage, Chain, TRTensorMap, TrtInferStage, Pipeline, PipelineTiming, BoxError};
