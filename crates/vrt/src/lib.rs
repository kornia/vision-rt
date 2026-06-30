//! Safe, idiomatic Rust wrapper for TensorRT 10.x.
//!
//! # Usage
//! ```no_run
//! use std::sync::Arc;
//! use vrt::{Logger, Runtime, Engine, Session};
//! use vrt::logger::Severity;
//!
//! let logger  = Logger::new(Severity::Warning)?;
//! let runtime = Runtime::new(logger)?;
//! let engine  = Engine::from_file(runtime, "model.fp16.engine")?;
//! let mut session = Session::new(engine)?;
//!
//! let input = vec![0.0f32; 3 * 640 * 640];
//! let outputs = session.run(&[("images", &input)])?;
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
pub mod engine;
pub mod error;
pub mod image;
pub mod logger;
pub mod model;
pub mod pipeline;
pub mod runtime;
pub mod session;
pub mod tensor;

pub use buffer::{DeviceBuffer, PinnedBuffer, Stream};
pub use cudarc;
pub use cudarc::driver::CudaStream;
pub use engine::{DataType, Engine, TensorMode, TensorSpec};
pub use error::{Result, TrtError};
pub use image::{Format, VrtImage};
pub use logger::Logger;
pub use model::ModelSession;
pub use pipeline::{
    BoxError, Chain, ExecCtx, Fork, FrameMeta, Operator, Pipeline, PipelineTiming, Sink, Source,
    TRTensorMap, TrtInferStage,
};
pub use runtime::Runtime;
pub use session::{OutputTensor, Session};
pub use tensor::{DType, MemKind, VrtTensor};
pub use vrt_sys::TENSORRT_VERSION;
