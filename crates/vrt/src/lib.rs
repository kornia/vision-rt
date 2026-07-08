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
pub mod camera;
pub mod cuda;
pub mod depth;
pub mod dtype;
pub mod engine;
pub mod error;
pub mod logger;
pub mod model;
pub mod runtime;
pub mod session;
pub mod stamp;

/// Boxed, thread-safe error — the convenient return type for algorithm
/// constructors that aggregate several error kinds.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub use buffer::{DeviceBuffer, PinnedBuffer, Stream};
pub use camera::Intrinsics;
pub use cudarc;
pub use cudarc::driver::CudaStream;
pub use depth::VrtDepthMap;
pub use dtype::DType;
pub use engine::{DataType, Engine, TensorMode, TensorSpec};
pub use error::{Result, TrtError};
pub use logger::Logger;
pub use model::{ModelSession, TRTensorMap};
pub use runtime::Runtime;
pub use session::{OutputTensor, OutputView, Session};
pub use stamp::{Clock, FrameMeta, MonotonicClock, Stamped};
pub use trt_sys::TENSORRT_VERSION;
