use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrtError {
    #[error("failed to create {0}")]
    Create(&'static str),
    #[error("engine deserialization failed (TRT version/arch mismatch? engine must be built on this exact Jetson with the same TRT version): {0}")]
    Deserialize(String),
    #[error("tensor '{0}' not found in engine")]
    UnknownTensor(String),
    #[error("shape/dtype mismatch: {0}")]
    Shape(String),
    #[error("CUDA error code {code}: {msg}")]
    Cuda { code: i32, msg: &'static str },
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("nvrtc compile: {0}")]
    Nvrtc(String),
    #[error("TensorRT error: {0}")]
    Trt(String),
}

pub type Result<T> = std::result::Result<T, TrtError>;

// Helper: pull the thread-local last error from the shim.
pub(crate) fn last_trt_error() -> String {
    unsafe {
        let ptr = trt_sys::btrt_last_error();
        if ptr.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
