use std::ffi::CStr;
use std::sync::Arc;
use trt_sys::{btrt_logger_create, btrt_logger_destroy, btrt_logger_set_callback, btrt_logger_t};

/// TRT log severity (mirrors ILogger::Severity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    InternalError = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Verbose = 4,
}

/// Thread-safe TRT logger that routes messages to the `log` crate.
/// Must outlive everything — enforced by the `Arc<Logger>` chain in Runtime/Engine/Session.
pub struct Logger {
    ptr: *mut btrt_logger_t,
}

// SAFETY: The logger's only mutable state is the callback pointer (set once at
// construction) and the TRT `ILogger` virtual dispatch table. Both are
// thread-safe; TRT calls `log()` from a single TRT internal thread and the
// mutex inside ShimLogger serialises access. We own the pointer exclusively.
unsafe impl Send for Logger {}
unsafe impl Sync for Logger {}

extern "C" fn rust_log_callback(severity: i32, msg: *const std::os::raw::c_char) {
    // SAFETY: TRT guarantees msg is a valid null-terminated string for the
    // duration of this call. We do NOT store the pointer.
    let msg_str = if msg.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(msg).to_str().unwrap_or("(invalid utf8)") }
    };
    // Wrap in catch_unwind — this is called from C++ and must not panic/unwind.
    let _ = std::panic::catch_unwind(|| match severity {
        0 | 1 => log::error!("[TensorRT] {}", msg_str),
        2 => log::warn!("[TensorRT] {}", msg_str),
        3 => log::info!("[TensorRT] {}", msg_str),
        _ => log::debug!("[TensorRT] {}", msg_str),
    });
}

impl Logger {
    /// Create a new logger forwarding TRT messages to the `log` crate.
    /// `min_severity`: the minimum severity to forward (e.g. `Severity::Warning`).
    pub fn new(min_severity: Severity) -> crate::error::Result<Arc<Self>> {
        let ptr = unsafe { btrt_logger_create(min_severity as i32) };
        if ptr.is_null() {
            return Err(crate::error::TrtError::Create("Logger"));
        }
        unsafe {
            btrt_logger_set_callback(ptr, Some(rust_log_callback));
        }
        Ok(Arc::new(Self { ptr }))
    }

    pub(crate) fn as_ptr(&self) -> *mut btrt_logger_t {
        self.ptr
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        // SAFETY: ptr is valid and this is the sole owner. All Runtimes
        // (and transitively Engines/Contexts) hold Arc<Logger>, so this
        // Drop only fires after all of them have dropped — correct ordering.
        unsafe {
            btrt_logger_destroy(self.ptr);
        }
    }
}
