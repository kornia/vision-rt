use crate::{
    error::{Result, TrtError},
    logger::Logger,
};
use std::sync::Arc;
use trt_sys::{btrt_runtime_create, btrt_runtime_destroy, btrt_runtime_t};

/// Wraps `nvinfer1::IRuntime`. Safe to clone (Arc) and share across threads.
///
/// # Ownership
/// Holds `Arc<Logger>` to guarantee the logger outlives the runtime.
pub struct Runtime {
    ptr: *mut btrt_runtime_t,
    _logger: Arc<Logger>, // keeps logger alive; Drop ordering guaranteed
}

// SAFETY: IRuntime is thread-safe for read operations and engine
// deserialization. We hold the only reference to the raw pointer.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn new(logger: Arc<Logger>) -> Result<Arc<Self>> {
        let ptr = unsafe { btrt_runtime_create(logger.as_ptr()) };
        if ptr.is_null() {
            return Err(TrtError::Create("Runtime"));
        }
        Ok(Arc::new(Self {
            ptr,
            _logger: logger,
        }))
    }

    pub(crate) fn as_ptr(&self) -> *mut btrt_runtime_t {
        self.ptr
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: unique owner; all Engines (which hold Arc<Runtime>) have
        // dropped before this fires.
        unsafe {
            btrt_runtime_destroy(self.ptr);
        }
    }
}
