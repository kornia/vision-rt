//! Link-time stubs for `TRT_STUB` builds.
//!
//! Stub mode exists so hosted CI can `check` and `clippy` without TensorRT headers. It
//! could not `test`, because a test binary links — which left the workspace's GPU-free
//! tests (the match-guard arithmetic, the benchmark harness's geometry) with no gate at
//! all outside the self-hosted Jetson job, and that job is disabled. A regression in
//! either shipped green.
//!
//! These definitions resolve the symbols so those binaries link. Every one aborts if
//! called: a stub build must never reach TensorRT, and returning a plausible value would
//! convert a link error into a wrong answer.
#![allow(clippy::missing_safety_doc, unused_variables, unused_imports)]

use super::*;
use std::os::raw::{c_char, c_void};

fn unreachable_in_stub(name: &str) -> ! {
    panic!("{name} called in a TRT_STUB build — there is no TensorRT to call into");
}

#[no_mangle]
pub unsafe extern "C" fn btrt_logger_create(min_severity: i32) -> *mut btrt_logger_t {
    unreachable_in_stub("btrt_logger_create")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_logger_set_callback(
    logger: *mut btrt_logger_t,
    callback: btrt_log_fn,
) {
    unreachable_in_stub("btrt_logger_set_callback")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_logger_get_ilogger(
    logger: *mut btrt_logger_t,
) -> *mut ::std::os::raw::c_void {
    unreachable_in_stub("btrt_logger_get_ilogger")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_logger_destroy(logger: *mut btrt_logger_t) {
    unreachable_in_stub("btrt_logger_destroy")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_last_error() -> *const ::std::os::raw::c_char {
    unreachable_in_stub("btrt_last_error")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_runtime_create(logger: *mut btrt_logger_t) -> *mut btrt_runtime_t {
    unreachable_in_stub("btrt_runtime_create")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_runtime_destroy(rt: *mut btrt_runtime_t) {
    unreachable_in_stub("btrt_runtime_destroy")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_engine_deserialize(
    rt: *mut btrt_runtime_t,
    blob: *const ::std::os::raw::c_void,
    len: usize,
) -> *mut btrt_engine_t {
    unreachable_in_stub("btrt_engine_deserialize")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_engine_destroy(engine: *mut btrt_engine_t) {
    unreachable_in_stub("btrt_engine_destroy")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_engine_num_io_tensors(engine: *mut btrt_engine_t) -> i32 {
    unreachable_in_stub("btrt_engine_num_io_tensors")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_engine_io_tensor_name(
    engine: *mut btrt_engine_t,
    idx: i32,
) -> *const ::std::os::raw::c_char {
    unreachable_in_stub("btrt_engine_io_tensor_name")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_engine_tensor_io_mode(
    engine: *mut btrt_engine_t,
    name: *const ::std::os::raw::c_char,
) -> i32 {
    unreachable_in_stub("btrt_engine_tensor_io_mode")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_engine_tensor_dtype(
    engine: *mut btrt_engine_t,
    name: *const ::std::os::raw::c_char,
) -> i32 {
    unreachable_in_stub("btrt_engine_tensor_dtype")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_engine_tensor_shape(
    engine: *mut btrt_engine_t,
    name: *const ::std::os::raw::c_char,
    out_dims: *mut i64,
    out_ndims: *mut i32,
) -> i32 {
    unreachable_in_stub("btrt_engine_tensor_shape")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_context_create(engine: *mut btrt_engine_t) -> *mut btrt_context_t {
    unreachable_in_stub("btrt_context_create")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_context_destroy(ctx: *mut btrt_context_t) {
    unreachable_in_stub("btrt_context_destroy")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_context_set_input_shape(
    ctx: *mut btrt_context_t,
    name: *const ::std::os::raw::c_char,
    dims: *const i64,
    ndims: i32,
) -> i32 {
    unreachable_in_stub("btrt_context_set_input_shape")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_context_get_tensor_shape(
    ctx: *mut btrt_context_t,
    name: *const ::std::os::raw::c_char,
    out_dims: *mut i64,
    out_ndims: *mut i32,
) -> i32 {
    unreachable_in_stub("btrt_context_get_tensor_shape")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_context_set_tensor_address(
    ctx: *mut btrt_context_t,
    name: *const ::std::os::raw::c_char,
    device_ptr: *mut ::std::os::raw::c_void,
) -> i32 {
    unreachable_in_stub("btrt_context_set_tensor_address")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_context_enqueue_v3(
    ctx: *mut btrt_context_t,
    stream: *mut ::std::os::raw::c_void,
) -> i32 {
    unreachable_in_stub("btrt_context_enqueue_v3")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_cuda_host_alloc(
    out_ptr: *mut *mut ::std::os::raw::c_void,
    bytes: usize,
) -> i32 {
    unreachable_in_stub("btrt_cuda_host_alloc")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_cuda_host_free(ptr: *mut ::std::os::raw::c_void) {
    unreachable_in_stub("btrt_cuda_host_free")
}

#[no_mangle]
pub unsafe extern "C" fn btrt_cuda_memcpy_d2h(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    bytes: usize,
    stream: *mut ::std::os::raw::c_void,
) -> i32 {
    unreachable_in_stub("btrt_cuda_memcpy_d2h")
}
