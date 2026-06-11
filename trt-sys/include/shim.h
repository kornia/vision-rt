#ifndef BTRT_SHIM_H
#define BTRT_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handles ─────────────────────────────────────────────────────── */
typedef struct btrt_logger_s   btrt_logger_t;
typedef struct btrt_runtime_s  btrt_runtime_t;
typedef struct btrt_engine_s   btrt_engine_t;
typedef struct btrt_context_s  btrt_context_t;

/* ── Logger ─────────────────────────────────────────────────────────────── */

/* Callback type: Rust registers one via btrt_logger_set_callback.
   severity: 0=INTERNAL_ERROR, 1=ERROR, 2=WARNING, 3=INFO, 4=VERBOSE (ILogger::Severity)
   msg: null-terminated UTF-8 string owned by TRT (caller must NOT free) */
typedef void (*btrt_log_fn)(int32_t severity, const char* msg);

/* Create a logger. min_severity: minimum severity to forward (0=all, 3=INFO+, 4=VERBOSE).
   TRT API: ILogger — NvInferRuntimeBase.h */
btrt_logger_t* btrt_logger_create(int32_t min_severity);

/* Register Rust log callback. Safe to call multiple times.
   The callback MUST be panic-safe (use catch_unwind on the Rust side). */
void btrt_logger_set_callback(btrt_logger_t* logger, btrt_log_fn callback);

/* Destroy the logger. MUST be called AFTER the runtime (and engine, context) is destroyed.
   TRT API: ILogger — NvInferRuntimeBase.h */
void btrt_logger_destroy(btrt_logger_t* logger);

/* ── Runtime ─────────────────────────────────────────────────────────────── */

/* Create an IRuntime. logger must outlive this runtime.
   Returns NULL on failure. Check btrt_last_error() for details.
   TRT API: createInferRuntime(ILogger&) — NvInferRuntime.h */
btrt_runtime_t* btrt_runtime_create(btrt_logger_t* logger);

/* Destroy runtime. MUST be called AFTER all engines derived from it are destroyed.
   TRT API: delete IRuntime — NvInferRuntime.h */
void btrt_runtime_destroy(btrt_runtime_t* rt);

/* ── Engine ──────────────────────────────────────────────────────────────── */

/* Deserialize a pre-built .engine blob into an ICudaEngine.
   blob/len: the raw bytes from the .engine file.
   Returns NULL on failure (version/arch mismatch, truncated file, etc.).
   TRT API: IRuntime::deserializeCudaEngine(const void*, size_t) — NvInferRuntime.h */
btrt_engine_t* btrt_engine_deserialize(btrt_runtime_t* rt,
                                        const void* blob, size_t len);

/* Destroy engine. MUST be called AFTER all contexts derived from it are destroyed.
   TRT API: delete ICudaEngine — NvInferRuntime.h */
void btrt_engine_destroy(btrt_engine_t* engine);

/* Number of I/O tensors (inputs + outputs combined).
   TRT API: ICudaEngine::getNbIOTensors() — NvInferRuntime.h */
int32_t btrt_engine_num_io_tensors(btrt_engine_t* engine);

/* Name of the i-th I/O tensor. Returned pointer is valid for the engine's lifetime.
   TRT API: ICudaEngine::getIOTensorName(int32_t index) — NvInferRuntime.h */
const char* btrt_engine_io_tensor_name(btrt_engine_t* engine, int32_t idx);

/* I/O mode of tensor named `name`: 0=NONE, 1=INPUT, 2=OUTPUT.
   TRT API: ICudaEngine::getTensorIOMode(const char*) — NvInferRuntime.h */
int32_t btrt_engine_tensor_io_mode(btrt_engine_t* engine, const char* name);

/* Data type of tensor: 0=FLOAT32, 1=FLOAT16, 2=INT8, 3=INT32, 4=BOOL, 5=UINT8.
   TRT API: ICudaEngine::getTensorDataType(const char*) — NvInferRuntime.h */
int32_t btrt_engine_tensor_dtype(btrt_engine_t* engine, const char* name);

/* Shape of tensor. out_dims: caller-allocated array of at least 8 int64_t.
   out_ndims: filled with number of valid dims (0-8). Dims may be -1 for dynamic.
   Returns 0 on success, -1 if tensor not found.
   TRT API: ICudaEngine::getTensorShape(const char*) -> Dims64 — NvInferRuntime.h */
int32_t btrt_engine_tensor_shape(btrt_engine_t* engine, const char* name,
                                  int64_t* out_dims, int32_t* out_ndims);

/* ── Execution context ───────────────────────────────────────────────────── */

/* Create an IExecutionContext from the engine.
   TRT API: ICudaEngine::createExecutionContext() — NvInferRuntime.h */
btrt_context_t* btrt_context_create(btrt_engine_t* engine);

/* Destroy context.
   TRT API: delete IExecutionContext — NvInferRuntime.h */
void btrt_context_destroy(btrt_context_t* ctx);

/* Set the runtime shape for a dynamic-shape input tensor (must call before enqueue).
   dims / ndims: the shape to use at this inference.
   Returns 0 on success, -1 on failure.
   TRT API: IExecutionContext::setInputShape(const char*, const Dims64&) — NvInferRuntime.h */
int32_t btrt_context_set_input_shape(btrt_context_t* ctx, const char* name,
                                      const int64_t* dims, int32_t ndims);

/* Query the resolved output shape (after setInputShape). Fill same out_dims/out_ndims.
   TRT API: IExecutionContext::getTensorShape(const char*) — NvInferRuntime.h */
int32_t btrt_context_get_tensor_shape(btrt_context_t* ctx, const char* name,
                                       int64_t* out_dims, int32_t* out_ndims);

/* Bind a device pointer to a named tensor (input or output).
   device_ptr must be a valid CUDA device pointer of the right size.
   TRT API: IExecutionContext::setTensorAddress(const char*, void*) — NvInferRuntime.h */
int32_t btrt_context_set_tensor_address(btrt_context_t* ctx,
                                         const char* name, void* device_ptr);

/* Enqueue inference on the given CUDA stream (void* = cudaStream_t).
   Asynchronous — caller MUST synchronize the stream before reading outputs.
   Returns 0 on success.
   TRT API: IExecutionContext::enqueueV3(cudaStream_t) — NvInferRuntime.h */
int32_t btrt_context_enqueue_v3(btrt_context_t* ctx, void* stream);

/* ── CUDA helpers ─────────────────────────────────────────────────────────── */
/* These thin wrappers let Rust own device memory without pulling in cudarc. */

/* cudaStreamCreate. out_stream: receives the new stream handle (cast to void*).
   Returns 0 on cudaSuccess. */
int32_t btrt_cuda_stream_create(void** out_stream);

/* cudaStreamSynchronize. Returns 0 on success. */
int32_t btrt_cuda_stream_sync(void* stream);

/* cudaStreamDestroy. */
void btrt_cuda_stream_destroy(void* stream);

/* cudaMalloc. Returns 0 on success. */
int32_t btrt_cuda_malloc(void** out_ptr, size_t bytes);

/* cudaFree. */
void btrt_cuda_free(void* ptr);

/* cudaMemcpyAsync host->device. Returns 0 on success. */
int32_t btrt_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, void* stream);

/* cudaMemcpyAsync device->host. Returns 0 on success. */
int32_t btrt_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, void* stream);

/* ── Plugin initialization (optional) ────────────────────────────────────── */
/* Call BEFORE deserializing any engine that uses TensorRT plugin layers
   (e.g. EfficientNMS_TRT). Safe to call even if plugins aren't used.
   TRT API: initLibNvInferPlugins(void* logger, const char* libNamespace) — NvInferPlugin.h */
int32_t btrt_init_plugins(btrt_logger_t* logger);

/* ── Error reporting ─────────────────────────────────────────────────────── */
/* Returns the last error message from any btrt_* call (thread-local). May be "".
   Pointer valid until the next btrt_* call on this thread. */
const char* btrt_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* BTRT_SHIM_H */
