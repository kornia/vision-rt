// TRT bridge: runtime, engine, execution context, and CUDA helpers.
// Compiled directly against TRT C++ headers — no hand-written abstract class
// subclassing needed here. autocxx-generated types are used for type resolution
// at compile time; the actual function bodies call the TRT C++ API directly.
//
// When TensorRT updates:
//   1. Install new TRT headers (apt)
//   2. Run `cargo build -p trt-sys` — fix any compile errors here
//   3. Most changes will be method renames or signature changes in the
//      IRuntime/ICudaEngine/IExecutionContext interfaces.
//   See UPDATING.md for the full checklist.

#include "NvInferRuntime.h"
#include "cuda_runtime_api.h"

#include "../include/trt_bridge.h"

#include <string>

// Shared thread-local error storage — defined in logger_shim.cpp.
extern thread_local std::string g_last_error;

static void set_error(const char* msg) {
    g_last_error = msg ? msg : "";
}

static void clear_error() {
    g_last_error.clear();
}

// ── ShimRuntime ──────────────────────────────────────────────────────────────

struct ShimRuntime {
    nvinfer1::IRuntime* rt{nullptr};
};

// ── ShimEngine ───────────────────────────────────────────────────────────────

struct ShimEngine {
    nvinfer1::ICudaEngine* engine{nullptr};
};

// ── ShimContext ──────────────────────────────────────────────────────────────

struct ShimContext {
    nvinfer1::IExecutionContext* ctx{nullptr};
};

extern "C" {

// ── Runtime API ──────────────────────────────────────────────────────────────

// TRT API: createInferRuntime(ILogger&) — NvInferRuntime.h
btrt_runtime_t* btrt_runtime_create(btrt_logger_t* logger) {
    clear_error();
    if (!logger) {
        set_error("btrt_runtime_create: null logger");
        return nullptr;
    }
    try {
        // Get the ILogger* from the logger shim via the pure-C accessor.
        nvinfer1::ILogger* ilogger =
            reinterpret_cast<nvinfer1::ILogger*>(btrt_logger_get_ilogger(logger));
        if (!ilogger) {
            set_error("btrt_runtime_create: null ilogger");
            return nullptr;
        }
        nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(*ilogger);
        if (!rt) {
            set_error("btrt_runtime_create: createInferRuntime returned null");
            return nullptr;
        }
        auto* sr = new ShimRuntime{rt};
        return reinterpret_cast<btrt_runtime_t*>(sr);
    } catch (std::exception const& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("btrt_runtime_create: unknown exception");
        return nullptr;
    }
}

// TRT API: delete IRuntime — NvInferRuntime.h
void btrt_runtime_destroy(btrt_runtime_t* rt) {
    if (!rt) return;
    auto* sr = reinterpret_cast<ShimRuntime*>(rt);
    delete sr->rt;
    delete sr;
}

// ── Engine API ───────────────────────────────────────────────────────────────

// TRT API: IRuntime::deserializeCudaEngine(const void* blob, std::size_t size) — NvInferRuntime.h
btrt_engine_t* btrt_engine_deserialize(btrt_runtime_t* rt,
                                        const void* blob, size_t len) {
    clear_error();
    if (!rt || !blob || len == 0) {
        set_error("btrt_engine_deserialize: invalid arguments");
        return nullptr;
    }
    try {
        auto* sr = reinterpret_cast<ShimRuntime*>(rt);
        nvinfer1::ICudaEngine* engine = sr->rt->deserializeCudaEngine(blob, len);
        if (!engine) {
            set_error("btrt_engine_deserialize: deserializeCudaEngine returned null");
            return nullptr;
        }
        auto* se = new ShimEngine{engine};
        return reinterpret_cast<btrt_engine_t*>(se);
    } catch (std::exception const& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("btrt_engine_deserialize: unknown exception");
        return nullptr;
    }
}

// TRT API: delete ICudaEngine — NvInferRuntime.h
void btrt_engine_destroy(btrt_engine_t* engine) {
    if (!engine) return;
    auto* se = reinterpret_cast<ShimEngine*>(engine);
    delete se->engine;
    delete se;
}

// TRT API: ICudaEngine::getNbIOTensors() — NvInferRuntime.h
int32_t btrt_engine_num_io_tensors(btrt_engine_t* engine) {
    if (!engine) return -1;
    auto* se = reinterpret_cast<ShimEngine*>(engine);
    return se->engine->getNbIOTensors();
}

// TRT API: ICudaEngine::getIOTensorName(int32_t index) — NvInferRuntime.h
const char* btrt_engine_io_tensor_name(btrt_engine_t* engine, int32_t idx) {
    if (!engine) return nullptr;
    auto* se = reinterpret_cast<ShimEngine*>(engine);
    return se->engine->getIOTensorName(idx);
}

// TRT API: ICudaEngine::getTensorIOMode(const char*) — NvInferRuntime.h
int32_t btrt_engine_tensor_io_mode(btrt_engine_t* engine, const char* name) {
    if (!engine || !name) return -1;
    auto* se = reinterpret_cast<ShimEngine*>(engine);
    auto mode = se->engine->getTensorIOMode(name);
    return static_cast<int32_t>(mode);
}

// TRT API: ICudaEngine::getTensorDataType(const char*) — NvInferRuntime.h
int32_t btrt_engine_tensor_dtype(btrt_engine_t* engine, const char* name) {
    if (!engine || !name) return -1;
    auto* se = reinterpret_cast<ShimEngine*>(engine);
    auto dtype = se->engine->getTensorDataType(name);
    return static_cast<int32_t>(dtype);
}

// TRT API: ICudaEngine::getTensorShape(const char*) -> Dims64 — NvInferRuntime.h
int32_t btrt_engine_tensor_shape(btrt_engine_t* engine, const char* name,
                                  int64_t* out_dims, int32_t* out_ndims) {
    if (!engine || !name || !out_dims || !out_ndims) return -1;
    auto* se = reinterpret_cast<ShimEngine*>(engine);
    nvinfer1::Dims64 dims = se->engine->getTensorShape(name);
    if (dims.nbDims < 0) {
        *out_ndims = 0;
        return -1;
    }
    *out_ndims = dims.nbDims;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        out_dims[i] = dims.d[i];
    }
    return 0;
}

// ── Context API ──────────────────────────────────────────────────────────────

// TRT API: ICudaEngine::createExecutionContext() — NvInferRuntime.h
btrt_context_t* btrt_context_create(btrt_engine_t* engine) {
    clear_error();
    if (!engine) {
        set_error("btrt_context_create: null engine");
        return nullptr;
    }
    try {
        auto* se = reinterpret_cast<ShimEngine*>(engine);
        nvinfer1::IExecutionContext* ctx = se->engine->createExecutionContext();
        if (!ctx) {
            set_error("btrt_context_create: createExecutionContext returned null");
            return nullptr;
        }
        auto* sc = new ShimContext{ctx};
        return reinterpret_cast<btrt_context_t*>(sc);
    } catch (std::exception const& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("btrt_context_create: unknown exception");
        return nullptr;
    }
}

// TRT API: delete IExecutionContext — NvInferRuntime.h
void btrt_context_destroy(btrt_context_t* ctx) {
    if (!ctx) return;
    auto* sc = reinterpret_cast<ShimContext*>(ctx);
    delete sc->ctx;
    delete sc;
}

// TRT API: IExecutionContext::setInputShape(const char*, const Dims64&) — NvInferRuntime.h
int32_t btrt_context_set_input_shape(btrt_context_t* ctx, const char* name,
                                      const int64_t* dims, int32_t ndims) {
    if (!ctx || !name || !dims || ndims < 0 || ndims > nvinfer1::Dims::MAX_DIMS) return -1;
    try {
        auto* sc = reinterpret_cast<ShimContext*>(ctx);
        nvinfer1::Dims64 d{};
        d.nbDims = ndims;
        for (int32_t i = 0; i < ndims; ++i) {
            d.d[i] = dims[i];
        }
        bool ok = sc->ctx->setInputShape(name, d);
        return ok ? 0 : -1;
    } catch (std::exception const& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("btrt_context_set_input_shape: unknown exception");
        return -1;
    }
}

// TRT API: IExecutionContext::getTensorShape(const char*) — NvInferRuntime.h
int32_t btrt_context_get_tensor_shape(btrt_context_t* ctx, const char* name,
                                       int64_t* out_dims, int32_t* out_ndims) {
    if (!ctx || !name || !out_dims || !out_ndims) return -1;
    try {
        auto* sc = reinterpret_cast<ShimContext*>(ctx);
        nvinfer1::Dims64 dims = sc->ctx->getTensorShape(name);
        if (dims.nbDims < 0) {
            *out_ndims = 0;
            return -1;
        }
        *out_ndims = dims.nbDims;
        for (int32_t i = 0; i < dims.nbDims; ++i) {
            out_dims[i] = dims.d[i];
        }
        return 0;
    } catch (std::exception const& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("btrt_context_get_tensor_shape: unknown exception");
        return -1;
    }
}

// TRT API: IExecutionContext::setTensorAddress(const char*, void*) — NvInferRuntime.h
int32_t btrt_context_set_tensor_address(btrt_context_t* ctx,
                                         const char* name, void* device_ptr) {
    if (!ctx || !name) return -1;
    try {
        auto* sc = reinterpret_cast<ShimContext*>(ctx);
        bool ok = sc->ctx->setTensorAddress(name, device_ptr);
        return ok ? 0 : -1;
    } catch (std::exception const& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("btrt_context_set_tensor_address: unknown exception");
        return -1;
    }
}

// TRT API: IExecutionContext::enqueueV3(cudaStream_t) — NvInferRuntime.h
int32_t btrt_context_enqueue_v3(btrt_context_t* ctx, void* stream) {
    if (!ctx) return -1;
    try {
        auto* sc = reinterpret_cast<ShimContext*>(ctx);
        bool ok = sc->ctx->enqueueV3(static_cast<cudaStream_t>(stream));
        return ok ? 0 : -1;
    } catch (std::exception const& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("btrt_context_enqueue_v3: unknown exception");
        return -1;
    }
}

// ── CUDA helpers ──────────────────────────────────────────────────────────────
// Device memory + streams are owned on the Rust side (cudarc); only the pinned
// host-buffer path and the result D2H go through this bridge.

/* Page-locked (pinned) host memory, cacheable (flags=0 — NOT write-combined),
   so D2H copies into it are truly async AND host reads of the result are fast.
   cudarc's alloc_pinned uses WRITECOMBINED, which is the wrong trade for the
   download-then-read case. */
int32_t btrt_cuda_host_alloc(void** out_ptr, size_t bytes) {
    if (!out_ptr) return -1;
    cudaError_t err = cudaHostAlloc(out_ptr, bytes, cudaHostAllocDefault);
    return static_cast<int32_t>(err);
}

void btrt_cuda_host_free(void* ptr) {
    cudaFreeHost(ptr);
}

int32_t btrt_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                      static_cast<cudaStream_t>(stream));
    return static_cast<int32_t>(err);
}

} // extern "C"
