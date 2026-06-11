// TENSORRT SHIM — compiled against TensorRT 10.3.x
//
// HOW TO UPDATE FOR A NEW TRT VERSION:
// 1. Install new TRT headers (apt). Confirm NV_TENSORRT_MAJOR in NvInferVersion.h.
// 2. Update TENSORRT_REQUIRED_MAJOR/MINOR/PATCH in build.rs.
// 3. Fix any compilation errors here (usually renamed/removed methods).
// 4. Run: cargo build -p trt-sys
// See UPDATING.md in the repo root for the full checklist.
//
// IMPORTANT TRT 10 CHANGES (vs TRT 8):
// - Use `delete obj` NOT `obj->destroy()` (destroy() removed in TRT 10)
// - Dims are Dims64 with int64_t d[8] (not int32_t as in TRT 8)
// - setMaxWorkspaceSize -> setMemoryPoolLimit(kWORKSPACE, n) (builder only)
// - enqueueV2 -> enqueueV3(stream) (named-tensor I/O, no binding indices)
// - kEXPLICIT_BATCH flag removed from createNetworkV2() (always explicit-batch)

#include "NvInferRuntime.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

#include "../include/shim.h"

#include <mutex>
#include <string>
#include <cstring>

// ── Thread-local error storage ───────────────────────────────────────────────

static thread_local std::string g_last_error;

static void set_error(const char* msg) {
    g_last_error = msg ? msg : "";
}

static void clear_error() {
    g_last_error.clear();
}

// ── ShimLogger ───────────────────────────────────────────────────────────────

// Wraps nvinfer1::ILogger and forwards log calls to an optional Rust callback.
struct ShimLogger {
    // Inner class that implements the TRT ILogger interface.
    class Inner final : public nvinfer1::ILogger {
    public:
        explicit Inner(int32_t min_severity) : m_min_severity(min_severity) {}

        // TRT API: ILogger::log(Severity, AsciiChar const*) — NvInferRuntimeBase.h
        void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
            int32_t sev = static_cast<int32_t>(severity);
            if (sev > m_min_severity) return;

            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_callback) {
                // Callback must be panic-safe; we catch C++ exceptions defensively.
                try {
                    m_callback(sev, msg);
                } catch (...) {
                    // Never propagate exceptions back into TRT.
                }
            }
        }

        void set_callback(btrt_log_fn cb) noexcept {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_callback = cb;
        }

        nvinfer1::ILogger* ilogger() noexcept { return this; }

    private:
        int32_t     m_min_severity;
        btrt_log_fn m_callback{nullptr};
        std::mutex  m_mutex;
    };

    explicit ShimLogger(int32_t min_severity)
        : inner(min_severity) {}

    Inner inner;
};

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

// ── Logger API ───────────────────────────────────────────────────────────────

extern "C" {

// TRT API: ILogger — NvInferRuntimeBase.h
btrt_logger_t* btrt_logger_create(int32_t min_severity) {
    clear_error();
    try {
        auto* sl = new ShimLogger(min_severity);
        return reinterpret_cast<btrt_logger_t*>(sl);
    } catch (std::exception const& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("btrt_logger_create: unknown exception");
        return nullptr;
    }
}

void btrt_logger_set_callback(btrt_logger_t* logger, btrt_log_fn callback) {
    if (!logger) return;
    auto* sl = reinterpret_cast<ShimLogger*>(logger);
    sl->inner.set_callback(callback);
}

// TRT API: ILogger — NvInferRuntimeBase.h
void btrt_logger_destroy(btrt_logger_t* logger) {
    if (!logger) return;
    auto* sl = reinterpret_cast<ShimLogger*>(logger);
    delete sl;
}

// ── Runtime API ──────────────────────────────────────────────────────────────

// TRT API: createInferRuntime(ILogger&) — NvInferRuntime.h
btrt_runtime_t* btrt_runtime_create(btrt_logger_t* logger) {
    clear_error();
    if (!logger) {
        set_error("btrt_runtime_create: null logger");
        return nullptr;
    }
    try {
        auto* sl = reinterpret_cast<ShimLogger*>(logger);
        nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(sl->inner);
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
        // nbDims == -1 signals tensor not found or unknown shape
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
    auto* sc = reinterpret_cast<ShimContext*>(ctx);
    nvinfer1::Dims64 d{};
    d.nbDims = ndims;
    for (int32_t i = 0; i < ndims; ++i) {
        d.d[i] = dims[i];
    }
    bool ok = sc->ctx->setInputShape(name, d);
    return ok ? 0 : -1;
}

// TRT API: IExecutionContext::getTensorShape(const char*) — NvInferRuntime.h
int32_t btrt_context_get_tensor_shape(btrt_context_t* ctx, const char* name,
                                       int64_t* out_dims, int32_t* out_ndims) {
    if (!ctx || !name || !out_dims || !out_ndims) return -1;
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
}

// TRT API: IExecutionContext::setTensorAddress(const char*, void*) — NvInferRuntime.h
int32_t btrt_context_set_tensor_address(btrt_context_t* ctx,
                                         const char* name, void* device_ptr) {
    if (!ctx || !name) return -1;
    auto* sc = reinterpret_cast<ShimContext*>(ctx);
    bool ok = sc->ctx->setTensorAddress(name, device_ptr);
    return ok ? 0 : -1;
}

// TRT API: IExecutionContext::enqueueV3(cudaStream_t) — NvInferRuntime.h
int32_t btrt_context_enqueue_v3(btrt_context_t* ctx, void* stream) {
    if (!ctx) return -1;
    auto* sc = reinterpret_cast<ShimContext*>(ctx);
    bool ok = sc->ctx->enqueueV3(static_cast<cudaStream_t>(stream));
    return ok ? 0 : -1;
}

// ── CUDA helpers ──────────────────────────────────────────────────────────────

int32_t btrt_cuda_stream_create(void** out_stream) {
    if (!out_stream) return -1;
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    if (err == cudaSuccess) {
        *out_stream = static_cast<void*>(s);
    }
    return static_cast<int32_t>(err);
}

int32_t btrt_cuda_stream_sync(void* stream) {
    cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    return static_cast<int32_t>(err);
}

void btrt_cuda_stream_destroy(void* stream) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

int32_t btrt_cuda_malloc(void** out_ptr, size_t bytes) {
    if (!out_ptr) return -1;
    cudaError_t err = cudaMalloc(out_ptr, bytes);
    return static_cast<int32_t>(err);
}

void btrt_cuda_free(void* ptr) {
    cudaFree(ptr);
}

int32_t btrt_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                      static_cast<cudaStream_t>(stream));
    return static_cast<int32_t>(err);
}

int32_t btrt_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                      static_cast<cudaStream_t>(stream));
    return static_cast<int32_t>(err);
}

// ── Plugin initialization ─────────────────────────────────────────────────────

// TRT API: initLibNvInferPlugins(void* logger, const char* libNamespace) — NvInferPlugin.h
int32_t btrt_init_plugins(btrt_logger_t* logger) {
    void* ilogger_ptr = nullptr;
    if (logger) {
        auto* sl = reinterpret_cast<ShimLogger*>(logger);
        ilogger_ptr = static_cast<void*>(sl->inner.ilogger());
    }
    bool ok = initLibNvInferPlugins(ilogger_ptr, "");
    return ok ? 0 : -1;
}

// ── Error reporting ──────────────────────────────────────────────────────────

const char* btrt_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
