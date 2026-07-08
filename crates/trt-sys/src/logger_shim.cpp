// Logger shim: the ONLY hand-written C++ that requires abstract class subclassing.
// Everything else (runtime/engine/context) is in trt_bridge.cpp and calls
// TRT headers directly. When TensorRT updates:
//   1. Install new TRT headers (apt)
//   2. Run `cargo build -p trt-sys` — fix any compile errors here first
//   3. If ILogger::log() signature changed, fix the `log()` override below.
//   See UPDATING.md for the full checklist.

#include "NvInferRuntime.h"
#include "../include/logger_shim.h"

#include <mutex>
#include <string>

// Thread-local error storage. Also read by trt_bridge.cpp via the extern declaration.
thread_local std::string g_last_error;

extern "C" const char* btrt_last_error() {
    return g_last_error.c_str();
}

// ── ShimLogger ───────────────────────────────────────────────────────────────
// Subclasses nvinfer1::ILogger — the ONLY reason a C++ shim exists at all.
// Rust cannot implement abstract C++ classes directly.

struct ShimLogger {
    class Inner final : public nvinfer1::ILogger {
    public:
        explicit Inner(int32_t min_sev) : m_min_sev(min_sev), m_callback(nullptr) {}

        // TRT API: ILogger::log(Severity, AsciiChar const*) — NvInferRuntimeBase.h
        void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
            int32_t sev = static_cast<int32_t>(severity);
            if (sev > m_min_sev) return;
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_callback) {
                // NOTE: callback must be panic-safe (Rust side wraps in catch_unwind)
                try {
                    m_callback(sev, msg ? msg : "");
                } catch (...) {
                    // Never propagate exceptions back into TRT.
                }
            }
        }

        void set_callback(btrt_log_fn cb) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_callback = cb;
        }

        nvinfer1::ILogger* ilogger() { return this; }

    private:
        int32_t m_min_sev;
        btrt_log_fn m_callback;
        std::mutex m_mutex;
    };

    explicit ShimLogger(int32_t min_sev) : inner(new Inner(min_sev)) {}
    ~ShimLogger() { delete inner; }

    Inner* inner;
};

extern "C" {

btrt_logger_t* btrt_logger_create(int32_t min_severity) {
    try {
        return reinterpret_cast<btrt_logger_t*>(new ShimLogger(min_severity));
    } catch (...) {
        g_last_error = "btrt_logger_create: allocation failed";
        return nullptr;
    }
}

void btrt_logger_set_callback(btrt_logger_t* logger, btrt_log_fn callback) {
    if (!logger) return;
    reinterpret_cast<ShimLogger*>(logger)->inner->set_callback(callback);
}

void* btrt_logger_get_ilogger(btrt_logger_t* logger) {
    if (!logger) return nullptr;
    return reinterpret_cast<ShimLogger*>(logger)->inner->ilogger();
}

void btrt_logger_destroy(btrt_logger_t* logger) {
    delete reinterpret_cast<ShimLogger*>(logger);
}

} // extern "C"
