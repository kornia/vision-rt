#ifndef BTRT_LOGGER_SHIM_H
#define BTRT_LOGGER_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The Rust-facing log callback. severity matches ILogger::Severity integer values:
   0=INTERNAL_ERROR, 1=ERROR, 2=WARNING, 3=INFO, 4=VERBOSE */
typedef void (*btrt_log_fn)(int32_t severity, const char* msg);

typedef struct btrt_logger_s btrt_logger_t;

/* Create a TRT ILogger subclass that forwards to the Rust callback.
   min_severity: minimum level to forward (0=all, 2=WARNING+, 3=INFO+). */
btrt_logger_t* btrt_logger_create(int32_t min_severity);

/* Register the Rust log callback. Must be wrapped in catch_unwind on Rust side. */
void btrt_logger_set_callback(btrt_logger_t* logger, btrt_log_fn callback);

/* Get the underlying nvinfer1::ILogger* as void* (for passing to TRT factory functions). */
void* btrt_logger_get_ilogger(btrt_logger_t* logger);

/* Destroy. Must outlive all TRT objects created with this logger. */
void btrt_logger_destroy(btrt_logger_t* logger);

/* Thread-local last error string. */
const char* btrt_last_error(void);


#ifdef __cplusplus
}
#endif

#endif /* BTRT_LOGGER_SHIM_H */
