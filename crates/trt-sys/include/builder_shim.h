#ifndef BTRT_BUILDER_SHIM_H
#define BTRT_BUILDER_SHIM_H

/* In-process ONNX -> serialized engine builder (feature = "builder").
   Wraps IBuilder + nvonnxparser + buildSerializedNetwork (TRT 10).
   Links libnvonnxparser in addition to libnvinfer. */

#include "logger_shim.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build a serialized engine from an ONNX file.
 *
 * logger:          must outlive this call.
 * onnx_path:       path to the .onnx file (external-data sidecars are
 *                  resolved relative to it by the parser).
 * fp16:            nonzero enables BuilderFlag::kFP16.
 * input_name:      name of the (single) dynamic input to attach an
 *                  optimization profile to, or NULL for static-shape models.
 * min/opt/max:     profile dims (length ndims each); ignored when
 *                  input_name is NULL.
 * workspace_bytes: IBuilderConfig memory-pool limit (kWORKSPACE).
 * out_blob/out_len: on success receive a malloc'd serialized engine;
 *                  free with btrt_blob_free.
 *
 * Returns 0 on success; nonzero on failure (details via btrt_last_error()
 * and the logger).
 *
 * TRT API: createInferBuilder, createNetworkV2(0) (explicit batch only),
 * nvonnxparser::createParser + parseFromFile, IOptimizationProfile,
 * setMemoryPoolLimit(kWORKSPACE), buildSerializedNetwork — NvInfer.h,
 * NvOnnxParser.h. All objects destroyed with `delete` (TRT 10). */
int32_t btrt_build_engine_from_onnx(
    btrt_logger_t* logger,
    const char*    onnx_path,
    int32_t        fp16,
    const char*    input_name,
    const int64_t* min_dims,
    const int64_t* opt_dims,
    const int64_t* max_dims,
    int32_t        ndims,
    int64_t        workspace_bytes,
    uint8_t**      out_blob,
    size_t*        out_len);

/* Free a blob returned by btrt_build_engine_from_onnx. */
void btrt_blob_free(uint8_t* blob);

#ifdef __cplusplus
}
#endif

#endif /* BTRT_BUILDER_SHIM_H */
