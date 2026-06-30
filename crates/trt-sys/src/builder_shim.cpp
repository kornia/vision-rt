// In-process ONNX -> serialized engine builder (feature = "builder").
//
// TRT 10 specifics honoured here:
// - createNetworkV2(0): explicit-batch is the only mode
// - setMemoryPoolLimit(kWORKSPACE, n) replaces setMaxWorkspaceSize
// - buildSerializedNetwork replaces buildEngineWithConfig
// - all objects destroyed with `delete` (no ->destroy())

#include "builder_shim.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

extern thread_local std::string g_last_error;

static void builder_set_error(const std::string& msg) {
    g_last_error = msg;
}

extern "C" int32_t btrt_build_engine_from_onnx(
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
    size_t*        out_len)
{
    if (!logger || !onnx_path || !out_blob || !out_len) {
        builder_set_error("btrt_build_engine_from_onnx: null argument");
        return -1;
    }
    *out_blob = nullptr;
    *out_len  = 0;

    try {
        auto* ilogger =
            reinterpret_cast<nvinfer1::ILogger*>(btrt_logger_get_ilogger(logger));
        if (!ilogger) {
            builder_set_error("btrt_build_engine_from_onnx: null ILogger");
            return -1;
        }

        std::unique_ptr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(*ilogger));
        if (!builder) {
            builder_set_error("createInferBuilder failed");
            return -2;
        }

        std::unique_ptr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(0));
        if (!network) {
            builder_set_error("createNetworkV2 failed");
            return -3;
        }

        std::unique_ptr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, *ilogger));
        if (!parser) {
            builder_set_error("nvonnxparser::createParser failed");
            return -4;
        }

        if (!parser->parseFromFile(
                onnx_path,
                static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::string msg = "ONNX parse failed for ";
            msg += onnx_path;
            for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
                msg += "; ";
                msg += parser->getError(i)->desc();
            }
            builder_set_error(msg);
            return -5;
        }

        std::unique_ptr<nvinfer1::IBuilderConfig> config(
            builder->createBuilderConfig());
        if (!config) {
            builder_set_error("createBuilderConfig failed");
            return -6;
        }

        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE,
            static_cast<std::size_t>(workspace_bytes));
        if (fp16) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        if (input_name && ndims > 0 && min_dims && opt_dims && max_dims) {
            if (ndims > nvinfer1::Dims::MAX_DIMS) {
                builder_set_error("input ndims exceeds Dims::MAX_DIMS");
                return -9;
            }
            nvinfer1::IOptimizationProfile* profile =
                builder->createOptimizationProfile();
            if (!profile) {
                builder_set_error("createOptimizationProfile failed");
                return -7;
            }
            nvinfer1::Dims dmin{}, dopt{}, dmax{};
            dmin.nbDims = dopt.nbDims = dmax.nbDims = ndims;
            for (int32_t i = 0; i < ndims; ++i) {
                dmin.d[i] = min_dims[i];
                dopt.d[i] = opt_dims[i];
                dmax.d[i] = max_dims[i];
            }
            bool ok = profile->setDimensions(
                          input_name, nvinfer1::OptProfileSelector::kMIN, dmin)
                   && profile->setDimensions(
                          input_name, nvinfer1::OptProfileSelector::kOPT, dopt)
                   && profile->setDimensions(
                          input_name, nvinfer1::OptProfileSelector::kMAX, dmax);
            if (!ok || config->addOptimizationProfile(profile) < 0) {
                builder_set_error(
                    std::string("optimization profile rejected for input '")
                    + input_name + "'");
                return -8;
            }
        }

        std::unique_ptr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized || serialized->size() == 0) {
            builder_set_error(
                "buildSerializedNetwork failed (see logger output)");
            return -9;
        }

        auto* blob = static_cast<uint8_t*>(std::malloc(serialized->size()));
        if (!blob) {
            builder_set_error("malloc failed for serialized engine");
            return -10;
        }
        std::memcpy(blob, serialized->data(), serialized->size());
        *out_blob = blob;
        *out_len  = serialized->size();
        return 0;
    } catch (const std::exception& e) {
        builder_set_error(std::string("exception during engine build: ") + e.what());
        return -11;
    } catch (...) {
        builder_set_error("unknown exception during engine build");
        return -12;
    }
}

extern "C" void btrt_blob_free(uint8_t* blob) {
    std::free(blob);
}
