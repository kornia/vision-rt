# vrt-xfeat

XFeat keypoints + descriptors on Jetson: GPU letterbox preprocessing → TensorRT
backbone → GPU post-processing (NMS, top-K selection, descriptor sampling,
mutual-NN matching). Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

`XFeat` is a single `Image<u8,3> → XFeatResult` algorithm sharing one CUDA stream
(one sync per frame). Construct it whichever way fits:

- `XFeat::from_hub(stream, params)` — feature `hub`: pull pinned weights from
  Hugging Face (`kornia/xfeat`), build/cache the engine on-device, construct.
- `XFeat::from_onnx(path, stream, params)` — feature `hub` (trtexec build) or
  `builder` (in-process build): build/cache from a local ONNX.
- `XFeat::from_engine_file(path, stream, params)` — no feature: load a prebuilt
  `.engine`.
- `XFeat::new(engine, stream, params)` — pass an `Engine` you already own.

The post-processing CUDA kernels are NVRTC-JIT-compiled at runtime.

License: Apache-2.0
