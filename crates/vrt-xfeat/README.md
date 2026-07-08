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

## Model & credits

The weights are **XFeat** by Potje, Cadar, Araujo, Martins & Nascimento —
*"XFeat: Accelerated Features for Lightweight Image Matching"*, CVPR 2024.

- Upstream model + `xfeat.pt` weights: https://github.com/verlab/accelerated_features
- The `xfeat_backbone.onnx` shipped here is a **backbone-only** export of that
  model (image → descriptors / heatmap / reliability; NMS/TopK moved out of the
  graph so TensorRT can parse it), produced by `scripts/export_xfeat_backbone.py`.
- Hosted at the [`kornia/xfeat`](https://huggingface.co/kornia/xfeat) HF repo,
  alongside the original `xfeat.pt`.

This crate re-implements XFeat's inference/matching on TensorRT + CUDA; all model
credit belongs to the original authors. Please cite their paper when using it.

License: Apache-2.0 (this crate). See upstream for the original model's license.
