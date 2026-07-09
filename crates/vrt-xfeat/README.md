# vrt-xfeat

XFeat keypoints + descriptors on Jetson: GPU resize/normalize (upstream XFeat's
floor-of-32 stretch) → TensorRT backbone → GPU post-processing (NMS, top-K
selection, descriptor sampling, L2-norm). Keypoints are returned in
original-image pixels. Mutual-NN matching is a separate [`Matcher`] (module
`matching`), so extraction and matching are decoupled but share one CUDA stream.
Part of the [`vision-rt`](https://github.com/kornia/vision-rt) workspace.

`XFeat` is a single `Image<u8,3> → XFeatResult` algorithm on one shared CUDA
stream. Construct it whichever way fits:

- `XFeat::from_hub(stream, params)` — feature `hub`: pull pinned weights from
  Hugging Face (`kornia/xfeat`), build/cache the engine on-device, construct.
- `XFeat::from_onnx(path, stream, params)` — feature `hub` (trtexec build) or
  `builder` (in-process build): build/cache from a local ONNX.
- `XFeat::from_engine_file(path, stream, params)` — no feature: load a prebuilt
  `.engine`.
- `XFeat::new(engine, stream, params)` — pass an `Engine` you already own.

The API is **fully async — the library never syncs for you** (VPI-style):

```rust
let mut res = xfeat.alloc_result()?;      // caller-owned output, reused
xfeat.submit(&image, &mut res)?;          // enqueue, returns immediately
stream.synchronize()?;                     // the caller owns the one sync
let kpts = res.kpts_to_host(&stream)?;     // original-image pixels
```

Match two results with `Matcher::new(stream)` → `submit_match(&a.descs, a.count(),
&b.descs, b.count(), cossim, &mut MatchResult)` → `stream.synchronize()` →
`MatchResult::pairs()`. All CUDA kernels are NVRTC-JIT-compiled at runtime.

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
