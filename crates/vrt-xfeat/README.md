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
- `XFeat::from_onnx_stereo` / `XFeat::from_hub_stereo` — same, with a batch-2
  engine for `submit_pair` (see Stereo pairs).
- `XFeat::from_engine_file(path, stream, params)` — no feature: load a prebuilt
  `.engine`.
- `XFeat::new(engine, stream, params)` — pass an `Engine` you already own.

The API is **fully async — the library never syncs for you** (VPI-style):

```rust
let mut res = xfeat.alloc_result()?;      // caller-owned output, reused
xfeat.submit(&image, &mut res)?;          // enqueue, returns immediately
stream.synchronize()?;                     // the caller owns the one sync
let kpts = res.kpts_to_host()?;     // original-image pixels
```

### Stereo pairs

`submit_pair` runs a same-size left/right pair as **one** batch-2 backbone run, with
every post-processing stage launched once for both. It needs a stereo engine built for
the camera's resolution (`from_onnx_stereo` / `from_hub_stereo`, or `stereo_shapes(w, h)`
for your own profile); the default engine is batch 1. A stereo engine serves only that
one size, for `submit` as well as `submit_pair`.

```rust
let mut xfeat = XFeat::from_onnx_stereo(onnx, stream.clone(), params, 640, 480)?;
let (mut l, mut r) = (xfeat.alloc_result()?, xfeat.alloc_result()?);
xfeat.submit_pair(&left, &right, &mut l, &mut r)?;
stream.synchronize()?;
```

Orin Nano (MAXN_SUPER), top_k 2048, p50 per pair (`examples/xfeat_stereo`):

| model size | 2× `submit`, default engine | `submit_pair`, stereo engine |
|---|---|---|
| 640×480 | 6.05 ms | 5.35 ms (1.13×) |
| 736×480 | 6.78 ms | 6.15 ms (1.10×) |

Opt-in because a batch-2 profile with a generic opt shape slows batch-1 runs 13–16%
and loses the pair's gain away from opt; sized to the camera it costs batch 1 nothing
— but use a separate `XFeat` for batch-1 calls: alternating `submit` and `submit_pair`
on one instance drains the stream at every switch.

Match two results with `Matcher::new(stream)` →
`submit(Descriptors::new(&a.descs, a.count(), a.desc_dim()), ..., cossim, &mut MatchResult)`
→ `stream.synchronize()` → `MatchResult::pairs()`. The descriptor width travels with the
buffer; see the breaking-change note below. All CUDA kernels are NVRTC-JIT-compiled at
runtime.

## Breaking change: `submit_match` is now `submit`

`Matcher::submit_match` and `LightGlue::submit_match` are both `submit`, matching every
other model in the workspace, and the descriptor arguments are now a single
`Descriptors::new(buf, count, dim)` instead of a `(&buf, count)` pair.

No deprecated forwarder is provided, deliberately. The old signature has nowhere to get
the descriptor width from, so a shim would have to assume the matcher's own — which is
exactly the tautology that let a 64-D buffer reach a 128-D kernel and return
plausible-looking nonsense. A compile error is the better outcome.

Known caller to update: `sensor-rt`, `crates/sensor-oak/examples/oakd_xfeat_stereo`
(line 232). That repo pins vision-rt by revision, so it keeps building until someone
repins it — **the repin and this rename must land together**, and nothing in this PR does
the sensor-rt half.

```rust
// before
matcher.submit_match(&l.descs, l.count(), &r.descs, r.count(), 0.82, &mut out)?;
// after — the width comes from whatever produced the descriptors
matcher.submit(
    Descriptors::new(&l.descs, l.count(), l.desc_dim()),
    Descriptors::new(&r.descs, r.count(), r.desc_dim()),
    0.82,
    &mut out,
)?;
```

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
