# Architecture

`vision-rt` is a set of **plain Rust libraries** for real-time vision on Jetson —
no orchestration framework. You construct a type, then call methods in a loop.
The design borrows its async shape from NVIDIA **VPI**: a *payload* holds
pre-compiled state, *caller-owned buffers* are filled by `submit`, one
`stream.sync()` completes the work, then you read.

## Crate stack

```
trt-sys      pure-C shim over TensorRT C++ (bindgen never sees C++); TRT_STUB off-Jetson
   ↓
vrt          safe core: Logger→Runtime→Engine→Session (Arc chain), ModelSession,
             cuda launch cfgs, PinnedBuffer/Stream, error types
   ↓
vrt-hub      weights (HF Hub, sha256-pinned) + on-device EngineCache (onnx→engine,
   ↓         optional guarded prebuilt-engine download)
vrt-xfeat    XFeat: preprocess → backbone → postprocess (+ matching submodule)
```

## The one-stream / one-sync model

Everything for a frame runs on **one shared `CudaStream`**. `run()` = enqueue all
GPU work **async** → a single `cudaStreamSynchronize` → read. There are **no
hidden syncs** in the hot path: the backbone (`Session::run_device_inputs_on_device`),
preprocessing (kornia `Preprocessor`), and post-processing kernels all enqueue and
return; the count/keypoints reach the host via **async pinned D2H** that completes
at the caller's sync. The only `synchronize()` calls are the explicit ones the
caller controls.

## VPI-style API (payload + caller-owned output)

| Role | Type | Notes |
|------|------|-------|
| Payload (created once) | `XFeat`, `Matcher` | own kernels + scratch, reused every frame |
| Output buffer (caller-owned) | `XFeatResult`, `MatchResult` | pre-allocated (`alloc`/`alloc_result`), reused |
| Submit (async) | `xfeat.submit(&img, &mut result)` | writes into `result`, no sync |
| Sync | `stream.sync()` | one call covers all submitted work |
| Read | `result.count()`, `result.kpts_to_host()`, `m.pairs()` | valid after the sync |

`run()` is a convenience = alloc + submit + sync. Holding several result buffers
lets **multiple frames stay outstanding** under one sync (see `xfeat_match`).

## XFeat data flow (`vrt-xfeat`)

```
Image<u8,3> (any size, device)
  → Preprocessor::stretch      resize to floor-of-32 (mh,mw)=(H/32*32,W/32*32), /255   [upstream XFeat]
  → TRT backbone               descriptors (1,64,mh/8,mw/8), heatmap (1,1,mh,mw), reliability
  → xfeat_score_nms            5×5 NMS, score = heatmap×reliability
  → histogram → cutoff → select   GPU top-K (capacity = result's top_k), atomic-append order
  → xfeat_sample_descs → l2_norm  64-D descriptors, unit-norm
  → async D2H of the count scalar → result.count_pin
XFeatResult: device kpts/descs/scores + count + scale(rw,rh).
kpts_to_host applies scale → keypoints in ORIGINAL pixels (upstream's mkpts*[rw,rh]).
```

Matching is a **separate** concern (`matching::Matcher`): the `xfeat_match_argmax`
kernel run twice (both directions) gives mutual-NN pairs by cosine (dot, since
descriptors are unit-norm). It shares the extractor's stream, so extract+match is
one continuous flow. All CUDA kernels are NVRTC-JIT-compiled once (kornia
`CudaKernel`); arch is auto-detected (no hardcoded sm_87).

## Models & engines

ONNX is the portable artifact (HF `kornia/xfeat`, sha256-pinned). **Engines are
machine-locked** (TRT version + GPU arch) — built on-device by `EngineCache`
(cached under `~/.cache/vision-rt/engines/…`), or a prebuilt engine downloaded
**only** when its `trt_version`+`sm` match the local box (`ModelHub::get_engine`).
`trt-sys` parses the installed TRT version into `TENSORRT_VERSION` (feeds cache
keys) and warns if it's outside the tested 10.3.x range.

## Errors & safety

Per-crate `thiserror` enums (`TrtError`, `HubError`, `XFeatError`); `vrt::BoxError`
for constructors that aggregate. `Session` is `Send` but not `Sync` (one per
thread from a shared `Arc<Engine>`). Device pointers from TRT outputs are borrowed
`OutputView`s valid until the next `run` or the session drops — the single
per-frame sync serialises access.
