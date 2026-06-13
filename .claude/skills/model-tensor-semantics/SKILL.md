---
name: model-tensor-semantics
description: Use when working on model pre/post-processing, keypoints, descriptors, detections, or matching — XFeat and YOLO tensor shapes, coordinate spaces, normalization, NMS, and the exact algorithm steps implemented in vrt-xfeat and vrt-yolo.
---

# Model Tensor Semantics (XFeat & YOLO)

## The two data types (post-2026-06 unification)

Device data crosses stage boundaries as one of two types in `vrt`:
- **`VrtTensor`** — dense N-D array: `shape` + element `strides` + `dtype` +
  `MemKind{Device,Unified,Imported}` + `byte_len`. Either owns its buffer
  (`owner: Some`, freed on drop) or borrows one (`owner: None`, e.g. a TRT
  session output valid only until the session's next `run_*`). Accessors:
  `dim(i)`, `shape()`/`shape_i64()`, `f32_ptr()` (dtype-checked), `as_ptr()`.
- **`Image`** — borrowed pitch-linear pixel surface: `width/height/pitch/
  Format/MemKind`. The camera-ingest input to the preprocessor (replaces the
  old `DeviceFrame`). `pitch` is bytes-per-row, ≥ width×bpp.

`NvmmFrame` (pre-import DMA-BUF descriptor) and host-side `OutputTensor`
are the only other carriers. The former `TRTensor`/`TensorView` split and
the 6-representation sprawl are gone.

## Coordinate spaces — the #1 source of bugs

Three spaces exist; always know which one a coordinate is in:

1. **Frame space** — camera pixels after VIC resize (e.g. 1280×720).
2. **Model space** — letterboxed/padded input (e.g. 1280×736 after pad32).
   XFeat keypoints come out in THIS space.
3. **Original space** — for YOLO, boxes are mapped back via `unletterbox`
   using the stored `LetterboxInfo { scale, pad_x, pad_y }`.

Viz code scales model→frame with `sx = fw/dst_w, sy = fh/dst_h`
(see rtsp_xfeat `save_kpts`). Skipping the letterbox pad offset when
mapping back is the classic off-by-pad bug.

## Preprocessing (vrt-preproc, GPU kernel `letterbox_rgba_to_chw`)

- Input: RGBA pitch-linear device buffer (NVMM import), read via
  `cudaTextureObject_t` (bilinear hardware sampling).
- Output: CHW FP32 `[1,3,H,W]`, values **/255 → [0,1]** (no mean/std for
  either model). Padding value: grey `114/255` (YOLO convention; harmless
  for XFeat).
- H and W must be multiples of 32 (`pad32`) — XFeat downsamples ×8 and
  TRT profiles assume it.

## XFeat (vrt-xfeat)

Backbone outputs (TRT engine, all FP32 on device):

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `descriptors` | (1, 64, H/8, W/8) | dense 64-D feature map |
| `heatmap`     | (1, 1, H, W)      | keypoint confidence |
| `reliability` | (1, 1, H, W)      | per-pixel reliability weight |

Postproc algorithm (postprocess.rs):
1. GPU `xfeat_score_nms` — 5×5 local-max NMS; score = heatmap×reliability,
   zeroed below `threshold` (default 0.05) or if any neighbour is greater.
2. GPU top-K (all on device — no CPU round trip): `xfeat_topk_histogram`
   bins survivor scores, `xfeat_topk_cutoff` finds the score threshold for
   ~K survivors, `xfeat_topk_select` atomically gathers survivors ≥ cutoff
   capped at K (default 4096). Approximate only at the boundary bucket
   (1024 bins). Output is in atomic-append order, NOT score-sorted.
3. GPU `xfeat_sample_descs` — bilinear sample 64-D descriptors at kpt/8
   positions, **align_corners=False** convention (matches PyTorch grid_sample).
4. GPU `xfeat_l2_norm` — in-place L2-normalize each descriptor row.

The whole postproc is a pure async tail: `launch_topk` enqueues everything
+ async D2H (count/scores/xy) with NO sync; the pipeline's single per-frame
sync makes it readable, and `finish_topk` (in the operator's finalize)
assembles the result. `XFeatResult` device buffers are **capacity top_k**;
the valid count is `scores.len()`. `kpts`/`descs`/`scores`/`kpts_cpu` share
the GPU-select order (use `scores.len()` to bound device-buffer access).

Matching: `match_mutual_nn_gpu` — cosine similarity (valid because descriptors
are L2-normalized, so dot = cosine), mutual nearest-neighbor check via two
calls of one tiled argmax kernel (`xfeat_match_argmax` — one thread per
query, candidates tiled through shared memory), with min-similarity cutoff.

## YOLO11/v8 (vrt-yolo)

- Input: `images` `[1,3,640,640]`, [0,1] RGB, letterboxed with 114-grey pad.
- Output: `[1, 84, 8400]` — 84 = 4 box (cx,cy,w,h in model space) + 80 class
  scores. NO objectness in v8/11 (that's v5). `decode_output` auto-detects
  `[1,84,N]` vs `[1,N,84]` orientation.
- Postproc (CPU, in `finalize`): decode (max class score > conf threshold,
  default 0.25) → greedy IoU NMS (default 0.45) → unletterbox.
- CPU NMS is fine here: ≤ a few hundred candidates post-threshold.

## When validating model changes

- XFeat sanity: static scene ≈ stable keypoint count frame-to-frame; kpts
  cluster on corners/texture, empty sky/walls ≈ none.
- YOLO sanity: point a camera at a person → stable "person" ≥0.5 score
  (`rtsp_yolo` prints per-frame detections).
- Wrong normalization symptom: detections/keypoints "almost work" with low
  scores — check /255 happened exactly once (not zero, not twice).
