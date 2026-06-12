---
name: model-tensor-semantics
description: Use when working on model pre/post-processing, keypoints, descriptors, detections, or matching — XFeat and YOLO tensor shapes, coordinate spaces, normalization, NMS, and the exact algorithm steps implemented in vrt-xfeat and vrt-yolo.
---

# Model Tensor Semantics (XFeat & YOLO)

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
2. GPU `xfeat_compact_scores` — stream-compact NMS survivors (atomic
   append), then D2H only the survivors (tens of KB, not the full map);
   CPU selects K best (default 4096). This is why the stage has `finalize`.
3. GPU `xfeat_sample_descs` — bilinear sample 64-D descriptors at kpt/8
   positions, **align_corners=False** convention (matches PyTorch grid_sample).
4. GPU `xfeat_l2_norm` — in-place L2-normalize each descriptor row.

`XFeatResult`: `kpts` (device, K×2 model-space xy), `descs` (device, K×64,
L2-normalized), `scores` (host), `kpts_cpu` (host copy, free — kept pre-upload).

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
