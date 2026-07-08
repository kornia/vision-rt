---
name: model-tensor-semantics
description: Use when working on XFeat pre/post-processing, keypoints, descriptors, or matching — tensor shapes, coordinate spaces, normalization, NMS/top-K, and the exact algorithm steps implemented in vrt-xfeat.
---

# Model Tensor Semantics (XFeat)

## Data flow & types

XFeat is a plain `Image<u8,3> → XFeatResult` type (no pipeline/operator
framework). Device data crosses boundaries as **kornia** types + TRT views:

- **Input**: a kornia `Image<u8,3>` (device). `XFeat` owns a kornia
  `Preprocessor::letterbox` that writes a reused `Tensor<f32,4>` `[1,3,H,W]`
  (the backbone input).
- **Backbone I/O**: `ModelSession::run(&Tensor<f32,4>) -> TRTensorMap`; outputs
  read by name via `TRTensorMap::get("descriptors"|"heatmap"|"reliability")` →
  `OutputView::f32_ptr()` (dtype-checked device pointer, valid until the next
  `run`). See `crates/vrt-xfeat/src/model.rs`.
- **Result**: `XFeatResult` holds device `kpts`/`descs`/`scores` + a host
  `count`; `kpts_to_host`/`scores_to_host` do explicit D2H.

## Coordinate spaces — the #1 source of bugs

1. **Frame space** — source image pixels (any resolution the caller hands in).
2. **Model space** — letterboxed/padded backbone input (H,W multiples of 32).
   **XFeat keypoints come out in THIS space.**

The caller maps model→frame if it needs source coords. In `examples/xfeat_match`
the source is pre-resized to the model size, so the internal letterbox is the
identity and kpts already align with the drawn image — don't assume that in
general; account for the letterbox scale + pad offset when mapping back.

## Preprocessing (kornia `Preprocessor::letterbox`)

- Output: CHW FP32 `[1,3,H,W]`, values **/255 → [0,1]** (no mean/std).
- H and W must be multiples of 32 — XFeat downsamples ×8 and the TRT shape
  profile assumes it. `XFeatParams { top_k, threshold, h, w }` sets H/W.

## XFeat backbone outputs (TRT engine, FP32 on device)

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `descriptors` | (1, 64, H/8, W/8) | dense 64-D feature map |
| `heatmap`     | (1, 1, H, W)      | keypoint confidence |
| `reliability` | (1, 1, H, W)      | per-pixel reliability weight |

Engine MUST expose exactly those three output names (`model.rs` errors with
`MissingOutput` otherwise).

## Post-processing (`crates/vrt-xfeat/src/postprocess.rs`, NVRTC-JIT kernels)

1. `xfeat_score_nms` — 5×5 local-max NMS; score = heatmap×reliability, zeroed
   below `params.threshold` or if any neighbour is greater.
2. GPU top-K, no CPU round trip: `xfeat_topk_histogram` bins survivor scores,
   `xfeat_topk_cutoff` finds the score cutoff for ~K survivors, `xfeat_topk_select`
   atomically gathers survivors ≥ cutoff, capped at `params.top_k`. Approximate
   only at the boundary bucket (1024 bins). Output is atomic-append order,
   **NOT score-sorted**.
3. `xfeat_sample_descs` — bilinear-sample 64-D descriptors at kpt/8 positions,
   **align_corners=False** (matches PyTorch `grid_sample`).
4. `xfeat_l2_norm` — in-place L2-normalize each descriptor row.

Async contract: `submit()` enqueues preproc → backbone → NMS → top-K with **no**
sync; `run()` does the single `stream.synchronize()` then `finish_topk` to
assemble `XFeatResult`. Device buffers are **capacity `top_k`**; the valid count
is `scores.len()` — bound all device-buffer access by it.

Matching: `match_mutual_nn_gpu` — cosine similarity (valid because descriptors
are L2-normalized, so dot = cosine), mutual nearest-neighbor via two calls of one
tiled argmax kernel (`xfeat_match_argmax`, one thread per query, candidates tiled
through shared memory), with a min-similarity cutoff. CPU fallback: `match_mutual_nn`.

## When validating XFeat changes

- Sanity: static scene ≈ stable keypoint count frame-to-frame; kpts cluster on
  corners/texture, empty sky/walls ≈ none.
- GPU vs CPU: `cargo test -p vrt-xfeat --release -- --ignored` runs
  `gpu_match_agrees_with_cpu_reference` + `gpu_topk_selects_correct_keypoints`.
- Wrong-normalization symptom: keypoints "almost work" with low scores — check
  `/255` happened exactly once (not zero, not twice).
