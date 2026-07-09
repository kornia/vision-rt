---
name: model-tensor-semantics
description: Use when working on XFeat pre/post-processing, keypoints, descriptors, or matching — tensor shapes, coordinate spaces, normalization, NMS/top-K, and the exact algorithm steps implemented in vrt-xfeat.
---

# Model Tensor Semantics (XFeat)

## Data flow & types

XFeat is a plain `Image<u8,3> → XFeatResult` type (no pipeline/operator
framework). Device data crosses boundaries as **kornia** types + TRT views:

- **Input**: a kornia `Image<u8,3>` (device), **any resolution**. `XFeat` owns a
  kornia `Preprocessor::stretch` and resizes each frame to its own floor-of-32
  dims `mh,mw = (H/32)*32, (W/32)*32` into a reused `Tensor<f32,4>` `[1,3,mh,mw]`
  (reallocated only when the size changes).
- **Backbone I/O**: `ModelSession::run(&Tensor<f32,4>) -> TRTensorMap`; outputs
  read by name via `TRTensorMap::get("descriptors"|"heatmap"|"reliability")` →
  `OutputView::f32_ptr()` (dtype-checked device pointer, valid until the next
  `run`). See `crates/vrt-xfeat/src/model.rs`.
- **VPI-style submit** (caller-owned output): pre-allocate an `XFeatResult` with
  `XFeat::alloc_result()`, then `xfeat.submit(&img, &mut result)` (async, no sync)
  → `stream.sync()` → read. `run()` = alloc + submit + sync convenience. Hold
  several results to keep multiple frames outstanding under one sync.
- **Result**: `XFeatResult` (device `kpts`/`descs`/`scores`, capacity `top_k`) —
  `count()` reads the pinned scalar (post-sync), `kpts_to_host` applies the
  `scale (rw,rh)` (returns original pixels), `scores_to_host` is a plain D2H.

## Coordinate spaces — the #1 source of bugs

This mirrors upstream XFeat exactly (`preprocess_tensor`):

1. **Original space** — the source image pixels the caller passes in.
2. **Model space** — the floor-of-32 resized backbone input. Keypoints are
   produced HERE on device (`XFeatResult.kpts`).

`XFeatResult.scale = (rw, rh) = (W/mw, H/mh)` maps model→original. `kpts_to_host`
applies it, so host keypoints are in **original pixels** (upstream's
`mkpts * [rw, rh]`). The resize is anisotropic but sub-32px, so aspect is
effectively preserved without any padding — no letterbox, no pad offset.
Descriptor matching uses descriptors only, so device `kpts` staying in model
space doesn't affect it.

## Preprocessing (kornia `Preprocessor::stretch`)

- Output: CHW FP32 `[1,3,mh,mw]`, values **/255 → [0,1]** (no mean/std),
  anisotropic resize to floor-of-32 (matches XFeat's `F.interpolate`).
- XFeat downsamples ×8, so mh,mw are forced to multiples of 32. `XFeatParams` is
  just `{ top_k, threshold }` — the input size is per-frame, not configured.

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

Matching lives in a **separate** `matching::Matcher` (module `crates/vrt-xfeat/src/matching.rs`),
decoupled from postproc but sharing the stream. Cosine similarity (descriptors
are L2-normalized, so dot = cosine), mutual nearest-neighbor via two calls of one
tiled argmax kernel (`xfeat_match_argmax`, one thread per query, candidates tiled
through shared memory), min-similarity cutoff. VPI-style: `submit_match(descs0,
n0, descs1, n1, cossim, &mut MatchResult)` (async) → sync → `MatchResult::pairs()`,
or sync one-shot `match_mutual_nn_gpu`.

## When validating XFeat changes

- Sanity: static scene ≈ stable keypoint count frame-to-frame; kpts cluster on
  corners/texture, empty sky/walls ≈ none.
- GPU vs CPU: `cargo test -p vrt-xfeat --release -- --ignored` runs
  `gpu_match_agrees_with_cpu_reference` + `gpu_topk_selects_correct_keypoints`.
- Wrong-normalization symptom: keypoints "almost work" with low scores — check
  `/255` happened exactly once (not zero, not twice).
