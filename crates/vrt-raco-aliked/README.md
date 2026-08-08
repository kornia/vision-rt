# vrt-raco-aliked

**RaCo** keypoint detection + **ALIKED** 128-D descriptors on TensorRT.

RaCo decides *where* to look — a rotation-robust detector with a learned ranker — and
ALIKED describes *what is there*. Versus [`vrt-xfeat`](../vrt-xfeat): ~8–16× slower, but it
still works under rotation, where XFeat collapses.

```rust
let mut raco = RaCoAliked::from_engine_file(engine, stream.clone())?;
let mut out  = raco.alloc_result()?;

raco.submit(&img, &mut out)?;   // async, no sync
stream.synchronize()?;          // the caller owns the one sync
let kpts  = out.keypoints_host()?;      // (x, y) in source pixels
let descs = out.descriptors_host()?;    // [K*128], L2-normalised
```

```text
images (B,3,H,W) f32   H,W multiples of 32, RGB in [0,1]
  -> keypoints            (B,K,2)   f32   model pixels
  -> normalized_keypoints (B,K,2)   f32   long-edge normalised, matcher input
  -> descriptors          (B,K,128) f32   L2-normalised
```

## Two ways to get this silently wrong

- **Feed RGB in `[0,1]` and nothing else.** The ImageNet mean/std live *inside* the graph,
  so this crate uses `Preprocessor::stretch` and must **not** apply `Normalize::imagenet()`.
- **`keypoints` and `normalized_keypoints` are not interchangeable.** Same shape, different
  spaces. Use `keypoints_host()` for geometry (it rescales to source pixels); pass
  `normalized_keypoints` to the matcher. RaCo normalises by the **long edge**, unlike
  SuperPoint/DISK.

## Choosing K

`K` is fixed at export and selects a **structurally different graph**: at K ≥ 3072 RaCo's
learned ranker — a second CNN over the image — is omitted entirely. Extraction is therefore
*non-monotonic* in K, while the matcher is O(K²) and moves the opposite way.

| K | ranker | extract/img | matcher/pair |
|---|---|---|---|
| 512 | dense | 49.1 ms | 7.9 ms |
| 1024 | boundary | 55.2 ms | 21.6 ms |
| **3072** | **bypass** | **28.5 ms** | 126.5 ms |

So use different K for each. The matcher accepts a result holding more keypoints than its
own `K` and matches the top-`K` prefix — **extract at k3072, match at k1024**:

| config | extract ×2 | match | E2E | 90° inliers |
|---|---|---|---|---|
| **k3072 ex + k1024 match** | 60.2 ms | 23.1 ms | **83.4 ms** | 878, **98.3%** |
| k1024 ex + k1024 match | 112.0 ms | 22.9 ms | 136.0 ms | 872, 97.8% |
| k3072 ex + k3072 match | 59.7 ms | 139.2 ms | 198.2 ms | 2297, 92.8% |

1.63× faster than k1024 throughout at equal-or-better accuracy; the ~52 ms saved is exactly
the ranker. Match at k3072 only if you want all ~2000 correspondences; use k512 only if
k3072 extraction will not fit.

## Getting the engine

Upstream publishes only a *fused* extractor+matcher graph, which cannot run under vrt at
all — int64 output, data-dependent shapes, no descriptors exposed.
`scripts/split_raco_pipeline.py` cuts it into two statically-shaped halves; the script
header explains where the cut lands and why.

```bash
python3 crates/vrt-raco-aliked/scripts/split_raco_pipeline.py \
    --input  models/onnx/raco/raco_aliked_lightglue_pipeline_k3072.onnx \
    --outdir models/onnx/raco

crates/vrt-raco-aliked/scripts/build_engine.sh \
    models/onnx/raco/raco_aliked_extractor_k3072.onnx
```

Or skip both: `RaCoAliked::from_hub(stream, 3072)` (feature `hub`) pulls the pinned ONNX —
and a prebuilt engine where one matches this box — from
[`kornia/raco-aliked`](https://huggingface.co/kornia/raco-aliked).

The split needs only `onnx`, so it runs on the Jetson's stock `python3`. Verify one with
`scripts/check_split_parity.py` (needs `onnxruntime`) before trusting it. The matcher half
is consumed by [`vrt-lightglue`](../vrt-lightglue).

## Benchmarks

Jetson Orin Nano, MAXN_SUPER, TRT 10.3.0.30, fp16, 640², engines built min=opt=max at that
resolution. Inliers are against a known ground-truth affine (within 2 px), *not*
self-consistency.

| rotation | RaCo k3072 | RaCo k1024 | XFeat + mutual-NN |
|---|---|---|---|
| 0° | 1913, 99.8% | 708, **100.0%** | 647, 98.0% |
| 45° | 1872, 94.3% | 690, **98.0%** | 261, 28.7% |
| 90° | 2297, 92.8% | 872, **97.8%** | 62, **0.0%** |

**On pure translation XFeat is nearly as good and ~16× cheaper — reaching for this crate
there is the wrong call.** It earns its cost only where orientation varies. Giving XFeat the
same keypoint budget does not rescue it (29–31% inliers at 45°); the gap is the model, not
the budget.

Two things will skew your own numbers: a **wide shape profile costs ~18–60%** (build
narrow), and **unrelated GPU load** inflated a first pass here from 110 ms to 235 ms on the
same input — measure on an idle box. Full harness: `vrt-lightglue`,
`examples/bench_vs_xfeat`. Single synthetic pair, so read the ordering rather than the exact
percentages.

## Accuracy on real data

The synthetic pair below is a sanity check, not evidence. Two real benchmarks with
published ground truth ship as examples.

> Measured 2026-08-08 on a Jetson Orin Nano (SM87, TensorRT 10.3.0.30, MAXN_SUPER),
> `raco-aliked-extractor-k3072` + `lightglue-matcher-k1024`, both fp16, images at long
> side 640. Every configuration below is reproducible with the commands given; no number
> here is carried over from an earlier run.

### Oxford/VGG affine — planar, ground-truth homographies

15 pairs; `bark`/`boat` are rotation+zoom, `graf` is viewpoint. A match is an inlier when
`H·left` lands within **3 px at original image resolution** — `prep_oxford` records the
uniform scale it applied and `eval_oxford` converts, so every sequence is scored under one
criterion rather than under whatever its own downscale factor happened to be. All three
columns run at the same k3072 keypoint budget. Format: inliers, precision.

| pair | rot | RaCo-ALIKED + LightGlue+ | RaCo-ALIKED + mutual-NN | XFeat + mutual-NN |
|---|---|---|---|---|
| bark/2 | −31° | 445, **91.6%** | 231, 21.6% | 601, 47.2% |
| bark/3 | **+150°** | 130, **79.8%** | 0, 0.0% | 0, **0.0%** |
| bark/4 | **−120°** | 128, **88.3%** | 0, 0.0% | 0, **0.0%** |
| bark/5 | −23° | 140, **94.6%** | 19, 2.0% | 1, 0.1% |
| bark/6 | +153° | 0, 0.0% | 0, 0.0% | 0, 0.0% |
| boat/2 | −14° | 684, **97.7%** | 1877, 88.4% | 1067, 59.7% |
| boat/3 | −40° | 591, **97.2%** | 241, 28.0% | 410, 36.5% |
| boat/4 | **−80°** | 405, **90.0%** | 0, 0.0% | 0, **0.0%** |
| boat/5 | +8° | 301, **90.9%** | 469, 52.5% | 74, 11.2% |
| boat/6 | −41° | 109, **59.6%** | 1, 0.2% | 7, 1.6% |
| graf/2 | −15° | 597, **94.3%** | 800, 61.4% | 808, 54.9% |
| graf/3 | +20° | 461, **81.9%** | 566, 47.2% | 500, 42.7% |
| graf/4 | −27° | 391, **82.1%** | 14, 2.1% | 187, 21.2% |
| graf/5 | +5° | 296, **84.6%** | 471, 51.8% | 214, 26.3% |
| graf/6 | +38° | 244, **80.5%** | 5, 0.8% | 17, 2.8% |
| **total** | | **4922** | 4694 | 3886 |

`rot` is the in-plane rotation recovered by polar decomposition of the ground-truth
homography — `bark` is a far harder rotation test than "rotation sequence" suggests.

### IMC 2021 phototourism — 3D scenes, ground-truth poses, stratified by difficulty

Oxford is planar: one homography maps every pixel. Phototourism is not, so ground truth
can only say a match must lie on its **epipolar line**, and a match is an inlier when its
**Sampson error** against `F = K2⁻ᵀ[t]ₓR K1⁻¹` is under 1 px (`kornia_3d::pose`). That is a
*weaker* test than Oxford's — a match on the right line at the wrong depth passes — so
these numbers are not comparable to the Oxford ones. Sampson mixes both images' frames, so
unlike Oxford this threshold is **not** converted to original resolution; it applies in the
640-long-side frame. 90 pairs from `reichstag`, `sacre_coeur` and `st_peters_square`, six
per scene per co-visibility band (0.1 = barely overlapping), sampled with a fixed seed.

| co-vis | pairs | RaCo-ALIKED + LightGlue+ | RaCo-ALIKED + mutual-NN | XFeat + mutual-NN |
|---|---|---|---|---|
| 0.1 | 18 | **87.0%** | 40.3% | 21.6% |
| 0.2 | 18 | **91.0%** | 56.9% | 39.4% |
| 0.3 | 18 | **89.7%** | 61.3% | 48.4% |
| 0.4 | 18 | **91.4%** | 68.5% | 54.4% |
| 0.5 | 18 | **92.1%** | 72.3% | 58.9% |
| **all** | **90** | **90.6%** (22642 inl) | 62.3% (57427) | 48.0% (39790) |

Loosening the threshold does not change the ordering, only the spread — at 3 px the totals
are 99.5% / 78.2% / 72.3%, at 5 px 99.9% / 82.2% / 79.5%.

### What the two benchmarks agree on

- **LightGlue wins on both counts, at equal budget.** With all three columns at k3072 it
  returns both the most correct correspondences on Oxford (4922 vs 4694 vs 3886) *and* the
  highest precision. An earlier revision of this table gave XFeat the most inliers; that
  was an artifact of running it at 4096 keypoints against RaCo's 3072 and of scoring in
  the downscaled frame. Both are fixed, and the conclusion reversed.
- **The rotation claim holds on real images, at large angles.** XFeat scores **0.0%** on
  every pair past ~79° — `bark/3` (+150°), `bark/4` (−120°), `boat/4` (−80°) — and is
  nonzero on every pair below it, while RaCo-ALIKED + LightGlue holds **79.8 / 88.3 /
  90.0%** on those three. What is *not* handled is extreme scale: `bark/6` combines 153°
  with a 4.2× zoom and every method returns nothing.
- **LightGlue buys precision, not recall.** Columns 1 and 2 use *identical* RaCo keypoints
  and *identical* ALIKED 128-D descriptors; only the matcher differs, so column 2 tests
  the descriptors alone. Mutual-NN finds nearly as many true correspondences (4694 vs
  4922) and buries them in outliers.
- **Difficulty separates them.** Across the co-visibility bands LightGlue decays
  gracefully (92.1% → 87.0%) while mutual-NN falls off a cliff (72.3% → 40.3%) and XFeat
  falls further (58.9% → 21.6%). Easy pairs hide this, which is why the bands are reported
  separately rather than as one mean.
- **Roughly where the literature sits, though not exactly comparable.** The IMC2021
  leaderboard's own ALIKED-2k + LightGlue entry reports a per-scene matching score at 3 px
  of 0.686–0.945. At *our* 3 px we measure 99.5%, above that range; at 1 px, 90.6%, inside
  it. The metrics are not the same quantity — different scenes (validation vs test),
  different keypoint budget, and their score is computed under its own thresholding — so
  treat this as an order-of-magnitude sanity check, not a like-for-like result.

> Mutual-NN's similarity gate is descriptor-specific and unforgiving: XFeat's tuned 0.82
> applied to ALIKED's 128-D descriptors returns **zero** matches on 10 of the 15 Oxford
> pairs. The two families therefore get separate CLI gates, both defaulting to ungated, so
> neither column is silently measured through a threshold picked for the other.

## Licences

Ships no weights. The ONNX combines three separately licensed upstreams, and **ALIKED is
BSD-3-Clause, which requires attribution in binary form** — reproduced in
[`LICENSE-NOTICE.md`](https://huggingface.co/kornia/raco-aliked/blob/main/LICENSE-NOTICE.md)
because it is absent upstream. Anything redistributing these engines must carry it too.

| Component | Source | Licence |
|---|---|---|
| RaCo detector + ranker | [`cvg/RaCo`](https://github.com/cvg/RaCo) | Apache-2.0 |
| ALIKED descriptors | [`Shiaoming/ALIKED`](https://github.com/Shiaoming/ALIKED) | **BSD-3-Clause** |
| LightGlue+ matcher | [`cvg/LightGlue`](https://github.com/cvg/LightGlue) | Apache-2.0 |
| ONNX export + TRT optimizations | [`fabio-sim/LightGlue-ONNX`](https://github.com/fabio-sim/LightGlue-ONNX) | Apache-2.0 |

**RaCo** — Shenoi, Lindenberger, Sarlin, Pollefeys, 3DV 2026,
[arXiv:2602.15755](https://arxiv.org/abs/2602.15755) · **ALIKED** — Zhao et al., IEEE TIM
2023 · **LightGlue** — Lindenberger, Sarlin, Pollefeys, ICCV 2023.

The TensorRT graph optimizations in these files are fabio-sim's work, preserved verbatim by
the split; this crate adds none of its own.
