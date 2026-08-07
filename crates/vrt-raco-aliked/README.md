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

## Accuracy on real data — Oxford/VGG affine

The synthetic pair below is a sanity check, not evidence. On the **Oxford/VGG affine
benchmark** (15 pairs, real ground-truth homographies; `bark`/`boat` are rotation+zoom,
`graf` is viewpoint), a match counts as an inlier only if `H·left` lands within 3 px of
`right` (`examples/eval_oxford`):

| pair | RaCo + LightGlue+ | RaCo + mutual-NN | XFeat + mutual-NN |
|---|---|---|---|
| bark/2 | 486, **96.6%** | 288, 27.8% | 841, 53.3% |
| bark/3 | 174, **89.7%** | 0, 0.0% | 0, **0.0%** |
| bark/4 | 149, **91.4%** | 1, 0.1% | 0, **0.0%** |
| bark/5 | 143, **98.6%** | 15, 1.6% | 3, **0.4%** |
| bark/6 | 4, 0.0% | 0, 0.0% | 0, 0.0% |
| boat/2 | 713, **99.2%** | 1987, 92.9% | 1800, 73.8% |
| boat/4 | 437, **93.4%** | 0, 0.0% | 0, **0.0%** |
| graf/3 | 485, **84.8%** | 601, 49.8% | 772, 51.6% |
| graf/6 | 256, **84.2%** | 6, 0.9% | 25, 3.6% |
| **total inliers** | **5224** | 5120 | 6176 |

Three things this shows that the synthetic pair could not:

- **The rotation claim holds on real images.** On `bark` — the rotation sequence — XFeat
  scores 0.0 / 0.0 / 0.4% while RaCo + LightGlue holds 89.7 / 91.4 / 98.6%.
- **LightGlue buys precision, not recall.** Columns 1 and 2 use *identical* keypoints and
  descriptors; only the matcher differs. Mutual-NN finds about as many true
  correspondences (5120 vs 5224) and buries them in outliers. At 0.9% inliers a robust
  estimator has nothing to lock onto.
- **Total inliers alone is misleading.** XFeat has the most (6176) and is the least
  usable, because precision is what a pose solver needs.

RaCo + LightGlue holds 69–99% on 14 of 15 pairs. It fails only on `bark/6` — extreme
rotation and zoom at 6% overlap — where it returns 4 matches and admits it rather than
emitting confident nonsense.

> Mutual-NN's similarity gate matters enormously and is descriptor-specific: at the
> XFeat-tuned 0.82 it returns **zero** matches on 11 of these 15 pairs; ungated it
> reaches 5120 inliers at poor precision. There is no setting that recovers both.

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
