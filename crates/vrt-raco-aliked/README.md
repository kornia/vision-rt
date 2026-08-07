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
published ground truth are shipped as examples.

### Oxford/VGG affine — planar, ground-truth homographies

15 pairs; `bark`/`boat` are rotation+zoom, `graf` is viewpoint. A match is an inlier only
if `H·left` lands within 3 px of `right`. Prepare with `scripts/get_oxford.sh` +
`examples/prep_oxford`, then run `examples/eval_oxford`:

| pair | RaCo-ALIKED + LightGlue+ | RaCo-ALIKED + mutual-NN | XFeat + mutual-NN |
|---|---|---|---|
| bark/2 | 476, **96.9%** | 253, 23.6% | 838, 53.4% |
| bark/3 | 158, **88.8%** | 0, 0.0% | 0, **0.0%** |
| bark/4 | 154, **91.1%** | 1, 0.1% | 0, **0.0%** |
| bark/5 | 142, **96.6%** | 18, 1.9% | 5, **0.6%** |
| bark/6 | 0, 0.0% | 0, 0.0% | 0, 0.0% |
| boat/2 | 692, **98.9%** | 1967, 92.6% | 1716, 71.9% |
| boat/4 | 424, **94.2%** | 0, 0.0% | 0, **0.0%** |
| graf/3 | 477, 84.7% | 615, 51.2% | 774, 51.2% |
| graf/6 | 257, **84.8%** | 7, 1.1% | 36, 4.8% |
| **total inliers** | **5157** | 5072 | 6072 |

### IMC 2021 phototourism — 3D scenes, ground-truth poses, stratified by difficulty

Oxford is planar: one homography maps every pixel. Phototourism is not, so ground truth
can only say a match must lie on its **epipolar line**, and a match is an inlier when the
symmetric epipolar distance under `F = K2⁻ᵀ[t]ₓR K1⁻¹` is under 3 px. That is a *weaker*
test — a match on the right line at the wrong depth passes — so these numbers are not
comparable to the Oxford ones. 90 pairs from `reichstag`, `sacre_coeur` and
`st_peters_square`, six per scene per co-visibility band (0.1 = barely overlapping):

| co-vis | pairs | RaCo-ALIKED + LightGlue+ | RaCo-ALIKED + mutual-NN | XFeat + mutual-NN |
|---|---|---|---|---|
| 0.1 | 18 | **85.2%** | 36.6% | 16.7% |
| 0.2 | 18 | **90.7%** | 54.3% | 36.1% |
| 0.3 | 18 | **90.4%** | 59.9% | 44.6% |
| 0.4 | 18 | **90.6%** | 63.1% | 49.3% |
| 0.5 | 18 | **91.9%** | 67.0% | 52.7% |
| **all** | **90** | **90.2%** (20700 inl) | 58.2% (50304) | 43.5% (35292) |

Prepare with `scripts/prep_imc.py` (metadata only — it converts the dataset's HDF5
calibration and .npy pair lists to text and never touches an image), then run
`examples/eval_imc`, which loads and resizes through `kornia_io`/`kornia_imgproc`.

### What the two benchmarks agree on

- **The rotation claim holds on real images.** On Oxford's `bark` — the rotation sequence
  — XFeat scores 0.0 / 0.0 / 0.6% where RaCo-ALIKED + LightGlue holds 88.8 / 91.1 / 96.6%.
- **LightGlue buys precision, not recall.** Columns 1 and 2 use *identical* RaCo keypoints
  and *identical* ALIKED 128-D descriptors; only the matcher differs, so column 2 tests
  the descriptors alone. On Oxford mutual-NN finds nearly as many true correspondences
  (5072 vs 5157) and buries them in outliers.
- **Total inliers alone is misleading.** XFeat has the most on Oxford (6072) and is the
  least usable, because precision is what a pose solver needs.
- **Difficulty separates them.** Across the co-visibility bands LightGlue decays
  gracefully (91.9% → 85.2%) while mutual-NN falls off a cliff (67.0% → 36.6%) and XFeat
  falls further (52.7% → 16.7%). Easy pairs hide this, which is why the bands are
  reported separately rather than as one mean.

> Mutual-NN's similarity gate matters enormously and is descriptor-specific: at the
> XFeat-tuned 0.82 it returns **zero** matches on 11 of the 15 Oxford pairs; ungated it
> reaches 5072 inliers at poor precision. There is no setting that recovers both.

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
