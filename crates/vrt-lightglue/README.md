# vrt-lightglue

**LightGlue+** transformer feature matching on TensorRT — two
[`vrt-raco-aliked`](../vrt-raco-aliked) results in, per-keypoint correspondences out,
entirely on the GPU.

```rust
// one shared stream: a single sync covers extraction AND matching
let mut raco = RaCoAliked::from_engine_file(extractor, stream.clone())?;
let mut glue = LightGlue::from_engine_file(matcher,  stream.clone())?;

raco.submit(&left,  &mut l)?;          // async
raco.submit(&right, &mut r)?;          // async
glue.submit(&l, &r, &mut m)?;    // async
stream.synchronize()?;                 // the caller owns the one sync

for (i0, i1) in m.pairs(0.0)? { /* left kp i0 <-> right kp i1 */ }
```

```text
normalized_keypoints (2P,1,K,2)   f32
descriptors          (2P,1,K,128) f32
  -> matches0 (P,K) i32   index into image 1, or -1 if unmatched
  -> mscores0 (P,K) f32   confidence in [0,1]
```

Splitting the matcher out of upstream's fused graph is the whole point: descriptors
extracted at *any* time — from a map, a relocalization database, a keyframe store — can be
matched against a live frame, instead of only the two images in one forward pass.

## Using it correctly

- **Pass `normalized_keypoints`, never `keypoints`.** Same shape, different spaces; mixing
  them degrades matching silently.
- **Pairs are interleaved** `[L0, R0, …]` and split *inside* the graph, so one pair is
  leading dim 2, not two inputs. `submit` packs both results on-device for you.
- **Build extractor and matcher on one shared stream.** A cross-stream result is rejected
  (`StreamMismatch`) rather than raced.
- **`pairs()` is the match count**, not `capacity()` — that returns the engine's fixed `K`.
- LightGlue already applies its own mutual-NN check and threshold inside the graph;
  `min_score` only tightens further, so `0.0` keeps everything.

## Matching cost is O(K²)

Unlike the extractor, the matcher punishes a large keypoint budget — and the two pull in
opposite directions, because RaCo's ranker is bypassed at K ≥ 3072.

| K | matcher/pair | extract ×2 | E2E pair |
|---|---|---|---|
| 512 | **7.9 ms** | 98.2 ms | **106.0 ms** |
| 1024 | 21.6 ms | 110.4 ms | 132.0 ms |
| 3072 | 126.5 ms | **57.0 ms** | 183.5 ms |

**Best of both: mix them.** `submit` accepts results holding *more* than the
matcher's `K` and uses the first `K`, which works because RaCo emits keypoints in
descending score order and both tensors are row-major — the prefix is the top-`K` and is
already contiguous. Extract with k3072 (ranker bypassed, cheap) and match with k1024:

| config | extract ×2 | match | E2E | 90° inliers |
|---|---|---|---|---|
| k3072 ex + k3072 match | 59.7 ms | 139.2 ms | 198.2 ms | 2297, 92.8% |
| **k3072 ex + k1024 match** | 60.2 ms | 23.1 ms | **83.4 ms** | 878, **98.3%** |
| k1024 ex + k1024 match | 112.0 ms | 22.9 ms | 136.0 ms | 872, 97.8% |

**1.63× faster than native k1024 at equal-or-better accuracy** — the ~52 ms it saves is
exactly RaCo's ranker, and the top-1024 by raw detector confidence turns out to match the
ranker's pick in quality here. All three rows measured in one run, so the comparison is on
equal footing; absolute values carry that run's load.

Use plain k512 only if you cannot extract at k3072 for memory reasons. Both halves must
still come from the same export family.

## Getting the engine

```bash
python3 crates/vrt-raco-aliked/scripts/split_raco_pipeline.py \
    --input  models/onnx/raco/raco_aliked_lightglue_pipeline_k3072.onnx \
    --outdir models/onnx/raco

crates/vrt-raco-aliked/scripts/build_engine.sh \
    models/onnx/raco/lightglue_matcher_k3072.onnx
```

Or `LightGlue::from_hub(stream, 3072)` (feature `hub`), which pulls the pinned ONNX — and a
prebuilt engine where one matches this box — from
[`kornia/lightglue`](https://huggingface.co/kornia/lightglue).

## Benchmark vs XFeat — `examples/bench_vs_xfeat`

Orin Nano, MAXN_SUPER, TRT 10.3.0.30, fp16, 640², K=1024, 30 iters. Both engines built
min=opt=max at the benchmark resolution.

| | extract ×2 | match | E2E |
|---|---|---|---|
| RaCo-ALIKED + LightGlue+ | 110.5 ms | 22.4 ms | **133.0 ms** |
| XFeat + mutual-NN | 6.8 ms | 0.9 ms | **7.7 ms** |

17.3× slower — and worth it only under rotation. Inliers against a ground-truth affine
(within 2 px): this pipeline holds 100 / 98.0 / 97.8% at 0 / 45 / 90°, XFeat gives
98.0 / 28.7 / **0.0%**. The benchmark sets XFeat's `top_k` from the RaCo engine's `K`, so
both run at the same budget; giving XFeat more keypoints does not rescue it.

```bash
cargo run --release -p vrt-lightglue --example bench_vs_xfeat -- \
    <raco.engine> <lightglue.engine> <xfeat.engine> left.png right.png 30 [gt_affine.txt]
```

Without a ground-truth affine it falls back to consistency against the pair's own median
displacement — a proxy that only rewards a matcher for agreeing with itself. Measure on an
idle box: unrelated GPU load inflated a first pass here by ~2×, and the example reports E2E
min alongside median so that shows up.

**Not comparable to the upstream blog post**, which reports a *speedup ratio* (2.67×) vs an
fp16 `torch.compile` baseline on an RTX 4080 Laptop, for the RaCo detector alone.

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

### Is the mutual-NN baseline handicapped?

A mutual-NN baseline with no similarity gate is low-precision by construction, so "the
matcher beats it" would be circular. Both gates were therefore swept, with LightGlue's
column as the control (it is invariant across every row below).

Oxford, macro-average precision — tuning the gate barely moves ALIKED:

| ALIKED gate | inliers | macro precision |
|---|---|---|
| 0.00 (ungated) | 4694 | 23.7% |
| 0.50 | 4536 | 24.7% |
| **0.70** | 2846 | **25.7%** |
| 0.80 | 1288 | 24.1% |
| 0.85 | 377 | 13.2% |
| 0.90 | 10 | 6.7% |

Best tuned mutual-NN reaches 25.7% against LightGlue's **80.9%**, and pays 40% of its
inliers to get there. XFeat's own sweep peaks at 25.0% (gate 0.90, half the inliers).

IMC is the case where the gate genuinely matters, and where the ungated number *was*
understating the baseline:

| configuration | inliers | micro | macro |
|---|---|---|---|
| ALIKED NN, ungated | 57427 | 62.3% | 57.7% |
| ALIKED NN, gate 0.85 | 16924 | **89.2%** | 71.4% |
| RaCo-ALIKED + LightGlue+ | **22642** | **90.6%** | **89.3%** |

At a tuned gate mutual-NN's pooled precision nearly reaches LightGlue's — but it returns
*fewer* correct correspondences doing so (16924 vs 22642), and its macro average stays 18
points behind because it collapses on the hard pairs specifically (co-visibility 0.1:
37.5% against 85.4%). So the honest statement is not "LightGlue buys precision": it is
that **the mutual-NN precision/recall curve does not reach LightGlue's operating point** —
LightGlue is simultaneously more precise and higher recall than any gate setting tested.

### Reproducing these numbers

```bash
# Oxford: fetch, prepare (records the per-axis scale it applied), evaluate
./crates/vrt-lightglue/scripts/get_oxford.sh /data/oxford
cargo run --release -p vrt-lightglue --example prep_oxford -- \
    /data/oxford /data/oxford_prepared 640 bilinear
cargo run --release -p vrt-lightglue --example eval_oxford -- \
    /data/oxford_prepared raco-aliked-extractor-k3072-...fp16.engine \
    lightglue-matcher-k1024-...fp16.engine 3.0 0.0 xfeat-backbone-...engine 0.0

# IMC: metadata only (needs h5py + numpy), then evaluate
python3 crates/vrt-lightglue/scripts/prep_imc.py /data/imc2021/phototourism 6 0
cargo run --release -p vrt-lightglue --example eval_imc -- \
    /data/imc2021/phototourism raco-aliked-extractor-k3072-...fp16.engine \
    lightglue-matcher-k1024-...fp16.engine 1.0 0.0 xfeat-backbone-...engine bilinear 0.0
```

The trailing arguments are the inlier threshold and the two mutual-NN gates; the tables
above use the values shown. Both examples print the configuration they ran with, so a
pasted run is self-describing.

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
- **LightGlue's operating point is outside mutual-NN's curve.** Columns 1 and 2 use
  *identical* RaCo keypoints and *identical* ALIKED 128-D descriptors; only the matcher
  differs, so column 2 tests the descriptors alone. Ungated, mutual-NN finds nearly as
  many true correspondences (4694 vs 4922) and buries them in outliers; gated, it gets
  precise but returns fewer. Neither setting reaches LightGlue on both axes at once — see
  the sweep above, which is why the gate is reported rather than assumed.
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

Ships no weights. LightGlue is Apache-2.0
([`cvg/LightGlue`](https://github.com/cvg/LightGlue) — Lindenberger, Sarlin, Pollefeys,
ICCV 2023); the checkpoint and ONNX export come from
[`fabio-sim/LightGlue-ONNX`](https://github.com/fabio-sim/LightGlue-ONNX) (Apache-2.0). The
engines derive from an asset also bundling RaCo (Apache-2.0) and ALIKED
(**BSD-3-Clause**, attribution required in binary form) — see
[`vrt-raco-aliked`](../vrt-raco-aliked) and the
[notice](https://huggingface.co/kornia/lightglue/blob/main/LICENSE-NOTICE.md).
