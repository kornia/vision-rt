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
published ground truth ship as examples, with their dataset preparation in-repo.

> Measured 2026-08-08 on a Jetson Orin Nano (SM87, TensorRT 10.3.0.30, MAXN_SUPER),
> `raco-aliked-extractor-k3072` fp16, images downscaled to long side 640 with an
> antialiased filter. Reproduce with the commands under *Reproducing these numbers*.

### Two LightGlue configurations, and why both are reported

The matcher's K is baked into its engine. Two are worth measuring:

* **k3072 — matched budget.** Every column sees the same 3072 keypoints. This is the fair
  comparison against mutual-NN and the one the conclusions below rest on.
* **k1024 — mixed-K.** The matcher takes the top-1024 prefix of RaCo's 3072. Cheaper, and
  the deployment default, but it is *not* an equal-budget comparison — an earlier revision
  of this README claimed it was.

### Oxford/VGG affine — planar, ground-truth homographies

15 pairs. A match is an inlier when `H·left` lands within **3 px at original resolution**
(the manifest records the per-axis scale; the evaluator converts, so every sequence is
scored under one criterion). `rot` is the in-plane rotation from a polar decomposition of
the ground-truth homography.

| pair | rot | LightGlue+ k3072 | LightGlue+ k1024 | RaCo-ALIKED NN | XFeat NN |
|---|---|---|---|---|---|
| bark/2 | −31° | 1251, **88.6%** | 445, 91.6% | 231, 21.6% | 602, 47.3% |
| bark/3 | **+150°** | 142, **67.3%** | 130, 88.3% | 0, 0.0% | 0, **0.0%** |
| bark/4 | **−120°** | 0, 0.0% | 128, **88.3%** | 0, 0.0% | 0, **0.0%** |
| bark/5 | −23° | 298, **87.1%** | 140, 94.6% | 20, 2.1% | 1, 0.1% |
| bark/6 | +153° | 0, 0.0% | 0, 0.0% | 0, 0.0% | 0, 0.0% |
| boat/2 | −14° | 1995, **96.8%** | 684, 97.7% | 1877, 88.4% | 1067, 59.6% |
| boat/3 | −40° | 1662, **96.3%** | 591, 97.2% | 241, 28.0% | 410, 36.5% |
| boat/4 | **−80°** | 1068, **87.0%** | 405, 90.0% | 0, 0.0% | 0, **0.0%** |
| boat/5 | +8° | 744, **92.3%** | 301, 90.9% | 469, 52.5% | 74, 11.2% |
| boat/6 | −41° | 211, 46.6% | 109, 59.6% | 1, 0.2% | 7, 1.6% |
| graf/2 | −15° | 1447, **90.6%** | 597, 94.3% | 800, 61.4% | 807, 54.8% |
| graf/3 | +20° | 1101, **79.3%** | 461, 81.9% | 566, 47.2% | 499, 42.6% |
| graf/4 | −27° | 900, **75.8%** | 391, 82.1% | 14, 2.1% | 187, 21.2% |
| graf/5 | +5° | 689, **75.9%** | 296, 84.6% | 471, 51.8% | 214, 26.3% |
| graf/6 | +38° | 517, **69.7%** | 244, 80.5% | 5, 0.8% | 17, 2.8% |
| **total inliers** | | **12025** | 4912 | 4695 | 3885 |
| **macro precision** | | 70.2% | **80.8%** | 23.7% | 20.3% |

### IMC 2021 phototourism — 3D scenes, ground-truth poses

Ground truth here is a camera pair, not a homography, so the strongest claim it supports
is that a match lies on its **epipolar line**: an inlier is a **Sampson error** under 1 px
against `F = K2⁻ᵀ[t]ₓR K1⁻¹` (`kornia_3d::pose`). That is a *weaker* test than Oxford's —
a match on the right line at the wrong depth passes — so the two sets of numbers are not
comparable. Sampson mixes both images' frames, so this threshold applies in the 640
frame rather than at original resolution. 90 pairs from `reichstag`, `sacre_coeur` and
`st_peters_square`, six per scene per co-visibility band, seeded sample.

Macro-average precision (each pair weighted equally), k1024:

| co-vis | pairs | LightGlue+ | RaCo-ALIKED NN | XFeat NN |
|---|---|---|---|---|
| 0.1 | 18 | **85.7%** | 39.2% | 21.2% |
| 0.2 | 18 | **86.3%** | 47.0% | 33.4% |
| 0.3 | 18 | **88.9%** | 61.0% | 45.3% |
| 0.4 | 18 | **90.8%** | 60.7% | 45.7% |
| 0.5 | 18 | **92.3%** | 69.0% | 55.2% |
| **all** | **90** | **88.8%** | 55.4% | 40.2% |

Totals, both configurations (inliers, pooled precision, macro precision):

| | inliers | micro | macro |
|---|---|---|---|
| LightGlue+ k3072 (matched) | **65283** | **89.2%** | **87.9%** |
| LightGlue+ k1024 (mixed-K) | 22174 | 90.1% | 88.8% |
| RaCo-ALIKED mutual-NN | 55102 | 60.4% | 55.4% |
| XFeat mutual-NN | 37062 | 46.6% | 40.2% |

### Is the mutual-NN baseline handicapped?

An ungated mutual-NN baseline is low-precision by construction, so "the matcher beats it"
would be circular. Both gates were swept, LightGlue invariant throughout as the control.

Oxford — tuning barely moves ALIKED: 23.7% macro ungated, peaking at **25.7%** (gate 0.70)
while giving up 40% of its inliers; 13.2% by 0.85. XFeat peaks at 25.0% (gate 0.90, half
its inliers).

IMC is where the gate genuinely matters and the ungated figure understates the baseline:

| IMC configuration | inliers | micro | macro |
|---|---|---|---|
| ALIKED NN, ungated | 55102 | 60.4% | 55.4% |
| ALIKED NN, gate 0.85 | ~16900 | ~89% | ~71% |
| LightGlue+ k3072 | **65283** | **89.2%** | **87.9%** |

At a tuned gate mutual-NN's *pooled* precision approaches LightGlue's — while returning a
quarter of the correct correspondences and staying ~17 points behind on the macro average,
because it collapses on the hard pairs specifically. **The mutual-NN precision/recall curve
does not reach LightGlue's operating point at any gate tested.**

### What the two benchmarks agree on

- **At a matched budget LightGlue dominates on both axes.** k3072 against k3072: 12025
  inliers vs 4695 on Oxford and 65283 vs 55102 on IMC, at roughly three times and 1.6
  times the precision. This is the claim an earlier revision asserted without running it.
- **Mixed-K trades recall for precision, and is better under extreme rotation.** k1024
  returns 2.4× fewer Oxford inliers but 10 points more macro precision — and holds
  `bark/4` (−120°) at 88.3% where the k3072 matcher returns 2 matches and scores 0%.
  Restricting to the most confident 1024 keypoints evidently helps where the detector is
  least reliable. Worth knowing before picking an engine; not something either number
  alone shows.
- **XFeat stops at rotation.** It scores **0.0%** on every pair past ~79° — `bark/3`
  (+150°), `bark/4` (−120°), `boat/4` (−80°) — and is nonzero on every pair below it.
  Extreme *scale* defeats everything: `bark/6` is 153° and a 4.2× zoom, and all four
  configurations return nothing.
- **Pooled and macro precision disagree, so both are reported.** Pooling weights by match
  volume, and the columns differ severalfold in volume; on Oxford k3072 the gap is 10
  points.

### Reproducing these numbers

```bash
# Oxford: fetch, prepare (records the per-axis scale it applied), evaluate
./crates/vrt-lightglue/scripts/get_oxford.sh /data/oxford
cargo run --release -p vrt-lightglue --example prep_oxford -- \
    /data/oxford /data/oxford_prepared 640 bilinear
cargo run --release -p vrt-lightglue --example eval_oxford -- \
    /data/oxford_prepared raco-aliked-extractor-k3072-...fp16.engine \
    <lightglue-k3072-or-k1024>.engine 3.0 0.0 xfeat-backbone-...engine 0.0

# IMC: metadata only (needs h5py + numpy), then evaluate
python3 crates/vrt-lightglue/scripts/prep_imc.py /data/imc2021/phototourism 6 0
cargo run --release -p vrt-lightglue --example eval_imc -- \
    /data/imc2021/phototourism raco-aliked-extractor-k3072-...fp16.engine \
    <lightglue-k3072-or-k1024>.engine 1.0 0.0 xfeat-backbone-...engine 0.0 bilinear
```

Arguments 1–7 mean the same thing in both harnesses, so a command line transfers between
them; `eval_imc` takes the interpolation kernel as an eighth. Both print the configuration
they ran with — including each column's keypoint budget — so a pasted run is
self-describing.

## Licences

Ships no weights. LightGlue is Apache-2.0
([`cvg/LightGlue`](https://github.com/cvg/LightGlue) — Lindenberger, Sarlin, Pollefeys,
ICCV 2023); the checkpoint and ONNX export come from
[`fabio-sim/LightGlue-ONNX`](https://github.com/fabio-sim/LightGlue-ONNX) (Apache-2.0). The
engines derive from an asset also bundling RaCo (Apache-2.0) and ALIKED
(**BSD-3-Clause**, attribution required in binary form) — see
[`vrt-raco-aliked`](../vrt-raco-aliked) and the
[notice](https://huggingface.co/kornia/lightglue/blob/main/LICENSE-NOTICE.md).
