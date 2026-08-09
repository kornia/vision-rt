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

> The tables in this section were taken on a **loaded** box and at a different resolution,
> so read them for their *ratios* only — they are internally consistent because each table
> is one run, but they are not comparable with each other or with the idle-box figures
> under [Latency](#latency-and-when-the-128-d-mutual-nn-kernel-is-worth-using) below.
> Where the two disagree on an absolute number, the idle-box one is the measurement.

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

## Latency, and when the 128-D mutual-NN kernel is worth using

The reason to have a mutual-NN kernel at all is speed: LightGlue's attention is O(K²) and
dominates at large K. This measures whether that pays off.

> Measured 2026-08-09 on a Jetson Orin Nano, **clocks locked at 1020 MHz**
> (`jetson_clocks`), MAXN_SUPER, with nothing else on the GPU — the camera nodes and the
> SfM build were stopped first. Min of 50 iterations after 5 warmups, one 640×512 pair,
> the same k3072 extractor throughout. Earlier revisions of this crate quoted no latency
> at all because every capture until now was taken under load.

| stage | time |
|---|---|
| RaCo-ALIKED extraction (×2) | **79.4 ms** |
| LightGlue+ k3072 — match | 132.4 ms |
| LightGlue+ k1024 — match | 22.5 ms |
| 128-D mutual-NN — match | **~7.4 ms** |

End-to-end (extract ×2 + match): mutual-NN **86.8 ms**, LightGlue k1024 **100.4 ms**,
LightGlue k3072 **207.3 ms**.

The mutual-NN figure is derived — `raco_mutualnn` reports only end-to-end (86.8 ms), and
79.4 ms of that is the extraction the other configurations also pay. Both are min-of-N on
the same engine and image, so the subtraction is sound, but it is a subtraction.

### The kernel is fast. It is still usually the wrong choice.

As a *matcher stage* it is ~18× faster than LightGlue at k3072 and ~3× faster at k1024, so
the speed claim holds. **Extraction is what dominates.** Against the k1024 default,
swapping LightGlue for mutual-NN saves 13.6 ms end-to-end — about **14%** — and costs
macro precision on Oxford falling from **80.5% to 24.3%**. That is a bad trade in a
one-pair-at-a-time pipeline, which is most of them.

**Where it does pay: many pairs per extraction.** Retrieval, one-to-many matching, or
re-matching against a cached descriptor bank — anywhere the 79 ms extraction amortises
across many matches and the 7.4 ms vs 22.5 ms difference is the entire marginal cost. That
is the case the kernel was worth adding for, and it is the case to use it in.

### And XFeat really is the throughput default

The same run puts XFeat's *entire* pipeline at **10.8 ms** end-to-end against RaCo's
79.4 ms of extraction alone — 8× faster than RaCo + mutual-NN and 19× faster than RaCo +
LightGlue k3072. The trade is the one the accuracy tables describe: XFeat scores 0.0% past
~79° of in-plane rotation, where this pipeline holds 98%+ to 180°. Reach for RaCo when
rotation is real; reach for XFeat when frame rate is.

## Accuracy on real data

The synthetic pair below is a sanity check, not evidence. Two real benchmarks with
published ground truth ship as examples, with their dataset preparation in-repo.

> Measured 2026-08-08 on a Jetson Orin Nano (SM87, TensorRT 10.3.0.30, MAXN_SUPER),
> `raco-aliked-extractor-k3072` fp16, images downscaled to long side 640 with
> **Lanczos + antialias** (`resize_fast_u8_aa`). The antialias flag only affects the
> separable kernels — bilinear and nearest ignore it — so bilinear, the previous
> default, was *not* antialiased despite passing the flag. Reproduce with the
> commands under *Reproducing these numbers*.

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
| bark/2 | −31° | 1253, **89.4%** | 436, 92.2% | 206, 19.9% | 576, 45.8% |
| bark/3 | **+150°** | 139, **64.4%** | 132, 78.1% | 0, 0.0% | 0, **0.0%** |
| bark/4 | **−120°** | 0, 0.0% | 69, **78.4%** | 0, 0.0% | 0, **0.0%** |
| bark/5 | −23° | 294, **89.1%** | 151, 96.2% | 26, 2.7% | 3, 0.4% |
| bark/6 | +153° | 0, 0.0% | 0, 0.0% | 0, 0.0% | 0, 0.0% |
| boat/2 | −14° | 2011, **97.8%** | 684, 98.3% | 1895, 88.9% | 1058, 60.1% |
| boat/3 | −40° | 1693, **96.1%** | 586, 96.9% | 240, 28.2% | 370, 33.3% |
| boat/4 | **−80°** | 1080, **87.6%** | 418, 91.5% | 0, 0.0% | 0, **0.0%** |
| boat/5 | +8° | 795, **94.9%** | 311, 91.7% | 480, 51.2% | 84, 12.7% |
| boat/6 | −41° | 219, 45.3% | 118, 59.6% | 3, 0.5% | 3, 0.7% |
| graf/2 | −15° | 1539, **93.0%** | 599, 95.1% | 856, 65.7% | 825, 55.4% |
| graf/3 | +20° | 1157, **80.3%** | 450, 82.1% | 572, 47.7% | 493, 41.8% |
| graf/4 | −27° | 951, **77.4%** | 386, 82.7% | 16, 2.3% | 198, 21.7% |
| graf/5 | +5° | 733, **78.6%** | 288, 85.7% | 522, 56.5% | 208, 26.4% |
| graf/6 | +38° | 548, **70.4%** | 240, 79.2% | 5, 0.9% | 19, 3.1% |
| **total inliers** | | **12412** | 4868 | 4821 | 3837 |
| **macro precision** | | 71.0% | **80.5%** | 24.3% | 20.1% |

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
| 0.1 | 18 | **86.7%** | 39.7% | 21.3% |
| 0.2 | 18 | **86.9%** | 47.5% | 33.8% |
| 0.3 | 18 | **88.9%** | 61.8% | 46.2% |
| 0.4 | 18 | **90.4%** | 61.4% | 46.1% |
| 0.5 | 18 | **92.3%** | 70.2% | 55.6% |
| **all** | **90** | **89.1%** | 56.1% | 40.6% |

Totals, both configurations (inliers, pooled precision, macro precision):

| | inliers | micro | macro |
|---|---|---|---|
| LightGlue+ k3072 (matched) | **66946** | 89.7% | 88.5% |
| LightGlue+ k1024 (mixed-K) | 22284 | **90.2%** | **89.1%** |
| RaCo-ALIKED mutual-NN | 56509 | 61.1% | 56.1% |
| XFeat mutual-NN | 37812 | 47.0% | 40.6% |

### Rotation and scale, isolated

`bark` is the usual evidence for rotation robustness and it cannot settle the question:
every large-rotation pair in it also carries a large zoom (`bark/3` 150° + 1.85×, `bark/4`
120° + 2.48×, `bark/6` 153° + 4.09×). `examples/eval_rotation` removes the confound by
matching one image against rotated and scaled copies of itself, inscribed by its diagonal
in a square canvas so no content is lost at any angle and the ground-truth homography is
exact by construction. 0° / 1.0× measures the resampling floor.

**Pure in-plane rotation** (precision, 3 px):

| deg | 0 | 15 | 30 | 45 | 60 | 90 | 120 | 150 | 180 |
|---|---|---|---|---|---|---|---|---|---|
| RaCo-ALIKED + LightGlue+ | 100% | **99.9%** | **99.6%** | **99.6%** | **99.6%** | **99.0%** | **98.5%** | **98.8%** | **99.4%** |
| RaCo-ALIKED + mutual-NN | 100% | 84.6% | 45.6% | 2.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| XFeat + mutual-NN | 100% | 79.1% | 58.7% | 26.6% | 1.5% | 0.3% | 0.0% | 0.0% | 0.0% |

**Pure scale** (precision, 3 px):

| zoom | 1.0× | 1.5× | 2.0× | 2.5× | 3.0× | 4.0× | 5.0× |
|---|---|---|---|---|---|---|---|
| RaCo-ALIKED + LightGlue+ | 100% | **100%** | **100%** | **100%** | **99.3%** | **100%** | **99.5%** |
| RaCo-ALIKED + mutual-NN | 100% | 93.2% | 89.1% | 84.4% | 70.7% | 52.4% | 23.1% |
| XFeat + mutual-NN | 100% | 78.3% | 55.5% | 32.9% | 18.4% | 7.4% | 9.7% |

Three things follow.

- **Rotation is fully handled, all the way to 180°.** No degradation is visible above the
  resampling floor across the entire sweep.
- **Scale is fully handled to 5×** as well, the widest this canvas supports.
- **The invariance is the matcher's, not the descriptors'.** Columns 1 and 2 share
  identical RaCo keypoints and identical ALIKED descriptors. Under raw mutual-NN those
  descriptors die at 45° — no better than XFeat's. LightGlue recovers the correspondences
  from the same descriptors, so what is rotation-invariant here is the learned matching
  over RaCo's repeatable keypoints, not the descriptor vectors.

Which explains the Oxford failures: `bark/6` combines 153°, 4.09× zoom **and** 6.3% image
overlap, and it is the combination that defeats it — neither nuisance alone does, at any
magnitude tested. `bark/4` (120°, 2.48×, 16.2% overlap) is a different story again: k1024
holds it at 78.4% while k3072 scores 0%, so that one is a matcher-budget artifact rather
than a robustness limit.

### Is the mutual-NN baseline handicapped?

An ungated mutual-NN baseline is low-precision by construction, so "the matcher beats it"
would be circular. Both gates were swept, LightGlue invariant throughout as the control.

Oxford, ALIKED gate (LightGlue invariant at 80.5% throughout, as the control):

| gate | inliers | macro precision |
|---|---|---|
| −1.0 (ungated) | 4821 | 24.3% |
| 0.50 | 4643 | 25.0% |
| **0.70** | 2935 | **27.9%** |
| 0.80 | 1278 | 24.2% |
| 0.85 | 368 | 11.5% |

XFeat's own sweep peaks at 24.9% (gate 0.90) against 20.1% ungated, for 39% of its
inliers. Best tuned mutual-NN therefore reaches 27.9% against LightGlue's 80.5%, and pays
39% of its correspondences for it.

IMC is where the gate genuinely matters and the ungated figure understates the baseline: a
0.85 gate lifts pooled precision to roughly LightGlue's while returning about a quarter of
the correct correspondences and staying ~17 points behind on the macro average, because it
collapses on the hard pairs specifically. **The mutual-NN precision/recall curve does not
reach LightGlue's operating point at any gate tested.**

### What the two benchmarks agree on

- **At a matched budget LightGlue dominates on both axes.** k3072 against k3072: 12412
  inliers vs 4821 on Oxford and 66946 vs 56509 on IMC, at roughly three times and 1.6
  times the precision. This is the claim an earlier revision asserted without running it.
- **Mixed-K trades recall for precision, and is better under extreme rotation.** k1024
  returns 2.5× fewer Oxford inliers but 10 points more macro precision — and holds
  `bark/4` (−120°) at 78.4% where the k3072 matcher scores 0%.
  Restricting to the most confident 1024 keypoints evidently helps where the detector is
  least reliable. Worth knowing before picking an engine; not something either number
  alone shows.
- **XFeat stops at rotation, and so do the ALIKED descriptors on their own.** On Oxford
  XFeat scores **0.0%** on every pair past ~79°. The controlled sweep above shows why, and
  shows it is not a descriptor property RaCo fixes: under raw mutual-NN, ALIKED dies at 45°
  too. LightGlue is what carries the rotation invariance.
- **Pooled and macro precision disagree, so both are reported.** Pooling weights by match
  volume, and the columns differ severalfold in volume; on Oxford the two differ by ~10 points
  between the matcher configurations.

### Reproducing these numbers

```bash
# Oxford: fetch, prepare (records the per-axis scale it applied), evaluate
./crates/vrt-lightglue/scripts/get_oxford.sh /data/oxford
cargo run --release -p vrt-lightglue --example prep_oxford -- \
    /data/oxford /data/oxford_prepared 640 lanczos
cargo run --release -p vrt-lightglue --example eval_oxford -- \
    /data/oxford_prepared raco-aliked-extractor-k3072-...fp16.engine \
    <lightglue-k3072-or-k1024>.engine 3.0 -1.0 xfeat-backbone-...engine -1.0

# IMC: metadata only (needs h5py + numpy), then evaluate
python3 crates/vrt-lightglue/scripts/prep_imc.py /data/imc2021/phototourism 6 0
cargo run --release -p vrt-lightglue --example eval_imc -- \
    /data/imc2021/phototourism raco-aliked-extractor-k3072-...fp16.engine \
    <lightglue-k3072-or-k1024>.engine 1.0 -1.0 xfeat-backbone-...engine -1.0 lanczos
```

```bash
# Controlled rotation and scale sweep (no dataset needed — one image against copies of itself)
cargo run --release -p vrt-lightglue --example eval_rotation -- \
    /data/oxford_prepared/graf/img1.png raco-aliked-extractor-k3072-...fp16.engine \
    lightglue-matcher-k1024-...fp16.engine xfeat-backbone-...engine 15 3.0 -1.0 -1.0
```

Arguments 1–7 mean the same thing in both dataset harnesses, so a command line transfers between
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
