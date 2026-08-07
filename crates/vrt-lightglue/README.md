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
glue.submit_match(&l, &r, &mut m)?;    // async
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
  leading dim 2, not two inputs. `submit_match` packs both results on-device for you.
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

**Best of both: mix them.** `submit_match` accepts results holding *more* than the
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

- **The rotation claim holds on real images, at larger angles than "rotation sequence"
  suggests.** Decomposing the ground-truth homographies gives the actual in-plane
  rotations: `bark/3` is **+150.2°**, `bark/4` is **-119.8°**, `boat/4` is **-79.8°**.
  XFeat scores **0.0%** on every pair beyond ~79° and is nonzero on every pair below it;
  RaCo-ALIKED + LightGlue holds **88.8 / 91.1 / 94.2%** on those three. What is *not*
  handled is extreme scale: `bark/6` combines 153° with a 4.2x zoom and returns 4 matches.
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

Ships no weights. LightGlue is Apache-2.0
([`cvg/LightGlue`](https://github.com/cvg/LightGlue) — Lindenberger, Sarlin, Pollefeys,
ICCV 2023); the checkpoint and ONNX export come from
[`fabio-sim/LightGlue-ONNX`](https://github.com/fabio-sim/LightGlue-ONNX) (Apache-2.0). The
engines derive from an asset also bundling RaCo (Apache-2.0) and ALIKED
(**BSD-3-Clause**, attribution required in binary form) — see
[`vrt-raco-aliked`](../vrt-raco-aliked) and the
[notice](https://huggingface.co/kornia/lightglue/blob/main/LICENSE-NOTICE.md).
