# vrt-raco-aliked

**RaCo** keypoint detection + **ALIKED** 128-D descriptors on TensorRT.

RaCo decides *where* to look — a rotation-robust detector with a learned ranker —
and ALIKED describes *what is there*. That separation of detection from description
is the main thing this buys over [`vrt-xfeat`](../vrt-xfeat), along with 128-D
descriptors instead of 64-D.

```rust
let mut raco = RaCoAliked::from_engine_file(engine, stream.clone())?;
let mut out  = raco.alloc_result()?;

raco.submit(&img, &mut out)?;   // async: resize → TRT → copy, no sync
stream.synchronize()?;          // the caller owns the one sync
let kpts  = out.keypoints_host()?;      // (x, y) in source pixels
let descs = out.descriptors_host()?;    // [K*128], L2-normalised
```

## Engine I/O

```text
images (B,3,H,W) f32   H,W multiples of 32, RGB in [0,1]
  -> keypoints            (B,K,2)   f32   model-resolution pixels (x,y)
  -> normalized_keypoints (B,K,2)   f32   long-edge normalised, matcher input
  -> descriptors          (B,K,128) f32   already L2-normalised
```

`K` is baked into the engine at export time (the k512…k3584 release assets), so
unlike XFeat there is no threshold-dependent survivor count to read back —
`count()` is always `K`.

## Choosing K — pick k3072 unless you match every frame

`K` is not just a keypoint count. Upstream's exporter selects a **structurally
different graph** per K: at **K ≥ 3072 RaCo's learned ranker is omitted entirely**
(`RankerMode.bypass`), at K=2560 and most of 1024–2560 it runs on a bounded boundary
window, and below that it runs dense. The ranker is a second CNN over the image —
nine residual conv blocks plus a 5×5 conv — that re-scores 2K candidates to pick the
best K, and it is what buys rotation robustness.

Dropping it makes extraction **roughly twice as fast while returning 3× the
keypoints**, which is why the default here is k3072 and not the middle of the range:

| K | ranker | extract (1 img) | matcher (1 pair) | E2E pair |
|---|---|---|---|---|
| 512 | dense | 49.1 ms | 7.9 ms | **106.0 ms** |
| 1024 | boundary | 55.2 ms | 21.6 ms | 132.0 ms |
| **3072** | **bypass** | **28.5 ms** | 126.5 ms | 183.5 ms |

Extraction is **non-monotonic in K** — k3072 costs half of k1024. The matcher is the
opposite: LightGlue's attention is O(K²), so it scales ~K^1.55 and dominates at k3072.

So the choice is really about which side you pay on:

- **Extraction-bound work — mapping, keyframe indexing, building a descriptor
  database: k3072.** Twice the throughput and 2.5× more correct correspondences than
  k1024 (see below). This is the default.
- **Matching every frame against one other frame: k512.** Fastest end to end
  (106 ms) and the *best* inlier rate of the three; you just get fewer matches.
- **k1024 is the worst extractor of the three** and only middling end-to-end. It is
  here for reference, not as a recommendation.

Extractor and matcher must be split from the *same* `kN` asset — `LightGlue::new`
rejects a mismatch.

## Two things that fail silently

**Preprocessing.** Feed RGB in `[0,1]` and nothing else. The ImageNet mean/std live
*inside* the graph as `extractor.raco.image_mean` / `image_std` buffers, so this crate
uses `Preprocessor::stretch` (resize + `/255`) and must **not** apply
`Normalize::imagenet()`. kornia's `Image<u8,3>` is already RGB, so no channel swap is
needed — upstream's Python preprocessor swaps only because OpenCV hands it BGR.

**Coordinate spaces.** `keypoints` and `normalized_keypoints` have identical shapes
and different meanings. `keypoints` is in model pixels (rescale to source pixels —
`keypoints_host()` does it for you). `normalized_keypoints` is RaCo's **long-edge**
normalisation `(kpts - size/2) / (size.max()/2)`, which differs from the per-axis
`2*kpts/size - 1` that SuperPoint and DISK use, and is what the LightGlue matcher
consumes verbatim.

## Getting the engine

Upstream publishes only the *fused* extractor+matcher graph, which cannot run under
vrt at all — it emits int64, has data-dependent output shapes, and never exposes
descriptors. `scripts/split_raco_pipeline.py` cuts the released ONNX into two
statically-shaped halves; see the script header for why the cut lands where it does.

```bash
curl -sSLO https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v3.0/\
raco_aliked_lightglue_pipeline_k3072.onnx

python3 crates/vrt-raco-aliked/scripts/split_raco_pipeline.py \
    --input  models/onnx/raco/raco_aliked_lightglue_pipeline_k3072.onnx \
    --outdir models/onnx/raco

crates/vrt-raco-aliked/scripts/build_engine.sh \
    models/onnx/raco/raco_aliked_extractor_k3072.onnx
```

The split needs only `onnx` — no torch, no onnxruntime, no Python 3.12 — so it runs
on the Jetson's stock `python3`. The matcher half feeds
[`vrt-lightglue`](../vrt-lightglue).

Verify a split before trusting it (requires `onnxruntime`):

```bash
python3 crates/vrt-raco-aliked/scripts/check_split_parity.py \
    --fused     models/onnx/raco/raco_aliked_lightglue_pipeline_k3072.onnx \
    --extractor models/onnx/raco/raco_aliked_extractor_k3072.onnx \
    --matcher   models/onnx/raco/lightglue_matcher_k3072.onnx \
    --size 256 --dump-ref models/onnx/raco/ref_k3072
```

## TensorRT notes

The released graph targets TRT 10.16 / CUDA 13, whose ONNX parser constant-folds
aggressively. JetPack 6's **TRT 10.3 does not**, and rejects two torch-dynamo
constructs — `Conv` with a computed zero bias, and `Reduce*` with computed `axes`.
The split script bakes both into initializers. This is not an artifact of the split:
the unmodified fused model fails to parse on TRT 10.3 with the identical error.

fp16 is the default and is what upstream recommends. The released graph already
carries the fix for the one known fp16 hazard: RaCo's flattened image indices exceed
fp16's 65504 limit, so the export lowers that division to integer floor-division.
Do **not** convert the ONNX itself to fp16 — let TensorRT pick precision from the
fp32 graph.

## Benchmarks

Jetson Orin Nano at **MAXN_SUPER**, TRT 10.3.0.30, fp16, 640×640. Engines built with
min=opt=max at the benchmark resolution so none is penalised for running off its optimum
profile. Extraction is **one image**; a pair costs twice this.

| | per image | engine build |
|---|---|---|
| **RaCo-ALIKED k3072** (default) | **28.5 ms** | 345 s |
| RaCo-ALIKED k1024 | 55.2 ms | 810 s @640² static · 1301 s @ dynamic profile |
| RaCo-ALIKED k512 | 49.1 ms | 570 s |
| XFeat (`vrt-xfeat`) | 3.4 ms | 114 s |

Even at its cheapest, extraction is **~8× the cost of XFeat** (16× at k1024). Latency is
data-independent, as CNN cost must be — measured 110.5 / 110.5 / 110.6 ms for two images
across three different inputs.

A **wide shape profile costs ~18%**: the dynamic 256²–640² engine is slower per image at
512² than the static engine is at 640², because TensorRT tunes tactics across the whole
range and sizes workspaces from the max. Build narrow.

### Why pay it

Because XFeat falls over under rotation and this does not. Matching two 640² crops
against a known ground-truth affine, counting a match as an inlier only if it lands
within 2 px of where the transform says it should:

| rotation | RaCo k3072 | RaCo k1024 | XFeat + mutual-NN |
|---|---|---|---|
| 0° | 1913, 99.8% | 708, **100.0%** | 647, 98.0% |
| 15° | 2098, 96.3% | 764, **99.2%** | 516, 59.3% |
| 45° | 1872, 94.3% | 690, **98.0%** | 261, 28.7% |
| 90° | 2297, 92.8% | 872, **97.8%** | 62, **0.0%** |

On pure translation XFeat is nearly as good and vastly cheaper — **reaching for this
crate there would be the wrong call**. It earns its cost only where orientation varies,
which is exactly what RaCo is for.

### What the ranker bypass actually costs

Upstream describes bypass as a "modest rotation-yield loss", and that holds: k3072 drops
from k1024's ~98% inliers to ~93% at 90°. But percentage is the wrong lens, because
k3072 also returns ~2.6× more matches. In **absolute correct correspondences** — which
is what a pose solver consumes:

| rotation | k3072 inliers | k1024 inliers |
|---|---|---|
| 45° | **1765** | 676 |
| 90° | **2132** | 853 |

k3072 yields ~2.5× more usable matches *and* extracts twice as fast. The ranker earns
its keep only if you need a high inlier *ratio* on few keypoints — e.g. feeding a solver
with no outlier rejection. With RANSAC downstream, bypass wins on both axes.

> Single image pair, synthetic rotations about the centre. The ordering is consistent
> across four rotations and three K values, but do not read the exact percentages as
> characteristic of real imagery.

Full end-to-end numbers, the intermediate rotations, and the harness live in
`vrt-lightglue` (`examples/bench_vs_xfeat`).

> Measure on an idle box. These figures come from a machine where unrelated GPU work
> (load average 4.2) inflated a first pass from 110 ms to 235 ms for the *same* input —
> contention, not thermal throttling; the SoC sat at 60 °C with no throttle asserted.
> The benchmark reports E2E min alongside median so that shows up rather than silently
> skewing the result.

## Model credit and licences

This crate ships no weights. The ONNX it consumes is derived from three separately
licensed upstreams, and the fused release asset combines all three:

| Component | Source | Licence |
|---|---|---|
| RaCo detector + ranker | [`cvg/RaCo`](https://github.com/cvg/RaCo) | Apache-2.0 |
| ALIKED descriptors | [`Shiaoming/ALIKED`](https://github.com/Shiaoming/ALIKED) | **BSD-3-Clause** |
| LightGlue+ matcher | [`cvg/LightGlue`](https://github.com/cvg/LightGlue) | Apache-2.0 |
| ONNX export tooling | [`fabio-sim/LightGlue-ONNX`](https://github.com/fabio-sim/LightGlue-ONNX) | Apache-2.0 |

All model credit belongs to the original authors:

- **RaCo** — Shenoi, Lindenberger, Sarlin, Pollefeys, *"RaCo: Ranking and Covariance
  for Practical Learned Keypoints"*, 3DV 2026, [arXiv:2602.15755](https://arxiv.org/abs/2602.15755).
- **ALIKED** — Zhao et al., *"ALIKED: A Lighter Keypoint and Descriptor Extraction
  Network via Deformable Transformation"*, IEEE TIM 2023.
- **LightGlue** — Lindenberger, Sarlin, Pollefeys, *"LightGlue: Local Feature Matching
  at Light Speed"*, ICCV 2023.

The ALIKED weights are BSD-3-Clause, which carries a binary-form attribution
requirement; that attribution is not present in the upstream export repo, so it is
reproduced here deliberately. Anything redistributing these engines must carry it too.

The TensorRT graph optimizations in the released ONNX (hierarchical chunked TopK,
BatchNorm folding, native SELU + logit-space NMS, ranker boundary-reranking, the
DeformConv→GridSample rewrite, and the integer-floor-div fp16 fix) are
fabio-sim's work and are preserved verbatim by the split — this crate adds no kernel
optimizations of its own.
