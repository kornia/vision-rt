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

## Why a separate matcher crate

Upstream publishes only the *fused* extractor+matcher graph, which can match only the
two images handed to it in a single forward pass. Splitting them means descriptors
extracted at any time — from a map, a relocalization database, a keyframe store — can
be matched against a live frame. That is the whole point; see
[`vrt-raco-aliked/scripts/split_raco_pipeline.py`](../vrt-raco-aliked/scripts/split_raco_pipeline.py).

## Engine I/O

```text
normalized_keypoints (2P,1,K,2)   f32   long-edge normalised
descriptors          (2P,1,K,128) f32   L2-normalised
  -> matches0 (P,K) i32   index into image 1, or -1 if unmatched
  -> mscores0 (P,K) f32   match confidence in [0,1]
```

**Rank-4 padding** is deliberate: the graph natively takes rank-3, but
`ModelSession::run_inputs` binds `&Tensor<f32,4>`, so the split script splices
`Reshape` nodes at the boundary rather than making vrt rank-generic for one model.

**`matches0` is int32 with the validity mask folded in as `-1`.** Upstream emits a
compacted `(M,3)` int64 list built by a `NonZero` — data-dependent (vrt has no
`IOutputAllocator`) *and* int64 (rejected at engine load). Cutting upstream of the
compaction yields the static per-query form, which is also the classic LightGlue
`matches0` convention. Reading it needs `OutputView::i32_ptr()`.

**Pairs are interleaved.** The leading dim is `2P` — images stacked
`[L0, R0, L1, R1, …]` — and the pair split happens inside the graph, so one pair is
leading dim 2, not two separate inputs. `submit_match` packs the two extractor results
into that layout on-device with no host round-trip.

**Feed `normalized_keypoints`, never `keypoints`.** They have identical shapes and
different meanings; mixing them up degrades matching silently. See
[`vrt-raco-aliked`](../vrt-raco-aliked) on the two coordinate spaces.

## Filtering

LightGlue applies its own mutual-nearest-neighbour check and filter threshold *inside*
the graph — everything with `matches0 >= 0` already passed both. `pairs(min_score)` is
an additional tightening, so `0.0` keeps all of them.

## Getting the engine

```bash
python3 crates/vrt-raco-aliked/scripts/split_raco_pipeline.py \
    --input  models/onnx/raco/raco_aliked_lightglue_pipeline_k1024.onnx \
    --outdir models/onnx/raco

crates/vrt-raco-aliked/scripts/build_engine.sh \
    models/onnx/raco/lightglue_matcher_k1024.onnx
```

Extractor and matcher must be split from the same `kN` asset — the `K` is baked into
both, and `LightGlue::new` rejects a mismatch.

## Benchmarks — `examples/bench_vs_xfeat`

Same pair, same box, both engines built min=opt=max at 640×640 so neither runs off its
optimum profile. Jetson Orin Nano at **MAXN_SUPER**, TRT 10.3.0.30, fp16, K=1024,
XFeat `top_k=1024`, 30 iterations:

| | extract ×2 | match | E2E median | E2E min |
|---|---|---|---|---|
| **RaCo-ALIKED + LightGlue+** | 110.5 ms | 22.4 ms | **133.0 ms** | 131.8 ms |
| XFeat + mutual-NN | 6.8 ms | 0.9 ms | **7.7 ms** | 7.7 ms |

**17.3× slower**, and that ratio holds on every input.

### Match quality

Scored against a **ground-truth affine** — a match counts as an inlier only if it lands
within 2 px of where the transform says it should, so this measures correctness rather
than self-consistency:

| rotation | RaCo-ALIKED + LightGlue+ | XFeat + mutual-NN |
|---|---|---|
| 0° | 708 matches, **100.0%** | 647 matches, 98.0% |
| 15° | 764 matches, **99.2%** | 516 matches, 59.3% |
| 30° | 688 matches, **98.3%** | 444 matches, 49.1% |
| 45° | 690 matches, **98.0%** | 261 matches, 28.7% |
| 90° | 872 matches, **97.8%** | 62 matches, **0.0%** |

So the trade is *not* "17× slower for a marginal gain". On pure translation XFeat is
nearly as good and vastly cheaper — **choosing this pipeline there would be the wrong
call**. Under rotation XFeat degrades and then fails outright, while this holds ~98%
inliers throughout. That is exactly the axis RaCo claims, and the only reason to pay.

```bash
cargo run --release -p vrt-lightglue --example bench_vs_xfeat -- \
    <raco_extractor.engine> <lightglue.engine> <xfeat_backbone.engine> \
    left.png right.png 30 [gt_affine.txt]
```

Without a ground-truth affine it falls back to consistency against the pair's own median
displacement — exact for pure translation, a proxy otherwise, and it only rewards a
matcher for agreeing with itself. Read it alongside the raw match count.

### Two caveats on the numbers

**Contention, not thermals.** Latency is data-independent, as CNN cost must be (extract
110.5 / 110.5 / 110.6 ms across three different inputs). A first pass showed extract
swinging 110→235 ms and XFeat 6.8→16.7 ms; that was unrelated GPU work on the box (load
average 4.2), *not* throttling — the SoC sat at 60 °C with no throttle asserted. The
example reports E2E min alongside median so contention is visible rather than silently
inflating the result. Measure on an idle machine.

**Not comparable to the upstream blog post.** [The
write-up](https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/)
reports a *speedup ratio* (2.67× median) against an fp16 `torch.compile` baseline on an
RTX 4080 Laptop; these are absolute milliseconds on an Orin Nano. Its "39.5 → 4.1 ms" is
the RaCo **detector alone** at 1280²/3584 keypoints, whereas the 110.5 ms here covers
RaCo **plus** ALIKED **plus** preprocessing for **two** images at 640²/1024 keypoints.
Reproducing that ratio would need a PyTorch baseline of the same graph.

## Model credit and licences

This crate ships no weights. LightGlue is Apache-2.0
([`cvg/LightGlue`](https://github.com/cvg/LightGlue)) — Lindenberger, Sarlin,
Pollefeys, *"LightGlue: Local Feature Matching at Light Speed"*, ICCV 2023. The
`raco_aliked` matcher checkpoint and the ONNX export are from
[`fabio-sim/LightGlue-ONNX`](https://github.com/fabio-sim/LightGlue-ONNX) (Apache-2.0).

The engines this crate consumes are derived from an asset that also bundles RaCo
(Apache-2.0) and ALIKED (**BSD-3-Clause**) weights; see
[`vrt-raco-aliked`](../vrt-raco-aliked) for the full attribution, which anything
redistributing these engines must carry.
