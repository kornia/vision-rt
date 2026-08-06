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
