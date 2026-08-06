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
raco_aliked_lightglue_pipeline_k1024.onnx

python3 crates/vrt-raco-aliked/scripts/split_raco_pipeline.py \
    --input  models/onnx/raco/raco_aliked_lightglue_pipeline_k1024.onnx \
    --outdir models/onnx/raco

crates/vrt-raco-aliked/scripts/build_engine.sh \
    models/onnx/raco/raco_aliked_extractor_k1024.onnx
```

The split needs only `onnx` — no torch, no onnxruntime, no Python 3.12 — so it runs
on the Jetson's stock `python3`. The matcher half is consumed by the `vrt-lightglue`
crate, which lands separately.

Verify a split before trusting it (requires `onnxruntime`):

```bash
python3 crates/vrt-raco-aliked/scripts/check_split_parity.py \
    --fused     models/onnx/raco/raco_aliked_lightglue_pipeline_k1024.onnx \
    --extractor models/onnx/raco/raco_aliked_extractor_k1024.onnx \
    --matcher   models/onnx/raco/lightglue_matcher_k1024.onnx \
    --size 256 --dump-ref models/onnx/raco/ref_k1024
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
