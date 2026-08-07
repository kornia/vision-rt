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

| K | ranker | extract/img | matcher/pair | E2E pair |
|---|---|---|---|---|
| 512 | dense | 49.1 ms | 7.9 ms | **106.0 ms** |
| 1024 | boundary | 55.2 ms | 21.6 ms | 132.0 ms |
| **3072** (default) | **bypass** | **28.5 ms** | 126.5 ms | 183.5 ms |

**k3072 when extraction dominates** (mapping, keyframe indexing) — half the cost of k1024
and ~2.5× more correct correspondences. **k512 when you match every frame** — fastest
end-to-end and the best inlier rate. k1024 is the worst extractor of the three.

Extractor and matcher must come from the same `kN` asset; `LightGlue::new` rejects a
mismatch.

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
