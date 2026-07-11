# vrt-depth-anything

**Depth Anything V2 metric** monocular depth on Jetson: GPU stretch +
ImageNet-normalize → TensorRT backbone → dense **metric** depth map (meters),
plus GPU **depth-at-box / depth-at-mask** fusion. Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

`DepthAnything` is `Image<u8,3> → DepthImage` (dense metric depth, meters). Like
the detector crates it is **GPU-resident + async / caller-owned**: `submit`
enqueues preprocess → TRT → a copy of the depth map into the caller-owned
`DepthResult`, with no sync and no host copy; the caller syncs once, then reads.

```rust
let mut depth = DepthAnything::from_engine_file(engine, stream.clone())?;
let mut out = depth.alloc_result()?;
depth.submit(&image, &mut out)?;        // enqueue, no sync
stream.synchronize()?;
let map = out.depth_host()?;            // DepthImage (meters), on demand
// or stay on device: out.depth_slice()
```

## Parallel detect + depth (the pattern)

Run a detector and depth on **one shared stream** from the **same** image; each
`submit` only enqueues; a **single** `synchronize()` drains both; then sample
per-detection metric depth **from the instance mask** (isolates the object — the
box bleeds background):

```rust
det.submit(&img, &mut d)?;                                    // enqueue, no sync
depth.submit(&img, &mut z)?;                                  // same stream, no sync
let zs = depth.sample_masks(&z, d.masks_slice(), d.mask_size(), d.count())?; // fusion
stream.synchronize()?;                                         // ONE sync drains all
let zs = stream.clone_dtoh(&zs)?;                              // per-instance metric z (m)
```

`sample_boxes` is the box-only fallback (mean of the inner-50% central patch).
Feed the sampled `z` to a tracker's `Detection::depth`.

Model: Depth Anything V2 Metric-Small (indoor/Hypersim) export — input `[1,3,S,S]`
(S multiple of 14), output `depth [1,S,S]` metric meters (~20 m indoor range). The
map spans the whole stretched frame, so box/mask coords scale to the map by
`map/src`. **Ships at S=392** (the speed/accuracy sweet spot — see Benchmark); the
crate reads S from the engine, so a 518 build works unchanged. Model credit to the
upstream authors.

## Benchmark

Jetson Orin (MAXN_SUPER, fp16, `trtexec` engine-only GPU compute):

| Input | GPU compute | Throughput | Note |
|-------|-------------|-----------:|------|
| **392×392** (shipped) | **10.1 ms** | **~98 fps** | fast; ~1.77× quicker than 518 |
| 518×518 (native) | 17.9 ms | ~56 fps | max accuracy |

The GPU **fusion kernels** (`sample_masks` / `sample_boxes`) are negligible next to
the engine — a per-instance masked reduction over ~200 slots. Verified end-to-end
(`detect_depth` on a COCO image): RF-DETR-Seg + DA2 + mask-sampling all complete in
**one** `synchronize()`, per-instance metric depth physically plausible (cats/remotes
on a couch ≈ 1.7–2.0 m). fp16 is numerically clean on this export (no norm-layer
overflow → no fp32 pinning needed). Run depth at **lower cadence** and let a tracker
coast between updates if you need to spend less GPU per frame.

## Building the weights

```bash
python3 crates/vrt-depth-anything/scripts/export_da2.py --out models/onnx/depth-anything-v2-metric-small
crates/vrt-depth-anything/scripts/build_engine.sh models/onnx/depth-anything-v2-metric-small/depth_anything_v2_metric.onnx
```

License: Apache-2.0.
