# vision-rt

**Real-time spatial & physical AI on NVIDIA Jetson Orin — in Rust.**

Turn a camera into **metric 3D perception**: detect, segment, range, and **track objects
in world coordinates**, on-device, at the sensor frame rate. TensorRT + GPU pre/post-
processing as plain Rust types — no orchestration framework, no Python in the loop, no
host round-trips mid-pipeline. GPU image/tensor types come from
[`kornia-rs`](https://github.com/kornia/kornia-rs); each model is its own crate.

![RF-DETR-Seg + Depth Anything V2 + a 3D tracker on one CUDA stream](assets/rtsp_track.png)

*One camera → instance masks, a **metric range per object**, stable track IDs, and a live
**top-down BEV** — the whole loop is **25.4 ms of GPU** per frame on an Orin Nano. Top:
masks with `id depth`. Bottom: each track at its real `(X, Z)` on a metre grid, camera at
the apex of the FoV cone. IDs match across views; the BEV is world-frame, not pixels.*

---

## Why vision-rt

- **Metric, not pixels.** Depth Anything V2 → a real range per pixel; mask-sampling → a
  metric `(X, Y, Z)` per object; the tracker's Kalman state is **3D**. Objects live in
  world coordinates — the substrate for BEV, keep-out zones, and multi-camera fusion.
- **Built for Orin.** One CUDA stream, **one `synchronize()` per frame**: enqueue every
  model + fusion kernel, sync once, read. No hidden syncs, no mid-pipeline host copies.
- **Libraries, not a framework.** Each model is a plain type with a caller-owned output
  buffer you reuse; threading, messaging, and back-pressure stay yours.
- **Honest numbers.** All timings are on a Jetson Orin Nano at MAXN, fp16 — engines via
  `trtexec`, pipeline via the example profiler; not desktop extrapolations.
- **Ships to a screen.** A built-in **H.264-over-WebSocket** live view (browser WebCodecs)
  streams the annotated feed + BEV to a phone at **~4 Mbit/s**, low-latency even remote.
- **Rust end to end.** No GC pauses in the hot loop; the only unsafe is a thin, audited C
  shim over TensorRT.

## The flagship pipeline — `examples/rtsp_track`

```
RTSP camera ─▶ GPU undistort ─▶ ┌ RF-DETR-Seg  (boxes + instance masks) ┐
                                │ Depth Anything V2 (metric depth)       │─▶ ONE sync
                                └ mask → per-instance metric depth       ┘
        ─▶ 3D Kalman tracker (depth-gated) ─▶ stable world-frame tracks
        ─▶ annotated view + top-down BEV ─▶ H.264 / WebCodecs live stream
```

```rust
let stream = CudaContext::new(0)?.default_stream();          // one shared CUDA stream
let mut seg     = RfDetrSeg::from_engine_file(seg_engine, stream.clone(), 0.4)?;
let mut depth   = DepthAnything::from_engine_file(depth_engine, stream.clone())?;
let mut tracker = Tracker::new(TrackerConfig::default())?;
let (mut d, mut z) = (seg.alloc_result()?, depth.alloc_result()?);

for frame in camera {                          // frame: device Image<u8,3>
    seg.submit(frame, &mut d)?;                // detect + instance masks  ┐ enqueued
    depth.submit(frame, &mut z)?;              // metric depth map         │ async —
    let zs = z.depth_image().sample_masks(     // mask → per-object depth  ┘ no sync yet
        d.masks_slice(), d.mask_size(), d.count_slice(), &stream)?;
    stream.synchronize()?;                     // ONE sync drains every GPU stage above

    let depth_m = stream.clone_dtoh(&zs.slice(0..d.count()))?;
    let dets: Vec<_> = d.detections()?.into_iter().zip(depth_m)
        .map(|(o, z)| Detection::new(o.bbox, o.score, o.class_id).with_depth(z))
        .collect();
    let tracks = tracker.update(&dets);        // stable, world-frame 3D tracks
}
```

Two nets + the depth-at-mask fusion enqueue on one stream and resolve in **one** sync; the
tracker and rendering are CPU and free by comparison.

## Benchmarks — Jetson Orin Nano, MAXN, fp16

Per-frame cost of the full detect + segment + depth + track pipeline (1280×720):

| Stage (per frame) | Time | Notes |
|---|---:|---|
| Depth Anything V2 — metric depth (392²) | 10.1 ms | trtexec engine-only; ~98 fps (17.9 ms @518²) |
| RF-DETR-Seg — detect + instance masks | ~15 ms | remaining GPU-wall share (wall − depth) |
| mask → per-instance metric depth (GPU) | 0.03 ms | one launch, ~200 masked reductions |
| **GPU wall — seg + depth + fusion, one sync** | **25.4 ms** | the real per-frame GPU cost |
| enqueue + readout (CPU, off the wall) | ~4.3 ms | truly async — ≪ the sync |
| 3D Kalman tracker — assoc + update | < 0.1 ms | pure CPU — negligible |
| **End-to-end** | **29.7 ms** | **→ 33.6 fps, GPU-bound** |

**~33 fps GPU ceiling** with a metric range for every object. Live on a 1280×720 RTSP
camera it held **~14.8 fps — sensor-capped** at 15 fps (RTSP receive ~36 ms/frame), i.e.
**~2× GPU headroom** for a faster sensor, a second camera, or another model. Spend less
GPU by running depth at a lower cadence and letting the tracker **coast** between updates.

### Feature extraction + matching (640²)

Two pipelines for the same job — keypoints, descriptors and correspondences between a pair
of frames (`vrt-lightglue`, `examples/bench_vs_xfeat`, 30 iters). Inliers are against a
known ground-truth affine, within 2 px.

| Pipeline | extract ×2 | match | End-to-end | Inliers @0° / 45° / 90° |
|---|---:|---:|---:|---|
| `vrt-xfeat` + mutual-NN | 6.8 ms | 0.9 ms | **7.7 ms** | 98.0% / 28.7% / **0.0%** |
| `vrt-raco-aliked` + `vrt-lightglue`, k512 | 98.2 ms | 7.9 ms | **106.0 ms** | 100.0% / 99.1% / 98.0% |
| …k3072 (extractor default) | **57.0 ms** | 126.5 ms | 183.5 ms | 99.8% / 94.3% / 92.8% |

**XFeat stays the right default** for throughput — ~14x faster and, on translation, as
accurate. What it does not survive is rotation. On the Oxford/VGG affine benchmark
(ground-truth homographies, 3 px at original resolution, every column at k3072), the
in-plane rotations recovered from the homographies are **+150°**, **-120°** and **-80°**
for `bark/3`, `bark/4` and `boat/4`; XFeat + mutual-NN scores **0.0%** on all three — and
on every pair past ~79°, while staying nonzero below it.

At a matched keypoint budget LightGlue+ dominates on both axes: **12412** Oxford inliers
against 4821 for mutual-NN on the same descriptors and 3835 for XFeat, at roughly three
times the precision. That holds against a *tuned* mutual-NN, not just an ungated one —
sweeping the similarity gate moves the baseline along a precision/recall curve that never
reaches LightGlue's operating point. The sweep is published alongside the tables.

On **IMC 2021 phototourism** (90 pairs, real 3D scenes, ground-truth poses, Sampson error
<= 1 px) macro precision across co-visibility bands from 0.5 down to 0.1 goes **92.3% ->
86.7%** for LightGlue+, **70.2% -> 39.7%** for mutual-NN on the same descriptors, and
**55.6% -> 21.3%** for XFeat. Both benchmarks ship as examples in `vrt-lightglue`.

`K` picks a structurally different graph — at K≥3072 RaCo's ranker is bypassed, halving
extraction, while the O(K²) matcher grows. Extraction and matching therefore want opposite
K, and you can have both: extract at k3072 and match at k1024 (the matcher takes the top-K
prefix), giving **83.4 ms** end-to-end at k1024's accuracy — 1.63× faster than using k1024
throughout. Details in [`vrt-raco-aliked`](crates/vrt-raco-aliked).

## Quickstart

**Requirements** — NVIDIA Jetson Orin (aarch64, SM87; Nano / NX / AGX), JetPack 6.x
(**TensorRT 10.3.x, CUDA 12.6**), Rust stable. Cap builds with `-j2` (the Orin Nano
OOM-kills parallel template builds); benchmark at MAXN (`sudo nvpmodel -m 2 && sudo
jetson_clocks`). The live demo also needs GStreamer + a software H.264 encoder
(`x264`/`openh264`), an RTSP camera, and `CARGO_NET_GIT_FETCH_WITH_CLI=true` for the
`kornia/sensor-rt` dep. Models are ONNX from HF (`kornia/*`, `HF_TOKEN` for gated repos);
engines are machine-locked (TRT + SM87) and built on-device on first run.

```bash
cargo build --release -j2                     # capped jobs (Orin Nano RAM)
TRT_STUB=1 cargo clippy --all-targets         # off-Jetson: committed bindings, no CUDA/TRT
```

Run the flagship tracking pipeline (a workspace-excluded example — needs GStreamer + the
private `sensor-rtsp` dep):

```bash
export CARGO_NET_GIT_FETCH_WITH_CLI=true
cargo run --release --manifest-path examples/rtsp_track/Cargo.toml -- \
    <rfdetr-seg.engine> <depth-anything.engine> rtsp://user:pass@camera/stream1 0.4 serve
# open http://<jetson-ip>:8080 (or your phone on the same network) — annotated view + BEV
```

The 5th arg picks the sink: `serve` / `:PORT` (live stream), `out.png` (one frame),
`out.gif` (~10 s clip). Engines are built on first run by the model crates, or with
`/usr/src/tensorrt/bin/trtexec`.

## Workspace

| Crate | Role |
|---|---|
| `trt-sys` | Raw FFI: pure-C shim over TensorRT (bindgen never sees C++) |
| `vrt` | Safe core: `Logger→Runtime→Engine→Session`, `ModelSession`, CUDA helpers |
| `vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |
| `vrt-types` | Shared leaf: `CameraIntrinsics`/`Extrinsics`, GPU `Undistorter`, depth-at-mask sampling |
| `vrt-rfdetr` | RF-DETR object detector (NMS-free) + GPU decode |
| `vrt-rfdetr-seg` | RF-DETR **instance segmentation** — boxes + per-instance masks |
| `vrt-rfdetr-kpts` | RF-DETR human pose: box + 17 COCO keypoints |
| `vrt-depth-anything` | Depth Anything V2 **metric depth** + depth-at-mask/box fusion |
| `vrt-xfeat` | XFeat keypoints + descriptors + GPU mutual-NN matching |
| `vrt-raco-aliked` | RaCo keypoint detection + ALIKED 128-D descriptors (rotation-robust) |
| `vrt-lightglue` | LightGlue+ transformer matching over two `vrt-raco-aliked` results |
| `vrt-track` | Pure-CPU **3D multi-object tracker** (ByteTrack assoc + depth-gated 3D Kalman) |
| `vrt-viz` | CPU render (masks / boxes / BEV) + **H.264 / WebSocket live view** (WebCodecs) |

`vrt-track` / `vrt-types` / `vrt-viz` are model-free and GPU-free — see
[ARCHITECTURE.md](ARCHITECTURE.md) for the crate DAG, the async / caller-owned contract,
multi-model composition, and multi-camera patterns. Model credit belongs to the upstream
authors — see each crate's README.

## Roadmap

- **Upstream reusable pieces to [`kornia-rs`](https://github.com/kornia/kornia-rs)** — the
  3D tracker and camera/undistort types are model-free algorithms.
- **More cameras** — beyond RTSP: USB **webcams**, Luxonis **OAK-D**, automotive **GMSL**,
  **D-Robotics RDK**, behind the `sensor-rt` layer.
- **Feature-reuse ReID** — re-identify from the detector/seg backbone's own features via
  the tracker's appearance hook, no second model on the GPU wall.
- **Quantization** — INT8 / lower-precision engines for more headroom and smaller Orins.

## Testing & feedback

vision-rt is early and moving fast — **try it** on your Jetson + camera and tell us how it
goes. Open an issue with your board, sensor, models, and numbers. Feedback shapes the roadmap.

## License

Apache-2.0.
