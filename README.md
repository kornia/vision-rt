# vision-rt

**Real-time spatial & physical AI on NVIDIA Jetson Orin — in Rust.**

Turn a camera into **metric 3D perception**: detect, segment, range, and **track
objects in world coordinates** on-device, at the sensor frame rate. TensorRT
inference + GPU pre/post-processing exposed as plain Rust types — no orchestration
framework, no Python in the loop, no host round-trips mid-pipeline. GPU image/tensor
types come from [`kornia-rs`](https://github.com/kornia/kornia-rs); each model is its
own crate over a shared safe core.

![RF-DETR-Seg + Depth Anything V2 + a 3D tracker on one CUDA stream](assets/rtsp_track.png)

*One camera → instance masks, a **metric range for every object**, stable track IDs,
and a live **top-down bird's-eye view** — the whole loop is **25.4 ms of GPU** per
frame on an Orin Nano. Top: per-instance masks with `id  depth`. Bottom: each tracked
object placed at its real `(X, Z)` on a metre grid, camera at the apex of the
field-of-view cone. IDs match across both views; the BEV is world-frame, not pixels.*

---

## Why vision-rt

- **Metric, not just pixels.** Depth Anything V2 gives a real range per pixel;
  per-instance mask sampling turns each detection into a metric `(X, Y, Z)`. The
  tracker's Kalman state is **3D** (`px, py, pz` + velocity), so objects live in world
  coordinates — the substrate for BEV, collision / keep-out zones, and multi-camera fusion.
- **Built for Orin, not ported to it.** One CUDA stream, **one `synchronize()` per
  frame**: enqueue every model + the fusion kernels, sync once, read. No hidden syncs,
  no mid-pipeline host copies. The API is VPI-shaped for the unified-memory GPU.
- **Production-shaped libraries, not a framework.** Each model is a plain type with a
  caller-owned output buffer you reuse across frames. Threading, messaging, and
  back-pressure are the application's job — vision-rt never owns your event loop.
- **Honest on-device numbers.** Everything below is measured on a Jetson Orin Nano at
  `nvpmodel -m 2 + jetson_clocks` (MAXN), fp16 — not desktop extrapolations.
- **Ships to a screen.** A built-in **H.264-over-WebSocket** live view (decoded in the
  browser via WebCodecs) streams the annotated feed + BEV to a phone at **~4 Mbit/s** —
  ~5× lighter than the equivalent MJPEG, and low-latency even over a remote link.
- **Rust end to end.** Memory-safe orchestration, no GC pauses in the hot loop; the only
  unsafe surface is a thin, audited C shim over TensorRT.

## The flagship pipeline — `examples/rtsp_track`

```
RTSP camera ─▶ GPU undistort ─▶ ┌ RF-DETR-Seg  (boxes + instance masks) ┐
                                │ Depth Anything V2 (metric depth)       │─▶ ONE sync
                                └ mask → per-instance metric depth       ┘
        ─▶ 3D Kalman tracker (depth-gated association, NSA noise) ─▶ stable world-frame tracks
        ─▶ annotated view + top-down BEV ─▶ H.264 / WebCodecs live stream
```

Two heavy nets and the depth-at-mask fusion all enqueue on **one** CUDA stream and
resolve in a **single** `synchronize()`; the tracker and rendering are CPU and free by
comparison. Every detection comes out with a metric range and a stable ID, placed in a
world-frame bird's-eye view — 14–16 objects/frame with coherent depths (chairs 2.0–2.9 m,
table 2.7 m, fridge 4.1 m, oven 4.9 m in the shot above).

## Benchmarks — Jetson Orin Nano, MAXN, fp16

Per-frame cost of the full detect + segment + depth + track pipeline (1280×720):

| Stage (per frame) | Time | Notes |
|---|---:|---|
| RF-DETR-Seg — detect + instance masks | ~15 ms | the larger engine share |
| Depth Anything V2 — metric depth (392²) | 10.1 ms | ~98 fps engine-only |
| mask → per-instance metric depth (GPU fusion) | 0.03 ms | one launch, ~200 masked reductions |
| **GPU wall — seg + depth + fusion, one sync** | **25.4 ms** | the real per-frame GPU cost |
| 3D Kalman tracker — association + update | ~0.08 ms | pure CPU, whole scene |
| enqueue + readout (CPU, off the GPU wall) | ~4.3 ms | truly async — ≪ the sync |
| **End-to-end** | **29.7 ms** | **→ 33.6 fps, GPU-bound** |

- **~33 fps GPU ceiling** with detection *and* a metric range for every object. Live on
  a 1280×720 RTSP camera the loop held **~14.8 fps — sensor-capped** at the camera's
  15 fps, not GPU-bound (the blocking RTSP receive is ~36 ms/frame), i.e. **~2× GPU
  headroom** for a faster sensor, a second camera, or another model.
- **Spend less GPU:** run depth at a lower cadence and let the tracker **coast** between
  updates — the 3D Kalman fills in metric motion for free.
- **Live view:** annotated frame + BEV as two H.264 streams, **~4 Mbit/s total**, encoded
  on a worker thread off the hot path (Orin Nano has no NVENC → software x264, still free
  next to the sensor cadence).

Single-model engine numbers (trtexec, engine-only): Depth Anything V2 **10.1 ms @392 /
17.9 ms @518**. See each crate's README for its own benchmarks.

## Quickstart

### Requirements

- **Hardware:** NVIDIA Jetson Orin (aarch64, SM87) — Nano / NX / AGX. Benchmarks above
  are on an Orin Nano (Super).
- **System:** JetPack 6.x — **TensorRT 10.3.x, CUDA 12.6**. Rust stable (aarch64).
- **For the live tracking demo only:** GStreamer + a software H.264 encoder
  (`x264` / `openh264`), an RTSP camera, and access to the `kornia/sensor-rt` git dep
  (`export CARGO_NET_GIT_FETCH_WITH_CLI=true`).
- **Model weights:** portable ONNX from Hugging Face (`kornia/*`), sha256-pinned;
  `export HF_TOKEN=…` for gated repos. Engines are **machine-locked** (TRT version + SM87)
  and built **on-device** on first run into `~/.cache/vision-rt/engines/` (or pulled
  prebuilt when TRT + SM match).
- **Memory:** the Orin Nano OOM-kills parallel template builds — cap jobs with `-j2`.
  Benchmark only at MAXN: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

### Build

```bash
cargo build --release -j2                          # capped jobs (Orin Nano RAM)
TRT_STUB=1 cargo clippy --all-targets              # off-Jetson: committed bindings, no CUDA/TRT
```

### Run a model in ~10 lines

```rust
let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
let mut det = RfDetr::from_hub(stream.clone(), 0.5)?; // conf threshold; ONNX → on-device engine, cached

let mut res = det.alloc_result()?;   // reuse across frames
det.submit(&image, &mut res)?;       // enqueue preprocess → engine → GPU decode, no sync
stream.synchronize()?;               // you own the one sync per frame
let objects: Vec<Detection> = res.detections()?; // boxes in original-image pixels
```

Construct any model with `from_hub` (pull ONNX from HF), `from_onnx` (local ONNX),
`from_engine_file` (prebuilt engine), or `new(engine, …)`.

### Run the flagship tracking pipeline

The `rtsp_track` example is kept out of the workspace (it needs GStreamer + the private
`sensor-rtsp` dep):

```bash
export CARGO_NET_GIT_FETCH_WITH_CLI=true
cargo run --release --manifest-path examples/rtsp_track/Cargo.toml -- \
    <rfdetr-seg.engine> <depth-anything.engine> \
    rtsp://user:pass@camera/stream1 0.4 serve
# then open http://<jetson-ip>:8080 in a browser (or your phone on the same network):
# the annotated view + top-down BEV, live over H.264 / WebCodecs.
```

The 5th arg selects the sink: `serve` (or `:PORT`) for the live stream, `out.png` for a
single annotated frame, `out.gif` for a ~10 s clip. Point it at your RF-DETR-Seg +
Depth Anything V2 engines (built on first run by the model crates, or with
`/usr/src/tensorrt/bin/trtexec`).

## Workspace

| Crate | Role |
|---|---|
| `trt-sys` | Raw FFI: pure-C shim over TensorRT (bindgen never sees C++) |
| `vrt` | Safe core: `Logger→Runtime→Engine→Session`, `ModelSession`, CUDA launch helpers |
| `vrt-hub` | Model weights (HF Hub, sha256-pinned) + on-device engine cache |
| `vrt-types` | Shared leaf: `CameraIntrinsics`/`Extrinsics`, GPU `Undistorter`, depth-at-mask sampling |
| `vrt-rfdetr` | RF-DETR object detector (NMS-free) + on-device GPU decode |
| `vrt-rfdetr-seg` | RF-DETR **instance segmentation** — boxes + per-instance masks |
| `vrt-rfdetr-kpts` | RF-DETR human pose: box + 17 COCO keypoints |
| `vrt-depth-anything` | Depth Anything V2 **metric depth** + depth-at-mask/box fusion |
| `vrt-xfeat` | XFeat keypoints + descriptors + GPU mutual-NN matching |
| `vrt-track` | Pure-CPU **3D multi-object tracker** (ByteTrack assoc + depth-gated 3D Kalman) |
| `vrt-viz` | CPU render (masks / boxes / BEV) + **H.264 / WebSocket live view** (WebCodecs client) |

## Compose your own

The single-model idiom extends to **N models on one frame, one stream, one sync**: build
each on the same `Arc<CudaStream>`, pass the same device image by reference to each
`submit`, enqueue any fusion kernel last, then one `synchronize()` drains everything. The
stream's FIFO order *is* the dependency edge — no CUDA events, no second stream. Every
model decodes back to source-pixel space, so coordinates line up across models and into
world-frame 3D. See [ARCHITECTURE.md](ARCHITECTURE.md) for the Arc chain, the async /
caller-owned contract, and multi-camera patterns.

## Models & engines

- **ONNX is the portable artifact** — hosted on Hugging Face (`kornia/*`), sha256-pinned,
  never committed to git. Private / gated repos need `HF_TOKEN`.
- **Engines are machine-locked** (TRT version + GPU arch) and built **on-device** into
  `~/.cache/vision-rt/engines/…` on first run; a matching prebuilt engine may instead be
  pulled from HF. Portable ONNX in, board-specific engine out.

Model credit belongs to the upstream authors — see each crate's README.

## Roadmap

- **Upstream the reusable pieces to [`kornia-rs`](https://github.com/kornia/kornia-rs).**
  Several crates are model-free, GPU-free algorithms (the 3D tracker, the camera model /
  undistort types) — a natural fit to land in kornia so the wider ecosystem gets them.
- **More cameras.** Beyond RTSP: USB **webcams**, Luxonis **OAK-D** (onboard stereo depth),
  automotive **GMSL** sensors, and **D-Robotics RDK** — plugged in behind the `sensor-rt`
  source layer, same one-stream pipeline downstream.
- **Feature-reuse ReID.** Re-identify objects from the detector/segmentation backbone's
  own features instead of a separate embedding network — the tracker already exposes the
  appearance-fusion hook, so this drops in without a second model on the GPU wall.
- **Quantization.** INT8 / lower-precision builds for the heavier engines to buy more GPU
  headroom (a second camera, higher fps) and to fit the smaller Orin modules.

## Testing & feedback

vision-rt is early and moving fast — **we'd love for you to try it** on your Jetson +
camera and tell us how it goes. Open an issue with your board, sensor, models, and
numbers; share what works and what's missing. Feedback directly shapes the roadmap above.

## License

Apache-2.0.
