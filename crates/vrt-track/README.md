# vrt-track

**BoT-SORT** multi-object tracker with a **3D** Kalman motion model — a pure-Rust,
CPU-only algorithm crate. No TensorRT, no CUDA, no model of its own. Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

Feed per-frame detections (box + score + class, plus optional depth / appearance
embedding), get stable track ids back. Construct once, reuse every frame:

```rust
use vrt_track::{BotSort, BotSortConfig, Detection};

let mut tracker = BotSort::new(BotSortConfig::default())?;
for frame in frames {
    let dets: Vec<Detection> = frame.boxes.iter()
        .map(|b| Detection::new(b.xyxy, b.score, b.class_id))
        .collect();
    let tracks = tracker.update(&dets);   // Vec<Track>: id + bbox + class + 3D state
}
# Ok::<(), vrt_track::TrackError>(())
```

## What it does

- **ByteTrack two-stage association** — high-confidence detections match first,
  then a recovery pass over low-confidence detections keeps established tracks
  alive through partial/occluded frames.
- **3D constant-velocity Kalman filter** — see below.
- **Track lifecycle** `Tentative → Confirmed → Lost → Removed` with age / hit /
  time-since-update counters and a re-identification buffer.
- **Appearance / ReID fusion** (feature `appearance`, off by default) — cosine
  distance on caller-supplied embeddings (`Detection::feature`), min-fused into the
  IoU cost with a per-track EMA feature bank. It is a *hook*: you provide the
  vectors (e.g. from a future `vrt-reid` crate), there is **no** hard dependency on
  any embedding model.
- **Camera-motion compensation (GMC) hook** — the `CameraMotion` trait + an
  identity stub; plug a real affine estimator into `update_with_motion`.

## The 3D Kalman model (the point of this crate)

State is **8-dimensional**:

```
x = [ px, py, pz,  w, h,  vx, vy, vz ]
      └─3D centre─┘ └box┘ └3D velocity┘
```

`px, py` are the box-centre pixels, **`pz` is depth**, `w, h` the box size. Motion
is **constant-velocity in 3D** (position integrates velocity; box size is a random
walk). This is deliberately *not* the classic image-plane `xywh` SORT filter — it
carries a real depth axis so a depth/lift source can be fused directly.

**Graceful degradation to the image plane.** The measurement is always
`[px, py, pz, w, h]`. When a detection has no depth (`Detection::depth == None`) the
depth measurement variance `R[z,z]` is inflated to a huge value, so the Kalman gain
on the depth row collapses to ≈ 0: `pz`/`vz` simply **coast on the motion model**
and are never corrupted by a fake measurement. The filter then behaves exactly like
an image-plane tracker on the observed axes, while still maintaining a
(growing-uncertainty) depth estimate. Supply a real `pz` — with the small
`meas_depth` variance — the moment depth is available (e.g. from a future
`vrt-lift`/depth crate) and the 3D estimate sharpens automatically. One fixed-size
code path, no per-frame matrix reshaping.

## Design notes

- **Linear algebra:** `nalgebra` fixed-size `SMatrix`/`SVector` (8-state, 5-measure).
  Statically sized → no heap in the hot loop, one monomorphisation, and a
  well-tested `try_inverse` for the 5×5 innovation covariance. Already in the
  workspace lockfile, so no new download.
- **Assignment:** a compact, dependency-free **Hungarian** (Kuhn–Munkres, O(n³)).
  Optimal, and at MOT scale (tens of tracks/dets) far cheaper than it needs to be —
  strictly better than greedy on crossing/overlapping targets. Rectangular problems
  are padded to square with a sentinel cost the gate rejects.

## Try it

```bash
cargo run -p vrt-track --example track_synthetic     # scripted two-target demo
cargo test -p vrt-track                               # Kalman + association + e2e
cargo test -p vrt-track --features appearance         # + ReID fusion path
```

License: Apache-2.0.
