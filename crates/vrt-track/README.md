# vrt-track

A robust **multi-object tracker** with a **3D** Kalman motion model — a pure-Rust,
CPU-only algorithm crate. No TensorRT, no CUDA, no model of its own. Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

**Design:** ByteTrack-style two-stage association + a BoT-SORT-style `w,h` Kalman +
a **depth-gated 3D** extension. It is *not* full BoT-SORT — camera-motion
compensation (GMC) is deliberately omitted (unneeded for fixed cameras; add it back
if you ever track from a moving/handheld camera).

> **Upstream candidate.** No model inference, no GPU — a self-contained algorithm
> (only `nalgebra`). A natural fit to upstream into **kornia-rs** as a tracking
> module (its `CameraIntrinsics`/`Detection` map onto kornia's own types).

Feed per-frame detections (box + score + class, plus optional depth / appearance
embedding), get stable track ids back. Construct once, reuse every frame:

```rust
use vrt_track::{Tracker, TrackerConfig, Detection};

let mut tracker = Tracker::new(TrackerConfig::default())?;
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
  alive through partial/occluded frames. The recovery pass min-fuses a
  **size-normalised centre-proximity** cost into IoU, so a partially-occluded
  object whose box shrinks (a half-hidden chair) still matches its coasting track
  on centre alone instead of churning its id — while the depth gate stays the hard
  veto against a cross-depth swap.
- **3D constant-velocity Kalman filter** — see below.
- **Track lifecycle** `Tentative → Confirmed → Lost → Removed` with age / hit /
  time-since-update counters and a re-identification buffer.
- **Appearance / ReID fusion** (feature `appearance`, off by default) — cosine
  distance on caller-supplied embeddings (`Detection::feature`), min-fused into the
  IoU cost with a per-track EMA feature bank. It is a *hook*: you provide the
  vectors (e.g. from the `vrt-osnet` crate), there is **no** hard dependency on
  any embedding model.

## Roadmap

- **Appearance ReID (next).** Geometry (IoU + centre + depth) re-acquires an object
  only while its box still overlaps its coasting track; a detection whose *shape*
  changed under heavy occlusion, or that re-enters after a long gap, needs
  appearance. The `appearance` feature already exposes the fusion hook
  (`Detection::feature` + per-track EMA bank); the next step is to port the OSNet
  embedder (`vrt-osnet`) into the workspace and feed its L2-normalised vectors in,
  updating the bank only from confident detections (StrongSORT) so a noisy
  embedding doesn't poison an identity.

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
