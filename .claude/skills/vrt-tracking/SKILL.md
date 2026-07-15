---
name: vrt-tracking
description: Use when working on multi-object tracking or metric-3D geometry — the vrt-track 3D Kalman, ByteTrack association, depth gating, NSA noise, occlusion recovery, coasting, world-frame BEV, camera intrinsics/extrinsics/undistort (vrt-types), or the appearance/ReID hook. Covers the tracker's design + tuning knobs and how metric depth feeds it, not GPU/model internals.
---

# vrt-track: 3D multi-object tracking + metric geometry

`vrt-track` is **pure CPU** (nalgebra + `vrt-types`, no TensorRT/CUDA) — a natural
upstream candidate. It turns per-frame detections (box + score + class, plus optional
**metric depth** and **appearance**) into stable track ids with a **3D** state. Feed it
downstream of the GPU pipeline (see `vrt-pipeline-compose`); it costs < 0.1 ms/frame.

## The 3D state (the point of the crate)

Kalman state is **8-D**: `[px, py, pz, w, h, vx, vy, vz]` — box-centre pixels `px,py`,
**depth `pz`**, box size `w,h`, 3D velocity. Constant-velocity in 3D; box size is a
random walk. This is *not* the classic image-plane `xywh` SORT filter — it carries a
real depth axis. Files: `kalman.rs` (the filter), `track.rs` (`Tracklet` + public
`Track`), `tracker.rs` (association + lifecycle), `association.rs` (cost matrices +
Hungarian).

**Depth is optional + graceful.** `Detection::with_depth(z)` attaches a mask-sampled
metric range; the measurement is always `[px,py,pz,w,h]`, but with no depth the depth
row's variance is inflated so its Kalman gain → 0 and `pz`/`vz` coast on the motion
model — the filter behaves exactly like an image-plane tracker on the observed axes.
Supply real depth and the 3D estimate sharpens. One fixed-size code path, no reshaping.

## Association (why ids stay stable)

ByteTrack two stages: high-confidence detections match first (IoU + optional
appearance), then a recovery pass over low-confidence detections keeps established
tracks alive through partial/occluded frames. Assignment is a dependency-free
**Hungarian** (optimal, MOT-scale is trivial). Three fusions layer onto the IoU cost,
each **only lowers** cost (rescue, never block); the depth gate is the hard veto:

- **Depth gate** (`association::gate_depth`, `tracker.rs` config `depth_gate_rel/abs`) —
  reject a pair when `|z_track − z_det| > max(abs, rel·z_track)`, both sides depth-known.
  Kills id-swaps between objects overlapping in the image but at different ranges.
  **Keep it loose** (`rel 0.35`, `abs 0.7 m`): monocular-depth noise on a static object
  is large, and a tight gate false-rejects a valid match → id churn. Genuine
  foreground/background separation is far larger and still vetoes a swap.
- **Centre-proximity rescue** (`association::fuse_center`, stage 2 only) — an occluded
  object's box shrinks to the visible fragment, so IoU drops below the gate even though
  the centre hasn't moved. A size-normalised centre cost re-acquires it. **Gated to
  near-static tracks** (image-plane speed < `STATIC_SPEED_PX`, in `tracker.rs`): a fast
  coasting track's centre has drifted, so centre-only matching would let it steal a
  neighbour. Applied in stage 2 only — stage-1 fusion let adjacent same-depth objects
  swap.
- **NSA measurement noise** (`track.rs`, `NSA_ALPHA`) — `R ← R·(1 + α·(1−score))`, α=2:
  a low-confidence box nudges the state gently, a crisp one updates firmly.

## Lifecycle + real dt

`Tentative → Confirmed → Lost → Removed`. A `Lost` track coasts on the Kalman for
`track_buffer` frames (default 60 ≈ 4 s at 15 fps) so intermittent detection doesn't
churn its id, then is reaped. `update(&dets)` steps at `dt=1`; `update_dt(&dets, dt)`
scales the Kalman F/Q by real inter-frame seconds (jitter-robust under variable fps).

## Metric 3D + world frame (`vrt-types`)

- `CameraIntrinsics::from_hfov(w, h, hfov_deg)` builds `fx,fy,cx,cy` from a spec'd FoV;
  `unproject(px,py,z)` → camera-frame `(X,Y,Z)` metres.
- `Undistorter` (`undistort.rs`) — one GPU `k1` barrel remap applied **before** seg/depth
  so boxes/masks/metric-3D are rectified-pinhole.
- `Track::metric_position(intr)` = camera-frame metres; `world_position(intr, extr)`
  applies a `CameraExtrinsics { r, t }` pose → shared world frame (`IDENTITY` =
  single-camera). This is the basis for the world-frame BEV (`vrt-viz::render_bev`) and
  multi-camera fusion.

## Appearance / ReID (feature `appearance`, off by default)

A **hook**, not a model: supply L2-normalised embeddings via `Detection::feature`;
cosine distance is min-fused into the IoU cost (BoT-SORT style) with a per-track EMA
bank. No embedder in the crate — `vrt-osnet` (or the detector's own backbone
features) provides the vectors. Update the bank only from confident detections.

## Tuning knobs (`TrackerConfig`)

`track_high_thresh`/`track_low_thresh` (the two association tiers), `match_thresh` /
`match_thresh_second` (IoU gates), `depth_gate_rel`/`depth_gate_abs`, `track_buffer`,
`min_hits`, `KalmanParams`.

## Gotchas

- **IoU alone fails on occlusion** — a partially-hidden object's box changes *shape*, so
  geometry can't always re-acquire it; the centre-fuse handles the static case, but the
  robust fix for shape-changed re-entry is appearance ReID.
- **Loose depth gate on purpose** — tightening it to "look precise" causes id churn on
  static objects from monocular-depth noise. Diagnose churn with the example's
  distinct-ids/100-frame counter.
- **Not full BoT-SORT** — camera-motion compensation (GMC) is deliberately omitted
  (unneeded for fixed cameras). Add it back only for a moving/handheld camera.

## Related skills

- `vrt-pipeline-compose` — where the GPU detections + mask-sampled depth come from.
- `jetson-benchmarking` — the tracker is CPU; measure it off the GPU wall.
