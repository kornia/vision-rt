# vrt-types

Shared vision types for the [`vision-rt`](https://github.com/kornia/vision-rt)
workspace — the data vocabulary model crates pass between each other.

- **`Detection`** — a detected object: COCO class + score + `[x1,y1,x2,y2]` box in
  source pixels.
- **`Mask`** — a binary instance mask (`Image<u8,1>` newtype, `1` = foreground),
  host or device.
- **`DepthImage`** — a **metric** depth map in meters (`Image<f32,1>` newtype),
  host or device.

`Mask` / `DepthImage` are zero-cost `#[repr(transparent)]` newtypes that `Deref`
to the inner `kornia_image::Image`, mirroring kornia's `color_spaces` types (`Rgb8`
etc.) — so every image op still applies, but the type name carries mask / depth
semantics.

Deliberately **dependency-light** (`kornia-image` / `kornia-tensor` / `cudarc`
only — no `vrt` / `trt-sys`), so it is a leaf every model crate depends on
**downward**, and a clean candidate to upstream into kornia-rs.

License: Apache-2.0.
