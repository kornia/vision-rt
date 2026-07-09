# vrt-rfdetr-kpts

RF-DETR **keypoint** (human pose) on Jetson: GPU stretch + ImageNet-normalize →
TensorRT backbone → CPU decode. Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

`RfDetrKpts` is `Image<u8,3> → Vec<PersonPose>` — each person: a box + **17 COCO
keypoints** `[x, y, confidence]` in original-image pixels (NMS-free set predictor,
class 1 = person). Decode is on the CPU (Q=100, ~110 KB — negligible next to the
transformer).

Same **async / caller-owned** API as the rest of the workspace (VPI-style):

```rust
// from_hub (kornia/rfdetr-kpts) / from_onnx / from_engine_file / new
let mut pose = RfDetrKpts::from_hub(stream.clone(), 0.5)?;
let mut out = pose.alloc_result()?;      // caller-owned, reused
pose.submit(&image, &mut out)?;          // enqueue, no sync
stream.synchronize()?;                    // caller owns the one sync
let people = out.poses();                // Vec<PersonPose>, original pixels
```

Model: the RF-DETR Keypoint Preview export (input `[1,3,576,576]`; outputs
`dets [1,Q,4]` + `labels [1,Q,2]` + `keypoints [1,Q,34,8]`). Per-keypoint
confidence folds the model's learned precision (visibility × spatial sharpness).
`COCO_KEYPOINT_NAMES` gives the joint order. Model credit to the upstream authors.

License: Apache-2.0.
