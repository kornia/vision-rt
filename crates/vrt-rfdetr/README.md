# vrt-rfdetr

RF-DETR object detection on Jetson: GPU stretch-resize → TensorRT backbone → GPU
decode. Part of the [`vision-rt`](https://github.com/kornia/vision-rt) workspace.

RF-DETR is a transformer **set predictor** — a fixed set of query boxes + class
logits, **NMS-free**. `RfDetr` is `Image<u8,3> → Vec<Detection>`; boxes are
returned in original-image pixels (COCO class ids 1–90).

Same **async / caller-owned** API as the rest of the workspace (VPI-style):

```rust
let mut det = RfDetr::from_engine_file("rfdetr.engine", stream.clone(), 0.5)?;
let mut out = det.alloc_result()?;      // caller-owned, reused
det.submit(&image, &mut out)?;          // enqueue (resize→backbone→decode), no sync
stream.synchronize()?;                   // caller owns the one sync
let dets = out.detections(&stream)?;     // survivors in original pixels
```

Model: the fixed-resolution official export (`rfdetr-small`, input `[1,3,512,512]`;
outputs `pred_boxes [1,300,4]` cxcywh-normalized + `pred_logits [1,300,91]`).
The decode kernel argmaxes the class logits (skips background), thresholds the
sigmoid score, and maps normalized cxcywh → xyxy in source pixels.

License: Apache-2.0.
