# vrt-rfdetr-seg

RF-DETR **instance segmentation** on Jetson: GPU stretch + ImageNet-normalize →
TensorRT backbone → **GPU decode**. Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

![RF-DETR-Seg on a COCO image: instance masks + boxes](https://huggingface.co/kornia/rfdetr/resolve/main/rfdetr-seg-preview-example.png)

*Masks + boxes drawn by the `rfdetr_seg_detect` example (2 cats, 2 remotes, couch).*

`RfDetrSeg` is `Image<u8,3> → Vec<Instance>` — each instance: a COCO class + score
+ box + a **binary mask** (NMS-free set predictor, class 0 = background). Like
`vrt-rfdetr`, everything stays on the **GPU**: `submit` enqueues two decode kernels
(boxes/labels, then a masks gather+threshold pass) and does **no host copy** — only
the survivor count is async-copied. Host transfers happen only when you ask.

Same **async / caller-owned** API as the rest of the workspace (VPI-style):

```rust
// from_hub (kornia/rfdetr-seg) / from_onnx / from_engine_file / new
let mut seg = RfDetrSeg::from_engine_file(engine, stream.clone(), 0.5)?;
let mut out = seg.alloc_result()?;       // caller-owned, reused
seg.submit(&image, &mut out)?;           // enqueue kernels, no sync, no host copy
stream.synchronize()?;                    // caller owns the one sync
let n = out.count();                     // survivors (pinned scalar)
let dets = out.detections()?;            // boxes → host, on demand
let masks = out.masks_host()?;           // binary masks → host, on demand
let all = out.instances()?;              // boxes + masks together (host)
// or stay on-device: out.dets_slice() / out.masks_slice() / out.qidx_slice()
```

Model: the RF-DETR Seg Preview export (input `[1,3,432,432]`; outputs
`dets [1,Q,4]` cxcywh + `labels [1,Q,91]` logits + `masks [1,Q,108,108]` raw mask
logits = `einsum(features, query_coeffs) + bias`). Decode thresholds the mask logit
at `0` (≡ `sigmoid ≥ 0.5`); the 108×108 grid covers the whole stretched frame, so
resize `mask_size → (src_w, src_h)` to overlay on the source. Model credit to the
upstream authors.

## Building the weights

The ONNX + engine are produced on-device (engines are machine-locked to the local
TensorRT + GPU arch):

```bash
# 1. Export ONNX (needs transformers>=5.1 — see the script header for an isolated
#    install if your env is pinned to 4.x).
python3 crates/vrt-rfdetr-seg/scripts/export_rfdetr_seg.py \
    --out models/onnx/rfdetr-seg-preview

# 2. Build the fp16 engine, named to the vrt convention
#    (rfdetr-seg-preview-trt<ver>-sm<cc>-fp16.engine).
crates/vrt-rfdetr-seg/scripts/build_engine.sh \
    models/onnx/rfdetr-seg-preview/inference_model.onnx
```

License: Apache-2.0.
