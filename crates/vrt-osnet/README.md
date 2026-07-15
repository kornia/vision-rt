# vrt-osnet

OSNet **person re-identification** on Jetson: GPU crop+resize+ImageNet-normalize →
TensorRT → **L2-normed** identity embeddings. Part of the
[`vision-rt`](https://github.com/kornia/vision-rt) workspace.

`OsNetReid` turns person boxes into one metric-learned appearance vector each, for the
tracker's identity **decisions** (re-id after occlusion, gallery resurrection after a
person leaves and re-enters). Unlike detector-backbone tokens — fine as a soft, IoU-gated
*tie-breaker* but collapsing toward each other under occlusion — OSNet is trained with a
re-id metric objective (MSMT17 pedestrians), so same-person-across-viewpoints stays closer
than different-person: what an identity *decision* needs.

**Caller-owned async**, like the other model crates: `submit(&frame, &boxes, &mut result)`
enqueues crop → TRT → GPU L2-normalize and returns **without syncing** — the embeddings
stay on device. The caller drains the stream with its **own** `stream.synchronize()`
(folding OSNet into whatever else is queued), then reads `result.embeddings_slice()`
(device, cosine-ready) or `result.embeddings_host()` (host `Vec<Vec<f32>>`, submit order).
Typically run after the frame's main readout, only when person boxes exist. ~1 ms for a
few crops on an Orin Nano.

```rust
let mut reid = OsNetReid::from_engine_file(engine, stream.clone())?;
let mut out = reid.alloc_result()?;         // reused every frame
reid.submit(&device_frame, &person_boxes, &mut out)?; // async, no sync
stream.synchronize()?;                        // caller's one sync
for emb in out.embeddings_host()? { /* feed the tracker via Detection::with_feature */ }
```

## Engine

Export `osnet_x0_25_msmt17` (torchreid) to ONNX at a fixed batch, then build on-device
with `trtexec` (SM87 / TRT 10.3.x — `.engine` files are machine-locked, never copy across
hosts). The crop preprocess matches torchreid's `Resize((256, 128))` (stretch, no aspect
preservation).

## Where it fits

Leaf-ish satellite: depends only on `vrt` (TRT session) + `kornia` (device image/tensor).
Consumed by `examples/rtsp_track`, which gates it to COCO person (class 1) and feeds the
embeddings to `vrt-track`'s appearance ReID hook (`reid_classes = [1]`).
