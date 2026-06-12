---
name: nvmm-gstreamer-debug
description: Use when modifying or debugging the GStreamer pipeline in trt-gst — RTSP connection failures, caps negotiation errors, NVMM/DMA-BUF import failures, VIC resize, tee branches, or frames not arriving.
---

# NVMM GStreamer Pipeline Debugging (trt-gst)

## Pipeline anatomy (crates/trt-gst/src/lib.rs)

```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvidconv → caps → tee
   ├→ queue leaky=upstream → appsink "sink"      (NVMM, zero-copy → CUDA)
   └→ queue leaky=upstream → nvvidconv → RGBA → appsink "sink_cpu" (viz snapshots)
```

- The NVMM caps after the first `nvvidconv` control the **VIC hardware scaler**:
  adding `width=W,height=H` makes VIC resize during the NVMM→NVMM conversion.
  This is what `connect_resized()` does. Free (fixed-function hw, not GPU).
- `leaky=upstream` queues are load-bearing: without them a slow branch
  blocks the decoder and the camera stalls.
- `max-buffers=1 drop=true sync=false` on appsinks = always-latest-frame.

## Debug recipes

```bash
# Test a pipeline string standalone before touching Rust:
gst-launch-1.0 rtspsrc location=<url> latency=100 ! rtph264depay ! h264parse ! \
  nvv4l2decoder ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA,width=1280,height=720' ! \
  fakesink silent=false -v 2>&1 | head -40

# Caps negotiation failures — see actual negotiated caps:
GST_DEBUG=3 <command> 2>&1 | grep -i "caps\|not-negotiated"

# Element-level tracing:
GST_DEBUG=nvvidconv:5,nvv4l2decoder:5 <command>
```

## Known failure modes

| Symptom | Cause / fix |
|---------|-------------|
| Hangs in `connect()` ("stream ended before first frame") | Wrong URL, camera offline, or H.265 stream (pipeline is H.264-only: `rtph264depay`) |
| `cudaImportExternalMemory failed` | NVMM buffer layout not pitch-linear; check `nvbuf_layout(surf) == 0` guard |
| `not-negotiated` after editing caps | VIC can't produce that format/size combo; oddball widths may need 2-px alignment |
| CPU branch frames lag NVMM frames | Expected — branches are independent; snapshots are approximate (fine for viz) |
| All frames drop under load | tegrastats shows VIC/decoder saturation; lower resolution or fps at the camera |

## Invariants when editing the pipeline string

1. Keep the tee + both queues; never let the CPU branch run unqueued.
2. NVMM appsink callback must validate `fd >= 0 && pitch != 0 && layout == 0`.
3. Dimensions reported by `width()`/`height()` come from the first sample's
   caps — after a resize they are the RESIZED size, which downstream
   `NvmmPreprocessStage::new(stream, src_w, src_h, ...)` depends on.
4. The `Sample` is kept alive inside `NvmmFrame` (`_keep_alive`) — the DMA-BUF
   fd is only valid while it lives. Don't "simplify" that away.
