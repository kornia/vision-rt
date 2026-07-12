---
name: vrt-live-stream
description: Use when streaming the annotated pipeline to a browser/phone, or touching vrt-viz's H.264/WebSocket/WebCodecs live view (crates/vrt-viz: stream.rs, serve.rs, h264.rs, client.html) — encoder pipeline, WebCodecs client, jitter buffer, bitrate, latency/choppiness, or the decoder-leak / no-NVENC / profile gotchas. Covers the streaming stack and its non-obvious failure modes, not rendering primitives.
---

# vrt-viz live view: H.264 over WebSocket → browser WebCodecs

Stream the annotated frame + world-frame BEV to a phone. Pipeline:
**render (CPU) → H.264 encode (worker thread) → WebSocket broadcast → browser
WebCodecs `VideoDecoder` → `<canvas>` with a jitter buffer.** Files: `stream.rs`
(`LiveStream` — owns encoders + worker + server), `serve.rs` (`StreamServer` WS
broadcast + handshake), `h264.rs` (`H264Encoder`, gstreamer), `client.html` (the
WebCodecs page, `include_str!`'d). Behind the `h264` feature (pulls gstreamer).

Use it from the app in two calls: `LiveStream::spawn(port, (w,h), (bw,bh), main_kbps,
bev_kbps, fps)?` then `sink.submit(main_rgb, bev_rgb)` per frame (drop-stale latest-only
slot; encode runs off your capture loop).

## Why H.264, not MJPEG

MJPEG is intra-only — every frame a full JPEG, **~22 Mbit/s** at 720p15, which buffers
on a cellular/remote link. H.264 inter-frame compression → **~4 Mbit/s** for both
streams (~5× measured here; the generic figure is 10–20×). This is *the* fix for remote
"buffering". A jitter buffer only smooths *variance*; it can't cure a bandwidth
shortfall — reach for the codec first.

## No NVENC on Orin Nano

The Orin **Nano has no hardware video encoder** (NVDEC decode only; NVIDIA's spec lists
encode as "1080p30 by 1–2 CPU cores"). Verified: the gstreamer nv plugin exposes
`nvv4l2decoder` **only**, no `/dev/*enc*`, no NVENC in sysfs. So `h264.rs` uses
**software `x264enc`**. On an NVENC Jetson (Orin NX/AGX) swap `x264enc` → `nvv4l2h264enc`
(NVMM path) — the only line that changes. Software x264 at 720p15 is cheap (~20 ms on a
worker thread, off the hot path).

## Encoder pipeline — the profile trap

`appsrc → videoconvert → video/x-raw,format=I420 → x264enc(tune=zerolatency
speed-preset=ultrafast) → video/x-h264,profile=main → h264parse →
video/x-h264,stream-format=avc,alignment=au → appsink`.

- **Force `I420` + a mainstream profile.** Feeding RGB straight to x264enc yields **High
  4:4:4** (profile `0xf4`), which browser WebCodecs **cannot decode**. I420 + zerolatency/
  ultrafast lands at **baseline 4:2:0** (`avc1.42…`) — decodes everywhere incl. iOS Safari.
- **`stream-format=avc`** (not Annex-B) → h264parse puts the **avcC** record in the caps
  `codec_data`; each sample is length-prefixed AVCC. That avcC is the WebCodecs
  `description` — the portable path (Safari won't reliably take raw Annex-B).
- **Zero-copy:** `encode(rgb: Vec<u8>)` uses `Buffer::from_slice` (not
  `from_mut_slice(rgb.to_vec())`) — saves a ~2.7 MB copy/frame; `get_mut` for pts still
  works (metadata, not the readonly memory).
- **GOP = fps** (1 s keyframe interval) so a new viewer joins/re-syncs within ~1 s.

## WebSocket wire + client

One WS carries both views. Binary message = `[kind, tag] + payload`: `kind` ∈
`C`(avcC config) / `K`(keyframe) / `D`(delta), `tag` ∈ `M`/`B` (a `Stream` enum, not
magic bytes). Handshake is hand-rolled SHA-1 + Base64 (dependency-free; no sha1 crate in
the tree). New viewer: cached config first, then skip deltas until the first keyframe.
H.264 is inter-frame → a viewer that falls a full bounded-channel behind is **dropped and
reconnects** (can't drop mid-GOP).

Client (`client.html`): configure `VideoDecoder` with the avcC `description` (codec
string derived from `avcC[1..3]`), decode key/delta chunks, push decoded frames to a
per-stream **jitter buffer** (`MINBUF`/`MAXBUF`), draw on a fixed ~15 fps tick, drop
stale. Auto-reconnect on close.

## The failure modes we actually hit

- **"EncodingError: Decoder failure" after minutes** = **leaked decoders**. The error
  handler must **`close()` the VideoDecoder before discarding it** (the `reset()`
  helper) — iOS Safari allows only a few HW decode sessions, so each un-closed decoder
  leaks one until the pool exhausts and the stream dies permanently. Use `reset()` on
  error, config-change, and reconnect.
- **Choppy, not laggy = cadence, not bandwidth.** A fixed-timer client push (`sleep(66ms)`)
  beats against the producer's jittery cadence → duplicate/dropped frames = judder. Fix:
  **event-driven delivery** (version + condvar), send each frame once when published.
- **Still choppy after that = CPU contention.** Encode competes for cores; a niced
  worker starved by other pegged processes delivers unevenly even though production is
  steady. Check per-core load, not just fps.
- **Still buffering at full res = bandwidth.** Drop bitrate (`RTSP_TRACK_MAIN_KBPS` /
  `_BEV_KBPS`) or resolution; the jitter buffer won't manufacture link capacity.
- **Stale client after an edit** — serve the index with `Cache-Control: no-cache`.

## Serving / viewing

`serve` or `:PORT` on the example → open `http://<jetson-ip>:PORT`. Encode time is
logged on the worker (`RUST_LOG=<bin>=debug`). `/main` `/bev` MJPEG were removed once
H.264 landed — `/` is the WebCodecs page, `/ws` the stream.

## Related skills

- `vrt-pipeline-compose` — produces the frames/tracks you render + stream.
- `vrt-tracking` — the tracks drawn in the main view + BEV.
- `jetson-benchmarking` — encode cost + power discipline.
