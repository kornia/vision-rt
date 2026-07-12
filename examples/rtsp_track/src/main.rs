//! RTSP → RF-DETR-Seg + Depth Anything V2 + **3D tracking**, device end-to-end.
//!
//! The seg + depth pipeline is the one-stream / one-sync flow of `rtsp_depth`; this
//! example adds the tracker on top. The mask-sampled metric depth feeds each
//! detection's `pz` (`Detection::with_depth`), so the 3D Kalman carries a real
//! metric-depth axis + approach velocity per track. All rendering / streaming lives
//! in the `vrt-viz` crate — this example just wires the pipeline to it.
//!
//! ```text
//!   source → seg.submit + depth.submit → sample_masks → ONE sync
//!   readout → Detection::with_depth(z) → Tracker::update → [Track]
//!   vrt_viz::render_main / render_bev → MJPEG / GIF / PNG
//! ```
//!
//! Build:  export CARGO_NET_GIT_FETCH_WITH_CLI=true
//!   cargo run --release --manifest-path examples/rtsp_track/Cargo.toml \
//!       -- <seg.engine> <depth.engine> rtsp://<camera>/stream [conf] [out]
//!
//! The optional 5th arg picks the output:
//!   `out.png`         — one annotated frame (main + BEV stacked) to that PNG
//!   `out.gif`         — a ~10 s animated-GIF clip, then exit
//!   `serve` | `:PORT` — MJPEG-over-HTTP live stream (main + BEV); open
//!                       `http://<jetson-ip>:PORT` in a phone browser (same LAN)

use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use sensor_rtsp::RtspSource;
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::RfDetrSeg;
use vrt_track::{CameraIntrinsics, Detection, TrackState, Tracker, TrackerConfig};
use vrt_types::Undistorter;
use vrt_viz::{
    downscale, encode_png, render_bev, render_main, stack_v, write_gif, H264Encoder, MaskOverlay,
    MjpegServer, TrailStore,
};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Approximate horizontal FoV of the Tapo C210 (deg). TP-Link publishes no angular
/// FoV — only the 3.83 mm F/2.4 lens — so this is computed from the lens on the
/// C210's 1/2.9" 16:9 sensor (~5.12 mm wide): `2·atan(5.12 / (2·3.83)) ≈ 67°`.
const TAPO_C210_HFOV_DEG: f32 = 67.0;
/// Eyeballed radial distortion coefficient for the Tapo C210 wide lens (barrel →
/// negative). Undistort runs before seg/depth so boxes/masks/metric-3D are pinhole.
/// Tune on a captured frame until straight edges look straight (replace with a
/// checkerboard calibration for accuracy).
const TAPO_C210_K1: f32 = -0.28;
/// Standalone BEV canvas size (matches the 1280-wide main frame so they stack clean).
const BEV_W: usize = 1280;
const BEV_H: usize = 640;

/// A rendered `(main_rgb, bev_rgb)` pair awaiting H.264 encode.
type FramePair = (Vec<u8>, Vec<u8>);
/// Latest-only handoff slot: newest pair + a condvar the worker waits on.
type FrameSlot = Arc<(Mutex<Option<FramePair>>, Condvar)>;

/// Nominal stream frame rate (camera-paced); also the encoder GOP (1 s keyframe
/// interval → a WebSocket viewer joins/re-syncs within ~1 s).
const STREAM_FPS: i32 = 15;

/// Off-thread **H.264** encoder + publisher. The render loop hands the two rendered RGB
/// buffers through a **latest-only** slot; a worker thread owns a per-stream
/// [`H264Encoder`] (software x264 — the Orin Nano has no NVENC) + the [`MjpegServer`]
/// and does the inter-frame encode + WebSocket broadcast off the hot path. H.264's
/// temporal compression cuts the bitrate ~10–20× vs MJPEG, which is what fixes remote
/// buffering; the browser decodes via WebCodecs.
struct EncodeSink {
    /// Newest un-encoded pair; `None` once the worker takes it.
    slot: FrameSlot,
    /// Worker's last measured encode×2 time (µs), for the profiling line.
    enc_us: Arc<AtomicU32>,
}

impl EncodeSink {
    /// Bind the server and spawn the H.264 encode worker. `w`/`h` size the main view,
    /// `bw`/`bh` the BEV; `main_kbps`/`bev_kbps` are the target bitrates.
    fn spawn(
        port: u16,
        w: usize,
        h: usize,
        bw: usize,
        bh: usize,
        main_kbps: u32,
        bev_kbps: u32,
    ) -> Res<Self> {
        let server = MjpegServer::spawn(port)?;
        let slot = Arc::new((Mutex::new(None), Condvar::new()));
        let enc_us = Arc::new(AtomicU32::new(0));
        let (wslot, wenc) = (slot.clone(), enc_us.clone());
        std::thread::spawn(move || {
            let mut menc = H264Encoder::new(w, h, STREAM_FPS, main_kbps, STREAM_FPS)
                .expect("h264 main encoder");
            let mut benc = H264Encoder::new(bw, bh, STREAM_FPS, bev_kbps, STREAM_FPS)
                .expect("h264 bev encoder");
            println!(
                "     stream: H.264 x264 sw — main {w}x{h}@{main_kbps}k + bev {bw}x{bh}@{bev_kbps}k \
                 (WebSocket/WebCodecs)"
            );
            let _ = std::io::stdout().flush();
            let (lock, cv) = &*wslot;
            loop {
                let pair: FramePair = {
                    let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
                    while g.is_none() {
                        g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
                    }
                    g.take().unwrap()
                };
                let (main_rgb, bev_rgb) = pair;
                let t = Instant::now();
                for (enc, tag, rgb) in [(&mut menc, b'M', main_rgb), (&mut benc, b'B', bev_rgb)] {
                    if let Ok(aus) = enc.encode(&rgb) {
                        if let Some(cd) = enc.codec_data() {
                            server.publish_h264_config(tag, cd);
                        }
                        for au in aus {
                            server.publish_h264_frame(tag, au.key, au.data);
                        }
                    }
                }
                wenc.store(t.elapsed().as_micros() as u32, Ordering::Relaxed);
            }
        });
        Ok(Self { slot, enc_us })
    }

    /// Hand the latest rendered pair to the worker, overwriting any pair not yet
    /// encoded (drop-stale → the browser always gets the freshest frame).
    fn submit(&self, main: Vec<u8>, bev: Vec<u8>) {
        let (lock, cv) = &*self.slot;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = Some((main, bev));
        cv.notify_one();
    }
}

fn main() -> Res<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: rtsp_track <seg.engine> <depth.engine> <rtsp://url> [conf] [out.png|out.gif|serve|:PORT]");
        std::process::exit(1);
    }
    let (seg_engine, depth_engine, url) = (&args[1], &args[2], &args[3]);
    let conf: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.4);
    let mut out = args.get(5).cloned();

    // One shared CUDA stream: source copy, seg, depth, and the depth-at-mask fusion
    // all enqueue on it, so a single sync completes the frame's GPU work.
    let stream = CudaContext::new(0)?.default_stream();
    let mut source = RtspSource::connect_resized(url, 1280, 720, stream.clone())?;
    let (w, h) = (source.width() as usize, source.height() as usize);
    println!("stream {w}x{h} → RF-DETR-Seg (conf ≥ {conf}) + Depth Anything V2 + 3D tracker");

    let mut seg = RfDetrSeg::from_engine_file(seg_engine, stream.clone(), conf)?;
    let mut depth = DepthAnything::from_engine_file(depth_engine, stream.clone())?;
    let mut d = seg.alloc_result()?;
    let mut z = depth.alloc_result()?;
    // Diagnostic A/B: `RTSP_TRACK_NO_DEPTH=1` disables the depth gate + soft cost so
    // ID stability can be compared with vs without the 3D association terms.
    let mut cfg = TrackerConfig::default();
    if std::env::var("RTSP_TRACK_NO_DEPTH").is_ok() {
        cfg.depth_gate = false; // A/B toggle for the depth-gate's effect on ID stability
        println!("(depth gate DISABLED for A/B)");
    }
    let mut tracker = Tracker::new(cfg)?;
    let intr = CameraIntrinsics::from_hfov(w as f32, h as f32, TAPO_C210_HFOV_DEG);
    // Lens undistort (before seg/depth) → rectified pinhole for the whole pipeline.
    let undist = Undistorter::new(&intr, TAPO_C210_K1, w, h, &stream)?;
    let mut rect = Image::<u8, 3>::zeros_cuda(
        ImageSize {
            width: w,
            height: h,
        },
        &stream,
    )?; // reused

    // Output mode from the 5th arg.
    let record = out.as_deref().is_some_and(|p| p.ends_with(".gif"));
    let serve_port = out.as_deref().and_then(parse_port);
    let enc_sink = match serve_port {
        Some(port) => {
            println!("live: open http://<this-jetson-ip>:{port} in a phone browser (same Wi-Fi)");
            // H.264 target bitrates (kbit/s). Full res streams comfortably at a few
            // Mbit/s thanks to inter-frame compression; tune for tighter links.
            let kbps = |var: &str, def: u32| {
                std::env::var(var)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .filter(|k: &u32| *k >= 100 && *k <= 20000)
                    .unwrap_or(def)
            };
            let (main_kbps, bev_kbps) = (
                kbps("RTSP_TRACK_MAIN_KBPS", 3500),
                kbps("RTSP_TRACK_BEV_KBPS", 1500),
            );
            Some(EncodeSink::spawn(
                port, w, h, BEV_W, BEV_H, main_kbps, bev_kbps,
            )?)
        }
        None => None,
    };
    // GIF = main stacked over the BEV, downscaled to `gw` wide.
    let (gw, rec_secs) = (640usize, 10.0);
    let (stack_w, stack_h) = (w.max(BEV_W), h + BEV_H);
    let gh = gw * stack_h / stack_w;
    let mut gif_frames: Vec<Vec<u8>> = Vec::new();

    let mut dets: Vec<Detection> = Vec::new();
    let mut trails = TrailStore::new(); // per-track metric path for the BEV
    let mut window_ids: HashSet<u64> = HashSet::new(); // distinct ids per 100-frame window
    let ms = |dur: std::time::Duration| dur.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_fus, mut a_sync, mut a_read, mut a_trk) =
        (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut prev, mut ema_dt, mut a_dt) = (Instant::now(), 0.0f64, 0.0f64);
    let mut a_render = 0.0f64;
    loop {
        let t0 = Instant::now();
        let Some(frame) = source.next_frame() else {
            break;
        };
        let t1 = Instant::now();
        undist.apply(frame.image(), &mut rect, &stream)?; // rectify on the shared stream
        seg.submit(&rect, &mut d)?;
        depth.submit(&rect, &mut z)?;
        let t2 = Instant::now();
        let zs = z.depth_image().sample_masks(
            d.masks_slice(),
            d.mask_size(),
            d.count_slice(),
            &stream,
        )?;
        let t3 = Instant::now();
        stream.synchronize()?; // the one sync completes source + detect + depth + fusion
        let t4 = Instant::now();

        // Readout: survivor boxes + per-instance metric depth (live-count prefix only).
        let inst_n = d.count();
        let detections = d.detections()?;
        let z_m = stream.clone_dtoh(&zs.slice(0..inst_n))?;
        let t5 = Instant::now();

        // Feed the tracker: each box carries its mask-sampled metric depth → `pz`.
        dets.clear();
        for (det, &zv) in detections.iter().zip(&z_m) {
            let mut det = Detection::new(det.bbox, det.score, det.class_id);
            if zv > 0.0 {
                det = det.with_depth(zv);
            }
            dets.push(det);
        }
        // Real inter-frame dt in nominal-frame units (EMA-calibrated) → jitter-robust.
        let interval = t0.duration_since(prev).as_secs_f64();
        prev = t0;
        let dt = if ema_dt > 0.0 {
            (interval / ema_dt).clamp(0.25, 4.0)
        } else {
            1.0
        };
        if interval > 1e-3 {
            ema_dt = if ema_dt > 0.0 {
                0.9 * ema_dt + 0.1 * interval
            } else {
                interval
            };
        }
        let tracks = tracker.update_dt(&dets, dt);
        let t6 = Instant::now();
        trails.update(&tracks, &intr); // accumulate BEV motion trails
        for t in tracks.iter().filter(|t| t.state == TrackState::Confirmed) {
            window_ids.insert(t.id); // churn: distinct ids over the window vs live count
        }

        // ── viz (vrt-viz): render main + BEV, then serve / record / dump ──
        // Only one of the three sinks is active per run; each renders the same pair.
        // Smoothed end-to-end loop rate for the on-frame HUD (EMA of the frame interval).
        let fps = if ema_dt > 0.0 {
            (1.0 / ema_dt) as f32
        } else {
            0.0
        };
        if let Some(sink) = &enc_sink {
            // Render on this thread (needs the GPU host-copies + trails); hand the two
            // RGB buffers to the worker, which encodes + publishes off the hot path.
            let (main, bev) = render_pair(&rect, &stream, &d, w, h, &tracks, &intr, &trails, fps)?;
            sink.submit(main, bev);
            a_render += ms(Instant::now() - t6);
        } else if record {
            if t_start.elapsed().as_secs_f64() < rec_secs {
                if n.is_multiple_of(2) {
                    let (main, bev) =
                        render_pair(&rect, &stream, &d, w, h, &tracks, &intr, &trails, fps)?;
                    let (st, sw, sh) = stack_v(&main, w, h, &bev, BEV_W, BEV_H);
                    gif_frames.push(downscale(&st, sw, sh, gw, gh));
                }
            } else if let Some(path) = out.take() {
                write_gif(&path, &gif_frames, gw as u16, gh as u16, 13)?;
                println!("     saved clip → {path} ({} frames)", gif_frames.len());
                break;
            }
        } else if let Some(path) = out.take_if(|_| n == 60) {
            let (main, bev) = render_pair(&rect, &stream, &d, w, h, &tracks, &intr, &trails, fps)?;
            let (st, sw, sh) = stack_v(&main, w, h, &bev, BEV_W, BEV_H);
            encode_png(&path, &st, sw, sh)?;
            println!("     saved overlay → {path}");
        }

        n += 1;
        a_src += ms(t1 - t0);
        a_enq += ms(t2 - t1);
        a_fus += ms(t3 - t2);
        a_sync += ms(t4 - t3);
        a_read += ms(t5 - t4);
        a_trk += ms(t6 - t5);
        a_dt += dt;
        if n.is_multiple_of(100) {
            // Per-window profiling + track/churn stats — silent by default, opt in with
            // `RUST_LOG=rtsp_track=debug`.
            if log::log_enabled!(log::Level::Debug) {
                let k = 100.0;
                let confirmed = tracks
                    .iter()
                    .filter(|t| t.state == TrackState::Confirmed)
                    .count();
                // Churn: distinct confirmed ids this window vs the live count. Static
                // scene ⇒ distinct ≈ live; distinct ≫ live ⇒ ids are switching.
                let distinct = window_ids.len();
                log::debug!(
                    "{n} frames | {:.1} fps | source {:.2} | enqueue {:.3} | fusion {:.3} | \
                     sync(GPU) {:.2} | readout {:.3} | track {:.3} | dt {:.2} | {inst_n} det → \
                     {confirmed} conf | {distinct} distinct-ids/100f",
                    n as f64 / t_start.elapsed().as_secs_f64(),
                    a_src / k,
                    a_enq / k,
                    a_fus / k,
                    a_sync / k,
                    a_read / k,
                    a_trk / k,
                    a_dt / k,
                );
                let spf = if ema_dt > 0.0 {
                    ema_dt as f32
                } else {
                    1.0 / 15.0
                };
                let shown: Vec<String> = tracks
                    .iter()
                    .filter(|t| t.state == TrackState::Confirmed)
                    .take(6)
                    .map(|t| {
                        let [x, y, zz] = t.metric_position(&intr);
                        let mv = t.metric_velocity(&intr);
                        let speed = (mv[0].powi(2) + mv[1].powi(2) + mv[2].powi(2)).sqrt() / spf;
                        format!(
                            "#{} {} [X{x:+.1} Y{y:+.1} Z{zz:.1}]m {speed:.1}m/s",
                            t.id,
                            coco_name(t.class_id)
                        )
                    })
                    .collect();
                log::debug!("tracks: {}", shown.join(", "));
                if let Some(sink) = &enc_sink {
                    let enc_ms = sink.enc_us.load(Ordering::Relaxed) as f64 / 1000.0;
                    log::debug!(
                        "serve: render {:.1} ms | encode×2 {enc_ms:.1} ms (worker, off hot path)",
                        a_render / k,
                    );
                }
            }
            window_ids.clear();
            (a_src, a_enq, a_fus, a_sync, a_read, a_trk, a_dt, a_render) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }
    Ok(())
}

/// The GPU/model-specific bridge to `vrt-viz`: host-copy the device frame, decode the
/// instance masks, and render the main + BEV pair (owned buffers the caller consumes).
#[allow(clippy::too_many_arguments)]
fn render_pair(
    rect: &Image<u8, 3>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d: &vrt_rfdetr_seg::SegResult,
    w: usize,
    h: usize,
    tracks: &[vrt_track::Track],
    intr: &CameraIntrinsics,
    trails: &TrailStore,
    fps: f32,
) -> Res<(Vec<u8>, Vec<u8>)> {
    let host = rect.to_host(stream)?;
    let insts = d.instances()?;
    let masks: Vec<MaskOverlay> = insts
        .iter()
        .map(|i| MaskOverlay {
            mask: &i.mask,
            mask_wh: i.mask_size,
            bbox: i.bbox,
        })
        .collect();
    let main = render_main(host.as_slice().to_vec(), w, h, &masks, tracks, fps);
    let bev = render_bev(tracks, intr, BEV_W, BEV_H, Some(trails));
    Ok((main, bev))
}

/// Parse the output arg into an MJPEG server port: `serve` → 8080, `:PORT` → PORT.
fn parse_port(s: &str) -> Option<u16> {
    if s == "serve" {
        Some(8080)
    } else {
        s.strip_prefix(':').and_then(|p| p.parse().ok())
    }
}

/// COCO 91-class category-id → name (for the stdout track list).
fn coco_name(id: u32) -> &'static str {
    COCO91.get(id as usize).copied().unwrap_or("?")
}

const COCO91: [&str; 91] = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
