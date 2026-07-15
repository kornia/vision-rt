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
//!   vrt_viz::render_main / render_bev → H.264/WebSocket live view / GIF / PNG
//! ```
//!
//! Build:  export CARGO_NET_GIT_FETCH_WITH_CLI=true
//!   cargo run --release --manifest-path examples/rtsp_track/Cargo.toml \
//!       -- <seg.engine> <depth.engine> rtsp://<camera>/stream [conf] [out]
//!
//! The optional 5th arg picks the output:
//!   `out.png`         — one annotated frame (main + BEV stacked) to that PNG
//!   `out.gif`         — a ~10 s animated-GIF clip, then exit
//!   `serve` | `:PORT` — H.264/WebSocket live stream (browser WebCodecs); open
//!                       `http://<jetson-ip>:PORT` in a phone browser (same LAN)

use std::collections::HashSet;
use std::time::Instant;

use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use sensor_rtsp::RtspSource;
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::RfDetrSeg;
use vrt_track::{CameraIntrinsics, Detection, TrackState, Tracker, TrackerConfig};
use vrt_types::Undistorter;
use vrt_viz::{
    downscale, encode_png, render_bev, render_main, stack_v, write_gif, LiveStream, MaskOverlay,
    TrailStore,
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

/// Nominal stream frame rate (camera-paced); also the encoder GOP (1 s keyframe
/// interval → a WebSocket viewer joins/re-syncs within ~1 s).
const STREAM_FPS: i32 = 15;

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
    // Monocular-honest measurement noise: a person's mask-sampled depth is ±0.3 m-ish
    // frame to frame (limbs/mask coverage vary), not the ±0.1 m the default assumes —
    // trusting it less smooths pz (and the BEV trail) without adding meaningful lag.
    // Static objects are unaffected (their z is constant). Image plane mildly smoothed
    // for the same reason; the velocity state keeps fast motion tracked.
    cfg.kalman.meas_depth = 0.35;
    cfg.kalman.std_depth = 0.1;
    cfg.kalman.meas_position = 1.5;
    cfg.kalman.std_position = 1.0;
    if std::env::var("RTSP_TRACK_NO_DEPTH").is_ok() {
        cfg.depth_gate = false; // A/B toggle for the depth-gate's effect on ID stability
        println!("(depth gate DISABLED for A/B)");
    }
    if std::env::var("RTSP_TRACK_NO_OCM").is_ok() {
        cfg.ocm_lambda = 0.0; // A/B: OC-SORT momentum off
        println!("(OCM DISABLED for A/B)");
    }
    if std::env::var("RTSP_TRACK_NO_ORU").is_ok() {
        cfg.oru = false; // A/B: OC-SORT re-update off
        println!("(ORU DISABLED for A/B)");
    }
    if std::env::var("RTSP_TRACK_DIOU").is_ok() {
        cfg.use_diou = true; // A/B: distance-aware IoU association
        println!("(DIoU association ENABLED)");
    }
    if std::env::var("RTSP_TRACK_DIOU3D").is_ok() {
        cfg.use_diou3d = true; // A/B: metric-3D DIoU association
        println!("(DIoU-3D association ENABLED)");
    }
    // Buffered-IoU margin: absorbs seg-mask box *shift* so static furniture whose box
    // lurches between frames re-associates instead of churning ids. Live A/B knob; the
    // synthetic regression (buffered_iou_absorbs_seg_box_shift_churn) shows ~0.5 kills a
    // 50px-shift ping-pong that plain IoU switches 29×.
    if let Ok(v) = std::env::var("RTSP_TRACK_IOU_BUFFER") {
        cfg.iou_buffer = v.parse().unwrap_or(0.0);
        println!("(Buffered-IoU margin = {})", cfg.iou_buffer);
    }
    // OSNet person re-id embedder (metric-learned — the only feature source trusted
    // for identity DECISIONS). `RTSP_TRACK_OSNET=<engine>` overrides; the default
    // on-box engine is picked up automatically. Without it, identity decisions are
    // disabled (reid_thresh=0) and appearance stays a tie-breaker only.
    const OSNET_DEFAULT: &str = "models/engines/osnet-reid-e78604f4-trt10.3.0.30-sm87.engine";
    let osnet_path =
        std::env::var("RTSP_TRACK_OSNET").unwrap_or_else(|_| OSNET_DEFAULT.to_string());
    let mut osnet = if std::path::Path::new(&osnet_path).exists() {
        match vrt_osnet::OsNetReid::from_engine_file(&osnet_path, stream.clone()) {
            Ok(r) => {
                println!(
                    "person re-id: OSNet ON (dim {}, batch {})",
                    r.dim(),
                    r.batch()
                );
                Some(r)
            }
            Err(e) => {
                println!("person re-id: OSNet failed to load ({e}) — identity decisions OFF");
                None
            }
        }
    } else {
        println!("person re-id: no OSNet engine — identity decisions OFF");
        None
    };
    // Reusable device buffer for OSNet embeddings (caller-owned async: submit → our sync
    // → host readout). Allocated once, only when OSNet is present.
    let mut reid_out = match &osnet {
        Some(o) => Some(o.alloc_result()?),
        None => None,
    };
    if osnet.is_some() {
        // OSNet cosine space, live-measured: same-person 0.005–0.18, different-person
        // ≈ 0.3+ — a real margin, unlike detector tokens. Identity decisions person-only.
        cfg.reid_thresh = 0.22;
        cfg.reid_classes = vec![1]; // COCO person
    } else {
        cfg.reid_thresh = 0.0; // no trained embedder → appearance never decides
    }
    // Lost-track re-id gate override; tune live via env, `0` disables the stage.
    if let Some(v) = std::env::var("RTSP_TRACK_REID")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        cfg.reid_thresh = v;
        println!("(reid_thresh = {v})");
    }
    let intr = CameraIntrinsics::from_hfov(w as f32, h as f32, TAPO_C210_HFOV_DEG);
    let mut tracker = Tracker::new(cfg, intr)?; // intr drives the metric OCM/ORU cues
                                                // Appearance ReID tie-breaker: attach the seg backbone's pooled block-11 embedding
                                                // to large detections so the tracker can re-associate through occlusion (min-fused
                                                // into IoU, depth-gated, never overrides geometry). A/B toggle via
                                                // `RTSP_TRACK_APPEARANCE=0`; auto-off if the engine has no `tokens` output.
    let appearance = std::env::var("RTSP_TRACK_APPEARANCE")
        .map(|v| v != "0")
        .unwrap_or(true)
        && seg.feat_dim() > 0;
    // Below ~64px a box spans too few backbone tokens for a reliable embedding
    // (Step-0 ablation: RF-DETR tokens collapse under ~32px) — those stay geometry-only.
    const MIN_FEAT_PX: f32 = 64.0;
    println!(
        "appearance ReID tie-breaker: {} (feat_dim {})",
        if appearance { "ON" } else { "off" },
        seg.feat_dim()
    );
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
    // `dump:<dir>` mode: per confirmed track per frame, write an RGB crop + mask crop +
    // a tracks.jsonl log — the pseudo-GT gallery source for the ReID token-source probe.
    let dump_dir = out
        .as_deref()
        .and_then(|p| p.strip_prefix("dump:").map(str::to_string));
    let dump_secs: f64 = std::env::var("RTSP_TRACK_DUMP_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20.0);
    let mut dump_log = match &dump_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir)?;
            println!("dump: per-track crops+masks+log → {dir} for {dump_secs}s");
            Some(std::fs::File::create(format!("{dir}/tracks.jsonl"))?)
        }
        None => None,
    };
    // `RTSP_TRACK_LOG=<path>`: lightweight per-frame JSONL of every detection + every
    // confirmed track (no crops — cheap enough to run alongside the live server). The
    // churn-diagnosis feed: id births/deaths/switches are derived offline.
    let mut track_log = match std::env::var("RTSP_TRACK_LOG") {
        Ok(p) => {
            println!("track log → {p}");
            Some(std::io::BufWriter::new(std::fs::File::create(p)?))
        }
        Err(_) => None,
    };
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
            println!(
                "     stream: H.264 x264 sw — main {w}x{h}@{main_kbps}k + bev {BEV_W}x{BEV_H}@{bev_kbps}k (WebSocket/WebCodecs)"
            );
            let live = LiveStream::spawn(
                port,
                (w, h),
                (BEV_W, BEV_H),
                main_kbps,
                bev_kbps,
                STREAM_FPS,
            )?;
            Some(live)
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
    // Previous frame's capture PTS (ns) — the tracker steps by real *capture* time when
    // the decoder supplies it, falling back to wall-clock arrival otherwise.
    let mut prev_pts: Option<u64> = None;
    let mut pts_mode_logged = false;
    let mut a_render = 0.0f64;
    loop {
        let t0 = Instant::now();
        let Some(frame) = source.next_frame() else {
            break;
        };
        let pts_ns = frame.meta.pts_ns; // camera capture timestamp (ns), if the decoder set one
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
        // Per-instance appearance embeddings (only when the tie-breaker is on).
        let feats = if appearance {
            d.features_host()?
        } else {
            Vec::new()
        };
        // OSNet person embeddings (metric-learned → drive identity decisions). Runs
        // AFTER the main sync on the still-alive device frame; only when persons exist.
        // Caller-owned async: submit enqueues crop→TRT→L2-norm, we drain with our own
        // sync, then read the host embeddings.
        let mut person_embs: std::collections::HashMap<usize, Vec<f32>> = Default::default();
        if let (Some(osnet), Some(reid_out)) = (&mut osnet, &mut reid_out) {
            let idx: Vec<usize> = detections
                .iter()
                .enumerate()
                .filter(|(_, det)| det.class_id == 1 && det.score >= 0.5)
                .map(|(i, _)| i)
                .take(osnet.batch())
                .collect();
            if !idx.is_empty() {
                let boxes: Vec<[f32; 4]> = idx.iter().map(|&i| detections[i].bbox).collect();
                match osnet.submit(&rect, &boxes, reid_out) {
                    Ok(()) => {
                        stream.synchronize()?; // drain the OSNet enqueue
                        match reid_out.embeddings_host() {
                            Ok(embs) => {
                                for (k, &i) in idx.iter().enumerate() {
                                    if let Some(e) = embs.get(k) {
                                        person_embs.insert(i, e.clone());
                                    }
                                }
                            }
                            Err(e) => log::debug!("osnet readout failed: {e}"),
                        }
                    }
                    Err(e) => log::debug!("osnet submit failed: {e}"),
                }
            }
        }
        let t5 = Instant::now();

        // Feed the tracker: each box carries its mask-sampled metric depth → `pz`, and
        // an appearance embedding — OSNet for persons (identity-grade), the pooled
        // backbone tokens for other large boxes (tie-breaker only).
        dets.clear();
        for (i, (det, &zv)) in detections.iter().zip(&z_m).enumerate() {
            let mut d2 = Detection::new(det.bbox, det.score, det.class_id);
            if zv > 0.0 {
                d2 = d2.with_depth(zv);
            }
            let (bw, bh) = (det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]);
            if let Some(e) = person_embs.remove(&i) {
                d2 = d2.with_feature(e);
            } else if appearance && bw.min(bh) >= MIN_FEAT_PX {
                // Skip zero embeddings (empty/stale mask slot).
                if let Some(f) = feats.get(i).filter(|f| f.iter().any(|&v| v != 0.0)) {
                    d2 = d2.with_feature(f.clone());
                }
            }
            dets.push(d2);
        }
        // Real inter-frame dt for the constant-velocity predict. Professional trackers
        // step by **capture** time (sensor PTS) so network + decoder jitter can't
        // masquerade as object motion; wall-clock arrival is only the fallback. Guard a
        // non-monotonic / absurd PTS delta (RTSP reconnect resets the timebase) by
        // dropping to arrival time. The EMA-normalise + [0.25,4] clamp stay as the final
        // backstop for either source.
        let arrival = t0.duration_since(prev).as_secs_f64();
        prev = t0;
        let interval = match (prev_pts, pts_ns) {
            (Some(p0), Some(p1)) if p1 > p0 => {
                let d = (p1 - p0) as f64 / 1e9;
                if d.is_finite() && d < 2.0 {
                    d
                } else {
                    arrival
                } // >2 s gap ⇒ reset, use arrival
            }
            _ => arrival, // no PTS, or backward/equal (reset) ⇒ arrival
        };
        prev_pts = pts_ns;
        if !pts_mode_logged {
            println!(
                "frame timing: {}",
                if pts_ns.is_some() {
                    "capture PTS (jitter-robust)"
                } else {
                    "wall-clock arrival (no decoder PTS)"
                }
            );
            pts_mode_logged = true;
        }
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
        for e in tracker.reid_events() {
            log::debug!(
                "REID id{} cos={:.3} jump={:.0}px lost={}f",
                e.track_id,
                e.cos_dist,
                e.jump_px,
                e.lost_frames
            );
        }
        if let Some(log) = &mut track_log {
            use std::io::Write;
            // One "r" line per re-id event, one "d" line per raw detection, one "k" line
            // per confirmed track.
            for e in tracker.reid_events() {
                writeln!(
                    log,
                    r#"{{"f":{n},"t":"r","id":{},"cos":{:.4},"jump":{:.0},"lost":{}}}"#,
                    e.track_id, e.cos_dist, e.jump_px, e.lost_frames
                )?;
            }
            for (det, &zv) in detections.iter().zip(&z_m) {
                writeln!(
                    log,
                    r#"{{"f":{n},"t":"d","c":{},"s":{:.3},"b":[{:.0},{:.0},{:.0},{:.0}],"z":{:.2}}}"#,
                    det.class_id, det.score, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], zv
                )?;
            }
            for t in tracks.iter().filter(|t| t.state == TrackState::Confirmed) {
                writeln!(
                    log,
                    r#"{{"f":{n},"t":"k","id":{},"c":{},"s":{:.3},"b":[{:.0},{:.0},{:.0},{:.0}],"z":{:.2},"h":{}}}"#,
                    t.id,
                    t.class_id,
                    t.score,
                    t.bbox[0],
                    t.bbox[1],
                    t.bbox[2],
                    t.bbox[3],
                    t.position_3d[2],
                    t.hits
                )?;
            }
        }
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
        if let Some(dir) = &dump_dir {
            dump_frame(
                dir,
                n,
                &rect,
                &stream,
                &d,
                &tracks,
                w,
                h,
                dump_log.as_mut().unwrap(),
            )?;
            if t_start.elapsed().as_secs_f64() > dump_secs {
                println!("dump: done ({n} frames)");
                break;
            }
        } else if let Some(sink) = &enc_sink {
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
                if enc_sink.is_some() {
                    // Encode time is logged by the worker thread; this is the main-thread
                    // render cost.
                    log::debug!("serve render {:.1} ms", a_render / k);
                }
            }
            window_ids.clear();
            (a_src, a_enq, a_fus, a_sync, a_read, a_trk, a_dt, a_render) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }
    Ok(())
}

/// Dump one frame's per-track crops for the ReID gallery: host-copy the device frame,
/// and for each **confirmed** track write an RGB crop (its bbox), the best-IoU instance's
/// mask crop (upsampled to the crop, gray RGB), and a `tracks.jsonl` row. `<dir>/<id>/`.
#[allow(clippy::too_many_arguments)]
fn dump_frame(
    dir: &str,
    n: u64,
    rect: &Image<u8, 3>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d: &vrt_rfdetr_seg::SegResult,
    tracks: &[vrt_track::Track],
    w: usize,
    h: usize,
    log: &mut std::fs::File,
) -> Res<()> {
    use std::io::Write;
    let host = rect.to_host(stream)?;
    let rgb = host.as_slice();
    let insts = d.instances()?;
    for t in tracks.iter().filter(|t| t.state == TrackState::Confirmed) {
        let x1 = t.bbox[0].max(0.0) as usize;
        let y1 = t.bbox[1].max(0.0) as usize;
        let x2 = (t.bbox[2] as usize).min(w);
        let y2 = (t.bbox[3] as usize).min(h);
        if x2 <= x1 + 8 || y2 <= y1 + 8 {
            continue; // skip degenerate / tiny boxes
        }
        let (cw, ch) = (x2 - x1, y2 - y1);
        // RGB crop (copy the bbox rows out of the host frame).
        let mut crop = vec![0u8; cw * ch * 3];
        for row in 0..ch {
            let src = ((y1 + row) * w + x1) * 3;
            let dst = row * cw * 3;
            crop[dst..dst + cw * 3].copy_from_slice(&rgb[src..src + cw * 3]);
        }
        let iddir = format!("{dir}/{}", t.id);
        std::fs::create_dir_all(&iddir)?;
        vrt_viz::encode_png(&format!("{iddir}/{n:06}.png"), &crop, cw, ch)?;
        // Mask crop from the best-IoU instance, sampled to the crop grid, gray RGB.
        let best = insts
            .iter()
            .map(|i| (vrt_track::iou(&t.bbox, &i.bbox), i))
            .filter(|(io, _)| *io > 0.2)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, i)| i);
        if let Some(inst) = best {
            let (mw, mh) = inst.mask_size;
            let (sx, sy) = (mw as f32 / w as f32, mh as f32 / h as f32);
            let mut m = vec![0u8; cw * ch * 3];
            for row in 0..ch {
                let my = (((y1 + row) as f32 * sy) as usize).min(mh - 1);
                for col in 0..cw {
                    let mx = (((x1 + col) as f32 * sx) as usize).min(mw - 1);
                    let v = if inst.mask[my * mw + mx] != 0 { 255 } else { 0 };
                    let o = (row * cw + col) * 3;
                    m[o] = v;
                    m[o + 1] = v;
                    m[o + 2] = v;
                }
            }
            vrt_viz::encode_png(&format!("{iddir}/{n:06}.mask.png"), &m, cw, ch)?;
        }
        writeln!(
            log,
            r#"{{"frame":{n},"id":{},"class_id":{},"bbox":[{:.1},{:.1},{:.1},{:.1}],"depth":{:.3},"hits":{},"score":{:.3}}}"#,
            t.id,
            t.class_id,
            t.bbox[0],
            t.bbox[1],
            t.bbox[2],
            t.bbox[3],
            t.position_3d[2],
            t.hits,
            t.score
        )?;
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
    // Render only confident instances: the detector now runs LOW (feeding ByteTrack's
    // weak-detection recovery tier), but weak masks would clutter the view.
    let render_conf = render_conf();
    let masks: Vec<MaskOverlay> = insts
        .iter()
        .filter(|i| i.score >= render_conf)
        .map(|i| MaskOverlay {
            mask: &i.mask,
            mask_wh: i.mask_size,
            bbox: i.bbox,
        })
        .collect();
    let main = render_main(host.into_vec(), w, h, &masks, tracks, fps);
    let bev = render_bev(tracks, intr, BEV_W, BEV_H, Some(trails));
    Ok((main, bev))
}

/// Mask-render confidence floor (`RTSP_TRACK_RENDER_CONF`, default 0.4): the detector
/// runs low so the tracker's weak-detection tier sees occluded objects, but only
/// confident instances are drawn. Cached — read once.
fn render_conf() -> f32 {
    use std::sync::OnceLock;
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("RTSP_TRACK_RENDER_CONF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.4)
    })
}

/// Parse the output arg into a live-stream port: `serve` → 8080, `:PORT` → PORT.
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
