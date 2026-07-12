//! RTSP → RF-DETR-Seg + Depth Anything V2 + **BoT-SORT tracking**, device end-to-end.
//!
//! The seg + depth pipeline is exactly the one-stream / one-sync flow of
//! `rtsp_depth`; this example adds the tracker on top. The mask-sampled metric depth
//! feeds each detection's `pz` (`Detection::with_depth`), so the 3D Kalman carries a
//! **real metric-depth axis + approach velocity `vz`** per track instead of coasting:
//!
//! ```text
//!   source → seg.submit ─┐
//!            depth.submit ─┼─ one stream, one sync
//!            sample_masks ─┘ → [count] metric z (depth-at-mask, on device)
//!   ── stream.synchronize() ──
//!   readout: boxes + score + class + per-instance z
//!   Detection::new(box, score, class).with_depth(z)  → BotSort::update → [Track]
//! ```
//!
//! Everything GPU stays async / caller-owned; the tracker is pure CPU (µs). Build:
//!   export CARGO_NET_GIT_FETCH_WITH_CLI=true
//!   cargo run --release --manifest-path examples/rtsp_track/Cargo.toml \
//!       -- <seg.engine> <depth.engine> rtsp://<camera>/stream [conf] [out]
//!
//! The optional 5th arg picks the output (annotated = masks tinted + tracked boxes
//! coloured by id, each labelled `<id> <depth>m`):
//!   `out.png`      — dump ONE annotated frame to that PNG
//!   `out.gif`      — record a ~10 s animated-GIF clip, then exit
//!   `serve` | `:PORT` — MJPEG-over-HTTP live stream; open `http://<jetson-ip>:PORT`
//!                       in a phone browser on the same LAN (no app, no ffmpeg)

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use kornia_io::jpegturbo::JpegTurboEncoder;
use kornia_io::png::write_image_png_rgb8;
use sensor_rtsp::RtspSource;
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::{Instance, RfDetrSeg};
use vrt_track::{BotSort, BotSortConfig, CameraIntrinsics, Detection, Track, TrackState};

/// Approximate horizontal FoV of the Tapo C210 (deg). TP-Link publishes no angular
/// FoV — only the 3.83 mm F/2.4 lens — so this is computed from the lens on the
/// C210's 1/2.9" 16:9 sensor (~5.12 mm wide): `2·atan(5.12 / (2·3.83)) ≈ 67°`.
/// Replace with a checkerboard calibration for accurate metres.
const TAPO_C210_HFOV_DEG: f32 = 67.0;

/// Standalone BEV canvas size (matches the 1280-wide main frame so they stack clean).
const BEV_W: usize = 1280;
const BEV_H: usize = 640;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Res<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: rtsp_track <seg.engine> <depth.engine> <rtsp://url> [conf] \
             [out.png | out.gif | serve | :PORT]"
        );
        std::process::exit(1);
    }
    let seg_engine = &args[1];
    let depth_engine = &args[2];
    let url = &args[3];
    let conf: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.4);
    let mut out_png = args.get(5).cloned(); // dump one annotated frame here, then keep running

    // One shared CUDA stream: source un-pitch copy, seg, depth, and the depth-at-mask
    // fusion all enqueue on it, so a single sync completes the frame's GPU work.
    let stream = CudaContext::new(0)?.default_stream();
    let mut source = RtspSource::connect_resized(url, 1280, 720, stream.clone())?;
    let (w, h) = (source.width(), source.height());
    println!("stream {w}x{h} → RF-DETR-Seg (conf ≥ {conf}) + Depth Anything V2 + BoT-SORT");

    let mut seg = RfDetrSeg::from_engine_file(seg_engine, stream.clone(), conf)?;
    let mut depth = DepthAnything::from_engine_file(depth_engine, stream.clone())?;
    let mut d = seg.alloc_result()?; // reused segmenter output
    let mut z = depth.alloc_result()?; // reused depth output
                                       // The tracker: pure CPU, reused across frames. Metric depth feeds `pz`.
    let mut tracker = BotSort::new(BotSortConfig::default())?;
    // Intrinsics for the metric-3D readout (back-project px,py,pz → X,Y,Z metres).
    let intr = CameraIntrinsics::from_hfov(w as f32, h as f32, TAPO_C210_HFOV_DEG);

    // Output mode from the 5th arg:
    //   *.gif    → record a ~10 s animated-GIF clip, then exit
    //   serve | :PORT → MJPEG-over-HTTP live stream (open http://<jetson-ip>:PORT
    //                   from a phone browser on the same LAN)
    //   *.png    → dump one annotated frame at frame 60
    let record = out_png.as_deref().is_some_and(|p| p.ends_with(".gif"));
    let serve_port = out_png.as_deref().and_then(parse_port);
    // GIF = main stacked over the BEV, downscaled to `gw` wide (height keeps aspect).
    let (gw, rec_secs) = (640usize, 10.0);
    let (stack_w, stack_h) = ((w as usize).max(BEV_W), h as usize + BEV_H);
    let gh = gw * stack_h / stack_w;
    let mut gif_frames: Vec<Vec<u8>> = Vec::new();
    // Live-stream shared frames: the loop writes the latest **main** (camera + masks
    // + boxes) and **BEV** (top-down map) JPEGs here; the HTTP server pushes each on
    // its own MJPEG endpoint (`/main`, `/bev`) and an index page stacks them.
    let latest_main: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let latest_bev: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let jenc = JpegTurboEncoder::new()?; // SIMD JPEG, reused every frame
    jenc.set_quality(72)?;
    jenc.set_subsamp(turbojpeg::Subsamp::Sub2x2)?; // 4:2:0 → ~2× faster + smaller
    if let Some(port) = serve_port {
        spawn_mjpeg_server(port, latest_main.clone(), latest_bev.clone());
        println!("live: open http://<this-jetson-ip>:{port} in a phone browser (same Wi-Fi)");
    }
    // Reused per-frame detection buffer (cleared, not reallocated).
    let mut dets: Vec<Detection> = Vec::new();

    // Per-stage profiler: enqueue (2× submit) + fusion (sample_masks) are CPU
    // kernel-launches, ≪ the single GPU `sync`; `track` is the CPU tracker cost.
    let ms = |dur: std::time::Duration| dur.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_fus, mut a_sync, mut a_read, mut a_trk) =
        (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    // Real per-frame dt in *nominal-frame units*: interval / EMA(interval). ≈1 at
    // steady fps, >1 after a dropped frame — keeps the Kalman predict consistent
    // under RTSP jitter without retuning the per-frame KalmanParams.
    let (mut prev, mut ema_dt, mut a_dt) = (Instant::now(), 0.0f64, 0.0f64);
    let (mut a_rmain, mut a_rbev, mut a_enc) = (0.0f64, 0.0, 0.0); // serve-mode render/encode
    loop {
        let t0 = Instant::now();
        let Some(frame) = source.next_frame() else {
            break;
        };
        let t1 = Instant::now();
        seg.submit(frame.image(), &mut d)?; // detect enqueue (async, no sync)
        depth.submit(frame.image(), &mut z)?; // depth enqueue (same frame + stream)
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
        let z_m = stream.clone_dtoh(&zs.slice(0..inst_n))?; // [inst_n] meters, aligned
        let t5 = Instant::now();

        // Feed the tracker: each box carries its mask-sampled metric depth → `pz`.
        dets.clear();
        for (det, &zv) in detections.iter().zip(&z_m) {
            let mut det = Detection::new(det.bbox, det.score, det.class_id);
            // Only attach depth that was actually measured (mask had valid pixels).
            if zv > 0.0 {
                det = det.with_depth(zv);
            }
            dets.push(det);
        }
        // Real inter-frame interval → dt in nominal-frame units (EMA-calibrated,
        // clamped). First frames run at dt=1 until the cadence is learned.
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

        if serve_port.is_some() {
            // Two independent JPEG streams: main (camera + masks + boxes) and the BEV.
            let r0 = Instant::now();
            let main = render_main(frame.image(), &stream, &d, &tracks)?;
            let r1 = Instant::now();
            let bev = render_bev(&tracks, &intr);
            let r2 = Instant::now();
            let jm = encode_jpeg(&jenc, &main, w as usize, h as usize)?;
            let jb = encode_jpeg(&jenc, &bev, BEV_W, BEV_H)?;
            let r3 = Instant::now();
            *latest_main.lock().unwrap_or_else(|e| e.into_inner()) = jm;
            *latest_bev.lock().unwrap_or_else(|e| e.into_inner()) = jb;
            a_rmain += ms(r1 - r0);
            a_rbev += ms(r2 - r1);
            a_enc += ms(r3 - r2);
        } else if record {
            if t_start.elapsed().as_secs_f64() < rec_secs {
                if n.is_multiple_of(2) {
                    // Stack main over the BEV → one tall frame; ~7.5 captured/s.
                    let main = render_main(frame.image(), &stream, &d, &tracks)?;
                    let bev = render_bev(&tracks, &intr);
                    let st = stack_v(&main, w as usize, h as usize, &bev, BEV_W, BEV_H);
                    gif_frames.push(downscale(&st, stack_w, stack_h, gw, gh));
                }
            } else if let Some(path) = out_png.take() {
                write_gif(&path, &gif_frames, gw as u16, gh as u16, 13)?;
                break;
            }
        } else if let Some(path) = out_png.take_if(|_| n == 60) {
            save_overlay(&path, frame.image(), &stream, &d, &tracks, &intr)?;
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
            let k = 100.0;
            let confirmed = tracks
                .iter()
                .filter(|t| t.state == TrackState::Confirmed)
                .count();
            println!(
                "── {n} frames | {:.1} fps | source {:.2} ms | enqueue {:.3} ms | \
                 fusion {:.3} ms | sync(GPU) {:.2} ms | readout {:.3} ms | track {:.3} ms | \
                 dt {:.2} | {inst_n} det → {confirmed} confirmed",
                n as f64 / t_start.elapsed().as_secs_f64(),
                a_src / k,
                a_enq / k,
                a_fus / k,
                a_sync / k,
                a_read / k,
                a_trk / k,
                a_dt / k,
            );
            // A few live tracks with their metric 3D position + speed. `metric_velocity`
            // is per nominal frame; divide by seconds/frame (the EMA interval) for m/s.
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
                        coco_name(t.class_id),
                    )
                })
                .collect();
            println!("     tracks: {}", shown.join(", "));
            if serve_port.is_some() {
                println!(
                    "     serve: render_main {:.1} ms | render_bev {:.1} ms | encode×2 {:.1} ms",
                    a_rmain / k,
                    a_rbev / k,
                    a_enc / k,
                );
            }
            let _ = std::io::stdout().flush(); // survive a timeout/SIGTERM kill
            a_src = 0.0;
            a_enq = 0.0;
            a_fus = 0.0;
            a_sync = 0.0;
            a_read = 0.0;
            a_trk = 0.0;
            a_dt = 0.0;
            a_rmain = 0.0;
            a_rbev = 0.0;
            a_enc = 0.0;
        }
    }
    Ok(())
}

/// Colour of a confirmed track by its id.
fn track_color(id: u64) -> [u8; 3] {
    PALETTE[(id as usize) % PALETTE.len()]
}

/// Render the **main** view: host-copy the frame, tint each instance mask in its
/// track's id colour (matched by box IoU), outline the track box + `<id> <depth>m`
/// label. Returns the `w×h` RGB buffer (no BEV — that is a separate image).
fn render_main(
    frame: &Image<u8, 3>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d: &vrt_rfdetr_seg::SegResult,
    tracks: &[Track],
) -> Res<Vec<u8>> {
    let (w, h) = (frame.width(), frame.height());
    let host = frame.to_host(stream)?;
    let mut buf = host.as_slice().to_vec();

    let confirmed: Vec<&Track> = tracks
        .iter()
        .filter(|t| t.state == TrackState::Confirmed)
        .collect();

    // Colour each mask by the id of the track it belongs to (best box-IoU match);
    // unmatched masks fall back to a neutral tint.
    for inst in &d.instances()? {
        let color = confirmed
            .iter()
            .map(|t| (iou(&t.bbox, &inst.bbox), t.id))
            .filter(|&(io, _)| io > 0.2)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, id)| track_color(id))
            .unwrap_or([110, 110, 110]);
        tint_mask(&mut buf, w, h, inst, color);
    }
    for t in &confirmed {
        let color = track_color(t.id);
        draw_box(&mut buf, w, h, t.bbox, color);
        let [x1, y1, ..] = t.bbox;
        let label = format!("{} {:.1}m", t.id, t.position_3d[2]);
        draw_label(&mut buf, w, h, x1 as i32 + 2, y1 as i32 + 2, &label, color);
    }
    Ok(buf)
}

/// Render the **BEV** as a standalone `BEV_W×BEV_H` top-down **floor plan**: an
/// orthographic metre grid (blueprint style) with the camera at the bottom edge
/// looking "up" into the room, and each track drawn as a **footprint rectangle**
/// sized by its real-world width (box width × depth ÷ fx), coloured by id.
fn render_bev(tracks: &[Track], intr: &CameraIntrinsics) -> Vec<u8> {
    let (w, h) = (BEV_W, BEV_H);
    let mut buf = vec![0u8; w * h * 3];
    let bg = [14u8, 18, 26];
    for p in buf.chunks_exact_mut(3) {
        p.copy_from_slice(&bg);
    }

    let zmax = 6.0f32; // depth into the room (metres)
    let margin = 34.0f32;
    let ppm = (h as f32 - 2.0 * margin) / zmax; // isotropic px/m (no distortion)
    let (ax, az) = (w as f32 / 2.0, h as f32 - margin); // camera at bottom-centre
    let xspan = (ax - 6.0) / ppm; // half-width visible (metres)
    let map = |x: f32, z: f32| ((ax + x * ppm) as i32, (az - z * ppm) as i32);

    // Metre grid (Z depth lines + labels, X lateral lines); axes brighter.
    let (grid, axis) = ([30u8, 38, 52], [64u8, 80, 104]);
    let ztop = (az - zmax * ppm) as i32;
    for zi in 0..=zmax as i32 {
        let (_, y) = map(0.0, zi as f32);
        draw_line(
            &mut buf,
            w,
            h,
            0,
            y,
            w as i32,
            y,
            if zi == 0 { axis } else { grid },
        );
        draw_label(&mut buf, w, h, 6, y - 8, &format!("{zi}m"), [96, 116, 140]);
    }
    for xi in -(xspan as i32)..=xspan as i32 {
        let (x, _) = map(xi as f32, 0.0);
        draw_line(
            &mut buf,
            w,
            h,
            x,
            ztop,
            x,
            az as i32,
            if xi == 0 { axis } else { grid },
        );
    }

    // Faint FoV cone outline for orientation (what the camera actually sees).
    let k = (intr.cx / intr.fx).max(0.05); // tan(half-FoV)
    for s in [-1.0f32, 1.0] {
        let (ex, ey) = map(s * k * zmax, zmax);
        draw_line(&mut buf, w, h, ax as i32, az as i32, ex, ey, [40, 58, 78]);
    }
    // Camera chevron.
    let (cx, cy) = (ax as i32, az as i32);
    fill_tri(
        &mut buf,
        w,
        h,
        (cx, cy - 12),
        (cx - 8, cy + 3),
        (cx + 8, cy + 3),
        [92, 212, 236],
    );

    // Object footprints: a metre-scaled rectangle per track at its (X, Z).
    for t in tracks {
        if t.state != TrackState::Confirmed {
            continue;
        }
        let [x_m, _, z_m] = t.metric_position(intr);
        if z_m <= 0.1 || z_m > zmax || x_m.abs() > xspan {
            continue;
        }
        let box_w = (t.bbox[2] - t.bbox[0]).max(1.0);
        let wm = (box_w * z_m / intr.fx).clamp(0.15, 3.0); // real width (m)
        let dm = (wm * 0.6).clamp(0.2, 1.5); // footprint depth guess
        let (fw, fd) = ((wm * ppm) as i32, (dm * ppm) as i32);
        let (mx, my) = map(x_m, z_m);
        let c = track_color(t.id);
        fill_rect_alpha(&mut buf, w, h, mx - fw / 2, my - fd / 2, fw, fd, c, 90);
        rect_outline(&mut buf, w, h, mx - fw / 2, my - fd / 2, fw, fd, c);
        draw_label(
            &mut buf,
            w,
            h,
            mx - 3,
            my - 3,
            &format!("{}", t.id),
            [240, 240, 248],
        );
    }
    buf
}

/// Alpha-blend a filled rectangle (`a` in 0..=255) at `(x, y)` size `rw×rh`.
fn fill_rect_alpha(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    color: [u8; 3],
    a: u16,
) {
    for yy in y.max(0)..(y + rh).min(h as i32) {
        for xx in x.max(0)..(x + rw).min(w as i32) {
            let o = (yy as usize * w + xx as usize) * 3;
            for k in 0..3 {
                buf[o + k] = ((buf[o + k] as u16 * (255 - a) + color[k] as u16 * a) / 255) as u8;
            }
        }
    }
}

/// Draw a rectangle outline at `(x, y)` size `rw×rh`.
fn rect_outline(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    color: [u8; 3],
) {
    draw_line(buf, w, h, x, y, x + rw, y, color);
    draw_line(buf, w, h, x, y + rh, x + rw, y + rh, color);
    draw_line(buf, w, h, x, y, x, y + rh, color);
    draw_line(buf, w, h, x + rw, y, x + rw, y + rh, color);
}

/// Stack `top` (`tw×th`) over `bot` (`bw×bh`) into one RGB buffer of
/// `max(tw,bw) × (th+bh)`, each centred horizontally on a black canvas.
fn stack_v(top: &[u8], tw: usize, th: usize, bot: &[u8], bw: usize, bh: usize) -> Vec<u8> {
    let w = tw.max(bw);
    let mut out = vec![0u8; w * (th + bh) * 3];
    let blit = |out: &mut [u8], src: &[u8], sw: usize, sh: usize, y0: usize| {
        let xoff = (w - sw) / 2;
        for y in 0..sh {
            let d = ((y0 + y) * w + xoff) * 3;
            let s = y * sw * 3;
            out[d..d + sw * 3].copy_from_slice(&src[s..s + sw * 3]);
        }
    };
    blit(&mut out, top, tw, th, 0);
    blit(&mut out, bot, bw, bh, th);
    out
}

/// Encode an RGB buffer to JPEG with kornia-io's SIMD TurboJPEG encoder (reused).
fn encode_jpeg(enc: &JpegTurboEncoder, rgb: &[u8], w: usize, h: usize) -> Res<Vec<u8>> {
    let img = Image::<u8, 3>::new(
        ImageSize {
            width: w,
            height: h,
        },
        rgb.to_vec(),
    )?;
    Ok(enc.encode_rgb8(&img)?)
}

/// Fill a triangle (barycentric sign test), clipped to the frame.
fn fill_tri(
    buf: &mut [u8],
    w: usize,
    h: usize,
    a: (i32, i32),
    b: (i32, i32),
    c: (i32, i32),
    color: [u8; 3],
) {
    let e = |p: (i32, i32), q: (i32, i32), r: (i32, i32)| {
        (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)
    };
    let (minx, maxx) = (
        a.0.min(b.0).min(c.0).max(0),
        a.0.max(b.0).max(c.0).min(w as i32 - 1),
    );
    let (miny, maxy) = (
        a.1.min(b.1).min(c.1).max(0),
        a.1.max(b.1).max(c.1).min(h as i32 - 1),
    );
    for y in miny..=maxy {
        for x in minx..=maxx {
            let p = (x, y);
            let (w0, w1, w2) = (e(b, c, p), e(c, a, p), e(a, b, p));
            if (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0) {
                let o = (y as usize * w + x as usize) * 3;
                buf[o..o + 3].copy_from_slice(&color);
            }
        }
    }
}

/// IoU of two `[x1,y1,x2,y2]` boxes (for matching masks to tracks).
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (ix1, iy1) = (a[0].max(b[0]), a[1].max(b[1]));
    let (ix2, iy2) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let ua = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let ub = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let uni = ua + ub - inter;
    if uni <= 0.0 {
        0.0
    } else {
        inter / uni
    }
}

/// Render main + BEV, stack them vertically, and write one PNG.
fn save_overlay(
    path: &str,
    frame: &Image<u8, 3>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d: &vrt_rfdetr_seg::SegResult,
    tracks: &[Track],
    intr: &CameraIntrinsics,
) -> Res<()> {
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let main = render_main(frame, stream, d, tracks)?;
    let bev = render_bev(tracks, intr);
    let st = stack_v(&main, w, h, &bev, BEV_W, BEV_H);
    let (sw, sh) = (w.max(BEV_W), h + BEV_H);
    let img = Image::<u8, 3>::new(
        ImageSize {
            width: sw,
            height: sh,
        },
        st,
    )?;
    write_image_png_rgb8(path, &img)?;
    println!("     saved tracked overlay → {path}");
    Ok(())
}

/// Parse the output arg into an MJPEG server port: `serve` → 8080, `:PORT` → PORT.
fn parse_port(s: &str) -> Option<u16> {
    if s == "serve" {
        Some(8080)
    } else {
        s.strip_prefix(':').and_then(|p| p.parse().ok())
    }
}

/// Index page: the two MJPEG streams (`/main`, `/bev`) stacked vertically.
const INDEX_HTML: &str = "<!doctype html><html><head><meta name=viewport \
content='width=device-width,initial-scale=1'><style>body{margin:0;background:#111}\
img{display:block;width:100%;height:auto}</style></head><body>\
<img src=/main><img src=/bev></body></html>";

/// Spawn a background MJPEG-over-HTTP server. `/` serves an index that stacks the two
/// live streams; `/main` and `/bev` are each a `multipart/x-mixed-replace` MJPEG —
/// open `http://<ip>:port` in a phone browser (same LAN). No app, no ffmpeg.
fn spawn_mjpeg_server(port: u16, main: Arc<Mutex<Vec<u8>>>, bev: Arc<Mutex<Vec<u8>>>) {
    std::thread::spawn(move || {
        let listener = match TcpListener::bind(("0.0.0.0", port)) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("mjpeg: bind :{port} failed: {e}");
                return;
            }
        };
        for stream in listener.incoming().flatten() {
            let (main, bev) = (main.clone(), bev.clone());
            std::thread::spawn(move || {
                let _ = serve_client(stream, &main, &bev);
            });
        }
    });
}

/// Route one client by request path: `/main` / `/bev` stream MJPEG, else the index.
fn serve_client(
    mut s: TcpStream,
    main: &Arc<Mutex<Vec<u8>>>,
    bev: &Arc<Mutex<Vec<u8>>>,
) -> std::io::Result<()> {
    let mut req = [0u8; 1024];
    let n = s.read(&mut req).unwrap_or(0);
    let path = std::str::from_utf8(&req[..n])
        .ok()
        .and_then(|r| r.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_string();
    let latest = match path.as_str() {
        "/main" => main,
        "/bev" => bev,
        _ => {
            // Serve the index HTML.
            write!(
                s,
                "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{}",
                INDEX_HTML.len(),
                INDEX_HTML
            )?;
            return Ok(());
        }
    };
    s.write_all(
        b"HTTP/1.0 200 OK\r\nConnection: close\r\nCache-Control: no-cache\r\n\
          Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n",
    )?;
    loop {
        let jpeg = latest.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if !jpeg.is_empty() {
            write!(
                s,
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg.len()
            )?;
            s.write_all(&jpeg)?;
            s.write_all(b"\r\n")?;
        }
        std::thread::sleep(std::time::Duration::from_millis(66)); // ~15 fps push
    }
}

/// Nearest-neighbour downscale an RGB buffer to `(dw, dh)` (keeps the GIF small).
fn downscale(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh * 3];
    for y in 0..dh {
        let sy = y * sh / dh;
        for x in 0..dw {
            let sx = x * sw / dw;
            let (s, o) = ((sy * sw + sx) * 3, (y * dw + x) * 3);
            out[o..o + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
    out
}

/// Encode collected RGB frames into an animated GIF (self-contained; no ffmpeg).
fn write_gif(path: &str, frames: &[Vec<u8>], w: u16, h: u16, delay_cs: u16) -> Res<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = gif::Encoder::new(std::io::BufWriter::new(file), w, h, &[])?;
    enc.set_repeat(gif::Repeat::Infinite)?;
    for rgb in frames {
        let mut f = gif::Frame::from_rgb_speed(w, h, rgb, 10); // quantise, speed 10
        f.delay = delay_cs;
        enc.write_frame(&f)?;
    }
    println!("     saved clip → {path} ({} frames)", frames.len());
    Ok(())
}

const PALETTE: [[u8; 3]; 6] = [
    [255, 60, 60],
    [60, 220, 60],
    [60, 120, 255],
    [255, 200, 40],
    [220, 60, 220],
    [40, 220, 220],
];

/// Alpha-tint an instance mask onto the frame (nearest-upsampled from the mask grid).
fn tint_mask(buf: &mut [u8], w: usize, h: usize, inst: &Instance, color: [u8; 3]) {
    let (mw, mh) = inst.mask_size;
    // Iterate only the instance's bbox (its mask foreground lives inside it), not the
    // whole frame — ~10× fewer pixels per mask on the serve/render hot path.
    let [bx1, by1, bx2, by2] = inst.bbox;
    let (x0, x1) = (
        (bx1.max(0.0) as usize).min(w),
        (bx2.ceil().max(0.0) as usize).min(w),
    );
    let (y0, y1) = (
        (by1.max(0.0) as usize).min(h),
        (by2.ceil().max(0.0) as usize).min(h),
    );
    for y in y0..y1 {
        let my = (y * mh) / h;
        for x in x0..x1 {
            let mx = (x * mw) / w;
            if inst.mask[my * mw + mx] == 1 {
                let o = (y * w + x) * 3;
                for k in 0..3 {
                    buf[o + k] = ((buf[o + k] as u16 + color[k] as u16) / 2) as u8;
                }
            }
        }
    }
}

/// Outline a box `[x1,y1,x2,y2]`.
fn draw_box(buf: &mut [u8], w: usize, h: usize, b: [f32; 4], color: [u8; 3]) {
    let [x1, y1, x2, y2] = b;
    for (a, c) in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ] {
        draw_line(
            buf, w, h, a.0 as i32, a.1 as i32, c.0 as i32, c.1 as i32, color,
        );
    }
}

/// Bresenham line, clipped to the frame.
#[allow(clippy::too_many_arguments)]
fn draw_line(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
            let o = (y as usize * w + x as usize) * 3;
            buf[o..o + 3].copy_from_slice(&color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a short label at `(x, y)` on a dark backdrop. 5×7 bitmap font, 2× scaled;
/// digits + `.` + `m` + space cover the `<id> <depth>m` label.
fn draw_label(buf: &mut [u8], w: usize, h: usize, x: i32, y: i32, text: &str, color: [u8; 3]) {
    const SCALE: i32 = 2;
    const GLYPH_W: i32 = 5;
    const GLYPH_H: i32 = 7;
    let advance = (GLYPH_W + 1) * SCALE;
    let bg_w = advance * text.chars().count() as i32;
    let bg_h = GLYPH_H * SCALE;
    for yy in (y - 1)..(y + bg_h + 1) {
        for xx in (x - 1)..(x + bg_w + 1) {
            if xx >= 0 && xx < w as i32 && yy >= 0 && yy < h as i32 {
                let o = (yy as usize * w + xx as usize) * 3;
                for k in 0..3 {
                    buf[o + k] /= 4;
                }
            }
        }
    }
    let mut cx = x;
    for ch in text.chars() {
        if let Some(glyph) = font5x7(ch) {
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..GLYPH_W {
                    if bits & (1 << (GLYPH_W - 1 - col)) != 0 {
                        for sy in 0..SCALE {
                            for sx in 0..SCALE {
                                let px = cx + col * SCALE + sx;
                                let py = y + row as i32 * SCALE + sy;
                                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                                    let o = (py as usize * w + px as usize) * 3;
                                    buf[o..o + 3].copy_from_slice(&color);
                                }
                            }
                        }
                    }
                }
            }
        }
        cx += advance;
    }
}

/// Minimal 5×7 bitmap font: digits `0`–`9`, `.`, `m` (space renders blank).
fn font5x7(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15],
        _ => return None,
    })
}

/// COCO 91-class category-id → name (RF-DETR emits these ids directly).
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
