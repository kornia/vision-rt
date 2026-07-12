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
//!       -- <seg.engine> <depth.engine> rtsp://<camera>/stream [conf] [out.png]
//!
//! A 5th arg dumps ONE annotated frame (masks tinted + tracked boxes coloured by id,
//! each labelled `<id> <depth>m`) to that PNG.

use std::io::Write;
use std::time::Instant;

use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use kornia_io::png::write_image_png_rgb8;
use sensor_rtsp::RtspSource;
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::{Instance, RfDetrSeg};
use vrt_track::{BotSort, BotSortConfig, Detection, Track, TrackState};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Res<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: rtsp_track <seg.engine> <depth.engine> <rtsp://url> [conf] [out.png]");
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
    // Reused per-frame detection buffer (cleared, not reallocated).
    let mut dets: Vec<Detection> = Vec::new();

    // Per-stage profiler: enqueue (2× submit) + fusion (sample_masks) are CPU
    // kernel-launches, ≪ the single GPU `sync`; `track` is the CPU tracker cost.
    let ms = |dur: std::time::Duration| dur.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_fus, mut a_sync, mut a_read, mut a_trk) =
        (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
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
        let tracks = tracker.update(&dets);
        let t6 = Instant::now();

        if let Some(path) = out_png.take_if(|_| n == 60) {
            save_overlay(&path, frame.image(), &stream, &d, &tracks)?;
        }

        n += 1;
        a_src += ms(t1 - t0);
        a_enq += ms(t2 - t1);
        a_fus += ms(t3 - t2);
        a_sync += ms(t4 - t3);
        a_read += ms(t5 - t4);
        a_trk += ms(t6 - t5);
        if n.is_multiple_of(100) {
            let k = 100.0;
            let confirmed = tracks
                .iter()
                .filter(|t| t.state == TrackState::Confirmed)
                .count();
            println!(
                "── {n} frames | {:.1} fps | source {:.2} ms | enqueue {:.3} ms | \
                 fusion {:.3} ms | sync(GPU) {:.2} ms | readout {:.3} ms | track {:.3} ms | \
                 {inst_n} det → {confirmed} confirmed",
                n as f64 / t_start.elapsed().as_secs_f64(),
                a_src / k,
                a_enq / k,
                a_fus / k,
                a_sync / k,
                a_read / k,
                a_trk / k,
            );
            // A few live tracks with their tracked metric depth + approach velocity.
            let shown: Vec<String> = tracks
                .iter()
                .filter(|t| t.state == TrackState::Confirmed)
                .take(6)
                .map(|t| {
                    let vz = t.velocity_3d[2];
                    let dir = if vz < -0.02 {
                        "→near"
                    } else if vz > 0.02 {
                        "→far"
                    } else {
                        ""
                    };
                    format!(
                        "#{} {} @ {:.2} m{dir}",
                        t.id,
                        coco_name(t.class_id),
                        t.position_3d[2]
                    )
                })
                .collect();
            println!("     tracks: {}", shown.join(", "));
            let _ = std::io::stdout().flush(); // survive a timeout/SIGTERM kill
            a_src = 0.0;
            a_enq = 0.0;
            a_fus = 0.0;
            a_sync = 0.0;
            a_read = 0.0;
            a_trk = 0.0;
        }
    }
    Ok(())
}

/// Host-copy the frame, tint instance masks (context), then draw each confirmed
/// track's box coloured by id with an `<id> <depth>m` label. One PNG.
fn save_overlay(
    path: &str,
    frame: &Image<u8, 3>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d: &vrt_rfdetr_seg::SegResult,
    tracks: &[Track],
) -> Res<()> {
    let (w, h) = (frame.width(), frame.height());
    let host = frame.to_host(stream)?;
    let mut buf = host.as_slice().to_vec();

    // Faint mask tint for scene context (neutral grey — identity is the track box).
    let instances = d.instances()?;
    for inst in &instances {
        tint_mask(&mut buf, w, h, inst, [90, 90, 90]);
    }
    // Tracked boxes coloured by id + `<id> <depth>m` label.
    let mut shown = 0;
    for t in tracks {
        if t.state != TrackState::Confirmed {
            continue;
        }
        shown += 1;
        let color = PALETTE[(t.id as usize) % PALETTE.len()];
        draw_box(&mut buf, w, h, t.bbox, color);
        let [x1, y1, ..] = t.bbox;
        let label = format!("{} {:.1}m", t.id, t.position_3d[2]);
        draw_label(&mut buf, w, h, x1 as i32 + 2, y1 as i32 + 2, &label, color);
    }
    let img = Image::<u8, 3>::new(
        ImageSize {
            width: w,
            height: h,
        },
        buf,
    )?;
    write_image_png_rgb8(path, &img)?;
    println!("     saved tracked overlay → {path} ({shown} confirmed tracks)");
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
    for y in 0..h {
        let my = (y * mh) / h;
        for x in 0..w {
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
