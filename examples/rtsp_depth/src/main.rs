//! RTSP → RF-DETR **instance segmentation** + **Depth Anything V2 metric** depth,
//! device end-to-end, reused-buffer pipeline. Detect *and* range every object in
//! one GPU pass, one sync.
//!
//! Both models share ONE CUDA stream and run on the SAME device frame; each `submit`
//! only enqueues. The per-instance depth (`sample_masks`) is enqueued too, so a
//! single `synchronize()` drains detect + depth + fusion:
//!
//! ```text
//!   1. source   RtspSource::next_frame() → &Image<u8,3>   (device RGB, model-ready)
//!   2. detect   RfDetrSeg::submit        → SegResult       (stretch + TRT + GPU decode)
//!   2. depth    DepthAnything::submit     → DepthResult     (stretch + TRT, same frame)
//!   3. fusion   DepthImage::sample_masks  → [count] z (m)   (depth-at-mask, on device)
//!   ── stream.synchronize()  (the single sync that completes the pipeline) ──
//!   4. output   count + per-instance metric depth (masks/boxes stay GPU-resident)
//! ```
//!
//! The engines are machine-locked (build them on-device with trtexec). This is a
//! workspace-excluded package (private `sensor-rtsp` git dep). Build directly:
//!   export CARGO_NET_GIT_FETCH_WITH_CLI=true
//!   cargo run --release --manifest-path examples/rtsp_depth/Cargo.toml \
//!       -- <seg.engine> <depth.engine> rtsp://<camera>/stream [conf] [out.png]
//!
//! Passing a 5th arg dumps ONE annotated frame (masks tinted + boxes + a per-object
//! metric-depth label) to that PNG, plus a colorized depth map alongside it — the
//! only path that host-copies the frame, masks, boxes, and depth.

use std::time::Instant;

use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::color::{apply_colormap, ColormapType};
use kornia_io::png::write_image_png_rgb8;
use sensor_rtsp::RtspSource;
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::{Instance, RfDetrSeg};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Res<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: rtsp_depth <seg.engine> <depth.engine> <rtsp://url> [conf] [out.png]");
        std::process::exit(1);
    }
    let seg_engine = &args[1];
    let depth_engine = &args[2];
    let url = &args[3];
    let conf: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.4);
    let mut out_png = args.get(5).cloned(); // dump one annotated frame here, then keep running

    // One shared CUDA stream: the source's un-pitch copy, RF-DETR-Seg, Depth Anything,
    // and the depth-at-mask fusion all enqueue on it, so a single sync completes the frame.
    let stream = CudaContext::new(0)?.default_stream();
    let mut source = RtspSource::connect_resized(url, 1280, 720, stream.clone())?;
    let (w, h) = (source.width(), source.height());
    println!("stream {w}x{h} → RF-DETR-Seg (conf ≥ {conf}) + Depth Anything V2, device end-to-end");

    let mut seg = RfDetrSeg::from_engine_file(seg_engine, stream.clone(), conf)?;
    let mut depth = DepthAnything::from_engine_file(depth_engine, stream.clone())?;
    let mut d = seg.alloc_result()?; // reused segmenter output
    let mut z = depth.alloc_result()?; // reused depth output

    // Per-stage profiler: the two `submit`s (`enqueue`) and `sample_masks` (`fusion`)
    // are CPU kernel-launches — they must be ≪ the single `sync` (the real GPU wall).
    // `read` folds the per-instance depth D2H + count read after sync.
    let ms = |dur: std::time::Duration| dur.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_fus, mut a_sync, mut a_read) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    loop {
        let t0 = Instant::now();
        let Some(frame) = source.next_frame() else {
            break;
        }; // recv(camera) + enqueue copy
        let t1 = Instant::now();
        seg.submit(frame.image(), &mut d)?; // detect enqueue (async, no sync)
        depth.submit(frame.image(), &mut z)?; // depth enqueue (same frame + stream, no sync)
        let t2 = Instant::now();
        // Depth-at-mask fusion: one masked reduction per instance, enqueued on the
        // same stream (reads the just-enqueued masks + depth map). Returns a device
        // [count] z buffer, valid after the sync below.
        let zs = z
            .depth_image()
            .sample_masks(d.masks_slice(), d.mask_size(), &stream)?;
        let t3 = Instant::now();
        stream.synchronize()?; // the one sync completes source + detect + depth + fusion
        let t4 = Instant::now();
        // Readout: per-instance metric depth (D2H) + the survivor count (pinned).
        let inst_n = d.count();
        // Post-sync count is valid — copy only the live prefix (sample_masks fills the
        // full capacity; the trailing stale slots are meaningless).
        let z_m = stream.clone_dtoh(&zs.slice(0..inst_n))?; // [inst_n] meters, aligned
        let t5 = Instant::now();

        // Debug: after warm-up, host-copy this frame + boxes + masks + depth and draw
        // one annotated PNG (masks tinted, boxes, per-object depth labels) plus a
        // colorized depth map. Off the timed path (takes the on-demand host copies).
        if let Some(path) = out_png.take_if(|_| n == 60) {
            save_overlays(&path, frame.image(), &stream, &d, &z, &z_m)?;
        }

        n += 1;
        a_src += ms(t1 - t0);
        a_enq += ms(t2 - t1);
        a_fus += ms(t3 - t2);
        a_sync += ms(t4 - t3);
        a_read += ms(t5 - t4);
        if n.is_multiple_of(100) {
            let k = 100.0;
            println!(
                "── {n} frames | {:.1} fps | source(recv+enqueue) {:.2} ms | \
                 enqueue(2× submit) {:.3} ms | fusion(sample_masks) {:.3} ms | \
                 sync(GPU) {:.2} ms | readout {:.3} ms | {inst_n} inst",
                n as f64 / t_start.elapsed().as_secs_f64(),
                a_src / k,
                a_enq / k,
                a_fus / k,
                a_sync / k,
                a_read / k,
            );
            // On-demand host copy of the boxes (no masks) to name what's in view, each
            // tagged with its sampled metric depth.
            let dets = d.detections()?;
            let shown: Vec<String> = dets
                .iter()
                .zip(&z_m)
                .take(6)
                .map(|(det, z)| format!("{} @ {z:.2} m", coco_name(det.class_id)))
                .collect();
            println!("     detected: {}", shown.join(", "));
            a_src = 0.0;
            a_enq = 0.0;
            a_fus = 0.0;
            a_sync = 0.0;
            a_read = 0.0;
        }
    }
    Ok(())
}

/// Host-copy the frame + detections + depth and write two PNGs: the annotated frame
/// (`path`) and a colorized depth map (`<path>` with `_depth` before the extension).
fn save_overlays(
    path: &str,
    frame: &Image<u8, 3>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    d: &vrt_rfdetr_seg::SegResult,
    z: &vrt_depth_anything::DepthResult,
    z_m: &[f32],
) -> Res<()> {
    let (w, h) = (frame.width(), frame.height());
    let host = frame.to_host(stream)?; // device Rgb8 → host
    let mut buf = host.as_slice().to_vec();
    let instances = d.instances()?; // boxes + masks to host
    for (i, inst) in instances.iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        draw_instance(&mut buf, w, h, inst, color);
        // Per-object metric-depth label near the box top-left (e.g. "1.8m").
        let z_val = z_m.get(i).copied().unwrap_or(0.0);
        let [x1, y1, _, _] = inst.bbox;
        draw_label(&mut buf, w, h, x1 as i32 + 2, y1 as i32 + 2, &format!("{z_val:.1}m"), color);
    }
    let img = Image::<u8, 3>::new(ImageSize { width: w, height: h }, buf)?;
    write_image_png_rgb8(path, &img)?;
    println!("     saved detect+depth overlay → {path} ({} instances)", instances.len());

    // Colorized depth map (Turbo) at the model's map resolution. Normalize valid
    // (positive) metric depths to 0..255, invert so near = warm.
    let depth_host = z.depth_host()?;
    let (mw, mh) = (depth_host.size().width, depth_host.size().height);
    let dvals = depth_host.as_slice();
    let (mut lo, mut hi) = (f32::INFINITY, 0.0f32);
    for &v in dvals {
        if v > 0.0 {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let span = (hi - lo).max(1e-3);
    let gray: Vec<u8> = dvals
        .iter()
        .map(|&v| {
            if v <= 0.0 {
                0
            } else {
                (255.0 * (1.0 - (v - lo) / span)).clamp(0.0, 255.0) as u8 // near = bright
            }
        })
        .collect();
    let gray_img = Image::<u8, 1>::new(ImageSize { width: mw, height: mh }, gray)?;
    let mut colored = Image::<u8, 3>::from_size_val(ImageSize { width: mw, height: mh }, 0)?;
    apply_colormap(&gray_img, &mut colored, ColormapType::Turbo)?;
    let depth_path = depth_png_path(path);
    write_image_png_rgb8(&depth_path, &colored)?;
    println!("     saved colorized depth   → {depth_path} ({mw}x{mh}, {lo:.2}–{hi:.2} m)");
    Ok(())
}

/// Colorized-depth companion path for the overlay: `…detect_depth.png` →
/// `…depth.png`, else insert `_depth` before the extension.
fn depth_png_path(path: &str) -> String {
    if path.contains("detect_depth") {
        return path.replace("detect_depth", "depth");
    }
    match path.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}_depth.{ext}"),
        None => format!("{path}_depth"),
    }
}

const PALETTE: [[u8; 3]; 6] = [
    [255, 60, 60],
    [60, 220, 60],
    [60, 120, 255],
    [255, 200, 40],
    [220, 60, 220],
    [40, 220, 220],
];

/// Draw one instance onto a tight RGB frame buffer: tint its mask (nearest-upsampled
/// from the mask grid, which spans the whole stretched frame) + outline its box.
fn draw_instance(buf: &mut [u8], w: usize, h: usize, inst: &Instance, color: [u8; 3]) {
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
    let [x1, y1, x2, y2] = inst.bbox;
    for (a, b) in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ] {
        draw_line(buf, w, h, a.0 as i32, a.1 as i32, b.0 as i32, b.1 as i32, color);
    }
}

/// Bresenham line, clipped to the frame.
#[allow(clippy::too_many_arguments)]
fn draw_line(buf: &mut [u8], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
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

/// Draw a short text label (metric depth) at `(x, y)` with a dark backdrop so it
/// stays legible over a tinted mask. 5×7 bitmap font, 2× scaled, digits + `.` + `m`.
fn draw_label(buf: &mut [u8], w: usize, h: usize, x: i32, y: i32, text: &str, color: [u8; 3]) {
    const SCALE: i32 = 2;
    const GLYPH_W: i32 = 5;
    const GLYPH_H: i32 = 7;
    let advance = (GLYPH_W + 1) * SCALE;
    // Dark backdrop rectangle behind the whole string.
    let bg_w = advance * text.chars().count() as i32;
    let bg_h = GLYPH_H * SCALE;
    for yy in (y - 1)..(y + bg_h + 1) {
        for xx in (x - 1)..(x + bg_w + 1) {
            if xx >= 0 && xx < w as i32 && yy >= 0 && yy < h as i32 {
                let o = (yy as usize * w + xx as usize) * 3;
                for k in 0..3 {
                    buf[o + k] /= 4; // darken to 25% for label contrast
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

/// Minimal 5×7 bitmap font: digits `0`–`9`, `.`, and `m` (all the depth label needs).
/// Each glyph is 7 rows, low 5 bits per row (bit 4 = leftmost column).
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

/// COCO 91-class category-id → name (RF-DETR emits these ids directly; `N/A` are
/// the gaps in the 91-index space, 0 = background is never emitted).
fn coco_name(id: u32) -> &'static str {
    COCO91.get(id as usize).copied().unwrap_or("?")
}

const COCO91: [&str; 91] = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];
