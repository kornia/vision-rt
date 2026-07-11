//! Parallel detect + depth on one image: run RF-DETR-Seg and Depth Anything V2 on
//! the SAME device image, one sync, then sample per-instance **metric depth from
//! the instance mask**. Demonstrates the workspace composition pattern — two models
//! sharing one stream, each `submit` only enqueues, a single `synchronize()` drains
//! both, then a GPU fusion kernel reads both models' device outputs.
//!
//! Usage:
//!   cargo run --release -p vrt-depth-anything --example detect_depth -- \
//!       <seg.engine> <depth.engine> <image> [conf] [out.png] [--bench [N]]
//!
//! With `[out.png]` it also writes a second `*.depth.png` Turbo depth map next to it.
//! `--bench [N]` (default 200) loops the one-sync pipeline on the static device
//! image and prints per-stage averages (enqueue ≪ sync proves it stays async).

use std::time::Instant;

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::{Instance, RfDetrSeg};

#[path = "common/mod.rs"]
mod common;

fn main() -> Result<(), vrt::BoxError> {
    // Split flags from positionals: [--bench [N]] may appear anywhere.
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (mut pos, mut bench): (Vec<String>, Option<usize>) = (Vec::new(), None);
    let mut it = raw.into_iter().peekable();
    while let Some(a) = it.next() {
        if a == "--bench" {
            // Optional count follows `--bench`; consume it only if it parses.
            let parsed = it.peek().and_then(|s| s.parse::<usize>().ok());
            let n = match parsed {
                Some(n) => {
                    it.next();
                    n
                }
                None => 200,
            };
            bench = Some(n);
        } else {
            pos.push(a);
        }
    }
    if pos.len() < 3 {
        eprintln!("Usage: detect_depth <seg.engine> <depth.engine> <image> [conf] [out.png] [--bench [N]]");
        std::process::exit(1);
    }
    let (seg_engine, depth_engine, image_path) = (&pos[0], &pos[1], &pos[2]);
    let conf: f32 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.4);
    let out_path = pos.get(4);

    // One shared stream: the detector, the depth net, and the fusion kernel all
    // enqueue on it, so a single sync completes the frame.
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut det = RfDetrSeg::from_engine_file(seg_engine, stream.clone(), conf)?;
    let mut depth = DepthAnything::from_engine_file(depth_engine, stream.clone())?;

    let src = read_image_any_rgb8(image_path)?;
    let dev = Image(src.0.to_cuda(&stream)?);

    let mut d = det.alloc_result()?;
    let mut z = depth.alloc_result()?;

    // Enqueue both models on the same image (no sync), then the fusion kernel that
    // reads the detector's GPU masks + the depth map. One sync drains everything.
    det.submit(&dev, &mut d)?;
    depth.submit(&dev, &mut z)?;
    let zs_dev = z
        .depth_image()
        .sample_masks(d.masks_slice(), d.mask_size(), &stream)?;
    stream.synchronize()?;

    let instances = d.instances()?;
    let zs = stream.clone_dtoh(&zs_dev)?; // per-instance metric depth (meters)

    let (mw, mh) = depth.map_size();
    println!(
        "{}x{} → {} instances | depth map {mw}x{mh} (metric meters)",
        src.0.width(),
        src.0.height(),
        instances.len()
    );
    for (i, (inst, z)) in instances.iter().zip(&zs).enumerate() {
        let [x1, y1, x2, y2] = inst.bbox;
        let area: usize = inst.mask.iter().map(|&m| m as usize).sum();
        println!(
            "  inst {i}: {:<12} score {:.3}  box [{:.0},{:.0},{:.0},{:.0}]  mask {area}px  depth {z:.2} m",
            coco_name(inst.class_id),
            inst.score,
            x1,
            y1,
            x2,
            y2
        );
    }

    // Optional: overlay (mask tint + box + `class z.zm` label) + a Turbo depth PNG.
    if let Some(out_path) = out_path {
        let (w, h) = (src.0.width(), src.0.height());
        let mut canvas = src.0.as_slice().to_vec();
        for (i, (inst, z)) in instances.iter().zip(&zs).enumerate() {
            let color = PALETTE[i % PALETTE.len()];
            overlay_mask(&mut canvas, w, h, inst, color);
            let label = format!("{} {z:.1}m", coco_name(inst.class_id));
            let [x1, y1, ..] = inst.bbox;
            draw_label(&mut canvas, w, h, x1 as i32, y1 as i32 - 9, &label, color);
        }
        let img = Image::<u8, 3>::new(
            ImageSize {
                width: w,
                height: h,
            },
            canvas,
        )?;
        write_image_png_rgb8(out_path, &img)?;
        println!("saved {out_path}");

        // Colorized depth map alongside.
        let rgb = common::depth_to_turbo(&z.depth_host()?)?;
        let depth_out = depth_png_path(out_path);
        write_image_png_rgb8(&depth_out, &rgb)?;
        println!("saved {depth_out}");
    }

    // Optional: profile the one-sync pipeline on the static device image.
    if let Some(n) = bench {
        run_bench(&mut det, &mut depth, &dev, &mut d, &mut z, &stream, n)?;
    }
    Ok(())
}

/// Per-stage profiler for the parallel detect+depth pipeline. Loops on the static
/// device image; the first ~20 iters are warm-up (discarded). Proves the pipeline
/// is async: `enqueue` (a pure CPU kernel-launch) must be ≪ the single `sync` (the
/// real GPU wall).
#[allow(clippy::too_many_arguments)]
fn run_bench(
    det: &mut RfDetrSeg,
    depth: &mut DepthAnything,
    dev: &Image<u8, 3>,
    d: &mut vrt_rfdetr_seg::SegResult,
    z: &mut vrt_depth_anything::DepthResult,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    n: usize,
) -> Result<(), vrt::BoxError> {
    let n = n.max(1); // at least one timed iter (avoids a 0-division in the fps line)
    let warmup = 20.min(n / 2);
    let ms = |dur: std::time::Duration| dur.as_secs_f64() * 1e3;
    let (mut a_enq, mut a_fus, mut a_sync, mut a_read, mut a_e2e) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut counted = 0u64;
    println!("── bench {n} iters ({warmup} warm-up) ──");
    for i in 0..n {
        let t0 = Instant::now();
        det.submit(dev, d)?;
        depth.submit(dev, z)?;
        let t1 = Instant::now();
        let zs_dev = z
            .depth_image()
            .sample_masks(d.masks_slice(), d.mask_size(), stream)?;
        let t2 = Instant::now();
        stream.synchronize()?; // the one sync completes both models + fusion
        let t3 = Instant::now();
        let _ = d.instances()?;
        let _ = stream.clone_dtoh(&zs_dev)?;
        let t4 = Instant::now();
        if i >= warmup {
            a_enq += ms(t1 - t0);
            a_fus += ms(t2 - t1);
            a_sync += ms(t3 - t2);
            a_read += ms(t4 - t3);
            a_e2e += ms(t4 - t0);
            counted += 1;
        }
    }
    let k = counted.max(1) as f64;
    println!(
        "enqueue(submit×2) {:.3} ms | fusion(sample_masks) {:.3} ms | sync(GPU) {:.2} ms | \
         read(instances+dtoh) {:.3} ms | end-to-end {:.2} ms | {:.1} fps",
        a_enq / k,
        a_fus / k,
        a_sync / k,
        a_read / k,
        a_e2e / k,
        1e3 / (a_e2e / k),
    );
    Ok(())
}

/// `foo.png` → `foo.depth.png` (or append `.depth.png` if no `.png` suffix).
fn depth_png_path(out: &str) -> String {
    match out.strip_suffix(".png") {
        Some(stem) => format!("{stem}.depth.png"),
        None => format!("{out}.depth.png"),
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

/// Tint an instance's mask onto the canvas + outline its box. The mask grid spans
/// the whole stretched frame, so a source pixel `(x,y)` samples mask cell
/// `(x/w*mw, y/h*mh)` (nearest — the stretch is full-frame and axis-aligned).
fn overlay_mask(buf: &mut [u8], w: usize, h: usize, inst: &Instance, color: [u8; 3]) {
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
        draw_line(
            buf, w, h, a.0 as i32, a.1 as i32, b.0 as i32, b.1 as i32, color,
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

/// Draw an upper-cased text label at `(x,y)` (top-left), 5×7 pixel font on a dark
/// backing bar for contrast. Only the glyphs used by COCO names + digits + `.`/`m`
/// are defined; unknowns render blank.
fn draw_label(buf: &mut [u8], w: usize, h: usize, x: i32, y: i32, text: &str, color: [u8; 3]) {
    let text = text.to_ascii_uppercase();
    let bar_w = text.len() as i32 * 6 + 1;
    // Backing bar (dark, semi-transparent) so the label reads over any scene.
    for by in y - 1..y + 8 {
        for bx in x - 1..x + bar_w {
            if bx >= 0 && bx < w as i32 && by >= 0 && by < h as i32 {
                let o = (by as usize * w + bx as usize) * 3;
                for k in 0..3 {
                    buf[o + k] = (buf[o + k] as u16 * 2 / 5) as u8;
                }
            }
        }
    }
    let mut cx = x;
    for ch in text.chars() {
        let g = glyph(ch);
        for (col, &bits) in g.iter().enumerate() {
            for row in 0..7i32 {
                if (bits >> row) & 1 == 1 {
                    let (px, py) = (cx + col as i32, y + row);
                    if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                        let o = (py as usize * w + px as usize) * 3;
                        buf[o..o + 3].copy_from_slice(&color);
                    }
                }
            }
        }
        cx += 6;
    }
}

/// COCO 91-class category-id → name (RF-DETR emits these ids directly).
fn coco_name(id: u32) -> &'static str {
    COCO91.get(id as usize).copied().unwrap_or("?")
}

#[rustfmt::skip]
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

/// Column-major 5×7 glyph: 5 columns, each a bitmask over 7 rows (bit `r` = row `r`,
/// top-to-bottom). Covers `A–Z`, `0–9`, space, `.`, `/`; unknowns render blank.
#[rustfmt::skip]
fn glyph(c: char) -> [u8; 5] {
    match c {
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x60, 0x60, 0x00],
        '/' => [0x60, 0x10, 0x08, 0x04, 0x03],
        '0' => [0x3E, 0x51, 0x49, 0x45, 0x3E],
        '1' => [0x00, 0x42, 0x7F, 0x40, 0x00],
        '2' => [0x42, 0x61, 0x51, 0x49, 0x46],
        '3' => [0x21, 0x41, 0x45, 0x4B, 0x31],
        '4' => [0x18, 0x14, 0x12, 0x7F, 0x10],
        '5' => [0x27, 0x45, 0x45, 0x45, 0x39],
        '6' => [0x3C, 0x4A, 0x49, 0x49, 0x30],
        '7' => [0x01, 0x71, 0x09, 0x05, 0x03],
        '8' => [0x36, 0x49, 0x49, 0x49, 0x36],
        '9' => [0x06, 0x49, 0x49, 0x29, 0x1E],
        'A' => [0x7E, 0x11, 0x11, 0x11, 0x7E],
        'B' => [0x7F, 0x49, 0x49, 0x49, 0x36],
        'C' => [0x3E, 0x41, 0x41, 0x41, 0x22],
        'D' => [0x7F, 0x41, 0x41, 0x22, 0x1C],
        'E' => [0x7F, 0x49, 0x49, 0x49, 0x41],
        'F' => [0x7F, 0x09, 0x09, 0x09, 0x01],
        'G' => [0x3E, 0x41, 0x49, 0x49, 0x7A],
        'H' => [0x7F, 0x08, 0x08, 0x08, 0x7F],
        'I' => [0x00, 0x41, 0x7F, 0x41, 0x00],
        'J' => [0x20, 0x40, 0x41, 0x3F, 0x01],
        'K' => [0x7F, 0x08, 0x14, 0x22, 0x41],
        'L' => [0x7F, 0x40, 0x40, 0x40, 0x40],
        'M' => [0x7F, 0x02, 0x0C, 0x02, 0x7F],
        'N' => [0x7F, 0x04, 0x08, 0x10, 0x7F],
        'O' => [0x3E, 0x41, 0x41, 0x41, 0x3E],
        'P' => [0x7F, 0x09, 0x09, 0x09, 0x06],
        'Q' => [0x3E, 0x41, 0x51, 0x21, 0x5E],
        'R' => [0x7F, 0x09, 0x19, 0x29, 0x46],
        'S' => [0x46, 0x49, 0x49, 0x49, 0x31],
        'T' => [0x01, 0x01, 0x7F, 0x01, 0x01],
        'U' => [0x3F, 0x40, 0x40, 0x40, 0x3F],
        'V' => [0x1F, 0x20, 0x40, 0x20, 0x1F],
        'W' => [0x7F, 0x20, 0x18, 0x20, 0x7F],
        'X' => [0x63, 0x14, 0x08, 0x14, 0x63],
        'Y' => [0x07, 0x08, 0x70, 0x08, 0x07],
        'Z' => [0x61, 0x51, 0x49, 0x45, 0x43],
        _ => [0x00, 0x00, 0x00, 0x00, 0x00],
    }
}
