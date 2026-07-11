//! Single-image RF-DETR instance segmentation: build/cache engine, run, print +
//! optionally overlay instance masks.
//!
//! Usage:
//!   cargo run --release -p vrt-rfdetr-seg --example rfdetr_seg_detect -- \
//!       <model.onnx|engine>  <image>  [conf]  [out.png]
//!   # or pull from Hugging Face (kornia/rfdetr-seg):
//!   cargo run --release -p vrt-rfdetr-seg --example rfdetr_seg_detect --features hub -- \
//!       hub  <image>  [conf]  [out.png]

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt_rfdetr_seg::{Instance, RfDetrSeg};

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rfdetr_seg_detect <model.onnx|engine> <image> [conf] [out.png]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let conf: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let out_path = args.get(4);

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut seg = if model_path == "hub" {
        #[cfg(feature = "hub")]
        {
            RfDetrSeg::from_hub(stream.clone(), conf)?
        }
        #[cfg(not(feature = "hub"))]
        {
            return Err("pass an .onnx/.engine path, or rebuild with --features hub".into());
        }
    } else if model_path.ends_with(".engine") {
        // Prebuilt, machine-locked engine — load as-is (no build).
        RfDetrSeg::from_engine_file(model_path, stream.clone(), conf)?
    } else {
        // ONNX — build + cache the engine on first run (keyed by TRT+SM).
        let profile = vrt_hub::EngineProfile {
            input: None,
            fp16: true,
            workspace_mb: 2048,
        };
        let engine_path =
            vrt_hub::EngineCache::default().resolve("rfdetr-seg", model_path, &profile)?;
        RfDetrSeg::from_engine_file(&engine_path, stream.clone(), conf)?
    };

    let src = read_image_any_rgb8(image_path)?;
    let dev = Image(src.0.to_cuda(&stream)?);

    // Async: submit (no sync, no host copy) → one caller sync → host-copy on request.
    let mut out = seg.alloc_result()?;
    seg.submit(&dev, &mut out)?;
    stream.synchronize()?;
    let instances = out.instances()?; // explicit host copy of boxes + masks

    println!(
        "{}x{} → {} instances (conf ≥ {conf})",
        src.0.width(),
        src.0.height(),
        instances.len()
    );
    for (i, inst) in instances.iter().enumerate().take(20) {
        let [x1, y1, x2, y2] = inst.bbox;
        let area: usize = inst.mask.iter().map(|&m| m as usize).sum();
        println!(
            "  inst {i}: class {:<3} score {:.3}  box [{:.0},{:.0},{:.0},{:.0}]  mask {}px",
            inst.class_id, inst.score, x1, y1, x2, y2, area
        );
    }

    // Optional: overlay masks (tinted) + boxes and save.
    if let Some(out_path) = out_path {
        let (w, h) = (src.0.width(), src.0.height());
        let mut canvas = src.0.as_slice().to_vec();
        for (i, inst) in instances.iter().enumerate() {
            overlay_mask(&mut canvas, w, h, inst, PALETTE[i % PALETTE.len()]);
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
    }
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
