//! Single-image RF-DETR keypoint (pose) detection: build/cache engine, run, print poses.
//!
//! Usage:
//!   cargo run --release -p vrt-rfdetr-kpts --example rfdetr_kpts_detect -- \
//!       <model.onnx|engine>  <image>  [conf]
//!   # or pull from Hugging Face (kornia/rfdetr-kpts):
//!   cargo run --release -p vrt-rfdetr-kpts --example rfdetr_kpts_detect --features hub -- \
//!       hub  <image>  [conf]

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt_rfdetr_kpts::{PersonPose, RfDetrKpts, COCO_KEYPOINT_NAMES};

// COCO 17-keypoint skeleton edges (index pairs).
const SKELETON: [(usize, usize); 18] = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
];

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rfdetr_kpts_detect <model.onnx|engine> <image> [conf] [out.png]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let conf: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let out_path = args.get(4);

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut pose = if model_path == "hub" {
        #[cfg(feature = "hub")]
        {
            RfDetrKpts::from_hub(stream.clone(), conf)?
        }
        #[cfg(not(feature = "hub"))]
        {
            return Err("pass an .onnx/.engine path, or rebuild with --features hub".into());
        }
    } else {
        let profile = vrt_hub::EngineProfile {
            input: None,
            fp16: true,
            workspace_mb: 2048,
        };
        let engine_path =
            vrt_hub::EngineCache::default().resolve("rfdetr-kpts", model_path, &profile)?;
        RfDetrKpts::from_engine_file(&engine_path, stream.clone(), conf)?
    };

    let src = read_image_any_rgb8(image_path)?;
    let dev = src.0.to_cuda(&stream)?;

    // Async: submit → one caller sync → decode.
    let mut out = pose.alloc_result()?;
    pose.submit(&dev, &mut out)?;
    stream.synchronize()?;
    let people = out.poses();

    println!(
        "{}x{} → {} people (conf ≥ {conf})",
        src.0.width(),
        src.0.height(),
        people.len()
    );
    for (i, p) in people.iter().enumerate().take(10) {
        let [x1, y1, x2, y2] = p.bbox;
        println!(
            "  person {i}: score {:.3}  box [{:.0},{:.0},{:.0},{:.0}]",
            p.score, x1, y1, x2, y2
        );
        // A few visible joints for a sanity check.
        for (j, kp) in p.keypoints.iter().enumerate() {
            if kp[2] >= 0.5 {
                println!(
                    "      {:14} ({:.0},{:.0})  {:.2}",
                    COCO_KEYPOINT_NAMES[j], kp[0], kp[1], kp[2]
                );
            }
        }
    }
    // Optional: draw boxes + skeletons and save.
    if let Some(out_path) = out_path {
        let (w, h) = (src.0.width(), src.0.height());
        let mut canvas = src.0.as_slice().to_vec();
        for p in &people {
            draw_pose(&mut canvas, w, h, p);
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

/// Draw a person's box (blue) + keypoint dots (green) + skeleton (green).
fn draw_pose(buf: &mut [u8], w: usize, h: usize, p: &PersonPose) {
    let [x1, y1, x2, y2] = p.bbox;
    for (a, b) in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ] {
        draw_line(
            buf,
            w,
            h,
            a.0 as i32,
            a.1 as i32,
            b.0 as i32,
            b.1 as i32,
            [60, 120, 255],
        );
    }
    for &(a, b) in &SKELETON {
        if p.keypoints[a][2] >= 0.5 && p.keypoints[b][2] >= 0.5 {
            let (pa, pb) = (p.keypoints[a], p.keypoints[b]);
            draw_line(
                buf,
                w,
                h,
                pa[0] as i32,
                pa[1] as i32,
                pb[0] as i32,
                pb[1] as i32,
                [40, 220, 40],
            );
        }
    }
    for kp in &p.keypoints {
        if kp[2] >= 0.5 {
            draw_dot(buf, w, h, kp[0] as i32, kp[1] as i32, [255, 60, 60]);
        }
    }
}

/// Fill a 5×5 block centred at (cx, cy).
fn draw_dot(buf: &mut [u8], w: usize, h: usize, cx: i32, cy: i32, color: [u8; 3]) {
    for dy in -2..=2 {
        for dx in -2..=2 {
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                let o = (y as usize * w + x as usize) * 3;
                buf[o..o + 3].copy_from_slice(&color);
            }
        }
    }
}

/// Bresenham line, clipped.
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
