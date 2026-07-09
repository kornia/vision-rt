//! Single-image XFeat detection — extract keypoints from one image and draw them.
//!
//! The simplest end-to-end use of the extractor: `Image → XFeat::run → keypoints`
//! (no matching). Keypoints come back in original-image pixels; this draws a green
//! dot at each and writes a PNG.
//!
//! Usage:
//!   cargo run --release -p vrt-xfeat --example xfeat_detect -- \
//!       <model.onnx|engine>  <image>  [out.png]

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime};
use vrt_xfeat::{XFeat, XFeatParams};

const TOP_K: usize = 4096;
const THRESHOLD: f32 = 0.05;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: xfeat_detect <model.onnx|engine> <image> [out.png]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let out_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("xfeat_detect.png");

    // .onnx → on-device engine cache (built once); .engine → used directly.
    let profile = vrt_hub::EngineProfile {
        input: Some((
            "image".into(),
            vec![1, 3, 240, 320],
            vec![1, 3, 640, 640],
            vec![1, 3, 1088, 1920],
        )),
        fp16: true,
        workspace_mb: 2048,
    };
    let engine_path =
        vrt_hub::EngineCache::default().resolve("xfeat-backbone", model_path, &profile)?;

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut xfeat = XFeat::new(engine, stream.clone(), XFeatParams::new(TOP_K, THRESHOLD))?;

    // Load native image; keep the host copy for drawing, upload a device copy.
    let src = read_image_any_rgb8(image_path)?; // Rgb8 (derefs to Image<u8,3>)
    let dev = Image(src.0.to_cuda(&stream)?); // device Image<u8,3>
    let host = src.0; // host Image<u8,3>

    // One submit + sync; keypoints returned in original-image pixels.
    let result = xfeat.run(&dev)?;
    let kpts = result.kpts_to_host(&stream)?;
    let scores = result.scores_to_host(&stream)?;

    let top = scores.iter().cloned().fold(f32::MIN, f32::max);
    println!(
        "{}x{} → {} keypoints (top score {:.3}, threshold {THRESHOLD})",
        host.width(),
        host.height(),
        result.len(),
        top
    );

    // Draw a green dot at each keypoint and save.
    let (w, h) = (host.width(), host.height());
    let mut canvas = host.as_slice().to_vec();
    for xy in kpts.chunks_exact(2) {
        draw_dot(&mut canvas, w, h, xy[0] as i32, xy[1] as i32, [40, 220, 40]);
    }
    let out = Image::<u8, 3>::new(
        ImageSize {
            width: w,
            height: h,
        },
        canvas,
    )?;
    write_image_png_rgb8(out_path, &out)?;
    println!("saved {out_path}");
    Ok(())
}

/// Fill a 3×3 block of `color` centred at `(cx, cy)` in an interleaved RGB buffer.
fn draw_dot(buf: &mut [u8], w: usize, h: usize, cx: i32, cy: i32, color: [u8; 3]) {
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
                let p = (y as usize * w + x as usize) * 3;
                buf[p..p + 3].copy_from_slice(&color);
            }
        }
    }
}
