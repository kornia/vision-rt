//! Single-image RaCo-ALIKED detection — extract keypoints from one image and draw them.
//!
//! The simplest end-to-end use of the extractor, showing the **async** API:
//! `alloc_result` → `submit` (no sync) → `stream.synchronize()` → read. Keypoints come
//! back in original-image pixels.
//!
//! The engine is the *extractor half* produced by `scripts/split_raco_pipeline.py`;
//! pass either that `.onnx` (built and cached on first run) or a prebuilt `.engine`.
//!
//! Usage:
//!   cargo run --release -p vrt-raco-aliked --example raco_detect -- \
//!       <raco_aliked_extractor_kN.onnx|engine>  <image>  [out.png]

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime};
use vrt_raco_aliked::RaCoAliked;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: raco_detect <model.onnx|engine> <image> [out.png]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let out_path = args.get(3).map(String::as_str).unwrap_or("raco_detect.png");

    // .onnx → on-device engine cache (built once); .engine → used directly.
    // H and W must be multiples of 32 (RaCo's input_dim_divisor).
    let profile = vrt_hub::EngineProfile {
        inputs: vec![(
            "images".into(),
            vec![1, 3, 256, 256],
            vec![1, 3, 512, 512],
            vec![1, 3, 768, 768],
        )],
        fp16: true,
        bf16: false,
        workspace_mb: 2048,
    };
    let engine_path =
        vrt_hub::EngineCache::default().resolve("raco-aliked-extractor", model_path, &profile)?;

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::new(engine, stream.clone())?;

    // Load native image; keep the host copy for drawing, upload a device copy.
    let src = read_image_any_rgb8(image_path)?; // Rgb8 (derefs to Image<u8,3>)
    let dev = src.to_cuda(&stream)?; // device Image<u8,3>
    let host = src.0; // host Image<u8,3>

    let mut result = raco.alloc_result()?;
    raco.submit(&dev, &mut result)?; // async: returns immediately
    stream.synchronize()?; // the caller owns the one sync
    let kpts = result.keypoints_host()?;

    let (rw, rh) = result.scale();
    println!(
        "{}x{} → {} keypoints (K is fixed by the engine; stretch ratio {rw:.3}x{rh:.3})",
        host.width(),
        host.height(),
        result.count(),
    );

    // Draw a green dot at each keypoint and save.
    let (w, h) = (host.width(), host.height());
    let mut canvas = host.as_slice().to_vec();
    for (x, y) in &kpts {
        draw_dot(&mut canvas, w, h, *x as i32, *y as i32, [40, 220, 40]);
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
