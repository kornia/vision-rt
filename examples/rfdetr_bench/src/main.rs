//! End-to-end benchmark of the **RF-DETR detector** (stretch preproc → TRT
//! backbone → decode) on a single static device-resident frame — no camera
//! transport. `RfDetr::run` in a tight loop, timed with the wall clock
//! (`run` syncs internally, so each call is the full per-frame latency).
//!
//! Usage:
//!   cargo run --release -p rfdetr_bench -- <image> [iters] [src_w src_h]

use std::time::Instant;

use kornia_image::{Image, ImageSize, InterpolationMode};
use kornia_imgproc::resize::resize_fast_rgb;
use kornia_io::functional::read_image_any_rgb8;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime};
use vrt_rfdetr::RfDetr;

const WARMUP: usize = 20;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rfdetr_bench <image> [iters] [src_w src_h]");
        std::process::exit(1);
    }
    let image_path = &args[1];
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let src_w: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1280);
    let src_h: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(720);

    let det_model = std::env::var("RFDETR_MODEL").unwrap_or_else(|_| "rfdetr-medium".into());
    let onnx = vrt_hub::ModelHub::get(&det_model)?;
    let engine_path = vrt_hub::EngineCache::default().resolve(
        &det_model,
        &onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut detr = RfDetr::new(engine, stream.clone(), 0.5)?;

    // Load RGB8, resize to the synthetic source size, upload to device once.
    let src = read_image_any_rgb8(image_path)?;
    let mut resized = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: src_w as usize,
            height: src_h as usize,
        },
        0,
    )?;
    resize_fast_rgb(&src, &mut resized, InterpolationMode::Bilinear)?;
    let img = Image(resized.0.to_cuda(&stream)?); // device-resident Image<u8,3>

    println!("{det_model} @ {src_w}×{src_h} (stretch)");
    println!("warmup {WARMUP}, measure {iters} iters\n");

    for _ in 0..WARMUP {
        let _ = detr.run(&img)?;
    }
    let n0 = detr.run(&img)?.len();

    let mut lat = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = detr.run(&img)?;
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let m = lat.len() as f64;
    let mean: f64 = lat.iter().sum::<f64>() / m;
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
    };
    println!("── end-to-end latency (ms)  [{n0} dets/frame] ──");
    println!(
        "  mean {:.2}   p50 {:.2}   p99 {:.2}   min {:.2}   max {:.2}",
        mean,
        pct(&mut lat, 0.50),
        pct(&mut lat, 0.99),
        lat.iter().cloned().fold(f64::MAX, f64::min),
        lat.iter().cloned().fold(0.0, f64::max)
    );
    println!("  throughput: {:.1} fps", 1000.0 / mean);
    Ok(())
}
