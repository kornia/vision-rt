//! End-to-end benchmark of the **XFeat detector** (preproc → backbone → GPU
//! top-K) on a single static device-resident frame — no RTSP / NVMM transport,
//! no orchestration. Just `XFeat::run` in a tight loop, timed with the wall clock
//! (`run` syncs internally, so each call is the full per-frame latency).
//!
//! Usage:
//!   cargo run --release -p xfeat_bench -- <model.onnx|engine> <image> [iters]

use std::time::Instant;

use kornia_io::functional::read_image_any_rgb8;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime};
use vrt_xfeat::{XFeat, XFeatParams};

const TOP_K: usize = 4096;
const THRESHOLD: f32 = 0.05;
const WARMUP: usize = 20;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: xfeat_bench <model.onnx|engine> <image> [iters]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);

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
    // XFeat resizes each frame to its own floor-of-32 model dims (upstream XFeat).
    let mut xfeat = XFeat::new(engine, stream.clone(), XFeatParams::new(TOP_K, THRESHOLD))?;

    // Load RGB8 at native resolution, upload to device once. Reused every iteration.
    let src = read_image_any_rgb8(image_path)?;
    let (sw, sh) = (src.width(), src.height());
    let img = src.0.to_cuda(&stream)?; // device-resident Image<u8,3>

    println!(
        "XFeat detector @ {sw}×{sh} → model {}×{} (floor-32), top_k={TOP_K}",
        (sw / 32) * 32,
        (sh / 32) * 32
    );
    println!("warmup {WARMUP}, measure {iters} iters\n");

    // Reuse one caller-owned output across all iterations (no per-frame alloc).
    let mut res = xfeat.alloc_result()?;

    for _ in 0..WARMUP {
        xfeat.submit(&img, &mut res)?;
        stream.synchronize()?;
    } // discard warm-up (TRT/CUDA cache fill)

    let mut lat = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        xfeat.submit(&img, &mut res)?; // resize + backbone + top-K (async)
        stream.synchronize()?; // one sync = full per-frame latency
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
        let _ = res.count();
    }

    let m = lat.len() as f64;
    let mean: f64 = lat.iter().sum::<f64>() / m;
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
    };
    println!("── end-to-end latency (ms) ──");
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
