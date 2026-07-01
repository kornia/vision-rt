//! Inference-latency benchmark for the **lingbot-depth refinement** model
//! (RGB + raw depth → refined depth) on TensorRT, via [`vrt_depth::DepthRefine`].
//!
//! Feeds synthetic device-resident inputs (a constant RGB frame + a zero depth
//! map — TRT latency is data-independent) and times `DepthRefine::run` in a
//! tight loop. `run` syncs the stream internally, so each wall-clock sample is
//! the full per-frame GPU latency (stretch preproc → 2-input inference → sync).
//!
//! The engine is supplied via `--engine` — build it externally with the
//! project's `tools/export_trt.py` (ONNX → TRT). No model download here.
//!
//! Usage:
//!   cargo run --release -p depth_bench -- --engine lingbot-depth.engine [--iters 200] [--src 1280x720]
//!
//! For representative numbers, pin the Jetson to max power first:
//!   sudo nvpmodel -m 2 && sudo jetson_clocks   # MAXN_SUPER

use std::time::Instant;

use kornia_image::{Image, ImageSize};
use kornia_tensor::zeros_cuda;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime, Stream};
use vrt_depth::DepthRefine;

const WARMUP: usize = 20;

fn main() -> Result<(), vrt::BoxError> {
    let mut engine_path: Option<String> = None;
    let mut iters = 200usize;
    let (mut src_w, mut src_h) = (1280u32, 720u32);

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--engine" => engine_path = args.next(),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            "--src" => {
                if let Some(s) = args.next() {
                    if let Some((w, h)) = s.split_once('x') {
                        src_w = w.parse().unwrap_or(src_w);
                        src_h = h.parse().unwrap_or(src_h);
                    }
                }
            }
            _ => {}
        }
    }
    let Some(engine_path) = engine_path else {
        eprintln!(
            "Usage: depth_bench --engine <path> [--iters N] [--src WxH]\n\
             build the engine with the lingbot-depth-trt export_trt.py"
        );
        std::process::exit(1);
    };

    let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
    let engine = Engine::from_file(runtime, &engine_path)?;
    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let mut refiner = DepthRefine::new(engine, stream.clone())?;

    let (mh, mw) = refiner.model_hw();

    // Synthetic device inputs: a constant-gray RGB frame at the source size and
    // a zero depth map at the model size. Values don't affect TRT latency.
    let rgb_host = Image::<u8, 3>::from_size_val(
        ImageSize {
            width: src_w as usize,
            height: src_h as usize,
        },
        128,
    )?;
    let rgb = Image(rgb_host.0.to_cuda(&stream)?);
    let depth = zeros_cuda::<f32, 4>([1, 1, mh, mw], &stream)?;

    println!("lingbot-depth @ model {mh}×{mw}, src {src_w}×{src_h} (stretch)");
    println!("warmup {WARMUP}, measure {iters} iters\n");

    for _ in 0..WARMUP {
        let _ = refiner.run(&rgb, &depth)?;
    }

    let mut lat = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = refiner.run(&rgb, &depth)?;
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let n = lat.len() as f64;
    let mean: f64 = lat.iter().sum::<f64>() / n;
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
    };
    let (p50, p99, max) = (pct(&mut lat, 0.50), pct(&mut lat, 0.99), pct(&mut lat, 1.0));

    println!("mean {mean:.2} ms  ({:.1} fps)", 1000.0 / mean);
    println!("p50  {p50:.2} ms");
    println!("p99  {p99:.2} ms");
    println!("max  {max:.2} ms");

    Ok(())
}
