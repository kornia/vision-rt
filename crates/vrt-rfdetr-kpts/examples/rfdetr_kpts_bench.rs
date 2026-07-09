//! End-to-end latency bench for RF-DETR keypoints: preproc → submit → sync → decode.
//!
//! Usage:
//!   cargo run --release -p vrt-rfdetr-kpts --example rfdetr_kpts_bench -- \
//!       <model.onnx|engine> <image> [iters] [conf]

use kornia_image::Image;
use kornia_io::functional::read_image_any_rgb8;
use std::time::Instant;
use vrt_rfdetr_kpts::RfDetrKpts;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rfdetr_kpts_bench <model.onnx|engine> <image> [iters] [conf]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let conf: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let warmup = 20;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let profile = vrt_hub::EngineProfile {
        input: None,
        fp16: true,
        workspace_mb: 2048,
    };
    let engine_path =
        vrt_hub::EngineCache::default().resolve("rfdetr-kpts", model_path, &profile)?;
    let mut pose = RfDetrKpts::from_engine_file(&engine_path, stream.clone(), conf)?;

    let src = read_image_any_rgb8(image_path)?;
    let dev = Image(src.0.to_cuda(&stream)?);
    let mut out = pose.alloc_result()?;

    // Per-stage accumulators (µs), tracking mean + peak over the timed window.
    let (mut sub, mut syn, mut dec, mut wall) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n_people = 0;
    for i in 0..(warmup + iters) {
        let t0 = Instant::now();
        pose.submit(&dev, &mut out)?;
        let t1 = Instant::now();
        stream.synchronize()?;
        let t2 = Instant::now();
        let people = out.poses();
        let t3 = Instant::now();
        if i >= warmup {
            sub.push((t1 - t0).as_secs_f64() * 1e3);
            syn.push((t2 - t1).as_secs_f64() * 1e3);
            dec.push((t3 - t2).as_secs_f64() * 1e3);
            wall.push((t3 - t0).as_secs_f64() * 1e3);
            n_people = people.len();
        }
    }

    let stat = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let peak = v.iter().cloned().fold(0.0_f64, f64::max);
        (mean, peak)
    };
    let show = |name: &str, v: &[f64]| {
        let (mean, peak) = stat(v);
        println!("  {name:10} mean {mean:6.2} ms   peak {peak:6.2} ms");
    };
    println!(
        "{}x{} → {n_people} people, {iters} iters (warmup {warmup}), conf ≥ {conf}",
        src.0.width(),
        src.0.height()
    );
    show("submit", &sub);
    show("sync", &syn);
    show("decode", &dec);
    show("end2end", &wall);
    let (m, _) = stat(&wall);
    println!("  → {:.1} fps end-to-end", 1000.0 / m);
    Ok(())
}
