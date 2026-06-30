//! Stage-by-stage benchmark of the full **2D tracking pipeline**
//! (RF-DETR detect → OSNet ReID → BoT-SORT update) on a single static
//! device-resident frame — no camera, so wall time is compute, not RTSP wait.
//!
//! Each model call syncs its stream internally, so a wall-clock `Instant` around
//! it is that stage's per-frame latency. The tracker stage is pure CPU. Together
//! they show where time goes and confirm the heavy stages are GPU-bound.
//!
//! Usage:
//!   cargo run --release -p track_bench -- <image> [iters] [src_w src_h]

use std::time::Instant;

use kornia_image::{Image, ImageSize, InterpolationMode};
use kornia_imgproc::resize::resize_fast_rgb;
use kornia_io::functional::read_image_any_rgb8;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime};
use vrt_reid::ReId;
use vrt_rfdetr::RfDetr;
use vrt_track::{Box2DTracker, Obs2D, TrackerConfig};

const WARMUP: usize = 20;

fn stats(name: &str, lat: &mut [f64]) {
    let m = lat.len() as f64;
    let mean: f64 = lat.iter().sum::<f64>() / m;
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| lat[((lat.len() as f64 * q) as usize).min(lat.len() - 1)];
    println!(
        "  {name:<8} mean {:7.3}   p50 {:7.3}   p99 {:7.3}   max {:7.3} ms",
        mean,
        p(0.50),
        p(0.99),
        lat[lat.len() - 1]
    );
}

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: track_bench <image> [iters] [src_w src_h]");
        std::process::exit(1);
    }
    let image_path = &args[1];
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let src_w: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1280);
    let src_h: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(720);

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;

    let det_model = std::env::var("RFDETR_MODEL").unwrap_or_else(|_| "rfdetr-medium".into());
    let det_onnx = vrt_hub::ModelHub::get(&det_model)?;
    let det_path = vrt_hub::EngineCache::default().resolve(
        &det_model,
        &det_onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;
    let det_engine = Engine::from_file(runtime.clone(), &det_path)?;

    let reid_onnx = vrt_hub::ModelHub::get("osnet-reid")?;
    let reid_path = vrt_hub::EngineCache::default().resolve(
        "osnet-reid",
        &reid_onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;
    let reid_engine = Engine::from_file(runtime, &reid_path)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut detr = RfDetr::new(det_engine, stream.clone(), 0.5)?;
    let mut reid = ReId::new(reid_engine, stream.clone())?;
    let mut tracker = Box2DTracker::new(TrackerConfig::default());

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

    println!("2D tracking pipeline @ {src_w}×{src_h}  ({det_model} → OSNet ReID → BoT-SORT)");
    println!("warmup {WARMUP}, measure {iters} iters\n");

    for _ in 0..WARMUP {
        let d = detr.run(&img)?;
        let b: Vec<[f32; 4]> = d.iter().map(|x| x.bbox).collect();
        let _ = reid.embed(&img, &b[..b.len().min(reid.batch())])?;
    }
    let ndet = detr.run(&img)?.len();

    let (mut t_det, mut t_reid, mut t_trk, mut t_all) = (
        Vec::with_capacity(iters),
        Vec::with_capacity(iters),
        Vec::with_capacity(iters),
        Vec::with_capacity(iters),
    );

    for _ in 0..iters {
        let t0 = Instant::now();

        let td = Instant::now();
        let dets = detr.run(&img)?;
        t_det.push(td.elapsed().as_secs_f64() * 1000.0);

        let boxes: Vec<[f32; 4]> = dets.iter().map(|d| d.bbox).collect();
        let k = boxes.len().min(reid.batch());
        let tr = Instant::now();
        let mut embeds = vec![Vec::new(); dets.len()];
        for (i, e) in reid.embed(&img, &boxes[..k])?.into_iter().enumerate() {
            embeds[i] = e;
        }
        t_reid.push(tr.elapsed().as_secs_f64() * 1000.0);

        let obs: Vec<Obs2D> = dets
            .iter()
            .map(|d| Obs2D {
                bbox: d.bbox,
                score: d.score,
                class_id: d.class_id,
            })
            .collect();
        let tt = Instant::now();
        let _ = tracker.update(&obs, 1.0 / 30.0, Some(&embeds));
        t_trk.push(tt.elapsed().as_secs_f64() * 1000.0);

        t_all.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let mean_all: f64 = t_all.iter().sum::<f64>() / t_all.len() as f64;
    println!("── per-stage latency  [{ndet} dets/frame] ──");
    stats("detect", &mut t_det);
    stats("reid", &mut t_reid);
    stats("track", &mut t_trk);
    stats("TOTAL", &mut t_all);
    println!(
        "\n  GPU-bound throughput: {:.1} fps  (camera-free; live is RTSP-bound)",
        1000.0 / mean_all
    );
    Ok(())
}
