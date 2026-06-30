//! End-to-end benchmark of the **XFeat detector** (preproc → backbone → GPU
//! top-K), driven by a synthetic static-image source so the numbers isolate
//! the detector itself — no RTSP / NVMM camera transport.
//!
//! The frame is uploaded to the device once; a [`Source`] re-emits a borrowed
//! [`VrtImage`] view of it for N iterations. Each `pipeline.next()` reports a
//! per-phase [`PipelineTiming`] (the profile), which we aggregate.
//!
//! Usage:
//!   cargo run --release -p xfeat_bench -- <model.onnx|engine> <image> [iters]

use std::ffi::c_void;
use std::sync::Arc;

use vrt::cudarc::driver::{CudaSlice, DevicePtr};
use vrt::logger::Severity;
use vrt::{Engine, Format, Logger, MemKind, Pipeline, PipelineTiming, Runtime, Source, VrtImage};
use vrt_preproc::Preprocessor;
use vrt_xfeat::{XFeat, XFeatParams};

const SRC_W: u32 = 1280; // camera frame after VIC resize (typical)
const SRC_H: u32 = 720;
const MODEL_W: u32 = 1280; // pad32(720) = 736; backbone runs at 1280×736
const MODEL_H: u32 = 736;
const TOP_K: usize = 4096;
const THRESHOLD: f32 = 0.05;
const WARMUP: usize = 20;

/// Re-emits a borrowed view of one device-resident RGBA frame, N times.
struct StaticImageSource {
    ptr: *mut c_void,
    remaining: usize,
    _backing: CudaSlice<u8>, // keeps the device frame alive
}
impl Source for StaticImageSource {
    type Frame = VrtImage;
    fn next_frame(&mut self) -> Option<VrtImage> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        // SAFETY: ptr is a device RGBA buffer kept alive by `_backing` for the
        // whole benchmark; pitch = SRC_W*4 (tightly packed, 32-byte aligned).
        Some(unsafe {
            VrtImage::borrowed(
                self.ptr,
                SRC_W,
                SRC_H,
                SRC_W * 4,
                Format::Rgba8,
                MemKind::Device,
            )
        })
    }
}

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
    let preproc = Preprocessor::new(stream.clone(), SRC_W, SRC_H, MODEL_W, MODEL_H)?;
    let xfeat = XFeat::with_stream(
        Arc::clone(&engine),
        stream.clone(),
        XFeatParams::new(TOP_K, THRESHOLD, MODEL_H as usize, MODEL_W as usize),
    )?;

    // Load + resize the frame to SRC, RGB→RGBA, upload to device once.
    let img = image::open(image_path)?.to_rgb8();
    let resized =
        image::imageops::resize(&img, SRC_W, SRC_H, image::imageops::FilterType::Triangle);
    let mut rgba = Vec::with_capacity((SRC_W * SRC_H * 4) as usize);
    for px in resized.pixels() {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    let backing: CudaSlice<u8> = stream.memcpy_stod(&rgba)?;
    let ptr = backing.device_ptr(stream.as_ref()).0 as *mut c_void;

    let source = StaticImageSource {
        ptr,
        remaining: WARMUP + iters,
        _backing: backing,
    };
    let mut pipeline = Pipeline::new(stream, source).pipe(preproc).pipe(xfeat);

    println!("XFeat detector @ {SRC_W}×{SRC_H} → model {MODEL_W}×{MODEL_H}, top_k={TOP_K}");
    println!("warmup {WARMUP}, measure {iters} iters\n");

    let mut totals = Vec::with_capacity(iters);
    let mut gpu = Vec::with_capacity(iters);
    let mut acc = PipelineTiming::default();
    let mut n = 0usize;

    while let Some(res) = pipeline.next() {
        let (kpts, t) = res?;
        n += 1;
        if n <= WARMUP {
            continue;
        } // discard warm-up (TRT/CUDA cache fill)
        let _ = kpts.scores.len();
        acc.source_ms += t.source_ms;
        acc.enqueue_ms += t.enqueue_ms;
        acc.gpu_ms += t.gpu_ms;
        acc.sync_ms += t.sync_ms;
        acc.finalize_ms += t.finalize_ms;
        totals.push(t.source_ms + t.enqueue_ms + t.sync_ms + t.finalize_ms);
        gpu.push(t.gpu_ms);
    }

    let m = totals.len() as f64;
    let mean = |s: f64| s / m;
    println!("── per-phase mean (ms) ──");
    println!(
        "  source   {:.3}   (synthetic — exclude)",
        mean(acc.source_ms)
    );
    println!(
        "  enqueue  {:.3}   (CPU kernel/TRT launch)",
        mean(acc.enqueue_ms)
    );
    println!(
        "  gpu      {:.3}   (CUDA events: letterbox + backbone + top-K)",
        mean(acc.gpu_ms)
    );
    println!(
        "  sync     {:.3}   (wall cudaStreamSynchronize)",
        mean(acc.sync_ms)
    );
    println!(
        "  finalize {:.3}   (host read: count/scores/xy)",
        mean(acc.finalize_ms)
    );

    println!("\n  note: results D2H into PINNED host memory → truly async, so");
    println!("  `enqueue` returns immediately and the GPU wait shows in `sync`");
    println!("  (where it belongs). The CPU is free during GPU compute.");

    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
    };
    let total_mean: f64 = totals.iter().sum::<f64>() / m;
    println!("\n── end-to-end latency (ms) ──");
    println!(
        "  mean {:.2}   p50 {:.2}   p99 {:.2}   min {:.2}   max {:.2}",
        total_mean,
        pct(&mut totals, 0.50),
        pct(&mut totals, 0.99),
        totals.iter().cloned().fold(f64::MAX, f64::min),
        totals.iter().cloned().fold(0.0, f64::max)
    );
    println!("  throughput: {:.1} fps", 1000.0 / total_mean);
    println!(
        "\n  gpu-only: mean {:.2}   p50 {:.2}   p99 {:.2} ms",
        gpu.iter().sum::<f64>() / m,
        pct(&mut gpu, 0.50),
        pct(&mut gpu, 0.99)
    );
    Ok(())
}
