//! GStreamer RTSP → NVMM → XFeat keypoint detection in real-time.
//!
//! ## Pipeline
//! ```text
//! RtspSource → NvmmPreprocessStage → XFeat
//! NvmmFrame     VrtTensor              XFeatResult
//! ```
//!
//! Every 30 frames a CPU RGBA snapshot is grabbed from the RtspSource tee branch,
//! keypoints are drawn on it as red circles, and the result is saved as a PNG.
//!
//! Usage:
//!   cargo run --release -p rtsp_xfeat -- \
//!       models/xfeat/xfeat_backbone_fp16.engine rtsp://camera/stream [--save-dir /tmp]

use std::sync::Arc;
use vrt::{Engine, Logger, Runtime, Stream, Pipeline};
use vrt::logger::Severity;
use vrt_xfeat::{XFeat, XFeatParams};
use vrt_gst::{RtspSource, NvmmPreprocessStage};
use image::{ImageBuffer, Rgba};

fn pad32(v: u32) -> u32 { ((v + 31) / 32) * 32 }

// ── Visualization ─────────────────────────────────────────────────────────────

fn draw_dot(img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, cx: i32, cy: i32, r: i32, color: [u8; 4]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let (x, y) = (cx + dx, cy + dy);
                if x >= 0 && x < w && y >= 0 && y < h {
                    img.put_pixel(x as u32, y as u32, Rgba(color));
                }
            }
        }
    }
}

fn save_kpts(
    rgba:     &[u8],
    fw:       u32,
    fh:       u32,
    kpts_cpu: &[f32],
    dst_w:    u32,
    dst_h:    u32,
    out_path: &str,
) {
    let Some(mut img) = ImageBuffer::<Rgba<u8>, _>::from_raw(fw, fh, rgba.to_vec()) else {
        eprintln!("[viz] bad frame buffer dimensions");
        return;
    };

    // keypoint coords are in model space (dst_w × dst_h); scale to frame space
    let sx = fw as f32 / dst_w as f32;
    let sy = fh as f32 / dst_h as f32;

    for chunk in kpts_cpu.chunks_exact(2) {
        let cx = (chunk[0] * sx) as i32;
        let cy = (chunk[1] * sy) as i32;
        draw_dot(&mut img, cx, cy, 4, [255, 50, 50, 255]);   // red filled circle
        draw_dot(&mut img, cx, cy, 2, [255, 255, 50, 255]);  // yellow centre dot
    }

    match img.save(out_path) {
        Ok(()) => println!("[viz] saved {out_path}  ({} kpts)", kpts_cpu.len() / 2),
        Err(e) => eprintln!("[viz] save failed: {e}"),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rtsp_xfeat <model.onnx|model.engine> <rtsp_url> [save_dir]");
        eprintln!("  .onnx   — built on-device into ~/.cache/vision-rt/engines (one-time)");
        eprintln!("  .engine — used directly (must match this machine's TRT + GPU)");
        std::process::exit(1);
    }
    let (model_path, rtsp_url) = (&args[1], &args[2]);
    let save_dir = args.get(3).map(String::as_str).unwrap_or(".");

    // .onnx → versioned engine cache (build on first run); .engine → as-is.
    let profile = vrt_hub::EngineProfile {
        input: Some(("image".into(),
            vec![1, 3, 240, 320], vec![1, 3, 640, 640], vec![1, 3, 1088, 1920])),
        fp16: true, workspace_mb: 2048,
    };
    let engine_path = vrt_hub::EngineCache::default().resolve("xfeat-backbone", model_path, &profile)?;

    let logger  = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine  = Engine::from_file(runtime, &engine_path)?;

    // Hardware-resize to 1280×720 via VIC before any CUDA work.
    // Matches the engine's opt profile area (640×640 ≈ 409k px; 1280×720 = 921k px — within max).
    // Change these to 640×480 / 640×360 to hit the opt profile more closely.
    const RESIZE_W: u32 = 1280;
    const RESIZE_H: u32 = 720;
    let source       = RtspSource::connect_resized(rtsp_url, RESIZE_W, RESIZE_H)?;
    let (src_w, src_h) = (source.width(), source.height());
    let (dst_w, dst_h) = (pad32(src_w), pad32(src_h));
    println!("Stream: resized to {src_w}×{src_h} (VIC) → model input {dst_w}×{dst_h}");

    // Keep a handle to the CPU frame snapshot BEFORE moving source into the pipeline.
    let cpu_snap = source.latest_cpu_frame();

    let stream  = Stream::new_standalone()?.cuda_stream().clone();
    let preproc = NvmmPreprocessStage::new(stream.clone(), src_w, src_h, dst_w, dst_h)?;
    let xfeat   = XFeat::with_stream(
        Arc::clone(&engine),
        stream.clone(),
        XFeatParams::new(4096, 0.05, dst_h as usize, dst_w as usize),
    )?;

    let mut pipeline = Pipeline::new(stream, source)
        .pipe(preproc)
        .pipe(xfeat);

    let mut n    = 0usize;
    let mut sum  = [0.0f64; 5];
    let mut peak = [0.0f64; 5];

    loop {
        let Some(result) = pipeline.next() else { break };

        match result {
            Ok((kpts, t)) => {
                let phases = [t.source_ms, t.enqueue_ms, t.gpu_ms, t.sync_ms, t.finalize_ms];
                for i in 0..5 {
                    sum[i]  += phases[i];
                    if phases[i] > peak[i] { peak[i] = phases[i]; }
                }
                n += 1;
                println!("[frame {n:06}] {t}  | {} kpts", kpts.scores.len());

                // Every 30 frames: grab the latest CPU snapshot and save a PNG.
                if n % 30 == 0 {
                    let maybe_frame = cpu_snap.lock().ok().and_then(|mut g| g.take());
                    if let Some((rgba, fw, fh)) = maybe_frame {
                        let path = format!("{save_dir}/xfeat_{n:06}.png");
                        save_kpts(&rgba, fw, fh, &kpts.kpts_cpu, dst_w, dst_h, &path);
                    } else {
                        eprintln!("[viz] no CPU frame yet at frame {n}");
                    }
                }
            }
            Err(e) => eprintln!("error: {e}"),
        }

        if n % 100 == 0 && n > 0 {
            let a: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();
            // a: [source, enqueue, gpu, sync, finalize]
            let wall = a[0] + a[1] + a[3] + a[4];
            println!(
                "── avg@{n}: source={:.1}  enqueue={:.2}  gpu={:.2}  sync={:.1}  finalize={:.2}  total={:.1}ms  fps={:.1}",
                a[0], a[1], a[2], a[3], a[4], wall, 1000.0 / wall
            );
        }
    }

    if n > 0 {
        let a: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();
                let wall = a[0] + a[1] + a[3] + a[4];
        println!("\n── final ({n} frames)");
        println!("   avg: source={:.1}  enqueue={:.2}  gpu={:.2}  sync={:.1}  finalize={:.2}  total={:.1}ms",
            a[0], a[1], a[2], a[3], a[4], wall);
        println!("  peak: source={:.1}  enqueue={:.2}  gpu={:.2}  sync={:.1}  finalize={:.2}",
            peak[0], peak[1], peak[2], peak[3], peak[4]);
        println!("   fps: {:.1}", 1000.0 / wall);
    }

    Ok(())
}
