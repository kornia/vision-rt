//! RTSP → NVMM RGBA → GPU letterbox → TRT YOLO inference.
//!
//! ## Copy budget per frame
//! ```text
//! nvv4l2decoder   H.264 → NV12 NVMM        [HW decode, on-chip]
//! nvvidconv       NV12 NVMM → RGBA NVMM    [HW colorspace]
//! cudaImport      NVMM DMA-BUF → dev_ptr   [~150 µs per frame]
//! NvmmPreprocessStage  dev_ptr → CHW FP32  [GPU letterbox + normalize]
//! YoloInferStage  CHW FP32 → Vec<Detection>[TRT inference + CPU NMS]
//! ```
//!
//! Usage:
//!   cargo run --example rtsp_yolo -- <engine_path> <rtsp_url>

use trt::{Engine, Logger, Runtime, Stream, Pipeline};
use trt::logger::Severity;
use trt_yolo::{YoloInferStage, LetterboxInfo};
use trt_gst::{RtspSource, NvmmPreprocessStage};

const MODEL_W: u32 = 640;
const MODEL_H: u32 = 640;

fn main() -> Result<(), trt::BoxError> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rtsp_yolo <engine_path> <rtsp_url>");
        std::process::exit(1);
    }
    let (engine_path, rtsp_url) = (&args[1], &args[2]);

    let logger  = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine  = Engine::from_file(runtime, engine_path)?;

    // Connect to stream — blocks until the first frame establishes resolution.
    let source = RtspSource::connect(rtsp_url)?;
    let (src_w, src_h) = (source.width(), source.height());
    println!("Stream: {src_w}×{src_h} | model: {MODEL_W}×{MODEL_H}");

    let stream  = Stream::new_standalone()?.cuda_stream().clone();
    let lb_info = LetterboxInfo::from_dims(src_w, src_h, MODEL_W, MODEL_H);
    let preproc = NvmmPreprocessStage::new(stream.clone(), src_w, src_h, MODEL_W, MODEL_H)?;
    let infer   = YoloInferStage::new(engine, stream.clone(), lb_info, 0.25, 0.45)
        ?;

    let mut pipeline = Pipeline::new(stream, source)
        .pipe(preproc)
        .pipe(infer);

    let mut frame_idx = 0usize;
    while let Some(result) = pipeline.next() {
        match result {
            Ok((detections, t)) => {
                println!("[frame {frame_idx:06}] {t}  | {} dets", detections.len());
                for d in detections.iter().take(3) {
                    println!(
                        "  class={} score={:.2} [{:.0},{:.0},{:.0},{:.0}]",
                        d.class_id, d.score,
                        d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                    );
                }
            }
            Err(e) => eprintln!("[frame {frame_idx:06}] error: {e}"),
        }
        frame_idx += 1;
    }

    Ok(())
}
