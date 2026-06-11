//! GPU smoke test: load a .engine, run YOLO on a static image, print detections.
//!
//! Usage: cargo run --example yolo_image --features yolo -- <engine_path> <image_path>
//!
//! On success, prints bounding boxes to stdout. On a typical COCO test image
//! (e.g. the "bus" image from ultralytics), expect class 0 (person) and
//! class 5 (bus) detections with score > 0.5.

use std::sync::Arc;
use trt::models::yolo;
use trt::logger::Severity;
use trt::{Engine, Logger, Runtime, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: yolo_image <engine_path> <image_path>");
        std::process::exit(1);
    }
    let engine_path = &args[1];
    let image_path = &args[2];

    // Load the image (any format supported by the `image` crate).
    let img = image::open(image_path)?;
    let img = img.to_rgba8();
    let (src_w, src_h) = (img.width(), img.height());
    let rgba_bytes = img.as_raw();

    // Pre-process: letterbox to 640×640.
    let model_size = 640u32;
    let (chw_input, lb_info) =
        yolo::letterbox_rgba_to_chw(rgba_bytes, src_w, src_h, model_size, model_size);

    println!(
        "Image: {}×{}, letterbox scale={:.3}, pad=({:.1},{:.1})",
        src_w, src_h, lb_info.scale, lb_info.pad_left, lb_info.pad_top
    );

    // Load engine and run inference.
    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, engine_path)?;

    // Print discovered I/O tensors.
    println!("Engine I/O tensors:");
    for spec in engine.specs() {
        println!(
            "  {:?} {:7?} {:?} shape={:?}",
            spec.mode, spec.dtype, spec.name, spec.dims
        );
    }

    let mut session = Session::new(Arc::clone(&engine))?;
    let outputs = session.run(&[("images", &chw_input)])?;

    // Post-process.
    let output_tensor = outputs.values().next().expect("no output tensors");
    let raw_f32 = output_tensor.as_f32();
    let detections = yolo::postprocess(
        raw_f32,
        &output_tensor.shape,
        &lb_info,
        0.25, // conf_threshold
        0.45, // iou_threshold
        None, // labels: pass a COCO list here for named output
    );

    println!("Detections ({})", detections.len());
    for d in &detections {
        println!(
            "  class={} score={:.3} bbox=[{:.0},{:.0},{:.0},{:.0}]",
            d.class_id, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
        );
    }

    if detections.is_empty() {
        println!("  (no detections — try lowering conf_threshold or check the engine)");
    }

    Ok(())
}
