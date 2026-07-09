//! Single-image RF-DETR detection: build/cache the engine, run, print boxes.
//!
//! Usage:
//!   cargo run --release -p vrt-rfdetr --example rfdetr_detect -- \
//!       <model.onnx|engine>  <image>  [conf]

use kornia_image::Image;
use kornia_io::functional::read_image_any_rgb8;
use vrt_rfdetr::RfDetr;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rfdetr_detect <model.onnx|engine> <image> [conf]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let conf: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);

    // .onnx → on-device engine cache (static shapes); .engine → used directly.
    let profile = vrt_hub::EngineProfile {
        input: None,
        fp16: true,
        workspace_mb: 2048,
    };
    let engine_path = vrt_hub::EngineCache::default().resolve("rfdetr", model_path, &profile)?;

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut det = RfDetr::from_engine_file(&engine_path, stream.clone(), conf)?;

    let src = read_image_any_rgb8(image_path)?; // Rgb8 (derefs to Image<u8,3>)
    let dev = Image(src.0.to_cuda(&stream)?);

    // Async: submit → one caller sync → read.
    let mut out = det.alloc_result()?;
    det.submit(&dev, &mut out)?;
    stream.synchronize()?;
    let dets = out.detections()?;

    println!(
        "{}x{} → {} detections (conf ≥ {conf}, {} queries)",
        src.0.width(),
        src.0.height(),
        dets.len(),
        det.num_queries()
    );
    for d in dets.iter().take(20) {
        let [x1, y1, x2, y2] = d.bbox;
        println!(
            "  class {:3}  score {:.3}  box [{:.0},{:.0},{:.0},{:.0}]",
            d.class_id, d.score, x1, y1, x2, y2
        );
    }
    Ok(())
}
