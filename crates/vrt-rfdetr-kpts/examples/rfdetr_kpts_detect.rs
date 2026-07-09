//! Single-image RF-DETR keypoint (pose) detection: build/cache engine, run, print poses.
//!
//! Usage:
//!   cargo run --release -p vrt-rfdetr-kpts --example rfdetr_kpts_detect -- \
//!       <model.onnx|engine>  <image>  [conf]
//!   # or pull from Hugging Face (kornia/rfdetr-kpts):
//!   cargo run --release -p vrt-rfdetr-kpts --example rfdetr_kpts_detect --features hub -- \
//!       hub  <image>  [conf]

use kornia_image::Image;
use kornia_io::functional::read_image_any_rgb8;
use vrt_rfdetr_kpts::{RfDetrKpts, COCO_KEYPOINT_NAMES};

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rfdetr_kpts_detect <model.onnx|engine> <image> [conf]");
        std::process::exit(1);
    }
    let (model_path, image_path) = (&args[1], &args[2]);
    let conf: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut pose = if model_path == "hub" {
        #[cfg(feature = "hub")]
        {
            RfDetrKpts::from_hub(stream.clone(), conf)?
        }
        #[cfg(not(feature = "hub"))]
        {
            return Err("pass an .onnx/.engine path, or rebuild with --features hub".into());
        }
    } else {
        let profile = vrt_hub::EngineProfile {
            input: None,
            fp16: true,
            workspace_mb: 2048,
        };
        let engine_path =
            vrt_hub::EngineCache::default().resolve("rfdetr-kpts", model_path, &profile)?;
        RfDetrKpts::from_engine_file(&engine_path, stream.clone(), conf)?
    };

    let src = read_image_any_rgb8(image_path)?;
    let dev = Image(src.0.to_cuda(&stream)?);

    // Async: submit → one caller sync → decode.
    let mut out = pose.alloc_result()?;
    pose.submit(&dev, &mut out)?;
    stream.synchronize()?;
    let people = out.poses();

    println!(
        "{}x{} → {} people (conf ≥ {conf})",
        src.0.width(),
        src.0.height(),
        people.len()
    );
    for (i, p) in people.iter().enumerate().take(10) {
        let [x1, y1, x2, y2] = p.bbox;
        println!(
            "  person {i}: score {:.3}  box [{:.0},{:.0},{:.0},{:.0}]",
            p.score, x1, y1, x2, y2
        );
        // A few visible joints for a sanity check.
        for (j, kp) in p.keypoints.iter().enumerate() {
            if kp[2] >= 0.5 {
                println!(
                    "      {:14} ({:.0},{:.0})  {:.2}",
                    COCO_KEYPOINT_NAMES[j], kp[0], kp[1], kp[2]
                );
            }
        }
    }
    Ok(())
}
