//! Validate `RfDetrKpts` end-to-end against the Python reference: load an image at native size,
//! run the TRT keypoint model, print people + 17 COCO keypoints. Compare to `RFDETRKeypointPreview`
//! (e.g. zidane.jpg @ 1280×720 → 2 people, noses ≈ (654,358) and (1000,194)).
//!
//!   cargo run --release -p rfdetr_kpts_check -- <image> <engine>

use kornia_image::Image;
use kornia_io::functional::read_image_any_rgb8;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime};
use vrt_rfdetr_kpts::{RfDetrKpts, COCO_KEYPOINT_NAMES};

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args.get(1).map(String::as_str).unwrap_or("/tmp/people.jpg");
    let engine_path = args.get(2).cloned().unwrap_or_else(|| {
        "/home/nvidia/vision-rt/models/engines/rfdetr-keypoint-preview-d969cac0-trt10.3.0.30-sm87.engine".into()
    });

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut pose = RfDetrKpts::new(engine, stream.clone(), 0.5)?;

    // Load at NATIVE size (so keypoint pixels match the Python reference), upload once.
    let src = read_image_any_rgb8(image_path)?;
    let sz = src.size();
    let (w, h) = (sz.width as u32, sz.height as u32);
    let img = Image(src.0.to_cuda(&stream)?); // device-resident Image<u8,3>

    let people = pose.run(&img)?;
    println!("image {w}×{h} → {} people\n", people.len());
    for (i, p) in people.iter().enumerate() {
        let nose = p.keypoints[0];
        println!(
            "person{i}: score {:.2}  nose ({:.0},{:.0})  bbox [{:.0},{:.0},{:.0},{:.0}]",
            p.score, nose[0], nose[1], p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]
        );
        for (j, kp) in p.keypoints.iter().enumerate() {
            println!(
                "    {:>15}  ({:>4.0},{:>4.0})  vis {:.2}",
                COCO_KEYPOINT_NAMES[j], kp[0], kp[1], kp[2]
            );
        }
    }
    Ok(())
}
