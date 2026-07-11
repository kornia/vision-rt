//! Parallel detect + depth on one image: run RF-DETR-Seg and Depth Anything V2 on
//! the SAME device image, one sync, then sample per-instance **metric depth from
//! the instance mask**. Demonstrates the workspace composition pattern — two models
//! sharing one stream, each `submit` only enqueues, a single `synchronize()` drains
//! both, then a GPU fusion kernel reads both models' device outputs.
//!
//! Usage:
//!   cargo run --release -p vrt-depth-anything --example detect_depth -- \
//!       <seg.engine> <depth.engine> <image> [conf]

use kornia_image::Image;
use kornia_io::functional::read_image_any_rgb8;
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::RfDetrSeg;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: detect_depth <seg.engine> <depth.engine> <image> [conf]");
        std::process::exit(1);
    }
    let (seg_engine, depth_engine, image_path) = (&args[1], &args[2], &args[3]);
    let conf: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.4);

    // One shared stream: the detector, the depth net, and the fusion kernel all
    // enqueue on it, so a single sync completes the frame.
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut det = RfDetrSeg::from_engine_file(seg_engine, stream.clone(), conf)?;
    let mut depth = DepthAnything::from_engine_file(depth_engine, stream.clone())?;

    let src = read_image_any_rgb8(image_path)?;
    let dev = Image(src.0.to_cuda(&stream)?);

    let mut d = det.alloc_result()?;
    let mut z = depth.alloc_result()?;

    // Enqueue both models on the same image (no sync), then the fusion kernel that
    // reads the detector's GPU masks + the depth map. One sync drains everything.
    det.submit(&dev, &mut d)?;
    depth.submit(&dev, &mut z)?;
    let zs_dev = depth.sample_masks(&z, d.masks_slice(), d.mask_size(), d.count())?;
    stream.synchronize()?;

    let instances = d.instances()?;
    let zs = stream.clone_dtoh(&zs_dev)?; // per-instance metric depth (meters)

    let (mw, mh) = depth.map_size();
    println!(
        "{}x{} → {} instances | depth map {mw}x{mh} (metric meters)",
        src.0.width(),
        src.0.height(),
        instances.len()
    );
    for (i, (inst, z)) in instances.iter().zip(&zs).enumerate() {
        let [x1, y1, x2, y2] = inst.bbox;
        let area: usize = inst.mask.iter().map(|&m| m as usize).sum();
        println!(
            "  inst {i}: class {:<3} score {:.3}  box [{:.0},{:.0},{:.0},{:.0}]  mask {area}px  depth {z:.2} m",
            inst.class_id, inst.score, x1, y1, x2, y2
        );
    }
    Ok(())
}
