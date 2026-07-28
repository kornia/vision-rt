//! Single-model metric depth: run Depth Anything V2 on one image → print depth
//! stats and save a Turbo-colorized depth PNG. The async / caller-owned flow with
//! one sync, then a host copy of the metric depth map for colorization.
//!
//! Usage:
//!   cargo run --release -p vrt-depth-anything --example depth -- \
//!       <depth.engine> <image> [out.png]

use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt_depth_anything::DepthAnything;

#[path = "common/mod.rs"]
mod common;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: depth <depth.engine> <image> [out.png]");
        std::process::exit(1);
    }
    let (depth_engine, image_path) = (&args[1], &args[2]);
    let out_path = args.get(3);

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut depth = if depth_engine == "hub" {
        #[cfg(feature = "hub")]
        {
            DepthAnything::from_hub(stream.clone())?
        }
        #[cfg(not(feature = "hub"))]
        {
            return Err("pass an .engine path, or rebuild with --features hub".into());
        }
    } else {
        DepthAnything::from_engine_file(depth_engine, stream.clone())?
    };

    let src = read_image_any_rgb8(image_path)?;
    let dev = src.0.to_cuda(&stream)?;

    // Async: submit (no sync, no host copy) → one caller sync → host copy of the
    // metric depth map (meters) for CPU colorization.
    let mut z = depth.alloc_result()?;
    depth.submit(&dev, &mut z)?;
    stream.synchronize()?;
    let dmap = z.depth_host()?; // vrt_types::DepthImage = Image<f32,1>, metric meters

    let (mw, mh) = depth.map_size();
    let vals = dmap.as_slice();
    let mut min = f32::INFINITY;
    for &v in vals {
        if v.is_finite() {
            min = min.min(v);
        }
    }
    let mut sorted: Vec<f32> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let center = vals[(mh / 2) * mw + mw / 2];
    println!(
        "{}x{} → depth map {mw}x{mh} (metric meters) | min {min:.2} m | median {median:.2} m | center {center:.2} m",
        src.0.width(),
        src.0.height(),
    );

    // Optional: Turbo-colorized depth PNG.
    if let Some(out_path) = out_path {
        write_image_png_rgb8(out_path, &common::depth_to_turbo(&dmap)?)?;
        println!("saved {out_path}");
    }
    Ok(())
}
