//! Single-model metric depth: run Depth Anything V2 on one image → print depth
//! stats and save a Turbo-colorized depth PNG. The async / caller-owned flow with
//! one sync, then a host copy of the metric depth map for colorization.
//!
//! Usage:
//!   cargo run --release -p vrt-depth-anything --example depth -- \
//!       <depth.engine> <image> [out.png]

use kornia_image::Image;
use kornia_imgproc::color::{apply_colormap, ColormapType};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt_depth_anything::DepthAnything;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: depth <depth.engine> <image> [out.png]");
        std::process::exit(1);
    }
    let (depth_engine, image_path) = (&args[1], &args[2]);
    let out_path = args.get(3);

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut depth = DepthAnything::from_engine_file(depth_engine, stream.clone())?;

    let src = read_image_any_rgb8(image_path)?;
    let dev = Image(src.0.to_cuda(&stream)?);

    // Async: submit (no sync, no host copy) → one caller sync → host copy of the
    // metric depth map (meters) for CPU colorization.
    let mut z = depth.alloc_result()?;
    depth.submit(&dev, &mut z)?;
    stream.synchronize()?;
    let dmap = z.depth_host()?; // vrt_types::DepthImage = Image<f32,1>, metric meters

    let (mw, mh) = depth.map_size();
    let vals = dmap.as_slice();
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in vals {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
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

    // Optional: Turbo-colorized depth PNG. Normalize meters over [min,max] → u8
    // gray (host path — vision-rt has cudarc, not gpu-cuda, so keep it on host),
    // then apply the Turbo colormap.
    if let Some(out_path) = out_path {
        let span = (max - min).max(1e-6);
        let gray_buf: Vec<u8> = vals
            .iter()
            .map(|&v| (((v - min) / span).clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        let gray = Image::<u8, 1>::new(dmap.size(), gray_buf)?;
        let mut rgb = Image::<u8, 3>::from_size_val(dmap.size(), 0)?;
        apply_colormap(&gray, &mut rgb, ColormapType::Turbo)?;
        write_image_png_rgb8(out_path, &rgb)?;
        println!("saved {out_path}");
    }
    Ok(())
}
