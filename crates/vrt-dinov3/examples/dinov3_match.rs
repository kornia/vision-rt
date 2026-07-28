//! Global-descriptor similarity between images: run DINOv3 on each, print the cosine
//! between their L2-normed CLS descriptors. The fastest end-to-end confidence check
//! before wiring up a camera.
//!
//! With two images it prints one number. With more, it prints the full similarity
//! matrix — useful for eyeballing whether the descriptor separates your actual data.
//!
//! Usage:
//!   cargo run --release -p vrt-dinov3 --example dinov3_match -- \
//!       <dinov3.engine> <image> <image> [image ...]

use kornia_io::functional::read_image_any_rgb8;
use vrt_dinov3::DinoV3;

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: dinov3_match <dinov3.engine> <image> <image> [image ...]");
        std::process::exit(1);
    }
    let engine = &args[1];
    let paths = &args[2..];

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut dino = if engine == "hub" {
        #[cfg(feature = "hub")]
        {
            DinoV3::from_hub(stream.clone())?
        }
        #[cfg(not(feature = "hub"))]
        {
            return Err("pass an .engine path, or rebuild with --features hub".into());
        }
    } else {
        DinoV3::from_engine_file(engine, stream.clone())?
    };

    let (gw, gh) = dino.grid();
    println!(
        "engine: input {}x{} → {gw}x{gh} patch grid, descriptor dim {}",
        gw * 16,
        gh * 16,
        dino.dim()
    );

    // One reused result buffer: each frame is read to host before the next submit, so a
    // single buffer is enough. Holding several would let multiple frames stay in flight.
    let mut r = dino.alloc_result()?;
    let mut descs: Vec<Vec<f32>> = Vec::with_capacity(paths.len());
    for p in paths {
        let src = read_image_any_rgb8(p)?;
        let dev = src.0.to_cuda(&stream)?;
        dino.submit(&dev, &mut r)?; // enqueue, no sync
        stream.synchronize()?; // the one sync
        descs.push(r.descriptor_host()?);
    }

    let cos = |a: &[f32], b: &[f32]| -> f32 {
        // Both are unit-norm from the device kernel, so the dot IS the cosine.
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    };

    if descs.len() == 2 {
        println!(
            "cos({}, {}) = {:.4}",
            short(&paths[0]),
            short(&paths[1]),
            cos(&descs[0], &descs[1])
        );
        return Ok(());
    }

    print!("{:>18}", "");
    for p in paths {
        print!("{:>10}", short(p));
    }
    println!();
    for (i, p) in paths.iter().enumerate() {
        print!("{:>18}", short(p));
        for j in 0..descs.len() {
            print!("{:>10.4}", cos(&descs[i], &descs[j]));
        }
        println!();
    }
    Ok(())
}

/// Trailing path component, truncated to keep the matrix columns aligned.
///
/// Counted in **chars**, not bytes: `&name[name.len() - 8..]` panics the moment a
/// filename contains a non-ASCII character whose encoding straddles that byte index.
fn short(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let n = name.chars().count();
    if n <= 9 {
        name.to_string()
    } else {
        format!("…{}", name.chars().skip(n - 8).collect::<String>())
    }
}
