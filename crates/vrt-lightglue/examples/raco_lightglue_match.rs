//! End-to-end RaCo-ALIKED + LightGlue+ matching between two images.
//!
//! Shows the intended composition: **one shared stream**, extract both images (each
//! `submit` only enqueues), match, then a **single** `stream.synchronize()` drains
//! everything. Draws the surviving correspondences into a side-by-side PNG.
//!
//! Both engines come from `vrt-raco-aliked/scripts/split_raco_pipeline.py` and must
//! have been exported with the same `K`.
//!
//! Usage:
//!   cargo run --release -p vrt-lightglue --example raco_lightglue_match -- \
//!       <raco_aliked_extractor_kN.engine> <lightglue_matcher_kN.engine> \
//!       <left.png> <right.png> [out.png]

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;
use vrt_lightglue::LightGlue;
use vrt_raco_aliked::RaCoAliked;

const MIN_SCORE: f32 = 0.0; // LightGlue already filtered internally; tighten if noisy

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: raco_lightglue_match <extractor.engine> <matcher.engine> \
             <left.png> <right.png> [out.png]"
        );
        std::process::exit(1);
    }
    let (ext_path, mat_path) = (&args[1], &args[2]);
    let (left_path, right_path) = (&args[3], &args[4]);
    let out_path = args.get(5).map(String::as_str).unwrap_or("raco_match.png");

    // One stream shared by extractor and matcher: a single sync covers all their work.
    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let mut raco = RaCoAliked::from_engine_file(ext_path, stream.clone())?;
    let mut glue = LightGlue::from_engine_file(mat_path, stream.clone())?;
    println!(
        "extractor K={}  matcher K={}",
        raco.num_keypoints(),
        glue.num_keypoints()
    );

    let left_src = read_image_any_rgb8(left_path)?;
    let right_src = read_image_any_rgb8(right_path)?;
    let left_dev = left_src.to_cuda(&stream)?;
    let right_dev = right_src.to_cuda(&stream)?;

    let mut left_out = raco.alloc_result()?;
    let mut right_out = raco.alloc_result()?;
    let mut matches = glue.alloc_result()?;

    // All async — nothing has executed yet when these return.
    raco.submit(&left_dev, &mut left_out)?;
    raco.submit(&right_dev, &mut right_out)?;
    glue.submit_match(&left_out, &right_out, &mut matches)?;
    stream.synchronize()?; // the caller owns the one sync

    let pairs = matches.pairs(MIN_SCORE)?;
    let lk = left_out.keypoints_host()?;
    let rk = right_out.keypoints_host()?;
    println!(
        "{} matches of {} keypoints",
        pairs.len(),
        raco.num_keypoints()
    );

    // Median displacement between matched keypoints. For a pure-translation pair this
    // recovers the shift, so it doubles as a cheap correctness check: a matcher that is
    // merely *running* produces a scattered median, a matcher that is *working* pins it.
    if !pairs.is_empty() {
        let mut dxs: Vec<f32> = pairs.iter().map(|(a, b)| lk[*a].0 - rk[*b].0).collect();
        let mut dys: Vec<f32> = pairs.iter().map(|(a, b)| lk[*a].1 - rk[*b].1).collect();
        dxs.sort_by(f32::total_cmp);
        dys.sort_by(f32::total_cmp);
        let mid = dxs.len() / 2;
        println!(
            "median match displacement: dx={:.1} dy={:.1}",
            dxs[mid], dys[mid]
        );
    }

    // Side-by-side canvas with a line per correspondence.
    let (lh, lw) = (left_src.height(), left_src.width());
    let (rh, rw) = (right_src.height(), right_src.width());
    let (cw, ch) = (lw + rw, lh.max(rh));
    let mut canvas = vec![0u8; cw * ch * 3];
    blit(&mut canvas, cw, left_src.as_slice(), lw, lh, 0);
    blit(&mut canvas, cw, right_src.as_slice(), rw, rh, lw);

    for (i0, i1) in &pairs {
        let (x0, y0) = lk[*i0];
        let (x1, y1) = rk[*i1];
        draw_line(
            &mut canvas,
            (cw, ch),
            (x0 as i32, y0 as i32),
            (x1 as i32 + lw as i32, y1 as i32),
            [40, 220, 40],
        );
    }

    let out = Image::<u8, 3>::new(
        ImageSize {
            width: cw,
            height: ch,
        },
        canvas,
    )?;
    write_image_png_rgb8(out_path, &out)?;
    println!("saved {out_path}");
    Ok(())
}

/// Copy an interleaved RGB image into `canvas` at horizontal offset `x_off`.
fn blit(canvas: &mut [u8], cw: usize, src: &[u8], w: usize, h: usize, x_off: usize) {
    for y in 0..h {
        let dst = (y * cw + x_off) * 3;
        let s = y * w * 3;
        canvas[dst..dst + w * 3].copy_from_slice(&src[s..s + w * 3]);
    }
}

/// Bresenham line into an interleaved RGB buffer, between points `a` and `b`.
fn draw_line(buf: &mut [u8], size: (usize, usize), a: (i32, i32), b: (i32, i32), c: [u8; 3]) {
    let ((w, h), (x0, y0), (x1, y1)) = (size, a, b);
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
            let p = (y as usize * w + x as usize) * 3;
            buf[p..p + 3].copy_from_slice(&c);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}
