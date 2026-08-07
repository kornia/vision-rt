//! Prepare the Oxford/VGG affine sequences for `examples/eval_oxford`.
//!
//! Fetch the sequences first with `scripts/get_oxford.sh`. This converts each sequence's
//! `.ppm`/`.pgm` frames to PNG, downscales them so the long side is <= `MAX_SIDE` and
//! both sides are multiples of 32 (RaCo's `input_dim_divisor`, and the published
//! engines' 640 shape profile), rescales the ground-truth homographies to match, and
//! writes the manifest `eval_oxford` reads.
//!
//! **Rescaling images invalidates the homographies.** If `p2 = H p1` in original pixels
//! and image `i` is scaled by `S_i`, then in resized pixels `p2' = S2 H S1⁻¹ p1'`.
//! Getting this wrong is silent and catastrophic in the same direction for every method:
//! the eval would report poor inlier rates that are the harness's fault. So each pair is
//! verified photometrically afterwards -- one image is warped onto the other through the
//! rescaled homography and the correlation over the overlap is reported. Low correlation
//! means the ground truth is broken, not the matcher.
//!
//! PPM/PGM is parsed here rather than through `kornia_io`, which does not read Netpbm;
//! everything after the header is a plain byte buffer handed to a `kornia` image.
//!
//! Usage:
//!   cargo run --release -p vrt-lightglue --example prep_oxford -- \
//!       <src_dir> <out_dir> [max_side] [nearest|bilinear|bicubic|lanczos]

use std::path::{Path, PathBuf};

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_rgb8;

use kornia_algebra::{Mat3F64, Vec3F64};

#[path = "common/mod.rs"]
mod common;
use common::{mat3_from_row_major, parse_interpolation, read_floats, resize_to_fit};

/// Below this, treat the rescaled ground truth as broken rather than the matcher.
const MIN_CORRELATION: f64 = 0.3;

/// Read a binary Netpbm file (P5 grayscale / P6 RGB, 8-bit) as an RGB image.
fn read_netpbm_rgb8(path: &Path) -> Result<Image<u8, 3>, vrt::BoxError> {
    let raw = std::fs::read(path)?;
    if raw.len() < 2 || raw[0] != b'P' {
        return Err(format!("{}: not a Netpbm file", path.display()).into());
    }
    let magic = raw[1];
    let mut fields: Vec<usize> = Vec::with_capacity(3);
    let mut i = 2usize;
    // Header: three whitespace-separated integers (width, height, maxval), with '#'
    // comments legal anywhere between them.
    while fields.len() < 3 {
        while i < raw.len() && (raw[i] as char).is_whitespace() {
            i += 1;
        }
        if i < raw.len() && raw[i] == b'#' {
            while i < raw.len() && raw[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < raw.len() && (raw[i] as char).is_ascii_digit() {
            i += 1;
        }
        if start == i {
            return Err(format!("{}: malformed Netpbm header", path.display()).into());
        }
        fields.push(std::str::from_utf8(&raw[start..i])?.parse()?);
    }
    i += 1; // exactly one whitespace byte separates the header from the payload
    let (w, h, maxval) = (fields[0], fields[1], fields[2]);
    if maxval != 255 {
        return Err(format!(
            "{}: only 8-bit Netpbm supported (maxval {maxval})",
            path.display()
        )
        .into());
    }

    let src_channels = match magic {
        b'5' => 1,
        b'6' => 3,
        other => return Err(format!("{}: unsupported P{}", path.display(), other as char).into()),
    };
    let need = w * h * src_channels;
    if raw.len() < i + need {
        return Err(format!(
            "{}: truncated ({} of {need} bytes)",
            path.display(),
            raw.len() - i
        )
        .into());
    }
    let body = &raw[i..i + need];
    let data = if src_channels == 3 {
        body.to_vec()
    } else {
        body.iter().flat_map(|&v| [v, v, v]).collect()
    };
    Ok(Image::<u8, 3>::new(
        ImageSize {
            width: w,
            height: h,
        },
        data,
    )?)
}

/// Image dimensions as (width, height).
type Dims = (usize, usize);

/// Convert one frame, returning original and resized dimensions.
fn convert(
    src: &Path,
    dst: &Path,
    max_side: usize,
    interpolation: InterpolationMode,
) -> Result<(Dims, Dims), vrt::BoxError> {
    let img = read_netpbm_rgb8(src)?;
    let (out, _, _) = resize_to_fit(&img, max_side, interpolation)?;
    let dims = (out.cols(), out.rows());
    write_image_png_rgb8(dst, &out)?;
    Ok(((img.cols(), img.rows()), dims))
}

fn luma(img: &Image<u8, 3>, x: usize, y: usize) -> f64 {
    let p = (y * img.cols() + x) * 3;
    let s = img.as_slice();
    0.299 * s[p] as f64 + 0.587 * s[p + 1] as f64 + 0.114 * s[p + 2] as f64
}

/// Warp `b`'s grid back through `h` into `a` and correlate. Returns (overlap %, corr).
fn photometric_check(a: &Image<u8, 3>, b: &Image<u8, 3>, h: &Mat3F64) -> Option<(f64, f64)> {
    if h.determinant().abs() < 1e-12 {
        return None;
    }
    let hi = h.inverse();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for y in 0..b.rows() {
        for x in 0..b.cols() {
            let p = hi * Vec3F64::new(x as f64, y as f64, 1.0);
            if p.z.abs() < 1e-12 {
                continue;
            }
            let (sx, sy) = (p.x / p.z, p.y / p.z);
            if sx >= 0.0 && sy >= 0.0 && (sx as usize) < a.cols() && (sy as usize) < a.rows() {
                xs.push(luma(a, sx as usize, sy as usize));
                ys.push(luma(b, x, y));
            }
        }
    }
    let n = xs.len();
    if n < 500 {
        return Some((100.0 * n as f64 / (b.rows() * b.cols()) as f64, f64::NAN));
    }
    let (mx, my) = (
        xs.iter().sum::<f64>() / n as f64,
        ys.iter().sum::<f64>() / n as f64,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (u, v) in xs.iter().zip(&ys) {
        let (du, dv) = (u - mx, v - my);
        sxy += du * dv;
        sxx += du * du;
        syy += dv * dv;
    }
    let denom = (sxx * syy).sqrt();
    let corr = if denom < 1e-12 { f64::NAN } else { sxy / denom };
    Some((100.0 * n as f64 / (b.rows() * b.cols()) as f64, corr))
}

fn main() -> Result<(), vrt::BoxError> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "Usage: prep_oxford <src_dir> <out_dir> [max_side] [nearest|bilinear|bicubic|lanczos]"
        );
        std::process::exit(1);
    }
    let (src_root, out_root) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let max_side: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(640);
    let interpolation = parse_interpolation(a.get(4).map(String::as_str).unwrap_or("bilinear"))?;
    std::fs::create_dir_all(&out_root)?;

    let mut seqs: Vec<String> = std::fs::read_dir(&src_root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    seqs.sort();

    let mut manifest = Vec::new();
    let mut suspect = 0usize;
    for seq in &seqs {
        let (sd, od) = (src_root.join(seq), out_root.join(seq));
        std::fs::create_dir_all(&od)?;

        let mut sizes = std::collections::BTreeMap::new();
        for i in 1..=6 {
            let src = ["ppm", "pgm"]
                .iter()
                .map(|e| sd.join(format!("img{i}.{e}")))
                .find(|p| p.exists());
            if let Some(src) = src {
                sizes.insert(
                    i,
                    convert(
                        &src,
                        &od.join(format!("img{i}.png")),
                        max_side,
                        interpolation,
                    )?,
                );
            }
        }
        let Some(&(orig1, new1)) = sizes.get(&1) else {
            println!("{seq}: no img1, skipped");
            continue;
        };
        println!("{seq}: {} images, {orig1:?} -> {new1:?}", sizes.len());

        let img1 = read_image_any_rgb8(od.join("img1.png"))?;
        for i in 2..=6 {
            let hp = sd.join(format!("H1to{i}p"));
            let Some(&(orig_i, new_i)) = sizes.get(&i) else {
                continue;
            };
            if !hp.exists() {
                continue;
            }
            let v = read_floats(&hp)?;
            if v.len() < 9 {
                return Err(format!("{}: expected 9 floats", hp.display()).into());
            }
            let h = mat3_from_row_major(&v[..9]);

            let s1 = Mat3F64::from_diagonal(Vec3F64::new(
                new1.0 as f64 / orig1.0 as f64,
                new1.1 as f64 / orig1.1 as f64,
                1.0,
            ));
            let s2 = Mat3F64::from_diagonal(Vec3F64::new(
                new_i.0 as f64 / orig_i.0 as f64,
                new_i.1 as f64 / orig_i.1 as f64,
                1.0,
            ));
            if s1.determinant().abs() < 1e-12 {
                return Err("degenerate image scale".into());
            }
            let hs = s2 * h * s1.inverse();
            // Fix the projective scale so the written matrix is comparable across pairs.
            let w = hs.z_axis.z;
            if w.abs() < 1e-12 {
                return Err(format!("{}: degenerate rescaled homography", hp.display()).into());
            }
            let hs = Mat3F64::from_cols_array(&hs.to_cols_array().map(|x| x / w));

            let name = format!("H1to{i}.txt");
            std::fs::write(
                od.join(&name),
                hs.transpose()
                    .to_cols_array()
                    .iter()
                    .map(|x| format!("{x:.10}"))
                    .collect::<Vec<_>>()
                    .join(" ")
                    + "\n",
            )?;

            let img_i = read_image_any_rgb8(od.join(format!("img{i}.png")))?;
            match photometric_check(&img1, &img_i, &hs) {
                Some((overlap, corr)) if corr >= MIN_CORRELATION => {
                    println!("  img{i}: overlap {overlap:5.1}%  corr {corr:+.3}")
                }
                Some((overlap, corr)) => {
                    suspect += 1;
                    println!("  img{i}: overlap {overlap:5.1}%  corr {corr:+.3}  <-- SUSPECT");
                }
                None => {
                    suspect += 1;
                    println!("  img{i}: singular homography  <-- SUSPECT");
                }
            }
            manifest.push(format!("{seq} img1.png img{i}.png {name}"));
        }
    }

    std::fs::write(out_root.join("manifest.txt"), manifest.join("\n") + "\n")?;
    println!(
        "\n{} evaluation pairs -> {}/manifest.txt",
        manifest.len(),
        out_root.display()
    );
    if suspect > 0 {
        return Err(format!(
            "{suspect} pairs failed the photometric check; ground truth is not usable"
        )
        .into());
    }
    Ok(())
}
