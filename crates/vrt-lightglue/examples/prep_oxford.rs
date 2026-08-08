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
use common::{
    arg_or, mat3_from_row_major, parse_interpolation, read_floats, resize_matrix, resize_to_fit,
};

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
    // Exactly one whitespace character separates the header from the payload — but on a
    // CRLF-terminated file that character is "\r\n", two bytes. Skipping a fixed one would
    // leave the '\n' as the first payload byte, shifting every pixel by one channel: the
    // length check below still passes (the file is one byte longer to match) and the
    // image comes out silently wrong rather than failing.
    match raw.get(i) {
        Some(b'\r') if raw.get(i + 1) == Some(&b'\n') => i += 2,
        Some(c) if (*c as char).is_whitespace() => i += 1,
        _ => {
            return Err(
                format!("{}: header is not terminated by whitespace", path.display()).into(),
            )
        }
    }
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
            "{}: truncated ({} of {need} payload bytes)",
            path.display(),
            // The header parse can leave `i` one past the end on a file truncated mid
            // header, so this subtraction must not be allowed to wrap.
            raw.len().saturating_sub(i)
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

/// Original pixel size of a frame, and the per-axis scale the resize applied to it.
type Prepared = ((usize, usize), (f64, f64));

/// Convert one frame, returning its original size and the per-axis scale applied.
fn convert(
    src: &Path,
    dst: &Path,
    max_side: usize,
    interpolation: InterpolationMode,
) -> Result<Prepared, vrt::BoxError> {
    let img = read_netpbm_rgb8(src)?;
    let scaled = resize_to_fit(&img, max_side, interpolation)?;
    write_image_png_rgb8(dst, &scaled.image)?;
    Ok(((img.cols(), img.rows()), (scaled.scale_x, scaled.scale_y)))
}

/// Rec.601 luma of one pixel. Takes the slice rather than the `Image` so the caller's
/// per-pixel loop does not re-borrow it a quarter of a million times per pair.
fn luma(s: &[u8], cols: usize, x: usize, y: usize) -> f64 {
    let p = (y * cols + x) * 3;
    0.299 * s[p] as f64 + 0.587 * s[p + 1] as f64 + 0.114 * s[p + 2] as f64
}

/// Warp `b`'s grid back through `h` into `a` and correlate. Returns (overlap %, corr).
fn photometric_check(a: &Image<u8, 3>, b: &Image<u8, 3>, h: &Mat3F64) -> Option<(f64, f64)> {
    if h.determinant().abs() < 1e-12 {
        return None;
    }
    let hi = h.inverse();
    // The overlap is most of the frame on a good pair, so size for it once instead of
    // growing two 260k-element vectors by doubling on every sequence.
    let n_px = b.rows() * b.cols();
    let (mut xs, mut ys) = (Vec::with_capacity(n_px), Vec::with_capacity(n_px));
    let (a_px, a_cols) = (a.as_slice(), a.cols());
    let (b_px, b_cols) = (b.as_slice(), b.cols());
    for y in 0..b.rows() {
        for x in 0..b_cols {
            let p = hi * Vec3F64::new(x as f64, y as f64, 1.0);
            if p.z.abs() < 1e-12 {
                continue;
            }
            let (sx, sy) = (p.x / p.z, p.y / p.z);
            if sx >= 0.0 && sy >= 0.0 && (sx as usize) < a_cols && (sy as usize) < a.rows() {
                xs.push(luma(a_px, a_cols, sx as usize, sy as usize));
                ys.push(luma(b_px, b_cols, x, y));
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
    let max_side: usize = arg_or(&a, 3, "max_side", 640)?;
    let interpolation = parse_interpolation(a.get(4).map(String::as_str).unwrap_or("lanczos"))?;
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
        let Some(&(orig1, scale1)) = sizes.get(&1) else {
            println!("{seq}: no img1, skipped");
            continue;
        };
        println!(
            "{seq}: {} images, {orig1:?} scaled by ({:.6}, {:.6})",
            sizes.len(),
            scale1.0,
            scale1.1
        );

        let img1 = read_image_any_rgb8(od.join("img1.png"))?;
        for i in 2..=6 {
            let hp = sd.join(format!("H1to{i}p"));
            let (has_img, has_h) = (sizes.contains_key(&i), hp.exists());
            // Neither present: this sequence legitimately has fewer than 6 frames.
            // Exactly one present: the download is truncated, and dropping the pair
            // silently is precisely what the `suspect` gate below exists to prevent —
            // the benchmark would shrink with nothing in the output saying so.
            if !has_img && !has_h {
                continue;
            }
            if !has_img {
                suspect += 1;
                println!("  img{i}: H1to{i}p present but img{i} is missing  <-- SUSPECT, excluded");
                continue;
            }
            if !has_h {
                suspect += 1;
                println!("  img{i}: image present but H1to{i}p is missing  <-- SUSPECT, excluded");
                continue;
            }
            let &(_orig_i, scale_i) = sizes.get(&i).expect("has_img was just checked");
            let v = read_floats(&hp)?;
            let h = mat3_from_row_major(&v).map_err(|e| format!("{}: {e}", hp.display()))?;

            // The scale actually applied, per axis, plus the half-pixel offset.
            let (s1, s2) = (
                resize_matrix(scale1.0, scale1.1),
                resize_matrix(scale_i.0, scale_i.1),
            );
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
            // A pair whose ground truth fails the check must not reach the manifest: an
            // operator who scrolls past the error would otherwise run eval_oxford against
            // homographies this tool just declared unusable and read the low inlier rate
            // as a matcher result.
            match photometric_check(&img1, &img_i, &hs) {
                Some((overlap, corr)) if corr >= MIN_CORRELATION => {
                    println!("  img{i}: overlap {overlap:5.1}%  corr {corr:+.3}");
                    manifest.push(format!(
                        "{seq} img1.png img{i}.png {name} {:.12} {:.12}",
                        scale_i.0, scale_i.1
                    ));
                }
                Some((overlap, corr)) => {
                    suspect += 1;
                    println!(
                        "  img{i}: overlap {overlap:5.1}%  corr {corr:+.3}  <-- SUSPECT, excluded"
                    );
                }
                None => {
                    suspect += 1;
                    println!("  img{i}: singular homography  <-- SUSPECT, excluded");
                }
            }
        }
    }

    // Refuse to leave a manifest behind at all if any pair failed: a partial manifest
    // silently shrinks the benchmark, and eval_oxford has no way to notice.
    if suspect > 0 {
        let _ = std::fs::remove_file(out_root.join("manifest.txt"));
        return Err(format!(
            "{suspect} pairs failed the photometric check; ground truth is not usable \
             and no manifest was written"
        )
        .into());
    }
    std::fs::write(out_root.join("manifest.txt"), manifest.join("\n") + "\n")?;
    println!(
        "\n{} evaluation pairs -> {}/manifest.txt",
        manifest.len(),
        out_root.display()
    );
    Ok(())
}
