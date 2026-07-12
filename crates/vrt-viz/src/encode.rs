//! Image encoding: a reusable JPEG encoder (SIMD TurboJPEG behind the `turbojpeg`
//! feature, else the pure-Rust encoder), plus PNG and animated-GIF writers.

use kornia_image::{Image, ImageSize};

use crate::VizError;

/// A reusable RGB→JPEG encoder. With the `turbojpeg` feature it uses kornia-io's SIMD
/// TurboJPEG compressor at 4:2:0; otherwise the pure-Rust encoder. Build once, reuse.
pub struct JpegEncoder {
    #[cfg(feature = "turbojpeg")]
    inner: kornia_io::jpegturbo::JpegTurboEncoder,
    #[cfg(not(feature = "turbojpeg"))]
    quality: u8,
}

impl JpegEncoder {
    /// New encoder at `quality` (0–100).
    pub fn new(quality: u8) -> Result<Self, VizError> {
        #[cfg(feature = "turbojpeg")]
        {
            let inner = kornia_io::jpegturbo::JpegTurboEncoder::new()
                .map_err(|e| VizError::Encode(e.to_string()))?;
            inner
                .set_quality(quality as i32)
                .map_err(|e| VizError::Encode(e.to_string()))?;
            inner
                .set_subsamp(turbojpeg::Subsamp::Sub2x2)
                .map_err(|e| VizError::Encode(e.to_string()))?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "turbojpeg"))]
        {
            Ok(Self { quality })
        }
    }

    /// Encode a host RGB buffer to JPEG bytes. Takes the buffer by value so the render
    /// output moves straight into the encode with no per-frame copy.
    pub fn encode(&self, rgb: Vec<u8>, w: usize, h: usize) -> Result<Vec<u8>, VizError> {
        let img = Image::<u8, 3>::new(
            ImageSize {
                width: w,
                height: h,
            },
            rgb,
        )?;
        #[cfg(feature = "turbojpeg")]
        {
            self.inner
                .encode_rgb8(&img)
                .map_err(|e| VizError::Encode(e.to_string()))
        }
        #[cfg(not(feature = "turbojpeg"))]
        {
            let mut out = Vec::new();
            kornia_io::jpeg::encode_image_jpeg_rgb8(&img, self.quality, &mut out)
                .map_err(|e| VizError::Encode(e.to_string()))?;
            Ok(out)
        }
    }
}

/// Write a host RGB buffer to a PNG file.
pub fn encode_png(path: &str, rgb: &[u8], w: usize, h: usize) -> Result<(), VizError> {
    let img = Image::<u8, 3>::new(
        ImageSize {
            width: w,
            height: h,
        },
        rgb.to_vec(),
    )?;
    kornia_io::png::write_image_png_rgb8(path, &img)
        .map_err(|e| VizError::Encode(e.to_string()))?;
    Ok(())
}

/// Encode RGB frames into a looping animated GIF (`delay_cs` = centiseconds/frame).
pub fn write_gif(
    path: &str,
    frames: &[Vec<u8>],
    w: u16,
    h: u16,
    delay_cs: u16,
) -> Result<(), VizError> {
    let file = std::fs::File::create(path)?;
    let mut enc = gif::Encoder::new(std::io::BufWriter::new(file), w, h, &[])
        .map_err(|e| VizError::Encode(e.to_string()))?;
    enc.set_repeat(gif::Repeat::Infinite)
        .map_err(|e| VizError::Encode(e.to_string()))?;
    for rgb in frames {
        let mut f = gif::Frame::from_rgb_speed(w, h, rgb, 10);
        f.delay = delay_cs;
        enc.write_frame(&f)
            .map_err(|e| VizError::Encode(e.to_string()))?;
    }
    Ok(())
}
