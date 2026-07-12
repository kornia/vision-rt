//! CPU rendering + streaming for the vision-rt tracker / perception demos.
//!
//! Takes **host** RGB buffers and [`vrt_track::Track`]s (the caller does any GPU→host
//! copies + decode) and produces annotated views:
//! - [`render_main`] — tint instance masks in their track's id colour + draw boxes.
//! - [`render_bev`] — a top-down **floor-plan** of the tracks' metric `(X, Z)`.
//! - [`StreamServer`] — stream the two views to a phone browser (H.264/WebSocket).
//! - [`JpegEncoder`] / [`encode_png`] / [`write_gif`] — encode/record.
//!
//! No model / TensorRT / sensor dependencies — a light leaf that model demos compose.
#![allow(clippy::too_many_arguments)] // pixel-drawing fns take (buf, w, h, geometry…)

use kornia_image::ImageError;

pub mod draw;
pub mod encode;
#[cfg(feature = "h264")]
pub mod h264;
pub mod render;
pub mod serve;
#[cfg(feature = "h264")]
pub mod stream;
pub mod trail;

pub use encode::{encode_png, write_gif, JpegEncoder};
#[cfg(feature = "h264")]
pub use h264::{EncodedAu, H264Encoder};
pub use render::{downscale, render_bev, render_main, stack_v};
pub use serve::{Stream, StreamServer};
#[cfg(feature = "h264")]
pub use stream::LiveStream;
pub use trail::TrailStore;

/// Errors from rendering / encoding.
#[derive(Debug, thiserror::Error)]
pub enum VizError {
    #[error(transparent)]
    Image(#[from] ImageError),
    #[error("encode: {0}")]
    Encode(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A distinct colour per track id (cycled).
pub fn track_color(id: u64) -> [u8; 3] {
    PALETTE[(id as usize) % PALETTE.len()]
}

/// The id colour palette.
pub const PALETTE: [[u8; 3]; 6] = [
    [255, 60, 60],
    [60, 220, 60],
    [60, 120, 255],
    [255, 200, 40],
    [220, 60, 220],
    [40, 220, 220],
];

/// One instance mask to overlay: a binary mask on its own grid + the source-pixel box
/// it belongs to. The renderer matches it to a track by box IoU to pick the colour.
pub struct MaskOverlay<'a> {
    /// Row-major binary mask (`1` = foreground) at `mask_wh` resolution.
    pub mask: &'a [u8],
    /// Mask grid `(width, height)`.
    pub mask_wh: (usize, usize),
    /// Bounding box `[x1, y1, x2, y2]` in source pixels.
    pub bbox: [f32; 4],
}
