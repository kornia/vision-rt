//! Low-level RGB-buffer drawing primitives (all clipped to the frame) + a minimal
//! 5×7 bitmap font (digits, `.`, `m` — the tracker labels are numeric).

/// Bresenham line.
#[allow(clippy::too_many_arguments)]
pub fn draw_line(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
            let o = (y as usize * w + x as usize) * 3;
            buf[o..o + 3].copy_from_slice(&color);
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

/// Outline a box `[x1,y1,x2,y2]`.
pub fn draw_box(buf: &mut [u8], w: usize, h: usize, b: [f32; 4], color: [u8; 3]) {
    let [x1, y1, x2, y2] = b;
    for (a, c) in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ] {
        draw_line(
            buf, w, h, a.0 as i32, a.1 as i32, c.0 as i32, c.1 as i32, color,
        );
    }
}

/// Rectangle outline at `(x, y)` size `rw×rh`.
pub fn rect_outline(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    color: [u8; 3],
) {
    draw_line(buf, w, h, x, y, x + rw, y, color);
    draw_line(buf, w, h, x, y + rh, x + rw, y + rh, color);
    draw_line(buf, w, h, x, y, x, y + rh, color);
    draw_line(buf, w, h, x + rw, y, x + rw, y + rh, color);
}

/// Alpha-blend a filled rectangle (`a` in 0..=255).
pub fn fill_rect_alpha(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    color: [u8; 3],
    a: u16,
) {
    for yy in y.max(0)..(y + rh).min(h as i32) {
        for xx in x.max(0)..(x + rw).min(w as i32) {
            let o = (yy as usize * w + xx as usize) * 3;
            for k in 0..3 {
                buf[o + k] = ((buf[o + k] as u16 * (255 - a) + color[k] as u16 * a) / 255) as u8;
            }
        }
    }
}

/// Fill a triangle (barycentric sign test), clipped to the frame.
pub fn fill_tri(
    buf: &mut [u8],
    w: usize,
    h: usize,
    a: (i32, i32),
    b: (i32, i32),
    c: (i32, i32),
    color: [u8; 3],
) {
    let e = |p: (i32, i32), q: (i32, i32), r: (i32, i32)| {
        (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)
    };
    let (minx, maxx) = (
        a.0.min(b.0).min(c.0).max(0),
        a.0.max(b.0).max(c.0).min(w as i32 - 1),
    );
    let (miny, maxy) = (
        a.1.min(b.1).min(c.1).max(0),
        a.1.max(b.1).max(c.1).min(h as i32 - 1),
    );
    for y in miny..=maxy {
        for x in minx..=maxx {
            let p = (x, y);
            let (w0, w1, w2) = (e(b, c, p), e(c, a, p), e(a, b, p));
            if (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0) {
                let o = (y as usize * w + x as usize) * 3;
                buf[o..o + 3].copy_from_slice(&color);
            }
        }
    }
}

/// Alpha-tint a binary mask (`mask_wh` grid, nearest-upsampled) over the frame, but
/// only within `bbox` (its foreground lives there — cheap on the hot path).
pub fn tint_mask(
    buf: &mut [u8],
    w: usize,
    h: usize,
    mask: &[u8],
    mask_wh: (usize, usize),
    bbox: [f32; 4],
    color: [u8; 3],
) {
    let (mw, mh) = mask_wh;
    let [bx1, by1, bx2, by2] = bbox;
    let (x0, x1) = (
        (bx1.max(0.0) as usize).min(w),
        (bx2.ceil().max(0.0) as usize).min(w),
    );
    let (y0, y1) = (
        (by1.max(0.0) as usize).min(h),
        (by2.ceil().max(0.0) as usize).min(h),
    );
    for y in y0..y1 {
        let my = (y * mh) / h;
        for x in x0..x1 {
            let mx = (x * mw) / w;
            if mask[my * mw + mx] == 1 {
                let o = (y * w + x) * 3;
                for k in 0..3 {
                    buf[o + k] = ((buf[o + k] as u16 + color[k] as u16) / 2) as u8;
                }
            }
        }
    }
}

/// Draw a short label at `(x, y)` on a dark backdrop; 5×7 font, 2× scaled.
pub fn draw_label(buf: &mut [u8], w: usize, h: usize, x: i32, y: i32, text: &str, color: [u8; 3]) {
    const SCALE: i32 = 2;
    const GLYPH_W: i32 = 5;
    const GLYPH_H: i32 = 7;
    let advance = (GLYPH_W + 1) * SCALE;
    let bg_w = advance * text.chars().count() as i32;
    let bg_h = GLYPH_H * SCALE;
    for yy in (y - 1)..(y + bg_h + 1) {
        for xx in (x - 1)..(x + bg_w + 1) {
            if xx >= 0 && xx < w as i32 && yy >= 0 && yy < h as i32 {
                let o = (yy as usize * w + xx as usize) * 3;
                for k in 0..3 {
                    buf[o + k] /= 4;
                }
            }
        }
    }
    let mut cx = x;
    for ch in text.chars() {
        if let Some(glyph) = font5x7(ch) {
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..GLYPH_W {
                    if bits & (1 << (GLYPH_W - 1 - col)) != 0 {
                        for sy in 0..SCALE {
                            for sx in 0..SCALE {
                                let px = cx + col * SCALE + sx;
                                let py = y + row as i32 * SCALE + sy;
                                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                                    let o = (py as usize * w + px as usize) * 3;
                                    buf[o..o + 3].copy_from_slice(&color);
                                }
                            }
                        }
                    }
                }
            }
        }
        cx += advance;
    }
}

/// Minimal 5×7 bitmap font: digits `0`–`9`, `.`, `m` (space/other render blank).
fn font5x7(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15],
        _ => return None,
    })
}

/// IoU of two `[x1,y1,x2,y2]` boxes.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (ix1, iy1) = (a[0].max(b[0]), a[1].max(b[1]));
    let (ix2, iy2) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let ua = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let ub = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let uni = ua + ub - inter;
    if uni <= 0.0 {
        0.0
    } else {
        inter / uni
    }
}
