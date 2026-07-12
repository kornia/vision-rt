//! Frame renderers: the annotated main view and the top-down floor-plan BEV.

use vrt_track::{CameraIntrinsics, Track, TrackState};

use crate::draw::*;
use crate::trail::TrailStore;
use crate::{track_color, MaskOverlay};

/// Render the **main** view: copy the host `rgb` frame, tint each mask in its track's
/// id colour (matched by box IoU; unmatched → grey), outline the track box +
/// `<id> <depth>m` label. Returns a new `w×h` RGB buffer.
pub fn render_main(
    rgb: &[u8],
    w: usize,
    h: usize,
    masks: &[MaskOverlay],
    tracks: &[Track],
) -> Vec<u8> {
    let mut buf = rgb.to_vec();
    let confirmed: Vec<&Track> = tracks
        .iter()
        .filter(|t| t.state == TrackState::Confirmed)
        .collect();

    for m in masks {
        let color = confirmed
            .iter()
            .map(|t| (iou(&t.bbox, &m.bbox), t.id))
            .filter(|&(io, _)| io > 0.2)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, id)| track_color(id))
            .unwrap_or([110, 110, 110]);
        tint_mask(&mut buf, w, h, m.mask, m.mask_wh, m.bbox, color);
    }
    for t in &confirmed {
        let color = track_color(t.id);
        draw_box(&mut buf, w, h, t.bbox, color);
        let [x1, y1, ..] = t.bbox;
        draw_label(
            &mut buf,
            w,
            h,
            x1 as i32 + 2,
            y1 as i32 + 2,
            &format!("{} {:.1}m", t.id, t.position_3d[2]),
            color,
        );
    }
    buf
}

/// Render the **BEV** as a `w×h` top-down **floor plan**: an orthographic metre grid
/// (camera at the bottom edge, faint FoV outline) with each confirmed track a
/// footprint rectangle sized by its real width (`box_w × Z ÷ fx`), coloured by id.
/// When `trails` is given, each track's recent metric path is drawn as a polyline.
pub fn render_bev(
    tracks: &[Track],
    intr: &CameraIntrinsics,
    w: usize,
    h: usize,
    trails: Option<&TrailStore>,
) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    for p in buf.chunks_exact_mut(3) {
        p.copy_from_slice(&[14u8, 18, 26]);
    }

    let zmax = 6.0f32;
    let margin = 34.0f32;
    let ppm = (h as f32 - 2.0 * margin) / zmax; // isotropic px/m
    let (ax, az) = (w as f32 / 2.0, h as f32 - margin);
    let xspan = (ax - 6.0) / ppm;
    let map = |x: f32, z: f32| ((ax + x * ppm) as i32, (az - z * ppm) as i32);

    let (grid, axis) = ([30u8, 38, 52], [64u8, 80, 104]);
    let ztop = (az - zmax * ppm) as i32;
    for zi in 0..=zmax as i32 {
        let (_, y) = map(0.0, zi as f32);
        draw_line(
            &mut buf,
            w,
            h,
            0,
            y,
            w as i32,
            y,
            if zi == 0 { axis } else { grid },
        );
        draw_label(&mut buf, w, h, 6, y - 8, &format!("{zi}m"), [96, 116, 140]);
    }
    for xi in -(xspan as i32)..=xspan as i32 {
        let (x, _) = map(xi as f32, 0.0);
        draw_line(
            &mut buf,
            w,
            h,
            x,
            ztop,
            x,
            az as i32,
            if xi == 0 { axis } else { grid },
        );
    }

    let k = (intr.cx / intr.fx).max(0.05); // tan(half-FoV)
    for s in [-1.0f32, 1.0] {
        let (ex, ey) = map(s * k * zmax, zmax);
        draw_line(&mut buf, w, h, ax as i32, az as i32, ex, ey, [40, 58, 78]);
    }
    let (cx, cy) = (ax as i32, az as i32);
    fill_tri(
        &mut buf,
        w,
        h,
        (cx, cy - 12),
        (cx - 8, cy + 3),
        (cx + 8, cy + 3),
        [92, 212, 236],
    );

    // Motion trails (drawn under the footprints): each track's recent metric path,
    // dimmed to ~55% of its id colour so the current footprint stands out.
    if let Some(trails) = trails {
        for t in tracks.iter().filter(|t| t.state == TrackState::Confirmed) {
            let Some(path) = trails.get(t.id) else {
                continue;
            };
            let c = track_color(t.id);
            let dim = [
                (c[0] as u16 * 55 / 100) as u8,
                (c[1] as u16 * 55 / 100) as u8,
                (c[2] as u16 * 55 / 100) as u8,
            ];
            let mut prev: Option<(i32, i32)> = None;
            for p in path {
                if p[1] <= 0.1 || p[1] > zmax || p[0].abs() > xspan {
                    prev = None;
                    continue;
                }
                let q = map(p[0], p[1]);
                if let Some(pr) = prev {
                    draw_line(&mut buf, w, h, pr.0, pr.1, q.0, q.1, dim);
                }
                prev = Some(q);
            }
        }
    }

    for t in tracks {
        if t.state != TrackState::Confirmed {
            continue;
        }
        let [x_m, _, z_m] = t.metric_position(intr);
        if z_m <= 0.1 || z_m > zmax || x_m.abs() > xspan {
            continue;
        }
        let box_w = (t.bbox[2] - t.bbox[0]).max(1.0);
        let wm = (box_w * z_m / intr.fx).clamp(0.15, 3.0);
        let dm = (wm * 0.6).clamp(0.2, 1.5);
        let (fw, fd) = ((wm * ppm) as i32, (dm * ppm) as i32);
        let (mx, my) = map(x_m, z_m);
        let c = track_color(t.id);
        fill_rect_alpha(&mut buf, w, h, mx - fw / 2, my - fd / 2, fw, fd, c, 90);
        rect_outline(&mut buf, w, h, mx - fw / 2, my - fd / 2, fw, fd, c);
        draw_label(
            &mut buf,
            w,
            h,
            mx - 3,
            my - 3,
            &format!("{}", t.id),
            [240, 240, 248],
        );
    }
    buf
}

/// Stack `top` (`tw×th`) over `bot` (`bw×bh`) into one RGB buffer of
/// `max(tw,bw) × (th+bh)`, each centred horizontally on a black canvas.
pub fn stack_v(
    top: &[u8],
    tw: usize,
    th: usize,
    bot: &[u8],
    bw: usize,
    bh: usize,
) -> (Vec<u8>, usize, usize) {
    let w = tw.max(bw);
    let mut out = vec![0u8; w * (th + bh) * 3];
    let mut blit = |src: &[u8], sw: usize, sh: usize, y0: usize| {
        let xoff = (w - sw) / 2;
        for y in 0..sh {
            let d = ((y0 + y) * w + xoff) * 3;
            let s = y * sw * 3;
            out[d..d + sw * 3].copy_from_slice(&src[s..s + sw * 3]);
        }
    };
    blit(top, tw, th, 0);
    blit(bot, bw, bh, th);
    (out, w, th + bh)
}

/// Nearest-neighbour downscale an RGB buffer to `(dw, dh)`.
pub fn downscale(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh * 3];
    for y in 0..dh {
        let sy = y * sh / dh;
        for x in 0..dw {
            let sx = x * sw / dw;
            let (s, o) = ((sy * sw + sx) * 3, (y * dw + x) * 3);
            out[o..o + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
    out
}
