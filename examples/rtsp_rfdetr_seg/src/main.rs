//! RTSP → RF-DETR **instance segmentation**, device end-to-end, reused-buffer pipeline.
//!
//! Same async shape as `rtsp_rfdetr` — enqueue on one shared stream, one
//! `synchronize()`, then decode:
//!
//! ```text
//!   1. source   RtspSource::next_frame() → &Image<u8,3>  (device RGB, model-ready)
//!   2. model    RfDetrSeg::submit        → SegResult      (stretch + TRT + GPU decode)
//!   ── stream.synchronize()  (the single sync that completes the pipeline) ──
//!   3. output   SegResult::count()       → survivors (masks stay GPU-resident)
//! ```
//!
//! Everything stays on the GPU: `submit` enqueues the decode + mask kernels and does
//! **no host copy** (only the survivor count is async-copied). This loop reads just
//! `count()`; `detections()` / `masks_host()` would copy boxes / masks on demand.
//!
//! The engine is machine-locked (build it on-device with trtexec). This is a
//! workspace-excluded package (private `sensor-rtsp` git dep). Build directly:
//!   export CARGO_NET_GIT_FETCH_WITH_CLI=true
//!   cargo run --release --manifest-path examples/rtsp_rfdetr_seg/Cargo.toml \
//!       -- rtsp://<camera>/stream <engine> [conf] [debug.png]
//!
//! Passing a 4th arg dumps ONE annotated frame (masks tinted + boxes drawn) to that
//! PNG for eyeballing — the only path that host-copies the frame, masks, and boxes.

use std::time::Instant;

use cudarc::driver::CudaContext;
use kornia_image::{Image, ImageSize};
use kornia_io::png::write_image_png_rgb8;
use sensor_rtsp::RtspSource;
use vrt_rfdetr_seg::{Instance, RfDetrSeg};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Res<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rtsp_rfdetr_seg <rtsp://url> <engine> [conf]");
        std::process::exit(1);
    }
    let url = &args[1];
    let engine = &args[2];
    let conf: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let mut debug_png = args.get(4).cloned(); // dump one annotated frame here, then keep running

    // One shared CUDA stream: the source's un-pitch copy and RF-DETR-Seg inference
    // both enqueue on it, so a single sync completes the frame.
    let stream = CudaContext::new(0)?.default_stream();
    let mut source = RtspSource::connect_resized(url, 1280, 720, stream.clone())?;
    let (w, h) = (source.width(), source.height());
    println!("stream {w}x{h} → RF-DETR-Seg (conf ≥ {conf}), device end-to-end");

    let mut seg = RfDetrSeg::from_engine_file(engine, stream.clone(), conf)?;
    let mut out = seg.alloc_result()?; // reused segmenter output

    // Per-stage profiler: `enqueue` (CPU kernel-launch) must be ≪ the single `sync`
    // (the real GPU wall). `read` folds the CPU mask decode.
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_sync, mut a_read) = (0.0f64, 0.0, 0.0, 0.0);
    loop {
        let t0 = Instant::now();
        let Some(frame) = source.next_frame() else {
            break;
        }; // recv(camera) + enqueue copy
        let t1 = Instant::now();
        seg.submit(frame.image(), &mut out)?; // model enqueue (async, no sync)
        let t2 = Instant::now();
        stream.synchronize()?; // the one sync completes source + model
        let t3 = Instant::now();
        // GPU-resident async path: masks stay on device; only the survivor count is
        // read (pinned). `out.detections()` / `out.masks_host()` copy on demand.
        let inst_n = out.count();
        let t4 = Instant::now();

        // Debug: after warm-up, host-copy this frame + boxes + masks and draw one
        // annotated PNG (takes the on-demand copies, so it's off the timed path).
        if let Some(path) = debug_png.take_if(|_| n == 60) {
            let (dw, dh) = (w as usize, h as usize);
            let host = frame.image().to_host(&stream)?; // device Rgb8 → host
            let mut buf = host.as_slice().to_vec();
            let instances = out.instances()?; // boxes + masks to host
            for (i, inst) in instances.iter().enumerate() {
                draw_instance(&mut buf, dw, dh, inst, PALETTE[i % PALETTE.len()]);
            }
            let img = Image::<u8, 3>::new(
                ImageSize {
                    width: dw,
                    height: dh,
                },
                buf,
            )?;
            write_image_png_rgb8(&path, &img)?;
            println!("     saved debug overlay → {path} ({} instances)", instances.len());
        }

        n += 1;
        a_src += ms(t1 - t0);
        a_enq += ms(t2 - t1);
        a_sync += ms(t3 - t2);
        a_read += ms(t4 - t3);
        if n.is_multiple_of(100) {
            let k = 100.0;
            println!(
                "── {n} frames | {:.1} fps | source(recv+enqueue) {:.2} ms | \
                 enqueue(submit) {:.3} ms | sync(GPU) {:.2} ms | decode {:.3} ms | {inst_n} inst",
                n as f64 / t_start.elapsed().as_secs_f64(),
                a_src / k,
                a_enq / k,
                a_sync / k,
                a_read / k,
            );
            // On-demand host copy of just the boxes (no masks) to name what's in view.
            let mut counts: std::collections::BTreeMap<&str, (usize, f32)> = Default::default();
            for d in out.detections()? {
                let e = counts.entry(coco_name(d.class_id)).or_insert((0, 0.0));
                e.0 += 1;
                e.1 = e.1.max(d.score);
            }
            let mut summary: Vec<_> = counts.into_iter().collect();
            summary.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            let shown: Vec<String> = summary
                .iter()
                .map(|(name, (c, s))| format!("{c}× {name} ({s:.2})"))
                .collect();
            println!("     detected: {}", shown.join(", "));
            a_src = 0.0;
            a_enq = 0.0;
            a_sync = 0.0;
            a_read = 0.0;
        }
    }
    Ok(())
}

const PALETTE: [[u8; 3]; 6] = [
    [255, 60, 60],
    [60, 220, 60],
    [60, 120, 255],
    [255, 200, 40],
    [220, 60, 220],
    [40, 220, 220],
];

/// Draw one instance onto a tight RGB frame buffer: tint its mask (nearest-upsampled
/// from the mask grid, which spans the whole stretched frame) + outline its box.
fn draw_instance(buf: &mut [u8], w: usize, h: usize, inst: &Instance, color: [u8; 3]) {
    let (mw, mh) = inst.mask_size;
    for y in 0..h {
        let my = (y * mh) / h;
        for x in 0..w {
            let mx = (x * mw) / w;
            if inst.mask[my * mw + mx] == 1 {
                let o = (y * w + x) * 3;
                for k in 0..3 {
                    buf[o + k] = ((buf[o + k] as u16 + color[k] as u16) / 2) as u8;
                }
            }
        }
    }
    let [x1, y1, x2, y2] = inst.bbox;
    for (a, b) in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ] {
        draw_line(buf, w, h, a.0 as i32, a.1 as i32, b.0 as i32, b.1 as i32, color);
    }
}

/// Bresenham line, clipped to the frame.
#[allow(clippy::too_many_arguments)]
fn draw_line(buf: &mut [u8], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
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

/// COCO 91-class category-id → name (RF-DETR emits these ids directly; `N/A` are
/// the gaps in the 91-index space, 0 = background is never emitted).
fn coco_name(id: u32) -> &'static str {
    COCO91.get(id as usize).copied().unwrap_or("?")
}

const COCO91: [&str; 91] = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];
