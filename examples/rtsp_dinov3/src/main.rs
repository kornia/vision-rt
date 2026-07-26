//! RTSP → DINOv3 global descriptor → keyframe-bank retrieval, device end-to-end.
//!
//! The loop flows like a TensorRT pipeline — **enqueue async on one shared stream, one
//! `synchronize()`, then read**:
//!
//! ```text
//!   1. source   RtspSource::next_frame() → &Image<u8,3>   (device RGB, model-ready)
//!   2. model    DinoV3::submit           → DinoV3Result   (stretch + TRT + L2-norm)
//!   3. match    DescriptorBank::match_into → scores[K]    (one dot per stored keyframe)
//!   ── stream.synchronize()  (the single sync that completes the pipeline) ──
//!   4. output   argmax over a K-float readout; enroll a new keyframe if nothing matched
//! ```
//!
//! **No host round-trip on the inference path.** The RTSP source (`sensor-rtsp`,
//! kornia/sensor-rt) hardware-decodes over NVMM, imports the dmabuf straight into the
//! CUDA address space (`cudaImportExternalMemory` — a true zero-copy), and its un-pitch
//! pass drops alpha (pitched RGBA → tight RGB) so the frame is model-ready. From there
//! it is device buffers all the way to the K-float score readout, which is 2 KB at the
//! default capacity. Stages 1–3 only *enqueue*.
//!
//! The live view (`--port`) is the one place a full frame comes back to the host, which
//! is why it is **opt-in**: without it the whole pipeline stays on the GPU. Keyframe
//! thumbnails cost one D2H each, on enrollment only — not per frame.
//!
//! What it demonstrates: walk the camera away from a scene and back. It should enroll
//! keyframes on the way out and **re-match the original** on return. That, not the frame
//! rate, is the proof the descriptor works.
//!
//! This is a workspace-excluded package (private `sensor-rtsp` git dep). Build it
//! directly, on-device:
//!   export CARGO_NET_GIT_FETCH_WITH_CLI=true CARGO_BUILD_JOBS=2
//!   cargo run --release --manifest-path examples/rtsp_dinov3/Cargo.toml \
//!       -- <dinov3.engine> rtsp://<camera>/stream [--tau 0.75] [--port 8080]

use std::time::Instant;

use cudarc::driver::CudaContext;
use sensor_rtsp::RtspSource;
use vrt_dinov3::{DescriptorBank, DinoV3};
use vrt_viz::{downscale, render_depth, stack_v, LiveStream};

// Send + Sync to match vrt's BoxError — `?` coerces cleanly.
type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Stored keyframe thumbnail size. 160x90 x 512 slots is ~22 MB of host RAM — cheap on
/// a 7.4 GB box, and enough to recognize a scene by eye.
const TH_W: usize = 160;
const TH_H: usize = 90;
/// Rolling best-match score history rendered as the timeline strip.
const HIST: usize = 256;
const STREAM_FPS: i32 = 30;

struct Cfg {
    engine: String,
    url: String,
    tau: f32,
    capacity: usize,
    min_gap: u64,
    port: Option<u16>,
}

fn parse_args() -> Res<Cfg> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "Usage: rtsp_dinov3 <dinov3.engine> <rtsp://url> \
             [--tau F] [--capacity N] [--min-gap N] [--port N]\n\
             \n  --tau       enroll when the best match falls below this cosine (default 0.75)\
             \n  --capacity  max keyframes in the bank (default 512)\
             \n  --min-gap   min frames between enrollments (default 15)\
             \n  --port      serve the live H.264/WebSocket view (default: off, stays fully on-GPU)"
        );
        std::process::exit(1);
    }
    let mut c = Cfg {
        engine: a[1].clone(),
        url: a[2].clone(),
        tau: 0.75,
        capacity: 512,
        min_gap: 15,
        port: None,
    };
    let mut i = 3;
    while i < a.len() {
        let val = || -> Res<String> {
            a.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", a[i]).into())
        };
        match a[i].as_str() {
            "--tau" => c.tau = val()?.parse()?,
            "--capacity" => c.capacity = val()?.parse()?,
            "--min-gap" => c.min_gap = val()?.parse()?,
            "--port" => c.port = Some(val()?.parse()?),
            other => return Err(format!("unknown flag {other}").into()),
        }
        i += 2;
    }
    Ok(c)
}

/// Decide whether the current frame becomes a new keyframe.
///
/// **TODO(you): this is the policy that decides whether the demo works**, and it is a
/// real trade-off rather than a detail. The default below is *threshold + minimum frame
/// gap*. Alternatives worth trying:
///
/// Measured separation on this engine (`examples/dinov3_match`, real images) says the
/// threshold is not delicate: two views of the same place score **~0.95–0.96**, unrelated
/// scenes **−0.001 … 0.11**. The default `tau = 0.75` sits in a wide empty band, so
/// anything around 0.5–0.9 behaves the same. The *gap* policy is the interesting knob.
///
/// - **fixed threshold alone** — simplest, but `tau` is scene-dependent. Too low and the
///   bank never grows, so nothing is ever recognized; too high and it enrolls on every
///   frame until capacity, which also recognizes nothing.
/// - **threshold + min gap** (the default) — the gap stops a burst of near-identical
///   keyframes while the camera is moving. Two constants instead of one.
/// - **adaptive** — track the running distribution of best-match scores and enroll on an
///   outlier. No magic constant and it travels across scenes, but it needs state and a
///   warm-up period.
///
/// Also open: what to do at capacity. This currently just stops enrolling (see the call
/// site); evicting the least-recently-matched, or reservoir-sampling, would keep the bank
/// representative over a long run.
fn should_enroll(bank_len: usize, best: f32, frames_since: u64, tau: f32, min_gap: u64) -> bool {
    // The first frame always becomes keyframe 0 — with an empty bank there is nothing to
    // match against, so `best` is meaningless.
    if bank_len == 0 {
        return true;
    }
    best < tau && frames_since >= min_gap
}

/// Place `left` (lw x h) and `right` (rw x h) side by side into one `(lw+rw) x h` RGB
/// buffer. `vrt-viz` ships a vertical stack but not a horizontal one.
fn stack_h(left: &[u8], lw: usize, right: &[u8], rw: usize, h: usize) -> Vec<u8> {
    let w = lw + rw;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        let d = y * w * 3;
        out[d..d + lw * 3].copy_from_slice(&left[y * lw * 3..(y + 1) * lw * 3]);
        out[d + lw * 3..d + w * 3].copy_from_slice(&right[y * rw * 3..(y + 1) * rw * 3]);
    }
    out
}

fn main() -> Res<()> {
    env_logger::init();
    let cfg = parse_args()?;

    // One shared CUDA stream: the source's un-pitch copy, DINOv3 inference and the bank
    // match all enqueue on it, so a single sync completes the frame.
    let stream = CudaContext::new(0)?.default_stream();
    let mut source = RtspSource::connect_resized(&cfg.url, 1280, 720, stream.clone())?;
    let (w, h) = (source.width() as usize, source.height() as usize);

    let mut dino = DinoV3::from_engine_file(&cfg.engine, stream.clone())?;
    let dim = dino.dim();
    let (gw, gh) = dino.grid();
    let mut bank = DescriptorBank::new(cfg.capacity, dim, stream.clone())?;

    // Allocated ONCE, reused every frame — the only per-frame allocation is none.
    let mut r = dino.alloc_result()?;
    let mut scores = stream.alloc_zeros::<f32>(cfg.capacity)?;

    println!(
        "stream {w}x{h} → DINOv3 {}x{} input ({gw}x{gh} patches, dim {dim}) | \
         bank capacity {} | tau {} | min-gap {}",
        gw * 16,
        gh * 16,
        cfg.capacity,
        cfg.tau,
        cfg.min_gap
    );

    // Live view is opt-in: it is the only full-frame D2H in the program, so leaving it
    // off keeps the pipeline entirely on the GPU.
    let strip_h = TH_H;
    let live = match cfg.port {
        Some(port) => {
            println!("     live view: http://<this-host>:{port}  (H.264/WebSocket → WebCodecs)");
            // Only the `main` view carries content — the composed frame+strip image. The
            // other two encoder slots get 16x16 blanks; the transport has three fixed
            // channels and we deliberately do not repurpose the BEV/depth ones, which
            // would just mislabel the view in the browser.
            Some(LiveStream::spawn(
                port,
                (w, h + strip_h),
                (16, 16),
                (16, 16),
                4000,
                64,
                64,
                STREAM_FPS,
            )?)
        }
        None => {
            println!("     live view: off (pass --port N to enable; adds a full-frame D2H)");
            None
        }
    };
    let blank = vec![0u8; 16 * 16 * 3];

    let mut thumbs: Vec<Vec<u8>> = Vec::new(); // host RGB, TH_W x TH_H, one per keyframe
    let mut hist: Vec<f32> = Vec::with_capacity(HIST);
    let mut frames_since_enroll = u64::MAX; // so the min-gap never blocks keyframe 0
    let mut capacity_warned = false;

    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_mat, mut a_sync, mut a_read) = (0.0f64, 0.0, 0.0, 0.0, 0.0);

    loop {
        let t0 = Instant::now();
        // `frame` must stay alive until after the sync — dropping it returns its buffer
        // to the source's ring, where the next frame's pack kernel would overwrite the
        // pixels the GPU is still reading.
        let Some(frame) = source.next_frame() else {
            break;
        };
        let t1 = Instant::now();
        dino.submit(frame.image(), &mut r)?; // enqueue (async, no sync)
        let t2 = Instant::now();
        bank.match_into(r.descriptor_slice(), &mut scores)?; // enqueue AFTER submit
        let t3 = Instant::now();
        stream.synchronize()?; // the one sync completes source + model + match
        let t4 = Instant::now();

        // Everything below is post-sync. Reading `len` floats is 2 KB at capacity 512 —
        // cheaper and simpler than a GPU argmax reduction at this size.
        let (mut best_id, mut best) = (0usize, 0.0f32);
        if !bank.is_empty() {
            let s = stream.clone_dtoh(&scores.slice(0..bank.len()))?;
            for (i, &v) in s.iter().enumerate() {
                if v > best {
                    best = v;
                    best_id = i;
                }
            }
        }
        let t5 = Instant::now();

        // The program's ONLY full-frame D2H, taken once and shared by the live view and
        // any thumbnail captured this frame. `None` when the view is off, which is what
        // keeps the default run entirely on the GPU — thumbnails exist purely to be
        // looked at, so capturing them with no viewer would be pure waste.
        let host_frame: Option<Vec<u8>> = match live {
            Some(_) => Some(frame.image().to_host(&stream)?.0.into_vec()),
            None => None,
        };

        frames_since_enroll = frames_since_enroll.saturating_add(1);
        if should_enroll(bank.len(), best, frames_since_enroll, cfg.tau, cfg.min_gap) {
            if bank.len() == bank.capacity() {
                if !capacity_warned {
                    println!("── bank full at {} keyframes; stopped enrolling", bank.capacity());
                    capacity_warned = true;
                }
            } else {
                let id = bank.enroll(r.descriptor_slice())?;
                // Pushed exactly when `host_frame` is Some, i.e. all-or-nothing across the
                // run — so `thumbs[i]` stays aligned with bank id `i` whenever it is used.
                if let Some(host) = &host_frame {
                    thumbs.push(downscale(host, w, h, TH_W, TH_H));
                }
                frames_since_enroll = 0;
                println!(
                    "── frame {n}: enrolled keyframe {id} (best match was {best:.3} < tau {})",
                    cfg.tau
                );
            }
        }

        if hist.len() == HIST {
            hist.remove(0);
        }
        hist.push(best);

        if let (Some(live), Some(host)) = (&live, &host_frame) {
            // Strip: best-match thumbnail on the left, score timeline on the right.
            // `render_depth` is a scalar-field → colormap → upsample, which is exactly a
            // timeline; it auto-scales to the window's own min/max, so the strip shows
            // relative variation rather than absolute cosine (those go to stdout).
            let thumb = thumbs
                .get(best_id)
                .cloned()
                .unwrap_or_else(|| vec![0u8; TH_W * TH_H * 3]);
            let timeline = render_depth(&hist, hist.len(), 1, w - TH_W, strip_h);
            let strip = stack_h(&thumb, TH_W, &timeline, w - TH_W, strip_h);
            let (composed, _cw, _ch) = stack_v(host, w, h, &strip, w, strip_h);
            live.submit(composed, blank.clone(), blank.clone());
        }

        n += 1;
        a_src += ms(t1 - t0);
        a_enq += ms(t2 - t1);
        a_mat += ms(t3 - t2);
        a_sync += ms(t4 - t3);
        a_read += ms(t5 - t4);
        if n.is_multiple_of(100) {
            let k = 100.0;
            // `enqueue`+`match` ≪ `sync` means the launches are genuinely async and the
            // GPU is the bottleneck. If they approach `sync`, something hidden-synced.
            println!(
                "── {n} frames | {:.1} fps | source {:.2} ms | enqueue {:.3} ms | \
                 match {:.3} ms | sync(GPU) {:.2} ms | read {:.3} ms | bank {} | best {best:.3}",
                n as f64 / t_start.elapsed().as_secs_f64(),
                a_src / k,
                a_enq / k,
                a_mat / k,
                a_sync / k,
                a_read / k,
                bank.len(),
            );
            a_src = 0.0;
            a_enq = 0.0;
            a_mat = 0.0;
            a_sync = 0.0;
            a_read = 0.0;
        }
    }
    Ok(())
}
