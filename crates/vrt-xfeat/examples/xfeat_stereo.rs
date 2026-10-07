//! Stereo XFeat: one batch-2 [`XFeat::submit_pair`] vs two [`XFeat::submit`]s.
//!
//! Runs three variants interleaved per iteration (so background load hits them
//! equally), each ending in one stream sync = full per-pair latency:
//!   `b1-engine 2x submit`  — the default engine (batch 1, opt 640×640)
//!   `b2-engine 2x submit`  — the stereo engine for this size, frames one at a time
//!   `b2-engine pair`       — the stereo engine, one batch-2 run
//!
//! Usage:
//!   cargo run --release -p vrt-xfeat --example xfeat_stereo -- \
//!       <xfeat_backbone.onnx> <left> [right|-] [WxH] [iters] [top_k]
//! `right` = `-` (default) uses the horizontally flipped left frame.

use std::time::Instant;

use kornia_image::{Image, ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime};
use vrt_xfeat::{stereo_shapes, XFeat, XFeatParams, XFeatResult, ENGINE_SHAPES};

const THRESHOLD: f32 = 0.05;
const WARMUP: usize = 30;

fn profile(shapes: [[i64; 4]; 3]) -> vrt_hub::EngineProfile {
    let [min, opt, max] = shapes;
    vrt_hub::EngineProfile {
        inputs: vec![("image".into(), min.to_vec(), opt.to_vec(), max.to_vec())],
        fp16: true,
        bf16: false,
        workspace_mb: 2048,
    }
}

fn free_mib() -> f64 {
    cudarc::driver::result::mem_get_info()
        .map(|(free, _)| free as f64 / (1 << 20) as f64)
        .unwrap_or(f64::NAN)
}

fn engine(onnx: &str, shapes: [[i64; 4]; 3]) -> Result<std::sync::Arc<Engine>, vrt::BoxError> {
    let path = vrt_hub::EngineCache::default().resolve("xfeat-backbone", onnx, &profile(shapes))?;
    let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
    Ok(Engine::from_file(runtime, &path)?)
}

fn sized(src: &Image<u8, 3>, wh: Option<(usize, usize)>) -> Result<Image<u8, 3>, vrt::BoxError> {
    let Some((width, height)) = wh else {
        return Ok(src.clone());
    };
    let mut dst = Image::<u8, 3>::from_size_val(ImageSize { width, height }, 0)?;
    kornia_imgproc::resize::resize_fast_rgb(
        src,
        &mut dst,
        kornia_imgproc::interpolation::InterpolationMode::Bilinear,
    )?;
    Ok(dst)
}

struct Stats(Vec<f64>);
impl Stats {
    fn line(&mut self, name: &str) -> f64 {
        self.0.sort_by(|a, b| a.total_cmp(b));
        let p = |q: f64| self.0[((self.0.len() as f64 * q) as usize).min(self.0.len() - 1)];
        let (p50, p99) = (p(0.50), p(0.99));
        println!(
            "  {name:<22} p50 {p50:6.2}  p99 {p99:6.2}  max {:6.2} ms",
            self.0[self.0.len() - 1]
        );
        p50
    }
}

fn main() -> Result<(), vrt::BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: xfeat_stereo <onnx> <left> [right|-] [WxH] [iters] [top_k]");
        std::process::exit(1);
    }
    let onnx = &args[1];
    let wh = args.get(4).and_then(|s| {
        let (w, h) = s.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    });
    let iters: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(300);
    let top_k: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(2048);

    let left = sized(&read_image_any_rgb8(&args[2])?.into_inner(), wh)?;
    let right = match args.get(3).map(String::as_str) {
        Some(p) if p != "-" => sized(&read_image_any_rgb8(p)?.into_inner(), wh)?,
        _ => {
            let mut r = left.clone();
            kornia_imgproc::flip::horizontal_flip(&left, &mut r)?;
            r
        }
    };

    let stream = vrt::Stream::new_standalone()?.cuda_stream().clone();
    let (l, r) = (left.to_cuda(&stream)?, right.to_cuda(&stream)?);

    // One XFeat per variant: alternating batch 1 and 2 on a single instance would
    // reshape its context (and drain the stream) on every call.
    let params = || XFeatParams::new(top_k, THRESHOLD);
    let f0 = free_mib();
    let mut b1 = XFeat::new(engine(onnx, ENGINE_SHAPES)?, stream.clone(), params())?;
    let mut o = [b1.alloc_result()?, b1.alloc_result()?];
    let two = |x: &mut XFeat, o: &mut [XFeatResult; 2]| -> Result<(), vrt::BoxError> {
        let [ol, or] = o;
        x.submit(&l, ol)?;
        x.submit(&r, or)?;
        stream.synchronize()?;
        Ok(())
    };
    two(&mut b1, &mut o)?;
    let f1 = free_mib();
    let e2 = engine(onnx, stereo_shapes(left.width(), left.height()))?;
    let mut b2_single = XFeat::new(e2.clone(), stream.clone(), params())?;
    let mut b2_pair = XFeat::new(e2, stream.clone(), params())?;
    let pair = |x: &mut XFeat, o: &mut [XFeatResult; 2]| -> Result<(), vrt::BoxError> {
        let [ol, or] = o;
        x.submit_pair(&l, &r, ol, or)?;
        stream.synchronize()?;
        Ok(())
    };
    two(&mut b2_single, &mut o)?;
    pair(&mut b2_pair, &mut o)?;
    let f2 = free_mib();

    let (w, h) = (left.width(), left.height());
    println!(
        "XFeat stereo {w}x{h} -> model {}x{}, top_k {top_k}, {iters} iters (+{WARMUP} warmup)",
        (w / 32) * 32,
        (h / 32) * 32
    );
    println!(
        "  free-mem delta: b1 engine {:.0} MiB, b2 engine {:.0} MiB (unified memory: noisy)",
        f0 - f1,
        f1 - f2
    );
    println!("  counts: left {} right {}", o[0].count(), o[1].count());

    let mut t = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..WARMUP + iters {
        let time = |f: &mut dyn FnMut() -> Result<(), vrt::BoxError>| {
            let s = Instant::now();
            f().map(|_| s.elapsed().as_secs_f64() * 1e3)
        };
        let a = time(&mut || two(&mut b1, &mut o))?;
        let b = time(&mut || two(&mut b2_single, &mut o))?;
        let c = time(&mut || pair(&mut b2_pair, &mut o))?;
        if i >= WARMUP {
            t[0].push(a);
            t[1].push(b);
            t[2].push(c);
        }
    }
    let [a, b, c] = t;
    let base = Stats(a).line("b1-engine 2x submit");
    Stats(b).line("b2-engine 2x submit");
    let p = Stats(c).line("b2-engine pair");
    println!("  pair vs b1 baseline: {:.2}x", base / p);
    Ok(())
}
