//! RTSP → RF-DETR object detection, device end-to-end, as a reused-buffer pipeline.
//!
//! The loop flows like a TensorRT pipeline — **enqueue async on one shared stream,
//! one `synchronize()`, then read**:
//!
//! ```text
//!   1. source   RtspSource::next_frame() → &Image<u8,3>  (device RGB, model-ready)
//!   2. model    RfDetr::submit           → DetectResult   (stretch + TRT + decode)
//!   ── stream.synchronize()  (the single sync that completes the pipeline) ──
//!   3. output   DetectResult::detections() → Vec<Detection>
//! ```
//!
//! The RTSP source (`sensor-rtsp`, kornia/sensor-rt) hardware-decodes over NVMM and
//! its un-pitch pass drops alpha (pitched-RGBA → tight RGB), so the frame is
//! model-ready — it goes straight to the detector, no preprocess copy. Every buffer
//! is reused; the frame never leaves the GPU; stages 1–2 only *enqueue*.
//!
//! The RF-DETR ONNX + prebuilt fp16 engine are pulled from Hugging Face
//! (`kornia/rfdetr`, sha256-pinned) on first run and cached on-device.
//!
//! This is a workspace-excluded package (private `sensor-rtsp` git dep). Build it
//! directly, on-device:
//!   export CARGO_NET_GIT_FETCH_WITH_CLI=true
//!   cargo run --release --manifest-path examples/rtsp_rfdetr/Cargo.toml \
//!       -- rtsp://<camera>/stream [conf]

use std::time::Instant;

use cudarc::driver::CudaContext;
use sensor_rtsp::RtspSource;
use vrt_rfdetr::RfDetr;

// Send + Sync to match vrt's BoxError (RfDetr::from_hub) — `?` coerces cleanly.
type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Res<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rtsp_rfdetr <rtsp://url> [conf]");
        std::process::exit(1);
    }
    let url = &args[1];
    let conf: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);

    // One shared CUDA stream: the source's un-pitch copy and RF-DETR inference
    // both enqueue on it, so a single sync completes the frame.
    let stream = CudaContext::new(0)?.default_stream();
    let mut source = RtspSource::connect_resized(url, 1280, 720, stream.clone())?;
    let (w, h) = (source.width(), source.height());
    println!("stream {w}x{h} → RF-DETR (conf ≥ {conf}), device end-to-end");

    // Pull weights (HF, sha256-pinned) + prebuilt engine, cached on-device.
    let mut detr = RfDetr::from_hub(stream.clone(), conf)?;
    let mut out = detr.alloc_result()?; // reused detector output

    // Per-stage profiler — proves the pipeline is async: `enqueue` (a pure CPU
    // kernel-launch) must be ≪ the single `sync` (the real GPU wall). If `sync`
    // is under the frame interval, the GPU has headroom and nothing hidden-syncs.
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_sync, mut a_read) = (0.0f64, 0.0, 0.0, 0.0);
    loop {
        let t0 = Instant::now();
        let Some(frame) = source.next_frame() else {
            break;
        }; // recv(camera) + enqueue copy
        let t1 = Instant::now();
        detr.submit(frame.image(), &mut out)?; // model enqueue (async, no sync)
        let t2 = Instant::now();
        stream.synchronize()?; // the one sync completes source + model
        let t3 = Instant::now();
        let dets_n = out.detections()?.len(); // small host readout
        let t4 = Instant::now();

        n += 1;
        a_src += ms(t1 - t0);
        a_enq += ms(t2 - t1);
        a_sync += ms(t3 - t2);
        a_read += ms(t4 - t3);
        if n.is_multiple_of(100) {
            let k = 100.0;
            println!(
                "── {n} frames | {:.1} fps | source(recv+enqueue) {:.2} ms | \
                 enqueue(submit) {:.3} ms | sync(GPU) {:.2} ms | read {:.3} ms | {dets_n} dets",
                n as f64 / t_start.elapsed().as_secs_f64(),
                a_src / k,
                a_enq / k,
                a_sync / k,
                a_read / k,
            );
            a_src = 0.0;
            a_enq = 0.0;
            a_sync = 0.0;
            a_read = 0.0;
        }
    }
    Ok(())
}
