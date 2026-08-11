//! On-device tests for the fit — `#[ignore]`d, run with:
//!
//! ```bash
//! cargo test -p vrt-raco-aliked --release --test gpu_fit -- --ignored
//! ```
//!
//! They need a CUDA device but NO engine: the fit is pure resize + crop, and keeping these free of
//! a `.engine` argument is what lets them run anywhere a GPU exists.
//!
//! ## What is actually being pinned
//!
//! That [`fit_to_engine`] and [`fit_to_engine_cuda`] are the same function on two devices.
//!
//! This matters more than it looks. `aliked_batch` runs the CUDA path while the benchmark harness
//! in `vrt-lightglue/examples/common` runs the host path, and both write coordinates into files
//! that are compared against each other. kornia documents its u8 resize as bit-identical across
//! residency — "the coordinate/weight tables come from the same host builders the CPU uses" — but a
//! documented guarantee that nothing exercises is a guarantee that breaks quietly on the next
//! kornia bump, and the failure would be a sub-pixel coordinate shift: far too small to notice by
//! eye and far too correlated to average out of a bundle adjust.

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use vrt_raco_aliked::{fit_to_engine, fit_to_engine_cuda};

/// A deterministic image with real high-frequency content.
///
/// A flat or smoothly-varying field would agree between two resamplers even if their coordinate
/// mapping were off by a fraction of a pixel, because there is nothing for the difference to bite
/// on. This has structure at the pixel scale specifically so a half-pixel disagreement shows up as
/// a byte difference.
fn noisy(w: usize, h: usize) -> Image<u8, 3> {
    let mut v = vec![0u8; w * h * 3];
    let mut s: u32 = 0x1234_5678;
    for p in v.iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *p = (s >> 24) as u8;
    }
    Image::new(
        ImageSize {
            width: w,
            height: h,
        },
        v,
    )
    .unwrap()
}

fn assert_same(w: usize, h: usize, max_side: usize) {
    let stream = vrt::Stream::new_standalone()
        .expect("CUDA stream")
        .cuda_stream()
        .clone();
    let src = noisy(w, h);
    let cpu = fit_to_engine(&src, max_side, InterpolationMode::Bilinear).expect("host fit");
    let gpu = fit_to_engine_cuda(
        &src.to_cuda(&stream).expect("upload"),
        max_side,
        InterpolationMode::Bilinear,
        &stream,
    )
    .expect("device fit");
    stream.synchronize().expect("sync");

    assert_eq!(
        (cpu.image.cols(), cpu.image.rows()),
        (gpu.image.cols(), gpu.image.rows()),
        "{w}x{h} @ {max_side}: geometry diverged"
    );
    assert_eq!(
        (cpu.scale_x, cpu.scale_y),
        (gpu.scale_x, gpu.scale_y),
        "{w}x{h} @ {max_side}: scales diverged"
    );

    // Device image back to the host to compare bytes; the inverse of `to_cuda`.
    let got = gpu.image.to_host_owned().expect("download");
    let (a, b) = (cpu.image.as_slice(), got.as_slice());
    let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    assert_eq!(
        diff,
        0,
        "{w}x{h} @ {max_side}: {diff} of {} bytes differ between host and device fit",
        a.len()
    );
}

/// Natural size: crop only, no resample. The default full-resolution path.
#[test]
#[ignore = "requires a CUDA device"]
fn crop_only_matches_host() {
    assert_same(1080, 1920, 1920);
}

/// The 640 fallback: a real >2x downscale, which is the path the crop-only test does NOT cover.
#[test]
#[ignore = "requires a CUDA device"]
fn downscale_matches_host() {
    assert_same(1080, 1920, 640);
    assert_same(480, 853, 640);
}

/// Landscape and an exact-2x reduction, which kornia routes to a different kernel
/// (`PyrDown2xRgb`) than the general bilinear path — so the equivalence has to hold there too.
#[test]
#[ignore = "requires a CUDA device"]
fn alternate_kernels_match_host() {
    assert_same(1920, 1080, 960);
    assert_same(1280, 704, 640);
}

/// Report host vs device fit cost. Not an assertion — a measurement, printed with `--nocapture`.
///
/// Min-of-N rather than a mean: this board routinely runs a reconstruction, several camera nodes
/// and a map service alongside the tests, so the mean measures the background load and the minimum
/// measures the code. Neither is a benchmark harness; the point is only to know whether moving the
/// fit to the GPU bought anything worth the second code path.
#[test]
#[ignore = "requires a CUDA device"]
fn report_fit_timing() {
    use std::time::Instant;
    const N: usize = 20;
    let stream = vrt::Stream::new_standalone()
        .expect("CUDA stream")
        .cuda_stream()
        .clone();

    for (w, h, cap, label) in [
        (1080, 1920, 1920, "1080x1920 natural (crop only)"),
        (1080, 1920, 640, "1080x1920 -> 640 (resize + crop)"),
    ] {
        let src = noisy(w, h);
        let dev = src.to_cuda(&stream).expect("upload");

        let mut host = f64::MAX;
        for _ in 0..N {
            let t = Instant::now();
            let s = fit_to_engine(&src, cap, InterpolationMode::Bilinear).expect("host fit");
            // The upload is part of the host path's cost: its output still has to reach the device.
            let _ = s.image.to_cuda(&stream).expect("upload");
            stream.synchronize().expect("sync");
            host = host.min(t.elapsed().as_secs_f64() * 1e3);
        }

        let mut devt = f64::MAX;
        for _ in 0..N {
            let t = Instant::now();
            let _ = fit_to_engine_cuda(&dev, cap, InterpolationMode::Bilinear, &stream)
                .expect("device fit");
            stream.synchronize().expect("sync");
            devt = devt.min(t.elapsed().as_secs_f64() * 1e3);
        }
        // The device path also needs one upload of the RAW frame, hoisted out of the retry loop by
        // the callers; counted here so the two columns describe the same total work.
        let mut up = f64::MAX;
        for _ in 0..N {
            let t = Instant::now();
            let _ = src.to_cuda(&stream).expect("upload");
            stream.synchronize().expect("sync");
            up = up.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "{label}: host {host:.2} ms  device {:.2} ms (fit {devt:.2} + upload {up:.2})",
            devt + up
        );
    }
}
