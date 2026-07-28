//! On-device tests — `#[ignore]`d, run with:
//!
//! ```bash
//! DINOV3_ENGINE=models/engines/dinov3-vits16-336-trt10.3.0.30-sm87.engine \
//! DINOV3_REF_DIR=models/onnx/dinov3-ref \
//!     cargo test -p vrt-dinov3 --release -- --ignored
//! ```
//!
//! `DINOV3_REF_DIR` is what `scripts/export_dinov3.py --dump-ref` writes. The bank
//! tests need only `DINOV3_ENGINE`; the pure-CPU-math ones need nothing.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageSize};
use vrt_dinov3::{DescriptorBank, DinoV3};

fn stream() -> Arc<CudaStream> {
    vrt::Stream::new_standalone()
        .expect("CUDA stream")
        .cuda_stream()
        .clone()
}

/// Resolve a relative path against the **workspace** root, not the crate root.
///
/// Integration tests run with CWD = the crate directory, so a bare
/// `models/engines/foo.engine` — the path you get from `build_engine.sh`, and what the
/// README tells you to paste — would silently resolve under `crates/vrt-dinov3/`.
/// Absolute paths pass through untouched.
fn from_workspace(p: String) -> std::path::PathBuf {
    let path = std::path::PathBuf::from(&p);
    if path.is_absolute() {
        return path;
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

fn engine_path() -> std::path::PathBuf {
    from_workspace(
        std::env::var("DINOV3_ENGINE")
            .expect("set DINOV3_ENGINE to a built .engine (see scripts/build_engine.sh)"),
    )
}

fn ref_dir() -> std::path::PathBuf {
    from_workspace(
        std::env::var("DINOV3_REF_DIR")
            .expect("set DINOV3_REF_DIR (scripts/export_dinov3.py --dump-ref)"),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "descriptor length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Deterministic **structured** RGB image: a smooth low-frequency plaid plus a bright
/// disc, both seeded. Reproducible without shipping a fixture.
///
/// Structure is the point. White noise does *not* work here — DINOv3 collapses it to
/// nearly a single descriptor (measured on this engine: 0.9934 same-scene vs 0.9923
/// different, a 0.001 margin), which makes a discrimination test pass while proving
/// essentially nothing. For reference, on real photos the same engine gives ~0.96 for
/// two views of one place vs ~0.01–0.11 for unrelated scenes.
fn synth_image(w: usize, h: usize, seed: u64) -> Image<u8, 3> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32) // [0, 1)
    };
    const TAU: f32 = std::f32::consts::TAU;
    let fx = 2.0 + 6.0 * rnd(); // 2..8 cycles across the frame — low frequency, so a
    let fy = 2.0 + 6.0 * rnd(); // small shift stays recognizably the same "scene"
    let ph = [rnd() * TAU, rnd() * TAU, rnd() * TAU];
    let cx = 0.2 + 0.6 * rnd();
    let cy = 0.2 + 0.6 * rnd();
    let rad = 0.12 + 0.1 * rnd();

    let mut buf = vec![0u8; w * h * 3];
    for y in 0..h {
        let v = y as f32 / h as f32;
        for x in 0..w {
            let u = x as f32 / w as f32;
            let inside = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt() < rad;
            let disc = if inside { 60.0 } else { 0.0 };
            for c in 0..3 {
                let plaid = (fx * u * TAU + ph[c]).sin() * (fy * v * TAU + ph[c]).sin();
                buf[(y * w + x) * 3 + c] = (128.0 + 90.0 * plaid + disc).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Image::new(
        ImageSize {
            width: w,
            height: h,
        },
        buf,
    )
    .expect("synthetic image")
}

/// **The fp16 gate.** Engine + preprocessor vs the PyTorch reference, compared by
/// cosine — the metric the application actually uses, not an element-wise tolerance.
///
/// The reference image is exactly the engine's input size, so kornia's Stretch resize is
/// the identity and the only thing under test is real numerics.
///
/// A silently-wrong low-precision ViT still emits a plausible-looking 384-d unit vector;
/// without this the failure surfaces only as mysteriously poor retrieval.
#[test]
#[ignore = "needs a GPU, a built engine and DINOV3_REF_DIR"]
fn descriptor_matches_pytorch_reference() {
    let dir = ref_dir();
    let stream = stream();
    let mut dino = DinoV3::from_engine_file(engine_path(), stream.clone()).expect("load engine");
    let (gw, gh) = dino.grid();
    let (w, h) = (gw * 16, gh * 16);

    let raw = std::fs::read(dir.join("ref_image.bin")).expect("ref_image.bin");
    assert_eq!(
        raw.len(),
        w * h * 3,
        "ref_image.bin is {}x{}x3 but the engine wants {w}x{h}x3 — \
         re-export with --input-size {w}",
        raw.len() / 3 / h.max(1),
        h
    );
    let ref_bytes = std::fs::read(dir.join("ref_descriptor.bin")).expect("ref_descriptor.bin");
    let want: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(want.len(), dino.dim(), "reference dim != engine dim");

    let img = Image::<u8, 3>::new(
        ImageSize {
            width: w,
            height: h,
        },
        raw,
    )
    .expect("ref image");
    let dev = Image(img.0.to_cuda(&stream).expect("h2d"));

    let mut r = dino.alloc_result().expect("alloc");
    dino.submit(&dev, &mut r).expect("submit");
    stream.synchronize().expect("sync");
    let got = r.descriptor_host().expect("d2h");

    let norm: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "descriptor should be L2-normed on device, got norm {norm}"
    );

    let cos = cosine(&got, &want);
    assert!(
        cos > 0.999,
        "descriptor cosine vs PyTorch is {cos:.6} (want > 0.999). \
         If the engine was built fp16, step down: fp16 + --layerPrecisions=*norm*:fp32, \
         then fp32 — and mirror the choice in DinoV3::engine_profile()."
    );
    println!("descriptor cosine vs PyTorch reference: {cos:.6}");
}

/// Catches a wrongly-bound output that parity alone might pass: the descriptor must
/// actually discriminate. Two crops of one scene should score well above an unrelated
/// image; a constant or mis-bound output scores identically on both.
#[test]
#[ignore = "needs a GPU and a built engine"]
fn descriptor_discriminates() {
    let stream = stream();
    let mut dino = DinoV3::from_engine_file(engine_path(), stream.clone()).expect("load engine");
    let (gw, gh) = dino.grid();
    let (w, h) = (gw * 16, gh * 16);

    // Two overlapping crops of one "scene" (same seed, shifted), vs a different scene.
    let base = synth_image(w, h, 7);
    let shifted = {
        let src = base.as_slice();
        let mut v = vec![0u8; w * h * 3];
        // Shift by 8 px horizontally — most content is shared.
        for y in 0..h {
            for x in 0..w {
                let sx = (x + 8) % w;
                let (d, s) = ((y * w + x) * 3, (y * w + sx) * 3);
                v[d..d + 3].copy_from_slice(&src[s..s + 3]);
            }
        }
        Image::new(
            ImageSize {
                width: w,
                height: h,
            },
            v,
        )
        .expect("shifted")
    };
    let other = synth_image(w, h, 99);

    let mut descs = Vec::new();
    for img in [&base, &shifted, &other] {
        let dev = Image(img.0.to_cuda(&stream).expect("h2d"));
        let mut r = dino.alloc_result().expect("alloc");
        dino.submit(&dev, &mut r).expect("submit");
        stream.synchronize().expect("sync");
        descs.push(r.descriptor_host().expect("d2h"));
    }

    let same = cosine(&descs[0], &descs[1]);
    let diff = cosine(&descs[0], &descs[2]);
    let margin = same - diff;
    println!("cos(same scene) = {same:.4}, cos(different) = {diff:.4}, margin = {margin:.4}");
    // A bare `same > diff` would pass on a fixture the model cannot tell apart at all,
    // so require a real margin. 0.05 is far below what a working engine produces here
    // and far above the ~0.001 a degenerate fixture or mis-bound output yields.
    assert!(
        margin > 0.05,
        "descriptor barely discriminates: same-scene {same:.4} vs different {diff:.4} \
         (margin {margin:.4}) — likely a mis-bound engine output"
    );
}

/// The bank's cosine kernel against a host dot product, and its enrollment bookkeeping.
#[test]
#[ignore = "needs a GPU and a built engine"]
fn bank_matches_host_cosine() {
    let stream = stream();
    let mut dino = DinoV3::from_engine_file(engine_path(), stream.clone()).expect("load engine");
    let (gw, gh) = dino.grid();
    let (w, h) = (gw * 16, gh * 16);
    let dim = dino.dim();

    let mut bank = DescriptorBank::new(8, dim, stream.clone()).expect("bank");
    assert!(bank.is_empty());

    let mut hosts = Vec::new();
    let mut r = dino.alloc_result().expect("alloc");
    for seed in [1u64, 2, 3] {
        let dev = Image(synth_image(w, h, seed).0.to_cuda(&stream).expect("h2d"));
        dino.submit(&dev, &mut r).expect("submit");
        stream.synchronize().expect("sync");
        hosts.push(r.descriptor_host().expect("d2h"));
        bank.enroll(r.descriptor_slice()).expect("enroll");
    }
    stream.synchronize().expect("sync enrollments");
    assert_eq!(bank.len(), 3);

    // Re-submit the first image; it must match bank row 0 at ~1.0.
    let dev = Image(synth_image(w, h, 1).0.to_cuda(&stream).expect("h2d"));
    dino.submit(&dev, &mut r).expect("submit");
    let mut scores = bank.alloc_scores().expect("scores");
    bank.match_into(r.descriptor_slice(), &mut scores)
        .expect("match");
    stream.synchronize().expect("sync");

    let host = scores.to_host_image(&stream).expect("d2h").into_vec();
    let got = &host[..bank.len()];
    let q = r.descriptor_host().expect("d2h");
    for (i, (g, stored)) in got.iter().zip(&hosts).enumerate() {
        let want = cosine(&q, stored);
        assert!(
            (g - want).abs() < 1e-3,
            "bank row {i}: kernel {g:.6} vs host {want:.6}"
        );
    }
    assert!(
        got[0] > 0.99,
        "same image should match its own bank row, got {:.4}",
        got[0]
    );
}

/// End-to-end proof that the **bf16 path through `EngineProfile` actually works** —
/// `engine_profile()` → `EngineCache` → `EngineBuilder::bf16` → the `kBF16` builder flag
/// → a numerically correct engine.
///
/// This is the test that distinguishes "the bf16 flag is plumbed" from "the bf16 flag has
/// an effect". If the flag were dropped anywhere in that chain the build would silently
/// fall back to fp16 (`engine_profile` sets `fp16: false`, so really fp32) and still
/// pass a smoke test — only the parity number catches it, and only because fp16 for this
/// model is catastrophically wrong rather than subtly wrong.
///
/// Slow on a cache miss (~2 min while TensorRT times kernels); instant afterwards.
#[test]
#[ignore = "needs a GPU, DINOV3_ONNX, DINOV3_REF_DIR, and --features builder"]
#[cfg(any(feature = "hub", feature = "builder"))]
fn from_onnx_builds_a_correct_bf16_engine() {
    let onnx = from_workspace(
        std::env::var("DINOV3_ONNX").expect("set DINOV3_ONNX to the exported .onnx"),
    );
    let dir = ref_dir();
    let stream = stream();

    let mut dino = DinoV3::from_onnx(&onnx, stream.clone()).expect("build engine from onnx");
    let (gw, gh) = dino.grid();
    let (w, h) = (gw * 16, gh * 16);

    let raw = std::fs::read(dir.join("ref_image.bin")).expect("ref_image.bin");
    let want: Vec<f32> = std::fs::read(dir.join("ref_descriptor.bin"))
        .expect("ref_descriptor.bin")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let img = Image::<u8, 3>::new(
        ImageSize {
            width: w,
            height: h,
        },
        raw,
    )
    .expect("ref image");
    let dev = Image(img.0.to_cuda(&stream).expect("h2d"));
    let mut r = dino.alloc_result().expect("alloc");
    dino.submit(&dev, &mut r).expect("submit");
    stream.synchronize().expect("sync");

    let cos = cosine(&r.descriptor_host().expect("d2h"), &want);
    assert!(
        cos > 0.999,
        "from_onnx() engine cosine vs PyTorch is {cos:.6} (want > 0.999). \
         NaN here means the bf16 flag never reached BuilderFlag::kBF16 and the graph \
         ran in fp16 — check EngineProfile::bf16 → EngineBuilder::bf16 → the shim."
    );
    println!("from_onnx (bf16 via EngineProfile) cosine vs PyTorch: {cos:.6}");
}

/// `enroll` must refuse to overrun its allocation rather than corrupting memory.
#[test]
#[ignore = "needs a GPU and a built engine"]
fn bank_rejects_overflow() {
    let stream = stream();
    let mut dino = DinoV3::from_engine_file(engine_path(), stream.clone()).expect("load engine");
    let (gw, gh) = dino.grid();
    let dim = dino.dim();

    let mut bank = DescriptorBank::new(1, dim, stream.clone()).expect("bank");
    let dev = Image(
        synth_image(gw * 16, gh * 16, 5)
            .0
            .to_cuda(&stream)
            .expect("h2d"),
    );
    let mut r = dino.alloc_result().expect("alloc");
    dino.submit(&dev, &mut r).expect("submit");
    stream.synchronize().expect("sync");

    bank.enroll(r.descriptor_slice()).expect("first fits");
    let err = bank.enroll(r.descriptor_slice()).unwrap_err();
    assert!(
        matches!(err, vrt_dinov3::DinoError::BankFull { capacity: 1 }),
        "expected BankFull, got {err}"
    );

    bank.clear();
    assert!(bank.is_empty());
    bank.enroll(r.descriptor_slice()).expect("room after clear");
}
