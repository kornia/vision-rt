//! `submit_pair` must return what two `submit`s return, image for image.
//!
//! Needs a GPU and the backbone ONNX: `XFEAT_ONNX=/path/xfeat_backbone.onnx
//! cargo test -p vrt-xfeat --release --features builder --test gpu_pair -- --ignored --nocapture`.
//! Both paths share one stereo engine, but TensorRT may pick different fp16 tactics for
//! batch 1 and 2, so equality is checked against a tolerance, not bit-exactly.
#![cfg(any(feature = "hub", feature = "builder"))]

use kornia_io::functional::read_image_any_rgb8;
use vrt_xfeat::{XFeat, XFeatParams, XFeatResult};

const TOP_K: usize = 2048;
const IMAGE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../vrt-depth-anything/assets/kitchen_detect_depth.png"
);

struct Host {
    kpts: Vec<f32>,
    descs: Vec<f32>,
}

fn host(r: &XFeatResult) -> Host {
    Host {
        kpts: r.kpts_to_host().unwrap(),
        descs: r.descs_to_host().unwrap(),
    }
}

/// Fraction of `a`'s keypoints that `b` has at the same pixel, and the min
/// descriptor cosine over those shared keypoints.
fn agreement(a: &Host, b: &Host) -> (f64, f32) {
    let key = |xy: &[f32]| (xy[0].round() as i32, xy[1].round() as i32);
    let index: std::collections::HashMap<_, usize> = b
        .kpts
        .chunks_exact(2)
        .enumerate()
        .map(|(i, xy)| (key(xy), i))
        .collect();
    let (mut shared, mut min_cos) = (0usize, 1.0f32);
    for (i, xy) in a.kpts.chunks_exact(2).enumerate() {
        if let Some(&j) = index.get(&key(xy)) {
            shared += 1;
            let da = &a.descs[i * 64..(i + 1) * 64];
            let db = &b.descs[j * 64..(j + 1) * 64];
            min_cos = min_cos.min(da.iter().zip(db).map(|(x, y)| x * y).sum());
        }
    }
    (shared as f64 / (a.kpts.len() / 2).max(1) as f64, min_cos)
}

#[test]
#[ignore = "needs a GPU, XFEAT_ONNX, and --features builder"]
fn pair_matches_single() {
    let onnx = std::env::var("XFEAT_ONNX").expect("set XFEAT_ONNX to xfeat_backbone.onnx");
    let left = read_image_any_rgb8(IMAGE).unwrap();
    let mut right = left.clone();
    kornia_imgproc::flip::horizontal_flip(&left, &mut right).unwrap();
    let stream = vrt::Stream::new_standalone().unwrap().cuda_stream().clone();
    let mut xfeat = XFeat::from_onnx_stereo(
        &onnx,
        stream.clone(),
        XFeatParams::new(TOP_K, 0.05),
        left.width(),
        left.height(),
    )
    .unwrap();

    let (l, r) = (
        left.to_cuda(&stream).unwrap(),
        right.to_cuda(&stream).unwrap(),
    );

    let mut s = [xfeat.alloc_result().unwrap(), xfeat.alloc_result().unwrap()];
    let mut p = [xfeat.alloc_result().unwrap(), xfeat.alloc_result().unwrap()];
    {
        let [s0, s1] = &mut s;
        xfeat.submit(&l, s0).unwrap();
        xfeat.submit(&r, s1).unwrap();
        let [p0, p1] = &mut p;
        xfeat.submit_pair(&l, &r, p0, p1).unwrap();
    }
    stream.synchronize().unwrap();

    // Device-side outputs a stereo matcher consumes without a host sync: the clamped
    // count, and keypoints in image pixels (720 rows floor to 704 → y scale != 1).
    for r in s.iter().chain(p.iter()) {
        let n = r.count();
        assert_eq!(stream.clone_dtoh(r.count_device()).unwrap(), vec![n as i32]);
        let dev = stream.clone_dtoh(&r.kpts_px().slice(0..2 * n)).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&dev), bits(&r.kpts_to_host().unwrap()));
    }

    for (side, (a, b)) in ["left", "right"].iter().zip(s.iter().zip(p.iter())) {
        let (ha, hb) = (host(a), host(b));
        let (shared, min_cos) = agreement(&ha, &hb);
        println!(
            "{side}: single {} pair {} shared {:.4} min_cos {:.5}",
            a.count(),
            b.count(),
            shared,
            min_cos
        );
        assert!(a.count() > 100, "{side}: too few keypoints to compare");
        let dc = a.count().abs_diff(b.count()) as f64 / a.count() as f64;
        assert!(dc <= 0.01, "{side}: counts differ by {:.2}%", dc * 100.0);
        assert!(
            shared >= 0.98,
            "{side}: only {shared:.4} of keypoints shared"
        );
        assert!(min_cos >= 0.99, "{side}: descriptor cosine {min_cos}");
    }
    // The two images differ, so a mixed-up batch slot cannot pass the above.
    let (cross, _) = agreement(&host(&s[0]), &host(&p[1]));
    assert!(
        cross < 0.5,
        "left matches right's slot ({cross:.3}): batch slots swapped?"
    );
}
