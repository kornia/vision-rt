/// A detected object in the original image coordinate space.
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: u32,
    pub label: Option<String>, // None if no label map provided
    pub score: f32,
    /// Bounding box in original image space: [x1, y1, x2, y2] pixels (f32).
    pub bbox: [f32; 4],
}

/// Letterbox (resize-with-padding) result — stores scale and padding for
/// un-letterboxing detections back to original image space.
#[derive(Debug, Clone)]
pub struct LetterboxInfo {
    pub scale: f32,    // scale factor applied (min of w_scale, h_scale)
    pub pad_left: f32, // pixels added to left
    pub pad_top: f32,  // pixels added to top
}

/// Preprocess: letterbox a raw RGBA frame into a flat f32 CHW tensor.
///
/// - `pixels`: raw bytes in RGBA order (4 bytes/px)
/// - `src_w`, `src_h`: source image dimensions
/// - `dst_w`, `dst_h`: target model input size (e.g. 640, 640)
/// - Returns: `(chw_f32_buffer: Vec<f32>, LetterboxInfo)` — the buffer is
///   [3, dst_h, dst_w] in CHW order, normalized to [0, 1].
pub fn letterbox_rgba_to_chw(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> (Vec<f32>, LetterboxInfo) {
    let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;
    let pad_left = (dst_w - new_w) as f32 / 2.0;
    let pad_top = (dst_h - new_h) as f32 / 2.0;

    let total = (3 * dst_h * dst_w) as usize;
    // Standard YOLO gray fill: 114/255
    let gray = 114.0_f32 / 255.0;
    let mut chw = vec![gray; total];

    let r_offset = 0usize;
    let g_offset = (dst_h * dst_w) as usize;
    let b_offset = 2 * (dst_h * dst_w) as usize;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let src_x_f = (dx as f32 - pad_left) / scale;
            let src_y_f = (dy as f32 - pad_top) / scale;

            // Only fill pixels that map into the source image.
            if src_x_f < 0.0
                || src_y_f < 0.0
                || src_x_f > (src_w - 1) as f32
                || src_y_f > (src_h - 1) as f32
            {
                // Padding area: already filled with gray.
                continue;
            }

            let sx = src_x_f.clamp(0.0, (src_w - 1) as f32) as u32;
            let sy = src_y_f.clamp(0.0, (src_h - 1) as f32) as u32;

            let src_idx = ((sy * src_w + sx) * 4) as usize;
            let r = pixels[src_idx] as f32 / 255.0;
            let g = pixels[src_idx + 1] as f32 / 255.0;
            let b = pixels[src_idx + 2] as f32 / 255.0;
            // alpha ignored

            let dst_idx = (dy * dst_w + dx) as usize;
            chw[r_offset + dst_idx] = r;
            chw[g_offset + dst_idx] = g;
            chw[b_offset + dst_idx] = b;
        }
    }

    let info = LetterboxInfo { scale, pad_left, pad_top };
    (chw, info)
}

/// Decode YOLO output tensor into detections before NMS.
///
/// `output`: f32 slice of the raw model output tensor.
/// `shape`: the output tensor shape (e.g. [1, 84, 8400] or [1, 8400, 84]).
/// `conf_threshold`: minimum class score to keep.
///
/// Returns list of (x1, y1, x2, y2, class_id, score) in letterboxed space.
pub fn decode_output(
    output: &[f32],
    shape: &[i64],
    conf_threshold: f32,
) -> Vec<(f32, f32, f32, f32, u32, f32)> {
    if shape.len() < 3 {
        return vec![];
    }

    // shape is [batch, A, B] — determine layout.
    let a = shape[1] as usize;
    let b = shape[2] as usize;

    // Determine layout:
    // [batch, num_cols, N]:  shape[1]=num_cols < shape[2]=N  → a < b → col-first
    //                        e.g. [1, 84, 8400]
    // [batch, N, num_cols]:  shape[1]=N > shape[2]=num_cols  → a > b → row-first
    //                        e.g. [1, 8400, 84]
    // When a > b: shape[1] is num_cols (small), shape[2] is num_anchors (large).
    // When a < b: shape[1] is num_anchors (small), shape[2] is num_cols (large) — unusual but supported.
    // For equal (a == b), default to col-first.
    let col_first = a >= b; // shape[1] is num_cols, shape[2] is num_anchors

    let (num_anchors, num_cols) = if col_first { (b, a) } else { (a, b) };
    let num_classes = num_cols.saturating_sub(4);
    if num_classes == 0 {
        return vec![];
    }

    let mut detections = Vec::new();

    for i in 0..num_anchors {
        let (cx, cy, w, h) = if col_first {
            // output[col * num_anchors + anchor]
            (
                output[0 * num_anchors + i],
                output[1 * num_anchors + i],
                output[2 * num_anchors + i],
                output[3 * num_anchors + i],
            )
        } else {
            // output[anchor * num_cols + col]
            (
                output[i * num_cols],
                output[i * num_cols + 1],
                output[i * num_cols + 2],
                output[i * num_cols + 3],
            )
        };

        // Find best class score.
        let mut best_score = -1.0f32;
        let mut best_class = 0u32;
        for c in 0..num_classes {
            let score = if col_first {
                output[(4 + c) * num_anchors + i]
            } else {
                output[i * num_cols + 4 + c]
            };
            if score > best_score {
                best_score = score;
                best_class = c as u32;
            }
        }

        if best_score < conf_threshold {
            continue;
        }

        // Convert cx,cy,w,h → x1,y1,x2,y2.
        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;

        detections.push((x1, y1, x2, y2, best_class, best_score));
    }

    detections
}

/// Non-maximum suppression. Removes overlapping boxes.
/// `boxes`: (x1,y1,x2,y2, class_id, score). `iou_threshold`: typically 0.45.
pub fn nms(
    mut boxes: Vec<(f32, f32, f32, f32, u32, f32)>,
    iou_threshold: f32,
) -> Vec<(f32, f32, f32, f32, u32, f32)> {
    // Sort by score descending.
    boxes.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<(f32, f32, f32, f32, u32, f32)> = Vec::new();

    for candidate in boxes {
        let suppress = kept.iter().any(|k| {
            // Only suppress within the same class.
            k.4 == candidate.4 && iou(k, &candidate) > iou_threshold
        });
        if !suppress {
            kept.push(candidate);
        }
    }

    kept
}

fn iou(a: &(f32, f32, f32, f32, u32, f32), b: &(f32, f32, f32, f32, u32, f32)) -> f32 {
    let ix1 = a.0.max(b.0);
    let iy1 = a.1.max(b.1);
    let ix2 = a.2.min(b.2);
    let iy2 = a.3.min(b.3);

    let inter_w = (ix2 - ix1).max(0.0);
    let inter_h = (iy2 - iy1).max(0.0);
    let inter = inter_w * inter_h;

    if inter == 0.0 {
        return 0.0;
    }

    let area_a = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let area_b = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Un-letterbox: map boxes from model-input space back to original image space.
pub fn unletterbox(
    boxes: Vec<(f32, f32, f32, f32, u32, f32)>,
    info: &LetterboxInfo,
    labels: Option<&[&str]>,
) -> Vec<Detection> {
    boxes
        .into_iter()
        .map(|(x1, y1, x2, y2, class_id, score)| {
            let orig_x1 = (x1 - info.pad_left) / info.scale;
            let orig_y1 = (y1 - info.pad_top) / info.scale;
            let orig_x2 = (x2 - info.pad_left) / info.scale;
            let orig_y2 = (y2 - info.pad_top) / info.scale;

            let label = labels
                .and_then(|l| l.get(class_id as usize))
                .map(|s| s.to_string());

            Detection {
                class_id,
                label,
                score,
                bbox: [orig_x1, orig_y1, orig_x2, orig_y2],
            }
        })
        .collect()
}

/// High-level all-in-one: run full postprocess pipeline.
///
/// `raw_output`: f32 slice of the raw YOLO output tensor.
/// `output_shape`: tensor shape.
/// `letterbox_info`: from the matching letterbox call.
/// `conf_threshold`: e.g. 0.25. `iou_threshold`: e.g. 0.45.
/// `labels`: optional COCO class name list.
pub fn postprocess(
    raw_output: &[f32],
    output_shape: &[i64],
    letterbox_info: &LetterboxInfo,
    conf_threshold: f32,
    iou_threshold: f32,
    labels: Option<&[&str]>,
) -> Vec<Detection> {
    let decoded = decode_output(raw_output, output_shape, conf_threshold);
    let after_nms = nms(decoded, iou_threshold);
    unletterbox(after_nms, letterbox_info, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_no_pad() {
        // Source 640x640 → target 640x640: scale=1, no padding, pixel values preserved.
        let pixels: Vec<u8> = (0..640 * 640 * 4).map(|i| ((i / 4) % 256) as u8).collect();
        let (chw, info) = letterbox_rgba_to_chw(&pixels, 640, 640, 640, 640);
        assert!((info.scale - 1.0).abs() < 1e-5);
        assert_eq!(chw.len(), 3 * 640 * 640);
    }

    #[test]
    fn test_letterbox_downscale_with_padding() {
        // Source 1280x720 → target 640x640: should add horizontal bars.
        let pixels = vec![128u8; 1280 * 720 * 4];
        let (chw, info) = letterbox_rgba_to_chw(&pixels, 1280, 720, 640, 640);
        assert!((info.scale - 0.5).abs() < 1e-5);
        assert!(info.pad_top > 0.0);
        assert_eq!(chw.len(), 3 * 640 * 640);
    }

    #[test]
    fn test_nms_removes_duplicates() {
        let boxes = vec![
            (10.0, 10.0, 100.0, 100.0, 0u32, 0.9),
            (12.0, 12.0, 102.0, 102.0, 0u32, 0.85), // heavily overlapping, same class
            (200.0, 200.0, 300.0, 300.0, 0u32, 0.8), // non-overlapping
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2); // keeps the 0.9 box and the non-overlapping 0.8
    }

    #[test]
    fn test_nms_different_classes_not_suppressed() {
        let boxes = vec![
            (10.0, 10.0, 100.0, 100.0, 0u32, 0.9), // class 0
            (10.0, 10.0, 100.0, 100.0, 1u32, 0.85), // class 1, same box
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2); // different classes: both kept
    }

    #[test]
    fn test_decode_output_shape_1_84_n() {
        // Construct a minimal fake output: batch=1, 84, 10 anchors.
        // One anchor has high score for class 0, others low.
        let n = 10usize;
        let mut data = vec![0.0f32; 84 * n];
        // Anchor 0: cx=320, cy=240, w=100, h=80, class0_score=0.9
        data[0 * n + 0] = 320.0; // cx
        data[1 * n + 0] = 240.0; // cy
        data[2 * n + 0] = 100.0; // w
        data[3 * n + 0] = 80.0; // h
        data[4 * n + 0] = 0.9; // class0 score
        let shape = vec![1i64, 84, n as i64];
        let dets = decode_output(&data, &shape, 0.25);
        assert!(!dets.is_empty(), "expected at least one detection");
        assert_eq!(dets[0].4, 0u32); // class_id 0
        assert!(dets[0].5 >= 0.25);
    }

    #[test]
    fn test_decode_output_shape_1_n_84() {
        // Transposed layout: [1, N, 84]
        let n = 10usize;
        let mut data = vec![0.0f32; n * 84];
        // Anchor 3: cx=100, cy=200, w=50, h=60, class5_score=0.7
        let anchor = 3;
        data[anchor * 84 + 0] = 100.0; // cx
        data[anchor * 84 + 1] = 200.0; // cy
        data[anchor * 84 + 2] = 50.0;  // w
        data[anchor * 84 + 3] = 60.0;  // h
        data[anchor * 84 + 4 + 5] = 0.7; // class5 score
        let shape = vec![1i64, n as i64, 84];
        let dets = decode_output(&data, &shape, 0.25);
        assert!(!dets.is_empty(), "expected at least one detection");
        assert_eq!(dets[0].4, 5u32); // class_id 5
        assert!((dets[0].5 - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_unletterbox_roundtrip() {
        let info = LetterboxInfo {
            scale: 0.5,
            pad_left: 0.0,
            pad_top: 40.0,
        };
        // Box in model space (640x640).
        let boxes = vec![(100.0f32, 80.0, 200.0, 160.0, 0u32, 0.9)];
        let dets = unletterbox(boxes, &info, None);
        assert_eq!(dets.len(), 1);
        // x coords: (100 - 0) / 0.5 = 200, (200 - 0) / 0.5 = 400
        assert!((dets[0].bbox[0] - 200.0).abs() < 1e-3);
        assert!((dets[0].bbox[2] - 400.0).abs() < 1e-3);
        // y coords: (80 - 40) / 0.5 = 80, (160 - 40) / 0.5 = 240
        assert!((dets[0].bbox[1] - 80.0).abs() < 1e-3);
        assert!((dets[0].bbox[3] - 240.0).abs() < 1e-3);
    }

    #[test]
    fn test_postprocess_pipeline() {
        let n = 10usize;
        let mut data = vec![0.0f32; 84 * n];
        data[0 * n + 0] = 320.0;
        data[1 * n + 0] = 240.0;
        data[2 * n + 0] = 100.0;
        data[3 * n + 0] = 80.0;
        data[4 * n + 0] = 0.9;
        let shape = vec![1i64, 84, n as i64];
        let info = LetterboxInfo { scale: 1.0, pad_left: 0.0, pad_top: 0.0 };
        let labels = ["person", "bicycle", "car"];
        let dets = postprocess(&data, &shape, &info, 0.25, 0.45, Some(&labels));
        assert!(!dets.is_empty());
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[0].label.as_deref(), Some("person"));
    }
}
