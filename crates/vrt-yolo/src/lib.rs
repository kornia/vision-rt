//! YOLO11/v8 detection pipeline: CPU letterbox, TRT inference, decode + NMS.

use std::sync::Arc;

use vrt::{BoxError, CudaStream, Engine, ExecCtx, ModelSession, Operator, VrtTensor};

/// Errors from YOLO pre/post-processing and inference.
#[derive(Debug, thiserror::Error)]
pub enum YoloError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("no output tensor '{0}' in engine")]
    MissingOutput(String),
}

// ── Public types ──────────────────────────────────────────────────────────────

/// A detected object in original-image coordinate space.
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: u32,
    pub label: Option<String>,
    pub score: f32,
    /// Bounding box: [x1, y1, x2, y2] pixels in the original image.
    pub bbox: [f32; 4],
}

/// Scale and padding applied during letterbox, used to map boxes back.
#[derive(Debug, Clone)]
pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_left: f32,
    pub pad_top: f32,
}

impl LetterboxInfo {
    /// Compute letterbox parameters from source → destination dimensions.
    /// Matches the geometry used by `Preprocessor` kernel exactly.
    pub fn from_dims(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
        let pad_left = (dst_w as f32 - src_w as f32 * scale) * 0.5;
        let pad_top = (dst_h as f32 - src_h as f32 * scale) * 0.5;
        Self {
            scale,
            pad_left,
            pad_top,
        }
    }
}

// ── CPU postprocessing ────────────────────────────────────────────────────────

/// Decode YOLO output tensor into raw detection candidates before NMS.
///
/// Handles both `[1, 84, N]` (col-first) and `[1, N, 84]` (row-first) layouts.
/// Returns `(x1, y1, x2, y2, class_id, score)` tuples in letterboxed space.
pub fn decode_output(
    output: &[f32],
    shape: &[i64],
    conf_threshold: f32,
) -> Vec<(f32, f32, f32, f32, u32, f32)> {
    if shape.len() < 3 {
        return vec![];
    }
    let a = shape[1] as usize;
    let b = shape[2] as usize;
    // YOLO11/v8 exports [1, num_features, num_anchors] e.g. [1, 84, 8400].
    let col_first = a < b;
    let (num_anchors, num_cols) = if col_first { (b, a) } else { (a, b) };
    let num_classes = num_cols.saturating_sub(4);
    if num_classes == 0 {
        return vec![];
    }

    let mut detections = Vec::new();
    for i in 0..num_anchors {
        let (cx, cy, w, h) = if col_first {
            (
                output[i],
                output[num_anchors + i],
                output[2 * num_anchors + i],
                output[3 * num_anchors + i],
            )
        } else {
            (
                output[i * num_cols],
                output[i * num_cols + 1],
                output[i * num_cols + 2],
                output[i * num_cols + 3],
            )
        };

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
        detections.push((
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
            best_class,
            best_score,
        ));
    }
    detections
}

/// Greedy per-class non-maximum suppression.
pub fn nms(
    mut boxes: Vec<(f32, f32, f32, f32, u32, f32)>,
    iou_threshold: f32,
) -> Vec<(f32, f32, f32, f32, u32, f32)> {
    boxes.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<(f32, f32, f32, f32, u32, f32)> = Vec::new();
    for candidate in boxes {
        let suppress = kept
            .iter()
            .any(|k| k.4 == candidate.4 && iou(k, &candidate) > iou_threshold);
        if !suppress {
            kept.push(candidate);
        }
    }
    kept
}

fn iou(a: &(f32, f32, f32, f32, u32, f32), b: &(f32, f32, f32, f32, u32, f32)) -> f32 {
    let inter_w = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let inter_h = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
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

/// Map boxes from model-input space back to original image coordinates.
pub fn unletterbox(
    boxes: Vec<(f32, f32, f32, f32, u32, f32)>,
    info: &LetterboxInfo,
    labels: Option<&[&str]>,
) -> Vec<Detection> {
    boxes
        .into_iter()
        .map(|(x1, y1, x2, y2, class_id, score)| {
            let label = labels
                .and_then(|l| l.get(class_id as usize))
                .map(|s| s.to_string());
            Detection {
                class_id,
                label,
                score,
                bbox: [
                    (x1 - info.pad_left) / info.scale,
                    (y1 - info.pad_top) / info.scale,
                    (x2 - info.pad_left) / info.scale,
                    (y2 - info.pad_top) / info.scale,
                ],
            }
        })
        .collect()
}

/// Full postprocess pipeline: decode → NMS → unletterbox.
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

// ── YoloInferStage ───────────────────────────────────────────────────────────

/// Pipeline stage: [`VrtTensor`] → `Vec<Detection>`.
///
/// Runs TRT YOLO inference async, schedules an async D2H copy in `enqueue`,
/// then runs decode + NMS on CPU in `finalize` (after the pipeline sync).
/// Owns its `Session` — share the CUDA stream with other stages via
/// [`Session::with_stream`](vrt::Session::with_stream).
pub struct YoloInferStage {
    model: ModelSession,
    lb_info: LetterboxInfo,
    conf_thresh: f32,
    iou_thresh: f32,
    output_name: String,
    output_cpu: Vec<f32>, // async-D2H target, reused every frame
    output_shape: Vec<i64>,
}

impl YoloInferStage {
    /// Create a YOLO inference stage sharing `cuda_stream` with other stages.
    ///
    /// `lb_info` must match the source → model dimension mapping used by the
    /// upstream [`NvmmPreprocessStage`](vrt_gst::NvmmPreprocessStage).
    pub fn new(
        engine: Arc<Engine>,
        cuda_stream: Arc<CudaStream>,
        lb_info: LetterboxInfo,
        conf_thresh: f32,
        iou_thresh: f32,
    ) -> Result<Self, BoxError> {
        let model = ModelSession::new(engine, cuda_stream)?;
        // Pick the first output tensor name from the engine (YOLO has one output).
        let output_name = model
            .output_names()
            .first()
            .cloned()
            .ok_or("engine has no output tensors")?;
        Ok(Self {
            model,
            lb_info,
            conf_thresh,
            iou_thresh,
            output_name,
            output_cpu: Vec::new(),
            output_shape: Vec::new(),
        })
    }

    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        self.model.cuda_stream()
    }
}

impl Operator for YoloInferStage {
    type Input = VrtTensor;
    type Pending = (); // async-D2H target lives in self; nothing to hand forward
    type Output = Vec<Detection>;

    fn enqueue(&mut self, input: &VrtTensor, _ctx: &ExecCtx) -> Result<(), BoxError> {
        let out = self.model.run(input)?;

        // The view carries pointer + resolved shape + byte length together.
        let view = out
            .get(&self.output_name)
            .ok_or_else(|| format!("no output tensor '{}'", self.output_name))?;
        let out_bytes = view.byte_len();

        // Resize host buffer and cache shape on first run (or shape change).
        let n_floats = out_bytes / 4;
        if self.output_cpu.len() != n_floats || self.output_shape != view.shape_i64() {
            self.output_cpu.resize(n_floats, 0.0);
            self.output_shape = view.shape_i64();
        }

        // Async D2H — data will be ready after the pipeline syncs the stream.
        unsafe {
            self.model.stream().memcpy_d2h_raw(
                self.output_cpu.as_mut_ptr() as *mut u8,
                view.as_ptr() as *const _,
                out_bytes,
            )?;
        }
        Ok(())
    }

    fn finalize(&mut self, _pending: (), _ctx: &ExecCtx) -> Result<Vec<Detection>, BoxError> {
        Ok(postprocess(
            &self.output_cpu,
            &self.output_shape,
            &self.lb_info,
            self.conf_thresh,
            self.iou_thresh,
            None,
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_removes_duplicates() {
        let boxes = vec![
            (10.0, 10.0, 100.0, 100.0, 0u32, 0.9),
            (12.0, 12.0, 102.0, 102.0, 0u32, 0.85),
            (200.0, 200.0, 300.0, 300.0, 0u32, 0.8),
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_different_classes_not_suppressed() {
        let boxes = vec![
            (10.0, 10.0, 100.0, 100.0, 0u32, 0.9),
            (10.0, 10.0, 100.0, 100.0, 1u32, 0.85),
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_decode_output_shape_1_84_n() {
        let n = 8400usize;
        let mut data = vec![0.0f32; 84 * n];
        data[0 * n + 0] = 320.0;
        data[1 * n + 0] = 240.0;
        data[2 * n + 0] = 100.0;
        data[3 * n + 0] = 80.0;
        data[4 * n + 0] = 0.9;
        let dets = decode_output(&data, &[1, 84, n as i64], 0.25);
        assert!(!dets.is_empty());
        assert_eq!(dets[0].4, 0u32);
    }

    #[test]
    fn test_decode_output_shape_1_n_84() {
        let n = 8400usize;
        let mut data = vec![0.0f32; n * 84];
        let anchor = 3;
        data[anchor * 84 + 0] = 100.0;
        data[anchor * 84 + 1] = 200.0;
        data[anchor * 84 + 2] = 50.0;
        data[anchor * 84 + 3] = 60.0;
        data[anchor * 84 + 4 + 5] = 0.7;
        let dets = decode_output(&data, &[1, n as i64, 84], 0.25);
        assert!(!dets.is_empty());
        assert_eq!(dets[0].4, 5u32);
        assert!((dets[0].5 - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_unletterbox_roundtrip() {
        let info = LetterboxInfo {
            scale: 0.5,
            pad_left: 0.0,
            pad_top: 40.0,
        };
        let boxes = vec![(100.0f32, 80.0, 200.0, 160.0, 0u32, 0.9)];
        let dets = unletterbox(boxes, &info, None);
        assert_eq!(dets.len(), 1);
        assert!((dets[0].bbox[0] - 200.0).abs() < 1e-3);
        assert!((dets[0].bbox[2] - 400.0).abs() < 1e-3);
        assert!((dets[0].bbox[1] - 80.0).abs() < 1e-3);
        assert!((dets[0].bbox[3] - 240.0).abs() < 1e-3);
    }

    #[test]
    fn test_postprocess_pipeline() {
        let n = 8400usize;
        let mut data = vec![0.0f32; 84 * n];
        data[0 * n + 0] = 320.0;
        data[1 * n + 0] = 240.0;
        data[2 * n + 0] = 100.0;
        data[3 * n + 0] = 80.0;
        data[4 * n + 0] = 0.9;
        let info = LetterboxInfo {
            scale: 1.0,
            pad_left: 0.0,
            pad_top: 0.0,
        };
        let labels = ["person", "bicycle", "car"];
        let dets = postprocess(&data, &[1, 84, n as i64], &info, 0.25, 0.45, Some(&labels));
        assert!(!dets.is_empty());
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[0].label.as_deref(), Some("person"));
    }
}
