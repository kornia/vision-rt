//! RF-DETR **Keypoint** (human pose): GPU stretch-resize + ImageNet-normalize → TRT → CPU decode.
//!
//! [`RfDetrKpts`] is an `Image<u8, 3> → Vec<PersonPose>` pose detector built on the RF-DETR Keypoint
//! Preview model (end-to-end, NMS-free, like the detection RF-DETR). Each person comes back with a
//! box + **17 COCO keypoints** in original-image pixels.
//!
//! Two things differ from [`vrt_rfdetr`](../vrt_rfdetr/index.html):
//! 1. **Normalization** — the detection ONNX bakes the `[0,1]`→ImageNet normalize into its graph, but
//!    this keypoint export expects *pre-normalized* input (verified: input range ≈ `[-2.12, 2.64]`).
//!    So we stretch to `[0,1]` ([`Preprocessor::stretch`]) then normalize in a tiny in-place kernel.
//! 2. **Decode is on the CPU** — only `Q=100` queries and ~112 KB of raw output, negligible next to
//!    the ~60 ms transformer (a GPU kernel would buy <0.05 % and be far harder to keep correct).
//!
//! Engine I/O (input `[1,3,576,576]`): `dets [1,Q,4]` (cxcywh, normalized), `labels [1,Q,2]` (logits;
//! **class 1 = person**, class 0 has no keypoints), `keypoints [1,Q,34,8]` (2 classes × 17 padded
//! slots; per keypoint: `x,y` normalized, `vis` logit, then uncertainty). The person's 17 keypoints
//! are slots `17..34`; `x*src_w, y*src_h, sigmoid(vis)` (the `×model` then `÷model` cancels — a
//! normalized coord maps straight to the source, exactly like the boxes).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::Image;
use kornia_imgproc::preprocess::Preprocessor;
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::{BoxError, DataType, Engine, OutputTensor, Session};

/// COCO 17-keypoint order (index → joint name), for downstream skeleton edges/labels.
pub const COCO_KEYPOINT_NAMES: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

/// A detected person with a box + 17 COCO keypoints, in original-image pixel coordinates.
#[derive(Debug, Clone)]
pub struct PersonPose {
    pub score: f32,
    /// `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
    /// Per joint: `[x_px, y_px, confidence]`, confidence `[0,1]` = visibility × spatial sharpness
    /// (the latter from the model's learned per-keypoint precision-Cholesky). Low ⇒ occluded or
    /// imprecisely localized — gate/anchor/smooth accordingly downstream.
    pub keypoints: [[f32; 3]; 17],
}

const PERSON_CLASS: usize = 1; // class 1 of 2 carries the keypoints (class 0 has none)
const NUM_KP: usize = 17;
const SIGMA_REF: f32 = 0.06; // normalized-position-std scale for the sharpness falloff (calibrated)

// In-place ImageNet normalization of the `[0,1]` CHW input: `(x - mean) / std` per channel.
const NORM_SRC: &str = r#"
extern "C" __global__ void imagenet_normalize(float* __restrict__ x, int chw, int hw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chw) return;
    int c = i / hw;                                   // CHW: 0=R, 1=G, 2=B
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    x[i] = (x[i] - mean[c]) * istd[c];
}
"#;

/// RF-DETR Keypoint pose detector: `Image<u8, 3> → Vec<PersonPose>`.
pub struct RfDetrKpts {
    session: Session,
    preproc: Preprocessor, // stretch mode (also owns the shared stream)
    input: Tensor<f32, 4>, // [1,3,H,W] CHW f32 device, reused
    normalize: CudaKernel,
    conf_thresh: f32,
    model_h: usize,
    model_w: usize,
}

impl RfDetrKpts {
    /// Build a pose detector sharing `cuda_stream` with the rest of the application. The model input
    /// size is read from the engine's static `[1,3,H,W]` input spec.
    pub fn new(
        engine: Arc<Engine>,
        cuda_stream: Arc<CudaStream>,
        conf_thresh: f32,
    ) -> Result<Self, BoxError> {
        let [_, _, model_h, model_w] = engine.static_input_nchw()?;

        let preproc = Preprocessor::stretch(cuda_stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, model_h, model_w], &cuda_stream)?;
        let normalize = CudaKernel::compile(cuda_stream.context(), NORM_SRC, "imagenet_normalize")?;
        let session = Session::with_stream(engine, cuda_stream)?;

        Ok(Self {
            session,
            preproc,
            input,
            normalize,
            conf_thresh,
            model_h,
            model_w,
        })
    }

    /// Detect people + 17 keypoints in `img`: stretch → ImageNet-normalize → TRT → CPU decode.
    /// Synchronous; keypoints come back in original-image pixels.
    pub fn run(&mut self, img: &Image<u8, 3>) -> Result<Vec<PersonPose>, BoxError> {
        // 1. Stretch to [0,1] CHW, then ImageNet-normalize in place — both kernels
        //    run on the shared stream before the inference sync.
        self.preproc.run(img, &mut self.input)?;
        let chw = 3 * self.model_h * self.model_w;
        let hw = (self.model_h * self.model_w) as i32;
        {
            let s = self.preproc.stream();
            let x_slice = self
                .input
                .as_cudaslice_mut()
                .ok_or("kpts: input tensor not device-resident")?;
            self.normalize
                .launch_builder(s)
                .arg(x_slice)
                .arg(&(chw as i32))
                .arg(&hw)
                .launch_1d(chw as u32)?;
        }

        // 2. TRT inference → HOST outputs (Q is small; CPU decode).
        let in_ptr = self.input.as_ptr() as *mut c_void;
        let shape: [i64; 4] = [1, 3, self.model_h as i64, self.model_w as i64];
        let outs = unsafe {
            self.session
                .run_device_inputs(&[("input", in_ptr, &shape)])?
        };

        // 3. Resolve outputs by shape: rank-4 = keypoints; rank-3 last-dim-4 = boxes; last-dim-2 = labels.
        let (mut dets, mut labels, mut kpts) = (None, None, None);
        for t in outs.values() {
            match t.shape.as_slice() {
                [_, _, _, _] => kpts = Some(t),
                [_, _, 4] => dets = Some(t),
                [_, _, 2] => labels = Some(t),
                _ => {}
            }
        }
        let (Some(dets), Some(labels), Some(kpts)) = (dets, labels, kpts) else {
            return Ok(Vec::new());
        };

        let num_classes = labels.shape[2] as usize; // 2
        let q = labels.shape[1] as usize; // 100
        let slots = kpts.shape[2] as usize; // 34
        let kp_ch = kpts.shape[3] as usize; // 8
        let max_kp = (slots / num_classes.max(1)).max(1); // 17
        let kp_offset = PERSON_CLASS * max_kp; // person slots start at 17

        let lab = to_f32(labels)?;
        let bx = to_f32(dets)?;
        let kp = to_f32(kpts)?;
        let (sw, sh) = (img.width() as f32, img.height() as f32);

        let mut out = Vec::new();
        for qi in 0..q {
            let score = sigmoid(lab[qi * num_classes + PERSON_CLASS]);
            if score < self.conf_thresh {
                continue;
            }

            // box: cxcywh normalized → xyxy pixels.
            let b = &bx[qi * 4..qi * 4 + 4];
            let (cx, cy, bw, bh) = (b[0] * sw, b[1] * sh, b[2] * sw, b[3] * sh);
            let bbox = [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5];

            // 17 person keypoints from class-1 slots. The 3rd value is a per-keypoint **confidence**
            // = visibility × spatial sharpness: chan 2 is the visibility logit (sigmoid'd), and chans
            // 4,5,6 are the Cholesky of the 2-D precision matrix (`log_l11, l21, log_l22`) — a learned
            // error ellipse. A tight ellipse (σ≈0.01, confident joint) keeps the confidence high; a
            // wide one (σ≈0.25, occluded/ambiguous) drives it toward 0. Downstream uses this single
            // value to gate the depth lift, weight the temporal smoothing, and fade the render.
            let mut keypoints = [[0.0f32; 3]; NUM_KP];
            let base = qi * slots * kp_ch;
            for (j, kpt) in keypoints.iter_mut().enumerate().take(max_kp) {
                let o = base + (kp_offset + j) * kp_ch;
                let vis = sigmoid(kp[o + 2]);
                let conf = if kp_ch >= 7 {
                    let (a, b, c) = (kp[o + 4].exp(), kp[o + 6].exp(), kp[o + 5]);
                    let det = (a * b).max(1e-6);
                    let sigma = (((a * a + b * b + c * c) / (det * det)) * 0.5).sqrt(); // normalized pos std
                    vis / (1.0 + (sigma / SIGMA_REF).powi(2)) // sharpness falls off past σ≈SIGMA_REF
                } else {
                    vis
                };
                // A huge precision logit overflows exp() → det=inf → sigma=NaN →
                // conf=NaN, which would poison downstream confidence-weighted
                // smoothing / bone learning. Clamp to a valid [0,1] confidence.
                let conf = if conf.is_finite() {
                    conf.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                *kpt = [kp[o] * sw, kp[o + 1] * sh, conf];
            }
            out.push(PersonPose {
                score,
                bbox,
                keypoints,
            });
        }
        Ok(out)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Host `OutputTensor` → `Vec<f32>` (engine I/O may be fp32 or, with `--fp16`, fp16).
///
/// Errors on any other dtype rather than returning an empty vec — a silent
/// empty would turn into an out-of-bounds panic in the decode loop with no clue
/// to the real cause (an unexpected output binding dtype).
fn to_f32(t: &OutputTensor) -> Result<Vec<f32>, BoxError> {
    match t.dtype {
        DataType::Float32 => Ok(t.as_f32().to_vec()),
        DataType::Float16 => Ok(t.as_f32_from_f16()),
        other => Err(format!("rfdetr-kpts: unsupported output dtype {other:?}").into()),
    }
}
