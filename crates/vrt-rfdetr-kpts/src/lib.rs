//! RF-DETR **Keypoint** (human pose): GPU stretch+ImageNet-normalize → TRT → CPU decode.
//!
//! [`RfDetrKpts`] is an `Image<u8,3> → Vec<PersonPose>` pose detector on the RF-DETR
//! Keypoint Preview model (end-to-end, NMS-free). Each person comes back with a box
//! + **17 COCO keypoints** in original-image pixels.
//!
//! Preprocessing is stretch-resize + **ImageNet mean/std** (this export expects
//! pre-normalized input, like `vrt-rfdetr`). Decode is on the **CPU**: only `Q=100`
//! queries and ~110 KB of output, negligible next to the transformer — after the
//! single stream sync, `poses()` reads the pinned host outputs and decodes.
//!
//! Engine I/O (input `[1,3,H,W]`): `dets [1,Q,4]` (cxcywh, normalized), `labels
//! [1,Q,2]` (logits; **class 1 = person**), `keypoints [1,Q,34,8]` (2 classes × 17
//! padded slots; per keypoint: `x,y` normalized, `vis` logit, then the precision
//! Cholesky). The person's 17 keypoints are slots `17..34`.
//!
//! Fully async / caller-owned (VPI-style): `alloc_result` → `submit` (no sync) →
//! `stream.synchronize()` → `poses()`.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::Image;
use kornia_imgproc::preprocess::{Normalize, Preprocessor, PreprocessorBuilder, ResizeMode};
use kornia_tensor::{zeros_cuda, Tensor};
use vrt::{BoxError, Engine, ModelSession};

/// COCO 17-keypoint order (index → joint name).
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

const PERSON_CLASS: usize = 1; // class 1 of 2 carries the keypoints
const NUM_KP: usize = 17;
const SIGMA_REF: f32 = 0.06; // normalized-position-std scale for the sharpness falloff

/// Errors from RF-DETR keypoint inference.
#[derive(Debug, thiserror::Error)]
pub enum KptsError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    Preproc(#[from] kornia_imgproc::preprocess::PreprocessError),
    #[error("engine output '{0}' missing")]
    MissingOutput(&'static str),
}

/// A detected person: box + 17 COCO keypoints, in original-image pixels.
#[derive(Debug, Clone)]
pub struct PersonPose {
    pub score: f32,
    /// `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
    /// Per joint `[x_px, y_px, confidence]`; confidence `[0,1]` = visibility ×
    /// spatial sharpness (from the model's learned per-keypoint precision).
    pub keypoints: [[f32; 3]; NUM_KP],
}

/// Caller-owned keypoint output (VPI-style): pinned host copies of the three
/// engine outputs, filled async by [`RfDetrKpts::submit`], decoded by [`poses`].
///
/// [`poses`]: KptsResult::poses
pub struct KptsResult {
    dets: vrt::PinnedBuffer<f32>,   // [q*4] cxcywh normalized
    labels: vrt::PinnedBuffer<f32>, // [q*num_classes] logits
    kpts: vrt::PinnedBuffer<f32>,   // [q*slots*kp_ch]
    q: usize,
    num_classes: usize,
    slots: usize,
    kp_ch: usize,
    src: (f32, f32), // original-image (w, h), stamped by submit
    conf: f32,       // confidence threshold, stamped by submit
}

impl KptsResult {
    fn alloc(q: usize, num_classes: usize, slots: usize, kp_ch: usize) -> Result<Self, KptsError> {
        Ok(Self {
            dets: vrt::PinnedBuffer::<f32>::alloc(q * 4)?,
            labels: vrt::PinnedBuffer::<f32>::alloc(q * num_classes)?,
            kpts: vrt::PinnedBuffer::<f32>::alloc(q * slots * kp_ch)?,
            q,
            num_classes,
            slots,
            kp_ch,
            src: (0.0, 0.0),
            conf: 0.0,
        })
    }

    /// Decode people + keypoints from the pinned outputs — pure host work, call
    /// **after** the stream sync following [`RfDetrKpts::submit`].
    pub fn poses(&self) -> Vec<PersonPose> {
        let (sw, sh) = self.src;
        let (lab, bx, kp) = (
            self.labels.as_slice(),
            self.dets.as_slice(),
            self.kpts.as_slice(),
        );
        let max_kp = (self.slots / self.num_classes.max(1)).max(1); // 17
        let kp_offset = PERSON_CLASS * max_kp; // person slots start at 17

        let mut out = Vec::new();
        for qi in 0..self.q {
            let score = sigmoid(lab[qi * self.num_classes + PERSON_CLASS]);
            if score < self.conf {
                continue;
            }

            // box: cxcywh normalized → xyxy pixels (stretch maps normalized → source).
            let b = &bx[qi * 4..qi * 4 + 4];
            let (cx, cy, bw, bh) = (b[0] * sw, b[1] * sh, b[2] * sw, b[3] * sh);
            let bbox = [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5];

            // 17 person keypoints. chan 2 = visibility logit; chans 4,5,6 are the
            // Cholesky (log_l11, l21, log_l22) of the 2-D precision → an error
            // ellipse; confidence = visibility × spatial sharpness.
            let mut keypoints = [[0.0f32; 3]; NUM_KP];
            let base = qi * self.slots * self.kp_ch;
            for (j, kpt) in keypoints.iter_mut().enumerate().take(max_kp) {
                let o = base + (kp_offset + j) * self.kp_ch;
                let vis = sigmoid(kp[o + 2]);
                let conf = if self.kp_ch >= 7 {
                    let (a, b, c) = (kp[o + 4].exp(), kp[o + 6].exp(), kp[o + 5]);
                    let det = (a * b).max(1e-6);
                    let sigma = (((a * a + b * b + c * c) / (det * det)) * 0.5).sqrt();
                    vis / (1.0 + (sigma / SIGMA_REF).powi(2))
                } else {
                    vis
                };
                // Guard against exp() overflow → NaN confidence.
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
        out
    }
}

/// RF-DETR keypoint pose detector (payload): backbone session + stretch/ImageNet
/// preprocessor + shared stream. Construct once, reuse every frame.
pub struct RfDetrKpts {
    model: ModelSession,
    preproc: Preprocessor,
    stream: Arc<CudaStream>,
    input: Tensor<f32, 4>, // [1,3,mh,mw] CHW f32 device, reused
    conf: f32,
    q: usize,
    num_classes: usize,
    slots: usize,
    kp_ch: usize,
    dets_name: String,
    labels_name: String,
    kpts_name: String,
}

impl RfDetrKpts {
    /// Build a pose detector sharing `stream`. Input size read from the engine's
    /// static `[1,3,H,W]`; the three outputs are identified by shape (rank-4 =
    /// keypoints, rank-3 last-dim-4 = boxes, rank-3 last-dim-2 = labels).
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>, conf: f32) -> Result<Self, BoxError> {
        let inp = engine.inputs().next().ok_or("kpts: engine has no input")?;
        let d = &inp.dims;
        if d.len() != 4 || d.iter().any(|&x| x <= 0) {
            return Err(format!("kpts: input must be static [1,3,H,W], got {d:?}").into());
        }
        let (mh, mw) = (d[2] as usize, d[3] as usize);

        // Positive-dim guards reject dynamic/unknown outputs (-1 → would wrap to a
        // huge usize); labels is rank-3 with last-dim ≠ 4 so a box-shaped tensor
        // can't be misbound as labels.
        let (mut dets_name, mut labels_name, mut kpts_name) = (None, None, None);
        let (mut q, mut num_classes, mut slots, mut kp_ch) = (0usize, 0usize, 0usize, 0usize);
        for s in engine.outputs() {
            match s.dims.as_slice() {
                [1, nq, ns, nc] if *nq > 0 && *ns > 0 && *nc > 0 => {
                    kpts_name = Some(s.name.clone());
                    (q, slots, kp_ch) = (*nq as usize, *ns as usize, *nc as usize);
                }
                [1, nq, 4] if *nq > 0 => dets_name = Some(s.name.clone()),
                [1, nq, ncl] if *nq > 0 && *ncl > 0 && *ncl != 4 => {
                    labels_name = Some(s.name.clone());
                    (q, num_classes) = (*nq as usize, *ncl as usize);
                }
                _ => {}
            }
        }
        let dets_name = dets_name.ok_or("kpts: no boxes output [1,Q,4]")?;
        let labels_name = labels_name.ok_or("kpts: no labels output [1,Q,2]")?;
        let kpts_name = kpts_name.ok_or("kpts: no keypoints output [1,Q,S,C]")?;
        if q == 0 || num_classes == 0 {
            return Err("kpts: dynamic/unknown output dims unsupported".into());
        }

        // Stretch + ImageNet normalization (this export has no baked-in mean/std).
        let preproc = PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build_cuda(stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, mh, mw], &stream)?;
        let model = ModelSession::new(engine, stream.clone())?;

        Ok(Self {
            model,
            preproc,
            stream,
            input,
            conf,
            q,
            num_classes,
            slots,
            kp_ch,
            dets_name,
            labels_name,
            kpts_name,
        })
    }

    /// Construct from a prebuilt `.engine` file.
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        conf: f32,
    ) -> Result<Self, BoxError> {
        Self::new(Engine::load(engine_path)?, stream, conf)
    }

    /// The engine build profile — fixed-resolution (static shapes).
    #[cfg(any(feature = "hub", feature = "builder"))]
    fn engine_profile() -> vrt_hub::EngineProfile {
        vrt_hub::EngineProfile {
            input: None,
            fp16: true,
            workspace_mb: 2048,
        }
    }

    /// Build (and cache) an engine from an ONNX file, then construct. Requires
    /// feature `hub` (trtexec build) or `builder` (in-process).
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        conf: f32,
    ) -> Result<Self, BoxError> {
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or("kpts: onnx path is not valid UTF-8")?;
        let engine_path = vrt_hub::EngineCache::default().resolve(
            "rfdetr-kpts",
            model_path,
            &Self::engine_profile(),
        )?;
        Self::from_engine_file(engine_path, stream, conf)
    }

    /// Pull from Hugging Face (`kornia/rfdetr-kpts`) and construct — a matching
    /// prebuilt engine if the registry has one, else the pinned ONNX built
    /// on-device. Requires feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>, conf: f32) -> Result<Self, BoxError> {
        let engine = vrt_hub::resolve_engine("rfdetr-kpts", &Self::engine_profile())?;
        Self::from_engine_file(engine, stream, conf)
    }

    /// Query slots (fixed by the engine) — the [`KptsResult`] capacity.
    pub fn num_queries(&self) -> usize {
        self.q
    }

    /// Allocate a reusable output for this detector.
    pub fn alloc_result(&self) -> Result<KptsResult, KptsError> {
        KptsResult::alloc(self.q, self.num_classes, self.slots, self.kp_ch)
    }

    /// Submit one frame — stretch+normalize → backbone → async D2H of the three
    /// outputs into `out`, with **no sync**. Sync the stream, then `out.poses()`.
    pub fn submit(&mut self, img: &Image<u8, 3>, out: &mut KptsResult) -> Result<(), KptsError> {
        self.preproc.run(img, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;

        let dets = tmap
            .get(&self.dets_name)
            .ok_or(KptsError::MissingOutput("dets"))?
            .f32_ptr()?;
        let labels = tmap
            .get(&self.labels_name)
            .ok_or(KptsError::MissingOutput("labels"))?
            .f32_ptr()?;
        let kpts = tmap
            .get(&self.kpts_name)
            .ok_or(KptsError::MissingOutput("keypoints"))?
            .f32_ptr()?;

        out.src = (img.width() as f32, img.height() as f32);
        out.conf = self.conf;

        // Async D2H each output into the caller's pinned buffers (no sync).
        let vstream = vrt::Stream::from_cuda_stream(self.stream.clone());
        let sz = std::mem::size_of::<f32>();
        unsafe {
            vstream.memcpy_d2h_raw(
                out.dets.as_mut_ptr() as *mut u8,
                dets as *const _,
                self.q * 4 * sz,
            )?;
            vstream.memcpy_d2h_raw(
                out.labels.as_mut_ptr() as *mut u8,
                labels as *const _,
                self.q * self.num_classes * sz,
            )?;
            vstream.memcpy_d2h_raw(
                out.kpts.as_mut_ptr() as *mut u8,
                kpts as *const _,
                self.q * self.slots * self.kp_ch * sz,
            )?;
        }
        Ok(())
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
