//! RF-DETR object detection: GPU stretch-resize → TRT backbone → **GPU decode**.
//!
//! [`RfDetr`] is a single `VrtImage → Vec<Detection>` detector. RF-DETR is a
//! transformer set-predictor: it emits a fixed set of query boxes + class
//! logits and is **NMS-free** (no duplicate suppression). The pipeline is
//! zero-copy and all-GPU: the NVMM camera frame stays on the device through a
//! stretch-resize kernel ([`Preprocessor`] in [`Stretch`](kornia_imgproc::preprocess::ResizeMode::Stretch)
//! mode — the export is `do_pad:false, do_normalize:false`, RGB `[0,1]`), the TRT
//! backbone, and a decode kernel; only the surviving detections (typically a
//! handful) are copied to the host.
//!
//! Model: the fixed-resolution official export (`PierreMarieCurie/rf-detr-onnx`,
//! registered in `vrt-hub` as `"rfdetr-small"`) — input `[1,3,512,512]`, outputs
//! `pred_boxes [1,300,4]` (cxcywh, normalized) + `pred_logits [1,300,91]`
//! (logits; class 0 = background).

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kornia_image::Image;
use kornia_imgproc::preprocess::Preprocessor;
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::{BoxError, Engine, ModelSession};

/// A detected object in original-image coordinate space.
#[derive(Debug, Clone)]
pub struct Detection {
    /// COCO category id (1–90); 0 (background) is never emitted.
    pub class_id: u32,
    pub score: f32,
    /// Bounding box `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
}

// One thread per query: argmax the class logits (sigmoid is monotonic, so argmax
// on the raw logit), skip background (class 0), threshold the sigmoid score, and
// — for survivors — atomically append the box (cxcywh normalized → xyxy in source
// pixels, since the stretch maps a normalized coord straight to the source).
const KERNEL_SRC: &str = r#"
extern "C" __global__ void rfdetr_decode(
    const float* __restrict__ boxes,    // [Q*4] cxcywh, normalized [0,1]
    const float* __restrict__ logits,   // [Q*C] raw logits
    int Q, int C, float conf,
    float src_w, float src_h,
    float* __restrict__ dets,           // [Q*6] out: x1,y1,x2,y2,class,score
    int*   __restrict__ count
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q) return;

    const float* lo = logits + (long)q * C;
    int   best_c = 0;
    float best_l = -1e30f;
    for (int c = 1; c < C; ++c) {        // class 0 = background, skip
        float l = lo[c];
        if (l > best_l) { best_l = l; best_c = c; }
    }
    float score = 1.0f / (1.0f + __expf(-best_l));
    if (best_c == 0 || score < conf) return;

    int slot = atomicAdd(count, 1);
    if (slot >= Q) return;

    const float* b = boxes + (long)q * 4;
    float cx = b[0] * src_w, cy = b[1] * src_h;
    float bw = b[2] * src_w, bh = b[3] * src_h;
    float* o = dets + (long)slot * 6;
    o[0] = cx - bw * 0.5f;
    o[1] = cy - bh * 0.5f;
    o[2] = cx + bw * 0.5f;
    o[3] = cy + bh * 0.5f;
    o[4] = (float)best_c;
    o[5] = score;
}
"#;

/// RF-DETR detector: `VrtImage → Vec<Detection>`.
///
/// [`run`](Self::run) does stretch-resize → TRT inference → GPU decode in one
/// synchronous call. No NMS (set predictor); boxes come back in original-image
/// pixels.
pub struct RfDetr {
    model: ModelSession,
    preproc: Preprocessor, // stretch mode (also owns the shared stream)
    input: Tensor<f32, 4>, // [1,3,model_h,model_w] CHW f32 device, reused
    decode: CudaKernel,
    conf_thresh: f32,
    box_name: String,
    logit_name: String,
}

impl RfDetr {
    /// Build a detector sharing `cuda_stream` with the rest of the application.
    ///
    /// The model input size is read from the engine (`[1,3,H,W]`), so any RF-DETR
    /// variant works unchanged (small 512², medium 576², …). The two engine
    /// outputs are bound by name: the one containing "box" is the boxes tensor,
    /// the one containing "logit" the class logits.
    pub fn new(
        engine: Arc<Engine>,
        cuda_stream: Arc<CudaStream>,
        conf_thresh: f32,
    ) -> Result<Self, BoxError> {
        // Derive the input H/W from the engine's static [1,3,H,W] input spec.
        let [_, _, model_h, model_w] = engine.static_input_nchw()?;

        // RF-DETR's processor stretches (no pad) and only rescales to [0,1].
        let preproc = Preprocessor::stretch(cuda_stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, model_h, model_w], &cuda_stream)?;
        let decode = CudaKernel::compile(cuda_stream.context(), KERNEL_SRC, "rfdetr_decode")?;
        let model = ModelSession::new(engine, cuda_stream)?;

        let outs = model.output_names();
        let box_name = outs
            .iter()
            .find(|n| n.contains("box"))
            .cloned()
            .ok_or("no boxes output (expected a name containing 'box')")?;
        let logit_name = outs
            .iter()
            .find(|n| n.contains("logit"))
            .cloned()
            .ok_or("no logits output (expected a name containing 'logit')")?;

        Ok(Self {
            model,
            preproc,
            input,
            decode,
            conf_thresh,
            box_name,
            logit_name,
        })
    }

    /// Detect objects in `img`: stretch-resize → TRT inference → GPU decode.
    ///
    /// Synchronous (syncs the stream internally); `img` is a device-resident RGBA
    /// surface of any resolution. Boxes come back in original-image coordinates.
    pub fn run(&mut self, img: &Image<u8, 3>) -> Result<Vec<Detection>, BoxError> {
        self.preproc.run(img, &mut self.input)?;
        let out = self.model.run(&self.input)?;

        let bview = out
            .get(&self.box_name)
            .ok_or_else(|| format!("no output '{}'", self.box_name))?;
        let lview = out
            .get(&self.logit_name)
            .ok_or_else(|| format!("no output '{}'", self.logit_name))?;
        let lshape = lview.shape_i64(); // [1, Q, C]
        if lshape.len() != 3 {
            return Ok(Vec::new());
        }
        let (q, c) = (lshape[1] as usize, lshape[2] as usize);
        // `f32_ptr()` rejects a non-F32 dtype with a `TrtError::Shape` Err, so an
        // `--fp16`-output engine fails loudly here instead of being misread as
        // garbage by the decode kernel (no separate dtype guard needed).
        let b_raw = bview.f32_ptr()? as usize as CUdeviceptr;
        let l_raw = lview.f32_ptr()? as usize as CUdeviceptr;

        // Per-frame device scratch: surviving dets [Q*6] + atomic count.
        let s = self.preproc.stream(); // the shared stream (also held by preproc/model)
        let count_dev: CudaSlice<i32> = s.alloc_zeros(1)?;
        let dets_dev: CudaSlice<f32> = unsafe { s.alloc(q * 6)? };
        let cnt_raw = count_dev.device_ptr(s.as_ref()).0;
        let dets_raw = dets_dev.device_ptr(s.as_ref()).0;

        let (qi, ci) = (q as i32, c as i32);
        let conf = self.conf_thresh;
        let (sw, sh) = (img.width() as f32, img.height() as f32);
        // One thread per query — the decode kernel is already 1-D.
        self.decode
            .launch_builder(s)
            .arg(&b_raw)
            .arg(&l_raw)
            .arg(&qi)
            .arg(&ci)
            .arg(&conf)
            .arg(&sw)
            .arg(&sh)
            .arg(&dets_raw)
            .arg(&cnt_raw)
            .launch_1d(q as u32)?;

        // Read the count, then D2H ONLY the surviving detections (n*6 floats).
        let count = s.clone_dtoh(&count_dev)?;
        let n = (count[0].max(0) as usize).min(q);
        if n == 0 {
            return Ok(Vec::new());
        }
        let flat = s.clone_dtoh(&dets_dev.slice(0..n * 6))?;

        Ok(flat
            .chunks_exact(6)
            .map(|d| Detection {
                class_id: d[4] as u32,
                score: d[5],
                bbox: [d[0], d[1], d[2], d[3]],
            })
            .collect())
    }
}
