//! RF-DETR **Segmentation** (instance masks): GPU stretch+ImageNet-normalize → TRT
//! → **GPU decode**.
//!
//! [`RfDetrSeg`] is an `Image<u8,3> → Vec<Instance>` instance segmenter on the
//! RF-DETR Seg Preview model (end-to-end, NMS-free). Each instance comes back with
//! a COCO class + score + box + a **binary mask**.
//!
//! Like [`vrt_rfdetr`], everything stays on the **GPU** and the pipeline is fully
//! **async / caller-owned** (VPI-style): `submit` enqueues stretch+normalize → TRT
//! → two decode kernels (boxes/labels, then masks) on the shared stream and returns
//! with **no sync and no host copy**. The caller syncs the stream once, then pulls
//! only what it needs — [`count`], [`detections`] (boxes), [`masks_host`] (masks),
//! or the GPU-resident [`SegResult::dets_slice`]/[`SegResult::masks_slice`] for
//! downstream on-device work. Host transfers happen **only when requested**.
//!
//! Engine I/O (input `[1,3,H,W]`): `dets [1,Q,4]` (cxcywh, normalized), `labels
//! [1,Q,C]` (logits; **class 0 = background**, COCO 1–90), `masks [1,Q,mh,mw]`
//! (raw per-query mask logits — `einsum(feats, query_coeffs) + bias`, no sigmoid).
//! The mask decode thresholds the logit at `0` (≡ `sigmoid ≥ 0.5`); the mask grid
//! covers the whole stretched frame, so a mask maps to the source by resizing
//! `mh×mw → src_h×src_w` (the stretch is full-frame).
//!
//! [`count`]: SegResult::count
//! [`detections`]: SegResult::detections
//! [`masks_host`]: SegResult::masks_host

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use kornia_image::Image;
use kornia_imgproc::preprocess::{Normalize, Preprocessor, PreprocessorBuilder, ResizeMode};
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};
use vrt::cuda::{cfg_1d, cfg_2d};
use vrt::{BoxError, Engine, ModelSession};

/// Errors from RF-DETR segmentation inference / decode.
#[derive(Debug, thiserror::Error)]
pub enum SegError {
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("kornia CUDA: {0}")]
    Cuda(#[from] kornia_tensor::CudaError),
    #[error(transparent)]
    Preproc(#[from] kornia_imgproc::preprocess::PreprocessError),
    #[error("engine output '{0}' missing")]
    MissingOutput(String),
}

/// A detected object (no mask) in original-image coordinate space.
#[derive(Debug, Clone)]
pub struct Detection {
    /// COCO category id (1–90); 0 (background) is never emitted.
    pub class_id: u32,
    pub score: f32,
    /// Bounding box `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
}

/// A segmented instance: [`Detection`] + its binary mask.
#[derive(Debug, Clone)]
pub struct Instance {
    pub class_id: u32,
    pub score: f32,
    pub bbox: [f32; 4],
    /// Row-major binary mask (`1` = foreground) at the model's mask grid,
    /// [`mask_size`](Self::mask_size) `= (width, height)`. The grid spans the whole
    /// stretched frame — resize it to `(src_w, src_h)` to overlay on the source.
    pub mask: Vec<u8>,
    /// Mask grid `(width, height)` — same for every instance (e.g. `(108, 108)`).
    pub mask_size: (usize, usize),
}

// One thread per query: argmax the class logits (sigmoid is monotonic → argmax on
// the raw logit), skip background (class 0), threshold the sigmoid score, and — for
// survivors — atomically append the box (cxcywh normalized → xyxy in source pixels,
// the stretch maps a normalized coord straight to the source) plus the surviving
// query's index, so the mask pass can gather its mask row.
const DECODE_SRC: &str = r#"
extern "C" __global__ void seg_decode(
    const float* __restrict__ boxes,    // [Q*4] cxcywh, normalized [0,1]
    const float* __restrict__ logits,   // [Q*C] raw logits
    int Q, int C, float conf,
    float src_w, float src_h,
    float* __restrict__ dets,           // [Q*6] out: x1,y1,x2,y2,class,score
    int*   __restrict__ qidx,           // [Q] out: survivor slot -> query index
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
    qidx[slot] = q;
}
"#;

// One thread per (survivor slot, mask pixel): gather the surviving query's mask row
// and threshold the raw logit at 0 (≡ sigmoid ≥ 0.5) → binary mask, packed by slot.
// Slots ≥ *count are skipped (their old contents are never read — `count` bounds
// every host/device reader).
const MASKS_SRC: &str = r#"
extern "C" __global__ void seg_masks(
    const float* __restrict__ masks_logits, // [Q*L] raw per-query mask logits
    const int*   __restrict__ qidx,          // [Q] survivor slot -> query index
    const int*   __restrict__ count,
    int L,
    unsigned char* __restrict__ out          // [Q*L] binary, packed by slot
) {
    int p    = blockIdx.x * blockDim.x + threadIdx.x; // mask pixel
    int slot = blockIdx.y * blockDim.y + threadIdx.y; // survivor slot
    if (p >= L || slot >= *count) return;
    long qi = qidx[slot];
    out[(long)slot * L + p] = (masks_logits[qi * L + p] >= 0.0f) ? 1 : 0;
}
"#;

/// Caller-owned segmentation output (VPI-style), allocated once and reused.
///
/// [`RfDetrSeg::submit`] fills these **GPU-resident** buffers async; after the
/// stream sync the accessors copy to host **only on request** ([`count`],
/// [`detections`], [`masks_host`], [`instances`]) or hand back device slices
/// ([`dets_slice`]/[`masks_slice`]/[`qidx_slice`]) for downstream GPU work.
///
/// [`count`]: Self::count
/// [`detections`]: Self::detections
/// [`masks_host`]: Self::masks_host
/// [`instances`]: Self::instances
/// [`dets_slice`]: Self::dets_slice
/// [`masks_slice`]: Self::masks_slice
/// [`qidx_slice`]: Self::qidx_slice
pub struct SegResult {
    dets: CudaSlice<f32>, // [q*6] x1,y1,x2,y2,class,score (survivors, packed by slot)
    qidx: CudaSlice<i32>, // [q] survivor slot -> query index
    masks: CudaSlice<u8>, // [q*mh*mw] binary masks (survivors, packed by slot)
    count_pin: vrt::PinnedBuffer<i32>,
    stream: Arc<CudaStream>,
    q: usize,
    mh: usize,
    mw: usize,
}

impl SegResult {
    fn alloc(stream: &Arc<CudaStream>, q: usize, mh: usize, mw: usize) -> Result<Self, SegError> {
        Ok(Self {
            dets: unsafe { stream.alloc::<f32>(q * 6)? },
            qidx: unsafe { stream.alloc::<i32>(q)? },
            masks: unsafe { stream.alloc::<u8>(q * mh * mw)? },
            count_pin: vrt::PinnedBuffer::<i32>::alloc(1)?,
            stream: stream.clone(),
            q,
            mh,
            mw,
        })
    }

    /// Number of surviving instances (reads the pinned scalar — call after sync).
    pub fn count(&self) -> usize {
        (self.count_pin.as_slice()[0].max(0) as usize).min(self.q)
    }

    /// Mask grid `(width, height)` (e.g. `(108, 108)`).
    pub fn mask_size(&self) -> (usize, usize) {
        (self.mw, self.mh)
    }

    /// Download the surviving boxes to host (original-image pixels), **no masks**.
    /// Call after the stream sync that follows [`RfDetrSeg::submit`].
    pub fn detections(&self) -> Result<Vec<Detection>, SegError> {
        let n = self.count();
        if n == 0 {
            return Ok(Vec::new());
        }
        let flat = self.stream.clone_dtoh(&self.dets.slice(0..n * 6))?;
        Ok(flat
            .chunks_exact(6)
            .map(|d| Detection {
                class_id: d[4] as u32,
                score: d[5],
                bbox: [d[0], d[1], d[2], d[3]],
            })
            .collect())
    }

    /// Download the surviving **binary masks** to host, one `mh*mw` row per instance
    /// (aligned with [`detections`](Self::detections)). Call after the stream sync.
    pub fn masks_host(&self) -> Result<Vec<Vec<u8>>, SegError> {
        let n = self.count();
        if n == 0 {
            return Ok(Vec::new());
        }
        let len = self.mh * self.mw;
        let flat = self.stream.clone_dtoh(&self.masks.slice(0..n * len))?;
        Ok(flat.chunks_exact(len).map(<[u8]>::to_vec).collect())
    }

    /// Download boxes **and** masks together as [`Instance`]s. Convenience that
    /// pairs [`detections`](Self::detections) with [`masks_host`](Self::masks_host);
    /// both are host copies, so this is the "give me everything on the CPU" path.
    pub fn instances(&self) -> Result<Vec<Instance>, SegError> {
        let dets = self.detections()?;
        let masks = self.masks_host()?;
        let size = (self.mw, self.mh);
        Ok(dets
            .into_iter()
            .zip(masks)
            .map(|(d, mask)| Instance {
                class_id: d.class_id,
                score: d.score,
                bbox: d.bbox,
                mask,
                mask_size: size,
            })
            .collect())
    }

    /// GPU-resident decoded boxes `[q*6]` (`x1,y1,x2,y2,class,score` per slot); only
    /// slots `0..count()` are valid. Stays on device — for downstream GPU work.
    pub fn dets_slice(&self) -> &CudaSlice<f32> {
        &self.dets
    }

    /// GPU-resident binary masks `[q*mh*mw]`, packed by slot; slots `0..count()`
    /// valid. Stays on device.
    pub fn masks_slice(&self) -> &CudaSlice<u8> {
        &self.masks
    }

    /// GPU-resident survivor-slot → query-index map `[q]`; slots `0..count()` valid.
    pub fn qidx_slice(&self) -> &CudaSlice<i32> {
        &self.qidx
    }
}

/// RF-DETR instance segmenter (payload): backbone session + stretch/ImageNet
/// preprocessor + the two decode kernels + shared stream. Build once, reuse per frame.
pub struct RfDetrSeg {
    model: ModelSession,
    preproc: Preprocessor,
    stream: Arc<CudaStream>,
    input: Tensor<f32, 4>, // [1,3,ih,iw] CHW f32 device, reused
    decode: CudaKernel,
    masks_k: CudaKernel,
    conf: f32,
    q: usize,
    c: usize,
    mh: usize,
    mw: usize,
    dets_name: String,
    labels_name: String,
    masks_name: String,
}

impl RfDetrSeg {
    /// Build a segmenter sharing `stream`. Input size read from the engine's static
    /// `[1,3,H,W]`; the three outputs are identified by shape (rank-4 = masks,
    /// rank-3 last-dim-4 = boxes, rank-3 last-dim ≠ 4 = labels).
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>, conf: f32) -> Result<Self, BoxError> {
        let inp = engine.inputs().next().ok_or("seg: engine has no input")?;
        let d = &inp.dims;
        if d.len() != 4 || d.iter().any(|&x| x <= 0) {
            return Err(format!("seg: input must be static [1,3,H,W], got {d:?}").into());
        }
        let (ih, iw) = (d[2] as usize, d[3] as usize);

        // Positive-dim guards reject dynamic/unknown outputs (-1 → would wrap to a
        // huge usize); labels is rank-3 with last-dim ≠ 4 so a box-shaped tensor
        // can't be misbound as labels.
        let (mut dets_name, mut labels_name, mut masks_name) = (None, None, None);
        let (mut q, mut c, mut mh, mut mw) = (0usize, 0usize, 0usize, 0usize);
        for s in engine.outputs() {
            match s.dims.as_slice() {
                [1, nq, nh, nw] if *nq > 0 && *nh > 0 && *nw > 0 => {
                    masks_name = Some(s.name.clone());
                    (q, mh, mw) = (*nq as usize, *nh as usize, *nw as usize);
                }
                [1, nq, 4] if *nq > 0 => dets_name = Some(s.name.clone()),
                [1, nq, ncl] if *nq > 0 && *ncl > 0 && *ncl != 4 => {
                    labels_name = Some(s.name.clone());
                    (q, c) = (*nq as usize, *ncl as usize);
                }
                _ => {}
            }
        }
        let dets_name = dets_name.ok_or("seg: no boxes output [1,Q,4]")?;
        let labels_name = labels_name.ok_or("seg: no labels output [1,Q,C]")?;
        let masks_name = masks_name.ok_or("seg: no masks output [1,Q,mh,mw]")?;
        if q == 0 || c == 0 || mh == 0 || mw == 0 {
            return Err("seg: dynamic/unknown output dims unsupported".into());
        }

        // Stretch + ImageNet normalization (this export has no baked-in mean/std).
        let preproc = PreprocessorBuilder::new()
            .mode(ResizeMode::Stretch)
            .normalize(Normalize::imagenet())
            .build_cuda(stream.clone())?;
        let input = zeros_cuda::<f32, 4>([1, 3, ih, iw], &stream)?;
        let decode = CudaKernel::compile(stream.context(), DECODE_SRC, "seg_decode")?;
        let masks_k = CudaKernel::compile(stream.context(), MASKS_SRC, "seg_masks")?;
        let model = ModelSession::new(engine, stream.clone())?;

        Ok(Self {
            model,
            preproc,
            stream,
            input,
            decode,
            masks_k,
            conf,
            q,
            c,
            mh,
            mw,
            dets_name,
            labels_name,
            masks_name,
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
            .ok_or("seg: onnx path is not valid UTF-8")?;
        let engine_path = vrt_hub::EngineCache::default().resolve(
            "rfdetr-seg",
            model_path,
            &Self::engine_profile(),
        )?;
        Self::from_engine_file(engine_path, stream, conf)
    }

    /// Pull from Hugging Face (`kornia/rfdetr-seg`) and construct — a matching
    /// prebuilt engine if the registry has one, else the pinned ONNX built
    /// on-device. Requires feature `hub`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>, conf: f32) -> Result<Self, BoxError> {
        let engine = vrt_hub::resolve_engine("rfdetr-seg", &Self::engine_profile())?;
        Self::from_engine_file(engine, stream, conf)
    }

    /// Query slots (fixed by the engine) — the [`SegResult`] capacity.
    pub fn num_queries(&self) -> usize {
        self.q
    }

    /// Mask grid `(width, height)` the engine emits (e.g. `(108, 108)`).
    pub fn mask_size(&self) -> (usize, usize) {
        (self.mw, self.mh)
    }

    /// Allocate a reusable output for this segmenter.
    pub fn alloc_result(&self) -> Result<SegResult, SegError> {
        SegResult::alloc(&self.stream, self.q, self.mh, self.mw)
    }

    /// Submit one frame's async GPU work — stretch-resize → backbone → decode boxes
    /// → gather+threshold masks — into the caller-owned `out`, with **no sync and no
    /// host copy** (only the tiny survivor count is async-copied to pinned host).
    /// Sync the stream, then read `out.count()` / `out.detections()` /
    /// `out.masks_host()`, or use the GPU-resident slices. `img` is a device image of
    /// any resolution; boxes come back in its pixel space.
    pub fn submit(&mut self, img: &Image<u8, 3>, out: &mut SegResult) -> Result<(), SegError> {
        self.preproc.run(img, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;

        let box_raw = tmap
            .get(&self.dets_name)
            .ok_or_else(|| SegError::MissingOutput(self.dets_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;
        let lab_raw = tmap
            .get(&self.labels_name)
            .ok_or_else(|| SegError::MissingOutput(self.labels_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;
        let msk_raw = tmap
            .get(&self.masks_name)
            .ok_or_else(|| SegError::MissingOutput(self.masks_name.clone()))?
            .f32_ptr()? as usize as CUdeviceptr;

        // Per-frame device count (atomic, zeroed). Survivors land in the caller's
        // buffers; `qidx` maps each survivor slot back to its query for the mask pass.
        let count_dev: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let dets_raw = out.dets.device_ptr(self.stream.as_ref()).0;
        let qidx_raw = out.qidx.device_ptr(self.stream.as_ref()).0;
        let masks_out_raw = out.masks.device_ptr(self.stream.as_ref()).0;
        let cnt_raw = count_dev.device_ptr(self.stream.as_ref()).0;

        let (qi, ci) = (self.q as i32, self.c as i32);
        let conf = self.conf;
        let (sw, sh) = (img.width() as f32, img.height() as f32);
        self.decode
            .launch_builder(&self.stream)
            .arg(&box_raw)
            .arg(&lab_raw)
            .arg(&qi)
            .arg(&ci)
            .arg(&conf)
            .arg(&sw)
            .arg(&sh)
            .arg(&dets_raw)
            .arg(&qidx_raw)
            .arg(&cnt_raw)
            .launch_cfg(cfg_1d(self.q, 256))?;

        // Mask pass: gather + threshold each survivor's mask row (reads the device
        // count → only decodes the survivors). Grid = (mask pixels × query slots).
        let li = (self.mh * self.mw) as i32;
        self.masks_k
            .launch_builder(&self.stream)
            .arg(&msk_raw)
            .arg(&qidx_raw)
            .arg(&cnt_raw)
            .arg(&li)
            .arg(&masks_out_raw)
            .launch_cfg(cfg_2d(self.mh * self.mw, self.q))?;

        // Async D2H of just the count into the caller's pinned buffer (no sync).
        let cnt_pin = out.count_pin.as_mut_ptr();
        let vstream = vrt::Stream::from_cuda_stream(self.stream.clone());
        unsafe {
            vstream.memcpy_d2h_raw(
                cnt_pin as *mut u8,
                cnt_raw as usize as *const _,
                std::mem::size_of::<i32>(),
            )?;
        }
        Ok(())
    }
}
