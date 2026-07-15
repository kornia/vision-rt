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
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig};
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
    // qidx[slot] is the same for all L pixel-threads of this slot; read it through
    // the read-only cache so the redundant loads coalesce in L1.
    long qi = __ldg(&qidx[slot]);
    out[(long)slot * L + p] = (masks_logits[qi * L + p] >= 0.0f) ? 1 : 0;
}
"#;

// One block per survivor slot: mask-pool the backbone's block-11 token grid
// ([C,gh,gw], NCHW) over the instance mask's foreground, then L2-normalize → a
// per-instance appearance embedding for the tracker's ReID tie-breaker. Coverage
// pooling: reduce the (packed) instance mask into a gh×gw coverage histogram (each
// foreground mask pixel votes for the token cell it lands in — "soft coverage",
// the mask grid is an integer multiple of the token grid so the map is exact), then
// take the coverage-weighted mean token and normalize. Stale slots (>= count) and
// empty masks yield a zero embedding (caller zeros `out` each frame + reads 0..count).
const POOL_SRC: &str = r#"
extern "C" __global__ void pool_tokens(
    const float* __restrict__ tokens,        // [C*gh*gw] NCHW (batch 1)
    int C, int gh, int gw,
    const unsigned char* __restrict__ masks, // [Q*mmh*mmw] binary, packed by slot
    int mmh, int mmw,
    const int* __restrict__ count,
    float* __restrict__ out                  // [Q*C] L2-normed per live slot
) {
    int m = blockIdx.x;
    if (m >= *count) return;                 // stale slot: leave pre-zeroed embedding
    const unsigned char* mask = masks + (long)m * mmh * mmw;
    extern __shared__ float cov[];           // [gh*gw] coverage histogram
    int G = gh * gw;
    for (int i = threadIdx.x; i < G; i += blockDim.x) cov[i] = 0.0f;
    __syncthreads();

    float sx = (float)gw / mmw, sy = (float)gh / mmh;
    for (int p = threadIdx.x; p < mmh * mmw; p += blockDim.x) {
        if (mask[p]) {
            int mx = p % mmw, my = p / mmw;
            int tx = min(gw - 1, (int)(mx * sx));
            int ty = min(gh - 1, (int)(my * sy));
            atomicAdd(&cov[ty * gw + tx], 1.0f);
        }
    }
    __syncthreads();

    __shared__ float total;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < G; ++i) s += cov[i];
        total = s;
    }
    __syncthreads();

    float* o = out + (long)m * C;
    if (total <= 0.0f) {                      // empty mask → zero embedding
        for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] = 0.0f;
        return;
    }
    // coverage-weighted mean token per channel (channel loop over the grid)
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float* tc = tokens + (long)c * G;
        float acc = 0.0f;
        for (int cell = 0; cell < G; ++cell) {
            float w = cov[cell];
            if (w > 0.0f) acc += w * tc[cell];
        }
        o[c] = acc / total;
    }
    __syncthreads();
    // L2-normalize across channels
    __shared__ float ss[256];
    float local = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float v = o[c]; local += v * v; }
    ss[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ss[threadIdx.x] += ss[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(ss[0] + 1e-12f);
    for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] *= inv;
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
    feats: CudaSlice<f32>, // [q*tc] per-instance L2-normed appearance embeddings (empty if no tokens output)
    count_dev: CudaSlice<i32>, // [1] survivor count on-device (atomic target, retained for GPU readers)
    count_pin: vrt::PinnedBuffer<i32>,
    stream: Arc<CudaStream>,
    q: usize,
    mh: usize,
    mw: usize,
    tc: usize, // appearance embedding dim (0 = engine has no tokens output)
}

impl SegResult {
    fn alloc(
        stream: &Arc<CudaStream>,
        q: usize,
        mh: usize,
        mw: usize,
        tc: usize,
    ) -> Result<Self, SegError> {
        Ok(Self {
            dets: unsafe { stream.alloc::<f32>(q * 6)? },
            qidx: unsafe { stream.alloc::<i32>(q)? },
            masks: unsafe { stream.alloc::<u8>(q * mh * mw)? },
            feats: stream.alloc_zeros::<f32>(q * tc)?,
            count_dev: stream.alloc_zeros::<i32>(1)?,
            count_pin: vrt::PinnedBuffer::<i32>::alloc(1)?,
            stream: stream.clone(),
            q,
            mh,
            mw,
            tc,
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

    /// GPU-resident survivor count `[1]` (the decode kernel's atomic target). Pass to
    /// the `vrt-types` fusion builtins (`DepthImage::sample_masks`/`sample_boxes`) so
    /// they gate stale capacity slots **on the GPU** — valid on the stream from the
    /// moment `submit` returns (no host sync needed), unlike the pinned [`count`].
    ///
    /// [`count`]: Self::count
    pub fn count_slice(&self) -> &CudaSlice<i32> {
        &self.count_dev
    }

    /// Appearance embedding dimension (`C`, e.g. 384), or `0` if the engine has no
    /// `tokens` output (then [`features_host`](Self::features_host) is always empty).
    pub fn feat_dim(&self) -> usize {
        self.tc
    }

    /// GPU-resident per-instance appearance embeddings `[q*C]`, L2-normed, packed by
    /// slot (slots `0..count()` valid, aligned with [`detections`](Self::detections)).
    /// Empty when the engine has no `tokens` output. Stays on device.
    pub fn feats_slice(&self) -> &CudaSlice<f32> {
        &self.feats
    }

    /// Download the surviving instances' appearance embeddings to host, one `C`-vector
    /// per instance (aligned with [`detections`](Self::detections)). Each vector is
    /// L2-normalized (ready for cosine ReID). Empty if the engine has no `tokens`
    /// output. Call after the stream sync that follows [`RfDetrSeg::submit`].
    pub fn features_host(&self) -> Result<Vec<Vec<f32>>, SegError> {
        let n = self.count();
        if n == 0 || self.tc == 0 {
            return Ok(Vec::new());
        }
        let flat = self.stream.clone_dtoh(&self.feats.slice(0..n * self.tc))?;
        Ok(flat.chunks_exact(self.tc).map(<[f32]>::to_vec).collect())
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
    pool_k: Option<CudaKernel>, // token mask-pool (only if the engine has a `tokens` output)
    conf: f32,
    q: usize,
    c: usize,
    mh: usize,
    mw: usize,
    tc: usize,  // token channels (0 = no tokens output)
    tgh: usize, // token grid height
    tgw: usize, // token grid width
    dets_name: String,
    labels_name: String,
    masks_name: String,
    tokens_name: Option<String>,
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
        let (mut dets_name, mut labels_name, mut masks_name, mut tokens_name) =
            (None, None, None, None);
        let (mut q, mut c, mut mh, mut mw) = (0usize, 0usize, 0usize, 0usize);
        let (mut tc, mut tgh, mut tgw) = (0usize, 0usize, 0usize);
        for s in engine.outputs() {
            // The optional backbone-token feature map is rank-4 like masks, so bind it
            // by NAME first (else it would be misclassified as the masks output).
            if s.name == "tokens" {
                if let [1, nc, nh, nw] = s.dims.as_slice() {
                    if *nc > 0 && *nh > 0 && *nw > 0 {
                        tokens_name = Some(s.name.clone());
                        (tc, tgh, tgw) = (*nc as usize, *nh as usize, *nw as usize);
                    }
                }
                continue;
            }
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
        // Compile the token mask-pool only if the engine exposes a `tokens` output.
        let pool_k = match &tokens_name {
            Some(_) => Some(CudaKernel::compile(
                stream.context(),
                POOL_SRC,
                "pool_tokens",
            )?),
            None => None,
        };
        let model = ModelSession::new(engine, stream.clone())?;

        Ok(Self {
            model,
            preproc,
            stream,
            input,
            decode,
            masks_k,
            pool_k,
            conf,
            q,
            c,
            mh,
            mw,
            tc,
            tgh,
            tgw,
            dets_name,
            labels_name,
            masks_name,
            tokens_name,
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

    /// Appearance embedding dimension (`C`) if the engine exposes a `tokens` output,
    /// else `0`. When `> 0`, [`SegResult::features_host`] returns a per-instance ReID
    /// embedding pooled from the backbone's block-11 tokens.
    pub fn feat_dim(&self) -> usize {
        self.tc
    }

    /// Allocate a reusable output for this segmenter.
    pub fn alloc_result(&self) -> Result<SegResult, SegError> {
        SegResult::alloc(&self.stream, self.q, self.mh, self.mw, self.tc)
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

        let out_ptr = |name: &str| -> Result<CUdeviceptr, SegError> {
            let p = tmap
                .get(name)
                .ok_or_else(|| SegError::MissingOutput(name.to_string()))?
                .f32_ptr()?;
            Ok(p as usize as CUdeviceptr)
        };
        let box_raw = out_ptr(&self.dets_name)?;
        let lab_raw = out_ptr(&self.labels_name)?;
        let msk_raw = out_ptr(&self.masks_name)?;

        // Per-frame device count (atomic). Zero the caller-owned counter (retained in
        // `SegResult` so GPU fusion readers can gate on it without a host sync);
        // survivors land in the caller's buffers; `qidx` maps each survivor slot back
        // to its query for the mask pass.
        self.stream.memset_zeros(&mut out.count_dev)?;
        let dets_raw = out.dets.device_ptr(self.stream.as_ref()).0;
        let qidx_raw = out.qidx.device_ptr(self.stream.as_ref()).0;
        let masks_out_raw = out.masks.device_ptr(self.stream.as_ref()).0;
        let cnt_raw = out.count_dev.device_ptr(self.stream.as_ref()).0;

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

        // Optional appearance pass: mask-pool the backbone token grid into a per-instance
        // L2-normed embedding (enqueued after the mask pass, so FIFO order guarantees it
        // sees this frame's masks + count). Zero `feats` first so stale slots read 0.
        if let Some(pool_k) = &self.pool_k {
            let tok_raw = out_ptr(self.tokens_name.as_deref().unwrap_or_default())?;
            self.stream.memset_zeros(&mut out.feats)?;
            let feats_raw = out.feats.device_ptr(self.stream.as_ref()).0;
            let (tci, tghi, tgwi) = (self.tc as i32, self.tgh as i32, self.tgw as i32);
            let (mmhi, mmwi) = (self.mh as i32, self.mw as i32);
            let cfg = LaunchConfig {
                grid_dim: (self.q as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (self.tgh * self.tgw * std::mem::size_of::<f32>()) as u32,
            };
            pool_k
                .launch_builder(&self.stream)
                .arg(&tok_raw)
                .arg(&tci)
                .arg(&tghi)
                .arg(&tgwi)
                .arg(&masks_out_raw)
                .arg(&mmhi)
                .arg(&mmwi)
                .arg(&cnt_raw)
                .arg(&feats_raw)
                .launch_cfg(cfg)?;
        }

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
