//! `XFeat` — GPU preprocessing + TRT backbone + GPU post-processing.

use crate::postprocess::{XFeatError, XFeatPostproc, XFeatResult};
use cudarc::driver::CudaSlice;
use kornia_image::Image;
use kornia_imgproc::preprocess::{PreprocessError, Preprocessor};
use kornia_tensor::{zeros_cuda, Tensor};
use std::sync::Arc;
use vrt::{BoxError, CudaStream, Engine, ModelSession};

// ── Params ────────────────────────────────────────────────────────────────────

/// Configuration for the XFeat feature extractor.
///
/// The backbone input size is NOT configured here — matching upstream XFeat, each
/// frame is resized to its own floor-of-32 dimensions (see [`XFeat::submit`]).
#[derive(Debug, Clone)]
pub struct XFeatParams {
    /// Maximum keypoints returned per frame.
    pub top_k: usize,
    /// Minimum NMS score for a keypoint candidate to be kept.
    pub threshold: f32,
}

impl XFeatParams {
    pub fn new(top_k: usize, threshold: f32) -> Self {
        Self { top_k, threshold }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// XFeat feature extractor: GPU resize/normalize + TRT backbone + GPU post-processing.
///
/// A single `Image<u8, 3> → XFeatResult` algorithm: it owns a kornia
/// [`Preprocessor`] in **stretch** mode and, matching upstream XFeat, resizes each
/// frame to its own floor-of-32 dimensions (`(H/32)*32 × (W/32)*32`), then rescales
/// keypoints back to original pixels. Callers hand it an image of any resolution.
///
/// Preprocess, TRT backbone, and post-processing all enqueue on the one CUDA
/// `stream` shared at construction — [`submit`](Self::submit) is fully async (no
/// sync) while the batch and frame size stay fixed; the caller owns the
/// `stream.synchronize()`, then reads the result.
pub struct XFeat {
    model: ModelSession,
    preproc: Preprocessor,
    postproc: XFeatPostproc,
    /// The one shared stream (== the backbone session's stream); used to (re)alloc
    /// the per-frame buffers so they are stream-ordered with all other GPU work.
    stream: Arc<CudaStream>,
    /// NMS score buffer, sized to the current model dims; reallocated on size change.
    score_dev: CudaSlice<f32>,
    /// Model input tensor `[B,3,mh,mw]` CHW FP32 device, written by the preprocessor.
    input: Tensor<f32, 4>,
    /// Batch and model dims `(B, mh, mw)` the buffers are currently sized for.
    cur: (usize, usize, usize),
    /// Keypoint capacity for results allocated by [`alloc_result`](Self::alloc_result).
    top_k: usize,
}

/// `image` input min/opt/max of the default engine profile: one frame of any size.
pub const ENGINE_SHAPES: [[i64; 4]; 3] = [[1, 3, 240, 320], [1, 3, 640, 640], [1, 3, 1088, 1920]];

/// `image` min/opt/max for a stereo engine serving [`XFeat::submit_pair`] on
/// `width × height` frames: batch 1..=2 at exactly that frame's floor-of-32 size.
///
/// Sized to the camera because a generic-opt batch-2 profile slows batch-1 runs
/// 13–16% on Orin; at the frame's size batch 1 is unaffected (see README, Stereo pairs).
pub fn stereo_shapes(width: usize, height: usize) -> [[i64; 4]; 3] {
    let (mw, mh) = ((width / 32 * 32) as i64, (height / 32 * 32) as i64);
    [[1, 3, mh, mw], [2, 3, mh, mw], [2, 3, mh, mw]]
}

/// Minimum model dimension (a multiple of 32) the reused buffers are seeded with
/// in [`XFeat::new`]; the first frame reallocates them to its real floor-32 size.
const SEED_DIM: usize = 32;

impl XFeat {
    /// Build an extractor sharing `stream` with the rest of the application
    /// (one CUDA stream so a single sync per frame covers all its GPU work,
    /// including matching via a `matching::Matcher` on the same stream).
    pub fn new(
        engine: Arc<Engine>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        let model = ModelSession::new(Arc::clone(&engine), Arc::clone(&stream))?;
        let preproc = Preprocessor::stretch(stream.clone())?;
        let postproc = XFeatPostproc::new(stream.clone(), params.threshold)?;
        // Seed the reused buffers at a minimal valid size so they're always live
        // (no Option/unwrap); the first frame reallocates them to its floor-32 size.
        let input = zeros_cuda::<f32, 4>([1, 3, SEED_DIM, SEED_DIM], &stream)?;
        let score_dev = stream.alloc_zeros::<f32>(SEED_DIM * SEED_DIM)?;
        Ok(XFeat {
            model,
            preproc,
            postproc,
            stream,
            score_dev,
            input,
            cur: (1, SEED_DIM, SEED_DIM),
            top_k: params.top_k,
        })
    }

    /// Allocate an output buffer sized for this extractor's `top_k`, to reuse
    /// across [`submit`](Self::submit) calls (VPI-style caller-owned output).
    pub fn alloc_result(&self) -> Result<XFeatResult, BoxError> {
        Ok(XFeatResult::alloc(&self.stream, self.top_k)?)
    }

    /// Construct from a prebuilt TensorRT `.engine` file (machine-locked to this
    /// TRT version + GPU arch). Creates its own `Logger`/`Runtime`; use
    /// [`new`](Self::new) when the application already owns an [`Engine`]. No
    /// `hub`/`builder` feature required.
    pub fn from_engine_file(
        engine_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        Self::new(Engine::load(engine_path)?, stream, params)
    }

    /// The default engine build profile — XFeat backbone: one frame, dynamic H×W
    /// input (downsampled ×8), fp16. See [`ENGINE_SHAPES`].
    #[cfg(any(feature = "hub", feature = "builder"))]
    fn engine_profile() -> vrt_hub::EngineProfile {
        Self::profile(ENGINE_SHAPES)
    }

    /// The stereo engine build profile for `width × height` frames (each side ≥ 32).
    #[cfg(any(feature = "hub", feature = "builder"))]
    fn stereo_profile(width: usize, height: usize) -> Result<vrt_hub::EngineProfile, XFeatError> {
        if width < 32 || height < 32 {
            return Err(XFeatError::InputTooSmall(width, height));
        }
        Ok(Self::profile(stereo_shapes(width, height)))
    }

    #[cfg(any(feature = "hub", feature = "builder"))]
    fn profile([min, opt, max]: [[i64; 4]; 3]) -> vrt_hub::EngineProfile {
        vrt_hub::EngineProfile {
            inputs: vec![("image".into(), min.to_vec(), opt.to_vec(), max.to_vec())],
            fp16: true,
            bf16: false,
            workspace_mb: 2048,
        }
    }

    /// Build (and cache) an engine from an ONNX file, then construct. First call
    /// builds on-device (~1–5 min); later calls are cache hits keyed by ONNX
    /// content + TRT version + GPU arch. Requires feature `hub` (trtexec build)
    /// or `builder` (in-process).
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
    ) -> Result<Self, BoxError> {
        Self::from_onnx_profile(onnx_path, stream, params, &Self::engine_profile())
    }

    /// [`from_onnx`](Self::from_onnx) with a stereo engine for `width × height` frames
    /// (each side ≥ 32). The engine accepts only `width/32*32 × height/32*32` model
    /// input, for [`submit`](Self::submit) as well as [`submit_pair`](Self::submit_pair).
    #[cfg(any(feature = "hub", feature = "builder"))]
    pub fn from_onnx_stereo(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
        width: usize,
        height: usize,
    ) -> Result<Self, BoxError> {
        let profile = Self::stereo_profile(width, height)?;
        Self::from_onnx_profile(onnx_path, stream, params, &profile)
    }

    #[cfg(any(feature = "hub", feature = "builder"))]
    fn from_onnx_profile(
        onnx_path: impl AsRef<std::path::Path>,
        stream: Arc<CudaStream>,
        params: XFeatParams,
        profile: &vrt_hub::EngineProfile,
    ) -> Result<Self, BoxError> {
        let model_path = onnx_path
            .as_ref()
            .to_str()
            .ok_or_else(|| BoxError::from("onnx path is not valid UTF-8"))?;
        let engine_path =
            vrt_hub::EngineCache::default().resolve("xfeat-backbone", model_path, profile)?;
        Self::from_engine_file(engine_path, stream, params)
    }

    /// Construct from Hugging Face (`kornia/xfeat`). Requires feature `hub`.
    ///
    /// Prefers a prebuilt engine matching this box's TRT+SM (skips the on-device
    /// build), else pulls the pinned ONNX and builds/caches it. Network only on
    /// the first run. For a private/gated HF repo set `HF_TOKEN`.
    #[cfg(feature = "hub")]
    pub fn from_hub(stream: Arc<CudaStream>, params: XFeatParams) -> Result<Self, BoxError> {
        let engine = vrt_hub::resolve_engine("xfeat-backbone", &Self::engine_profile())?;
        Self::from_engine_file(engine, stream, params)
    }

    /// [`from_hub`](Self::from_hub) with a stereo engine for `width × height` frames
    /// (built on-device on first use; no prebuilt is published). The engine accepts
    /// only `width/32*32 × height/32*32` model input, for [`submit`](Self::submit) as
    /// well as [`submit_pair`](Self::submit_pair).
    #[cfg(feature = "hub")]
    pub fn from_hub_stereo(
        stream: Arc<CudaStream>,
        params: XFeatParams,
        width: usize,
        height: usize,
    ) -> Result<Self, BoxError> {
        let profile = Self::stereo_profile(width, height)?;
        let engine = vrt_hub::resolve_engine("xfeat-backbone", &profile)?;
        Self::from_engine_file(engine, stream, params)
    }

    /// Submit one frame's async GPU work — resize/normalize → backbone → NMS →
    /// top-K — into the caller-owned `out`, all enqueued on the shared stream with
    /// **no sync** (VPI-style). Sync the stream once (covering any other work on
    /// it), then read `out` (its `count()`/`kpts_to_host` are valid after the
    /// sync). Reuse one `out` per frame, or hold several to keep multiple frames
    /// outstanding.
    pub fn submit(&mut self, img: &Image<u8, 3>, out: &mut XFeatResult) -> Result<(), XFeatError> {
        self.submit_batch(&[img], &mut [out])
    }

    /// Submit a same-size stereo pair as **one** batch-2 backbone run — every
    /// stage (preprocess, TensorRT, NMS, top-K, sampling) is enqueued once for
    /// both images instead of twice. Same async contract as
    /// [`submit`](Self::submit): sync the stream once, then read both results.
    ///
    /// Needs a stereo engine ([`from_onnx_stereo`](Self::from_onnx_stereo),
    /// [`stereo_shapes`]) for this frame size; the default engine is batch 1 and
    /// TensorRT rejects the batch-2 shape.
    ///
    /// Switching between `submit` and `submit_pair` (or changing frame size) on one
    /// instance drains the stream and reallocates the buffers; keep one `XFeat` per
    /// batch size if you mix them.
    pub fn submit_pair(
        &mut self,
        left: &Image<u8, 3>,
        right: &Image<u8, 3>,
        out_left: &mut XFeatResult,
        out_right: &mut XFeatResult,
    ) -> Result<(), XFeatError> {
        if (left.width(), left.height()) != (right.width(), right.height()) {
            return Err(XFeatError::StereoSizeMismatch(
                left.width(),
                left.height(),
                right.width(),
                right.height(),
            ));
        }
        self.submit_batch(&[left, right], &mut [out_left, out_right])
    }

    fn submit_batch(
        &mut self,
        imgs: &[&Image<u8, 3>],
        outs: &mut [&mut XFeatResult],
    ) -> Result<(), XFeatError> {
        // Validate before anything is enqueued or reallocated.
        self.postproc.check_outs(outs)?;
        let batch = imgs.len();
        // Upstream XFeat: resize to floor-of-32 dims, keypoints scaled back by (rw,rh).
        let (sw, sh) = (imgs[0].width(), imgs[0].height());
        let (mw, mh) = ((sw / 32) * 32, (sh / 32) * 32);
        if mw == 0 || mh == 0 {
            return Err(XFeatError::InputTooSmall(sw, sh));
        }
        let (rw, rh) = (sw as f32 / mw as f32, sh as f32 / mh as f32);

        // A model-size change reconfigures the execution context: `set_input_shape` and
        // the output-buffer reallocation are host-side calls, not stream-ordered, so
        // performing them while a previous `enqueue_v3` is still in flight mutates a live
        // context and frees buffers it is reading. Draining here makes every caller safe
        // by construction. Costs one sync only when the shape (size or batch) changes.
        if self.cur != (batch, mh, mw) {
            self.stream.synchronize()?;
            self.input = zeros_cuda::<f32, 4>([batch, 3, mh, mw], &self.stream)?;
            self.score_dev = self.stream.alloc_zeros::<f32>(batch * mh * mw)?;
            self.cur = (batch, mh, mw);
        }

        // Preprocess (stretch resize + /255) into the reused input, then backbone.
        let frames = imgs
            .iter()
            .map(|i| i.as_cudaslice().ok_or(PreprocessError::NotDeviceImage))
            .collect::<Result<Vec<_>, _>>()?;
        self.preproc
            .run_raw_batch(&frames, sw, sh, &mut self.input)?;
        let tmap = self.model.run(&self.input)?;
        let desc_ptr = tmap
            .get("descriptors")
            .ok_or(XFeatError::MissingOutput("descriptors"))?
            .f32_ptr()?;
        let heat_ptr = tmap
            .get("heatmap")
            .ok_or(XFeatError::MissingOutput("heatmap"))?
            .f32_ptr()?;
        let rel_ptr = tmap
            .get("reliability")
            .ok_or(XFeatError::MissingOutput("reliability"))?
            .f32_ptr()?;
        self.postproc
            .launch_score_nms(heat_ptr, rel_ptr, &self.score_dev, batch, mh, mw)?;
        // Before the launch: the post-processing bakes the scale into `kpts_px`.
        for out in outs.iter_mut() {
            out.set_scale((rw, rh));
        }
        self.postproc
            .launch_topk_batch(desc_ptr, &self.score_dev, mh, mw, outs)?;
        Ok(())
    }
}
