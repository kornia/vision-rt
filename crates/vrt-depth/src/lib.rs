//! Monocular **depth refinement** on TensorRT: RGB frame + a raw depth map →
//! a refined depth map, all on the GPU.
//!
//! [`DepthRefine`] wraps a two-input engine exported from
//! [lingbot-depth-trt](https://github.com/Ar-Ray-code/lingbot-depth-trt): a
//! transformer that takes a color image and a coarse/raw depth map and predicts
//! a cleaned-up depth map. Unlike the single-input detectors in this workspace
//! (`vrt-rfdetr`, `vrt-xfeat`), the engine binds **two** device inputs by name,
//! so this crate drives it through [`ModelSession::run_inputs`].
//!
//! Model contract (from the export recipe, `tools/export_trt.py`):
//! - input `image`  `[1, 3, H, W]` f32, RGB in `[0,1]`, CHW — the ImageNet
//!   mean/std normalize is **baked into the graph**, so no extra normalize here.
//! - input `depth`  `[1, 1, H, W]` f32, raw metric depth.
//! - output `depth_refined` `[1, 1, H, W]` (or `[1, H, W]`) f32, refined depth.
//!
//! `H`/`W` are read from the engine's `image` input spec (the reference export
//! is 640×480), so any re-export resolution works unchanged.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use kornia_image::Image;
//! # use kornia_tensor::Tensor;
//! # use vrt::{Engine, Runtime, Logger, Stream, logger::Severity};
//! # fn main() -> Result<(), vrt::BoxError> {
//! let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
//! let engine = Engine::from_file(runtime, "lingbot-depth.engine")?;
//! let stream = Stream::new_standalone()?.cuda_stream().clone();
//! let mut refiner = vrt_depth::DepthRefine::new(engine, stream.clone())?;
//!
//! let (h, w) = refiner.model_hw();
//! # let rgb: Image<u8, 3> = todo!();          // device-resident RGB frame
//! # let depth: Tensor<f32, 4> = todo!();      // device [1,1,h,w] raw depth
//! let out = refiner.run(&rgb, &depth)?;       // syncs; refined depth on device
//! let refined = out.get(refiner.output_name()).unwrap();
//! assert_eq!(refined.shape_i64().last(), Some(&(w as i64)));
//! let _ = h;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use kornia_image::Image;
use kornia_imgproc::preprocess::Preprocessor;
use kornia_tensor::{zeros_cuda, Tensor};
use vrt::{BoxError, CudaStream, Engine, ModelSession, TRTensorMap};

/// A two-input depth-refinement model (RGB + raw depth → refined depth).
///
/// Owns a reused CHW `image` input tensor and a stretch [`Preprocessor`]; the
/// raw `depth` input is supplied per call by the caller (already at model
/// resolution). `run` is synchronous — it syncs the stream before returning, so
/// the returned device views are ready to read.
pub struct DepthRefine {
    model: ModelSession,
    preproc: Preprocessor,
    /// Reused device tensor for the preprocessed RGB input, `[1,3,H,W]`.
    image: Tensor<f32, 4>,
    stream: Arc<CudaStream>,
    image_name: String,
    depth_name: String,
    out_name: String,
    /// Expected `depth` input shape `[1,1,H,W]` — validated per call.
    depth_shape: [usize; 4],
    model_h: usize,
    model_w: usize,
}

impl DepthRefine {
    /// Build from a deserialized engine and a CUDA stream.
    ///
    /// The two inputs are bound by name: the input whose name contains `"image"`
    /// is the color tensor, the one containing `"depth"` is the raw depth
    /// tensor; the output containing `"depth"` is the refined map. Both inputs
    /// must be static 4-D (`[1,C,H,W]`), and both must agree on `H`/`W`.
    pub fn new(engine: Arc<Engine>, stream: Arc<CudaStream>) -> Result<Self, BoxError> {
        let (image_name, [_, ic, ih, iw]) = find_input(&engine, "image")?;
        let (depth_name, [dn, dc, dh, dw]) = find_input(&engine, "depth")?;
        if ic != 3 {
            return Err(format!("'image' input must be [1,3,H,W], got channels {ic}").into());
        }
        if (dn, dc) != (1, 1) {
            return Err(
                format!("'depth' input must be [1,1,H,W], got [{dn},{dc},{dh},{dw}]").into(),
            );
        }
        if (dh, dw) != (ih, iw) {
            return Err(
                format!("'image' {ih}×{iw} and 'depth' {dh}×{dw} inputs must share H×W").into(),
            );
        }

        let preproc = Preprocessor::stretch(stream.clone())?;
        let image = zeros_cuda::<f32, 4>([1, 3, ih, iw], &stream)?;
        let model = ModelSession::new(engine, stream.clone())?;

        let out_name = model
            .output_names()
            .iter()
            .find(|n| n.contains("depth"))
            .cloned()
            .ok_or("no depth output (expected a name containing 'depth')")?;

        Ok(Self {
            model,
            preproc,
            image,
            stream,
            image_name,
            depth_name,
            out_name,
            depth_shape: [dn, dc, dh, dw],
            model_h: ih,
            model_w: iw,
        })
    }

    /// Model input resolution `(H, W)` — size the `depth` input to match.
    pub fn model_hw(&self) -> (usize, usize) {
        (self.model_h, self.model_w)
    }

    /// Name of the refined-depth output binding (for `TRTensorMap::get`).
    pub fn output_name(&self) -> &str {
        &self.out_name
    }

    /// Refine `depth` using `rgb`: stretch-resize RGB → TRT inference → device
    /// outputs. Synchronous (syncs the stream internally).
    ///
    /// `rgb` is a device-resident color frame of any resolution (stretched to
    /// the model size). `depth` is a device-resident `[1,1,H,W]` f32 raw depth
    /// map already at the model resolution ([`model_hw`](Self::model_hw)). The
    /// refined depth stays on the device; read it via
    /// `out.get(self.output_name())`.
    pub fn run(
        &mut self,
        rgb: &Image<u8, 3>,
        depth: &Tensor<f32, 4>,
    ) -> Result<TRTensorMap, BoxError> {
        if depth.shape != self.depth_shape {
            return Err(format!(
                "depth input shape {:?} != expected {:?}",
                depth.shape, self.depth_shape
            )
            .into());
        }
        self.preproc.run(rgb, &mut self.image)?;
        let out = self.model.run_inputs(&[
            (self.image_name.as_str(), &self.image),
            (self.depth_name.as_str(), depth),
        ])?;
        // `run_inputs` only enqueues; sync so the returned device views are
        // valid for the caller to read (and so a bench times full inference).
        self.stream.synchronize()?;
        Ok(out)
    }
}

/// Find a static 4-D input whose name contains `needle`, returning its name and
/// `[N,C,H,W]` dims as `usize`.
fn find_input(engine: &Engine, needle: &str) -> Result<(String, [usize; 4]), BoxError> {
    let spec = engine
        .inputs()
        .find(|s| s.name.contains(needle))
        .ok_or_else(|| format!("no input containing '{needle}'"))?;
    let d = &spec.dims;
    if d.len() != 4 || d.iter().any(|&x| x <= 0) {
        return Err(format!("input '{}' must be static [N,C,H,W], got {d:?}", spec.name).into());
    }
    Ok((
        spec.name.clone(),
        [d[0] as usize, d[1] as usize, d[2] as usize, d[3] as usize],
    ))
}
