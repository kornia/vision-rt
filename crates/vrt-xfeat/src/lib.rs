//! XFeat local feature extraction and matching for TensorRT on Jetson Orin.
//!
//! # Modules
//! - [`postprocess`] — GPU kernels (NMS, descriptor sampling, L2 norm, matching) + `XFeatResult`
//! - [`model`]       — `XFeat` extractor (preproc + backbone + postproc)

pub mod model;
pub mod postprocess;

pub use kornia_imgproc::preprocess::Preprocessor;
pub use model::{XFeat, XFeatParams, XFeatPending};
pub use postprocess::{TopkBufs, XFeatError, XFeatPostproc, XFeatResult};
