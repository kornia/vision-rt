//! XFeat local feature extraction and matching for TensorRT on Jetson Orin.
//!
//! # Modules
//! - [`postprocess`] — GPU kernels (NMS, descriptor sampling, L2 norm, matching) + `XFeatResult`
//! - [`model`]       — `XFeat` struct (backbone + postproc stage) + builder

pub mod postprocess;
pub mod model;

pub use postprocess::{XFeatResult, XFeatPostproc, XFeatError, TopkBufs, match_mutual_nn};
pub use model::{XFeat, XFeatBuilder, XFeatParams, XFeatInferStage, XFeatPostprocStage};
pub use vrt_preproc::{Preprocessor, TextureGuard};
