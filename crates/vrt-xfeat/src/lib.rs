//! XFeat local feature extraction and matching for TensorRT on Jetson Orin.
//!
//! # Modules
//! - [`postprocess`] — GPU kernels (NMS, descriptor sampling, L2 norm, matching) + `XFeatResult`
//! - [`model`]       — `XFeat` struct (backbone + postproc stage) + builder

pub mod model;
pub mod postprocess;

pub use model::{XFeat, XFeatParams};
pub use postprocess::{match_mutual_nn, TopkBufs, XFeatError, XFeatPostproc, XFeatResult};
pub use vrt_preproc::{Preprocessor, TextureGuard};
