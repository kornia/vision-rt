//! XFeat local feature extraction and matching for TensorRT on Jetson Orin.
//!
//! # Modules
//! - [`postprocess`] — GPU kernels (NMS, top-K, descriptor sampling, L2 norm) + `XFeatResult`
//! - [`matching`]    — GPU mutual-NN descriptor matcher (`Matcher`), decoupled from postproc
//! - [`model`]       — `XFeat` extractor (preproc + backbone + postproc)

pub mod matching;
pub mod model;
pub mod postprocess;

pub use kornia_imgproc::preprocess::Preprocessor;
pub use matching::{MatchResult, Matcher};
pub use model::{XFeat, XFeatParams};
pub use postprocess::{XFeatError, XFeatPostproc, XFeatResult};
