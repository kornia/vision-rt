//! Model weights distribution + on-device TensorRT engine cache.
//!
//! ## The shipping model
//! - **ONNX weights** are the portable artifact, hosted on Hugging Face Hub
//!   and pinned by sha256 — never committed to the repo (feature `hub`).
//! - **Engines** are machine-locked (TRT version + GPU arch). They are normally
//!   built **on-device** from ONNX into a versioned cache. A registry MAY also
//!   list prebuilt engines for exact-match environments; [`ModelHub::get_engine`]
//!   downloads one only when its `trt_version` + `sm` match the local box **and its
//!   [`Precision`] matches what the caller's [`EngineProfile`] asks for**, so a
//!   mismatched engine is never fetched — it falls back to an on-device build.
//!   Precision is part of that guard because it changes the engine's *outputs*, not
//!   just its speed: an fp16 artifact served to a bf16 request deserializes fine and
//!   then returns wrong numbers.
//!
//! ```no_run
//! use vrt_hub::{ModelHub, EngineCache, EngineProfile};
//!
//! // Resolve weights: explicit local path, or HF Hub download (feature "hub").
//! let onnx = ModelHub::get("xfeat-backbone")?;          // hub download + sha256 verify
//! // let onnx = std::path::PathBuf::from("my.onnx");    // offline path also fine
//!
//! // Engine: cache hit returns instantly; miss builds on-device (~minutes, once).
//! let profile = EngineProfile {
//!     inputs: vec![("image".into(),
//!                   vec![1,3,240,320], vec![1,3,640,640], vec![1,3,1088,1920])],
//!     fp16: true,
//!     bf16: false,   // transformers want bf16 instead — see EngineProfile::bf16
//!     workspace_mb: 2048,
//! };
//! let engine_path = EngineCache::default().get_or_build("xfeat-backbone", &onnx, &profile)?;
//! # Ok::<(), vrt::BoxError>(())
//! ```

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
#[cfg(not(feature = "builder"))]
use std::process::Command;

use sha2::{Digest, Sha256};

/// Re-exported from `vrt` core, where it sits alongside `DType` so the builder and
/// this registry reason about precision through one type.
pub use vrt::Precision;

/// Errors from model resolution and engine building.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("unknown model '{0}' — see vrt_hub::REGISTRY for known names")]
    UnknownModel(String),
    #[error("invalid model name '{0}' — must be a single path component (no '/', '\\', or '..')")]
    InvalidName(String),
    #[error("model '{0}' has no files in the registry")]
    EmptyModel(String),
    #[error("sha256 mismatch for {path}: expected {expected}, got {actual} (corrupted download? delete and retry)")]
    Sha256Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[cfg(feature = "hub")]
    #[error("Hugging Face Hub: {0}")]
    Hf(#[from] hf_hub::api::sync::ApiError),
    #[cfg(not(feature = "hub"))]
    #[error("vrt-hub built without the 'hub' feature — enable it to download '{0}', or pass an explicit ONNX path")]
    HubFeatureDisabled(String),
    #[error(transparent)]
    Trt(#[from] vrt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("engine build: {0}")]
    Build(String),
}

// ── Model registry ────────────────────────────────────────────────────────────

/// One file inside a hub model: name + sha256 pin.
pub struct ModelFile {
    pub filename: &'static str,
    pub sha256: &'static str,
}

/// An OPTIONAL prebuilt TensorRT engine, guarded by the exact environment it was
/// serialized for. An engine only deserializes on a matching `trt_version` +
/// GPU `sm` (compute capability), and is only *correct* for a caller wanting the
/// same `precision`; it is downloaded only when all three match, otherwise the
/// engine is built on-device from the ONNX instead.
pub struct EngineArtifact {
    pub filename: &'static str, // e.g. "xfeat_backbone-trt10.3.0.30-sm87-fp16.engine"
    pub sha256: &'static str,
    pub trt_version: &'static str, // must equal `vrt::TENSORRT_VERSION`, e.g. "10.3.0.30"
    pub sm: &'static str,          // GPU compute capability, e.g. "87"
    /// What this artifact was built at — must equal the requesting
    /// [`EngineProfile::precision`] or the prebuilt is skipped in favour of an
    /// on-device build.
    pub precision: Precision,
    /// The optimization profile this artifact was BUILT at, as
    /// [`EngineProfile::shape_tag`] renders it — e.g.
    /// `"images:1x3x256x256|2x3x512x512|2x3x640x640"`, or `""` for a static-shape
    /// engine.
    ///
    /// Guarded because a TensorRT engine only accepts shapes inside the profile it was
    /// built with. Without this an engine built for one resolution range is served to a
    /// caller asking for another, deserializes happily, and then rejects perfectly
    /// ordinary frames at `setInputShape` — with an empty TensorRT message. The failure
    /// looks like a bug in the caller's code, not a mismatched download.
    pub shape_profile: &'static str,
}

/// A distributable model: where it lives on the Hub and what it contains.
///
/// `files[0]` is the entry-point .onnx; the rest are sidecars (e.g.
/// `.onnx.data` external weights) that must land in the same directory.
/// `engines` are optional prebuilt engines for exact-match environments (a
/// convenience that skips the on-device build); empty = always build from ONNX.
pub struct ModelSpec {
    pub name: &'static str,
    pub hf_repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [ModelFile],
    pub engines: &'static [EngineArtifact],
}

/// Static registry of known models.
///
/// To add a model: export ONNX (scripts/), upload to the HF repo, add an
/// entry here with `sha256sum` pins.
pub static REGISTRY: &[ModelSpec] = &[
    // ── RaCo-ALIKED extractor ─────────────────────────────────────────────────
    //
    // The extractor half of fabio-sim/LightGlue-ONNX's fused RaCo-ALIKED-LightGlue+
    // export, cut out by crates/vrt-raco-aliked/scripts/split_raco_pipeline.py.
    // Three separately licensed upstreams are combined in these weights — RaCo
    // (Apache-2.0), ALIKED (BSD-3-Clause, which requires attribution in binary form)
    // and the export tooling (Apache-2.0); the HF repo carries the notice.
    //
    // `K` is baked into each export and is NOT just a keypoint budget: at K >= 3072
    // RaCo's learned ranker is omitted, which halves extraction cost while returning
    // 3x the keypoints. Hence one entry per K rather than a single default.
    //
    // Engines are pinned to the shape profile RaCoAliked::engine_profile() declares
    // (min 1x3x256x256 / opt 2x3x512x512 / max 2x3x640x640). pick_artifact matches on
    // trt_version + sm + precision ONLY and does not check the profile, so an engine
    // built at any other profile would be served here and then reject frames it cannot
    // handle. Do not add one without matching the declared profile.
    ModelSpec {
        name: "raco-aliked-extractor-k3072",
        hf_repo: "kornia/raco-aliked",
        revision: "main",
        files: &[ModelFile {
            filename: "raco_aliked_extractor_k3072.onnx",
            sha256: "acafa7a3d54e9aa5fb1ec0c20f593e2b6879c5570fde65819a5119301eef680f",
        }],
        engines: &[EngineArtifact {
            filename: "raco-aliked-extractor-k3072-trt10.3.0.30-sm87-fp16.engine",
            sha256: "ca529ee8ab2e3dc919f8201331d5d4d44a300a61e8ffef4ec6f243a82e53bffc",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "images:1x3x256x256|2x3x512x512|2x3x640x640",
        }],
    },
    ModelSpec {
        name: "raco-aliked-extractor-k1024",
        hf_repo: "kornia/raco-aliked",
        revision: "main",
        files: &[ModelFile {
            filename: "raco_aliked_extractor_k1024.onnx",
            sha256: "33d788905259eba848f77a099b30089c425f72f7afb9c07f67cc79e8ad2949a6",
        }],
        engines: &[EngineArtifact {
            filename: "raco-aliked-extractor-k1024-trt10.3.0.30-sm87-fp16.engine",
            sha256: "b7f8b972118bcf46215ddc613d637c29cfa49bf5451cb3a531d67444a6aa3073",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "images:1x3x256x256|2x3x512x512|2x3x640x640",
        }],
    },
    ModelSpec {
        name: "raco-aliked-extractor-k512",
        hf_repo: "kornia/raco-aliked",
        revision: "main",
        files: &[ModelFile {
            filename: "raco_aliked_extractor_k512.onnx",
            sha256: "83209276023a54ff4e820cb69900aed4ecc3cc39b072411ff20fe0062e4c2026",
        }],
        engines: &[],
    },
    // ── LightGlue+ matcher ────────────────────────────────────────────────────
    //
    // The matcher half of the same split. Its engines are pinned to
    // LightGlue::engine_profile(k)'s declared min = opt = max of 2x1xKx2 and
    // 2x1xKx128, for the same reason as above.
    //
    // Matching cost is O(K^2), so unlike the extractor it gets rapidly worse with K:
    // 7.9 ms at k512, 21.6 ms at k1024, 126.5 ms at k3072 on an Orin Nano.
    ModelSpec {
        name: "lightglue-matcher-k3072",
        hf_repo: "kornia/lightglue",
        revision: "main",
        files: &[ModelFile {
            filename: "lightglue_matcher_k3072.onnx",
            sha256: "8479ea7ed24b18e8f5f0ccbb4d09d0942c51185ebb1bf429204e6a26f87d46b0",
        }],
        engines: &[EngineArtifact {
            filename: "lightglue-matcher-k3072-trt10.3.0.30-sm87-fp16.engine",
            sha256: "52dcff20850492a90b82b35952c6b4c7b61458d3a79d9b96e5c28c6f05f18e8f",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "descriptors:2x1x3072x128|2x1x3072x128|2x1x3072x128;normalized_keypoints:2x1x3072x2|2x1x3072x2|2x1x3072x2",
        }],
    },
    ModelSpec {
        name: "lightglue-matcher-k1024",
        hf_repo: "kornia/lightglue",
        revision: "main",
        files: &[ModelFile {
            filename: "lightglue_matcher_k1024.onnx",
            sha256: "0b95d616137367a50b5b1c656672a6375ce6facf6a68040744c4dec2fabab499",
        }],
        engines: &[EngineArtifact {
            filename: "lightglue-matcher-k1024-trt10.3.0.30-sm87-fp16.engine",
            sha256: "090987df46a1f969b386affa10f7f286fb30f9d5dbb9d47d7603c66a554e28de",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "descriptors:2x1x1024x128|2x1x1024x128|2x1x1024x128;normalized_keypoints:2x1x1024x2|2x1x1024x2|2x1x1024x2",
        }],
    },
    ModelSpec {
        name: "lightglue-matcher-k512",
        hf_repo: "kornia/lightglue",
        revision: "main",
        files: &[ModelFile {
            filename: "lightglue_matcher_k512.onnx",
            sha256: "4e5348ecffd09e428ae3368a44e5b6c8800a45ae2d2b3a72691a6504dc2138fe",
        }],
        engines: &[EngineArtifact {
            filename: "lightglue-matcher-k512-trt10.3.0.30-sm87-fp16.engine",
            sha256: "9eea3c1455bdbcc4e5d618a5b4ecc1a1af9bd8c94b6d7ef090d6860d360c0a9c",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "descriptors:2x1x512x128|2x1x512x128|2x1x512x128;normalized_keypoints:2x1x512x2|2x1x512x2|2x1x512x2",
        }],
    },
    ModelSpec {
        // Source: XFeat (Potje et al., CVPR 2024) — https://github.com/verlab/accelerated_features
        // The .onnx is a backbone-only export of the upstream `xfeat.pt`, produced by
        // crates/vrt-xfeat/scripts/export_xfeat_backbone.py. Model credit is the authors'.
        name: "xfeat-backbone",
        hf_repo: "kornia/xfeat",
        revision: "main",
        files: &[
            ModelFile {
                filename: "xfeat_backbone.onnx",
                sha256: "86d7d549b380405f208933efb5202e1584d9762f3a72e06e7ed81ca1436972e0",
            },
            ModelFile {
                filename: "xfeat_backbone.onnx.data",
                sha256: "d4498528d37bf7c737cce9c135f9b0340d828bab7dc808339e50553ac8c1b7d9",
            },
        ],
        engines: &[EngineArtifact {
            filename: "xfeat_backbone-trt10.3.0.30-sm87-fp16.engine",
            sha256: "2190ad0e8daf7356708f91a2c18b89fa481082646c79b25fab91f5af6a912e6d",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "image:1x3x240x320|1x3x640x640|1x3x1088x1920",
        }],
    },
    ModelSpec {
        // RF-DETR (NMS-free transformer detector). Fixed-resolution official export
        // (input [1,3,512,512]) + a prebuilt engine for this Orin config (trt+sm
        // guarded; other boxes build from the ONNX on-device).
        name: "rfdetr",
        hf_repo: "kornia/rfdetr",
        revision: "main",
        files: &[ModelFile {
            filename: "rf-detr-small.onnx",
            sha256: "0e0817f4cafa479ccba17662a142092932b0b10c98947e7cf60f3badd0f5c219",
        }],
        engines: &[EngineArtifact {
            filename: "rf-detr-small-trt10.3.0.30-sm87-fp16.engine",
            sha256: "0caa4fa8c1852d22ed044e6a4d8c87f7695538ed38a555b7a41eb15ef0833181",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "",
        }],
    },
    ModelSpec {
        // RF-DETR Keypoint (human pose): box + 17 COCO keypoints. Fixed-resolution
        // export (input [1,3,576,576]) + a prebuilt engine for this Orin config
        // (trt+sm guarded; other boxes build from the ONNX on-device). Shares the
        // kornia/rfdetr HF repo with the detector (distinct filenames).
        name: "rfdetr-kpts",
        hf_repo: "kornia/rfdetr",
        revision: "main",
        files: &[ModelFile {
            filename: "rfdetr-keypoint-preview-folded.onnx",
            sha256: "d969cac0266cbbd335bc818ea186d6f91ad7d5730002b40d8287651abc95b406",
        }],
        engines: &[EngineArtifact {
            filename: "rfdetr-keypoint-preview-trt10.3.0.30-sm87-fp16.engine",
            sha256: "0c4595bf0689ba509a33be9ab7eea02320a57bc2b59ad33f694c181e4bd54cf2",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "",
        }],
    },
    ModelSpec {
        // RF-DETR Segmentation (instance masks): box + class + per-instance mask.
        // Fixed-resolution export (input [1,3,432,432]) + a prebuilt engine for this
        // Orin config (trt+sm guarded; other boxes build from the ONNX on-device).
        // Shares the kornia/rfdetr HF repo with the detector (distinct filenames).
        name: "rfdetr-seg",
        hf_repo: "kornia/rfdetr",
        revision: "main",
        files: &[ModelFile {
            filename: "rfdetr-seg-preview.onnx",
            sha256: "82c5c032cf5e7c97d00dff59b72f67cc8f8f0a481b350193bba29cb7fe51c111",
        }],
        engines: &[EngineArtifact {
            filename: "rfdetr-seg-preview-trt10.3.0.30-sm87-fp16.engine",
            sha256: "57582be75a56411ffe1900165d2dc7860e8709498bbd5b669cda4f88a673d753",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "",
        }],
    },
    ModelSpec {
        // Depth Anything V2 Metric-Small, indoor (Hypersim, ~20 m) — dense metric
        // depth. Fixed-resolution export (input [1,3,392,392]) + a prebuilt engine
        // for this Orin config (trt+sm guarded; other boxes build from the ONNX
        // on-device).
        name: "depth-anything-v2-metric-small",
        hf_repo: "kornia/depth-anything",
        revision: "main",
        files: &[ModelFile {
            filename: "depth-anything-v2-metric-small-indoor.onnx",
            sha256: "50dbcac7a6d667e365a3ceffdf51cc497aa5e06b6d2c1d5824c252640fbf5bf3",
        }],
        engines: &[EngineArtifact {
            filename: "depth-anything-v2-metric-small-indoor-trt10.3.0.30-sm87-fp16.engine",
            sha256: "a6255e66b01b11239dff9045df25e278f06afc7c2d63691ced8be1eecafca655",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: "",
        }],
    },
    ModelSpec {
        // OSNet x0.25 person re-id (torchreid `osnet_x0_25_msmt17`, MSMT17). Fixed-batch
        // export (input [16,3,256,128]) + a prebuilt engine for this Orin config (trt+sm
        // guarded; other boxes build from the ONNX on-device). The engine filename
        // carries the ONNX sha256 prefix (`e78604f4`) it was built from.
        name: "osnet-reid",
        hf_repo: "kornia/osnet",
        revision: "main",
        files: &[ModelFile {
            filename: "osnet_x0_25_msmt17.onnx",
            sha256: "e78604f4ccda49b8f41cd0f8f7303800ce75d2361895ebb0729513c1bf53d277",
        }],
        engines: &[EngineArtifact {
            filename: "osnet-reid-e78604f4-trt10.3.0.30-sm87.engine",
            sha256: "893b4dae9af84aeb0d6309e8a19fc06f3b759ee40a82f8af091729aa686166e3",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp32,
            shape_profile: "",
        }],
    },
    ModelSpec {
        // DINOv3 ViT-S/16 (Siméoni et al., Meta AI) — global CLS descriptors. Static
        // square export (input [1,3,336,336]) via crates/vrt-dinov3/scripts/export_dinov3.py.
        //
        // Redistribution is permitted: Meta's DINOv3 Licence grants the right to
        // "distribute, copy, create derivative works", and an ONNX export is a derivative
        // work — provided it ships under the same Agreement with a copy attached. The
        // kornia/dinov3 repo carries LICENSE.md and attribution for exactly that reason.
        //
        // The pins are of the artifacts produced by `scripts/export_dinov3.py
        // --input-size 336` on torch 2.11 / transformers 4.57.6, and match the uploaded
        // files (verified end-to-end through `from_hub`). Re-exporting under a different
        // stack yields different bytes, which fails closed (Sha256Mismatch) as it should.
        //
        // The `.onnx.data` sidecar is REQUIRED, not optional: torch's dynamo exporter
        // externalizes weights regardless of the 2 GB protobuf limit, so the .onnx alone
        // is a 1.2 MB graph with no weights. Entry ONNX first, sidecar after — same shape
        // as the xfeat-backbone entry above; the parser resolves it next to the .onnx.
        //
        // The prebuilt engine is **bf16**, matching DinoV3::engine_profile(), and its
        // filename carries the ONNX sha prefix (95753abe) it was built from.
        //
        // Never add an fp16 prebuilt here: fp16 is 2.56x faster on Orin but emits
        // all-NaN for this model (attention logits reach 2.1e6 vs fp16's 65504 ceiling).
        // `get_engine` now also matches on `precision`, so an fp16 artifact would simply
        // never be served to this crate's bf16 profile — but it would still be a trap for
        // anyone who flipped `engine_profile()` without re-running the parity test.
        name: "dinov3-vits16-336",
        hf_repo: "kornia/dinov3",
        revision: "main",
        files: &[
            ModelFile {
                filename: "dinov3-vits16-336.onnx",
                sha256: "95753abe620709f2df043e682a333ba29f6827bab493d4d3cb027dd2fe3b71f6",
            },
            ModelFile {
                filename: "dinov3-vits16-336.onnx.data",
                sha256: "73961e69729798b4e58f761051f88d099eadf869eae467ae7a3e8d26b1271db4",
            },
        ],
        engines: &[EngineArtifact {
            filename: "dinov3-vits16-336-95753abe-trt10.3.0.30-sm87-bf16.engine",
            sha256: "86d2e7853a8d98bb445a808056e8c868a057786e4ab8714ed97214c592ed04c5",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Bf16,
            shape_profile: "",
        }],
    },
];

/// Look up a model spec by name.
pub fn spec(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().find(|m| m.name == name)
}

// ── ModelHub (feature = "hub") ────────────────────────────────────────────────

/// Downloads pinned ONNX weights from Hugging Face Hub.
pub struct ModelHub;

impl ModelHub {
    /// Resolve a registry model to a local .onnx path.
    ///
    /// With feature `hub`: downloads via hf-hub into the standard HF cache
    /// (`~/.cache/huggingface`), verifies every file against its sha256 pin,
    /// and returns the entry-point path.  Re-runs are cache hits (no network).
    ///
    /// Without the feature this returns an error — pass explicit paths instead.
    #[cfg(feature = "hub")]
    pub fn get(name: &str) -> Result<PathBuf, HubError> {
        let spec = spec(name).ok_or_else(|| HubError::UnknownModel(name.into()))?;

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            spec.hf_repo.to_string(),
            hf_hub::RepoType::Model,
            spec.revision.to_string(),
        ));

        let mut entry: Option<PathBuf> = None;
        for f in spec.files {
            let path = repo.get(f.filename)?;
            if let Err(e) = verify_sha256(&path, f.sha256) {
                // Remove the bad file so a retry re-downloads instead of
                // re-verifying the same corrupt bytes from the HF cache forever.
                let _ = fs::remove_file(&path);
                return Err(e);
            }
            if entry.is_none() {
                entry = Some(path);
            }
        }
        entry.ok_or_else(|| HubError::EmptyModel(name.into()))
    }

    #[cfg(not(feature = "hub"))]
    pub fn get(name: &str) -> Result<PathBuf, HubError> {
        Err(HubError::HubFeatureDisabled(name.into()))
    }

    /// Try to fetch a prebuilt engine matching THIS box (feature `hub`).
    ///
    /// Returns `Ok(Some(path))` only when the registry lists an engine whose
    /// `trt_version` + `sm` equal the local TensorRT version and GPU compute
    /// capability — the sole configuration on which a serialized engine will
    /// deserialize. Otherwise `Ok(None)`: the caller should build from the ONNX.
    /// The downloaded engine is verified against its sha256 pin.
    #[cfg(feature = "hub")]
    pub fn get_engine(name: &str, profile: &EngineProfile) -> Result<Option<PathBuf>, HubError> {
        let spec = spec(name).ok_or_else(|| HubError::UnknownModel(name.into()))?;
        if spec.engines.is_empty() {
            return Ok(None);
        }
        profile.validate()?;
        let (local_trt, local_sm) = (vrt::TENSORRT_VERSION, compute_capability()?);
        let art = match pick_artifact(
            spec.engines,
            local_trt,
            &local_sm,
            profile.precision(),
            &profile.shape_tag(),
        ) {
            Some(a) => a,
            None => return Ok(None), // no matching prebuilt for this env → build from ONNX
        };

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            spec.hf_repo.to_string(),
            hf_hub::RepoType::Model,
            spec.revision.to_string(),
        ));
        let path = repo.get(art.filename)?;
        if let Err(e) = verify_sha256(&path, art.sha256) {
            let _ = fs::remove_file(&path);
            return Err(e);
        }
        Ok(Some(path))
    }

    #[cfg(not(feature = "hub"))]
    pub fn get_engine(_name: &str, _profile: &EngineProfile) -> Result<Option<PathBuf>, HubError> {
        Ok(None)
    }
}

/// Resolve a registry model to a usable engine path (feature `hub`): a matching
/// prebuilt engine if the registry lists one for this box's TRT+SM, otherwise the
/// pinned ONNX downloaded and built/cached on-device. This is the one call behind
/// each model crate's `from_hub` constructor.
#[cfg(feature = "hub")]
pub fn resolve_engine(name: &str, profile: &EngineProfile) -> Result<String, HubError> {
    if let Some(engine) = ModelHub::get_engine(name, profile)? {
        return Ok(engine.to_string_lossy().into_owned());
    }
    let onnx = ModelHub::get(name)?;
    EngineCache::default().resolve(name, &onnx.to_string_lossy(), profile)
}

/// Pick the prebuilt engine usable on this box for a given precision, if any.
///
/// All three of `trt_version`, `sm` and `precision` must match. The first two are
/// about whether the engine *deserializes*; the third is about whether it is
/// *right*. Precision changes an engine's outputs, so a mismatched artifact loads
/// cleanly and then returns wrong numbers — for DINOv3, an fp16 engine served to a
/// bf16 request is all-NaN, with no error anywhere to attribute it to. Returning
/// `None` falls back to an on-device build at the requested precision: slower, but
/// correct.
///
/// Split out from `get_engine` so the matching rule is testable without a network
/// round-trip or a GPU — hence `test` in the cfg: `get_engine` itself is hub-only.
#[cfg(any(feature = "hub", test))]
/// Select a prebuilt engine that is usable *as-is* for this request.
///
/// All four guards matter, and each one exists because violating it fails silently or
/// confusingly rather than loudly: `trt_version` and `sm` because an engine will not
/// deserialize elsewhere, `precision` because an fp16 artifact answering a bf16 request
/// returns wrong numbers, and `shape_profile` because an engine only accepts shapes
/// inside the profile it was built with. A miss is not an error — it just means the
/// engine gets built on-device from the ONNX instead.
fn pick_artifact<'a>(
    engines: &'a [EngineArtifact],
    trt: &str,
    sm: &str,
    want: Precision,
    shapes: &str,
) -> Option<&'a EngineArtifact> {
    engines.iter().find(|e| {
        e.trt_version == trt && e.sm == sm && e.precision == want && e.shape_profile == shapes
    })
}

/// Verify a file against an expected sha256 hex digest.
pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), HubError> {
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(HubError::Sha256Mismatch {
            path: path.to_path_buf(),
            expected: expected.into(),
            actual,
        });
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, HubError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

// ── EngineCache ───────────────────────────────────────────────────────────────

/// `(input_name, min_dims, opt_dims, max_dims)` for a dynamic-shape input.
pub type ShapeProfile = (String, Vec<i64>, Vec<i64>, Vec<i64>);

/// Optimization profile + build options for an engine.
pub struct EngineProfile {
    /// One profile per dynamic-shape input; empty = static shapes.
    ///
    /// Multi-input models (e.g. a matcher taking keypoints *and* descriptors) need a
    /// profile for each. The trtexec path supports any number; the in-process
    /// `builder` path is limited to one and errors above that.
    pub inputs: Vec<ShapeProfile>,
    pub fp16: bool,
    /// Enable BF16 kernels (Ampere+/SM80+, which includes the Orin's SM87).
    ///
    /// **The right choice for transformers.** FP16 caps at 65504 and ViT
    /// attention logits exceed it (measured 2.1e6 on DINOv3 ViT-S/16),
    /// overflowing to `inf` and making `softmax` produce NaN.  BF16 keeps
    /// fp32's exponent range at similar speed.  Setting `fp16` and `bf16`
    /// together lets TensorRT choose per layer — it chooses on speed, not
    /// range, and can reintroduce the overflow.  Pick one.
    pub bf16: bool,
    pub workspace_mb: i64,
}

impl Default for EngineProfile {
    fn default() -> Self {
        Self {
            inputs: vec![],
            fp16: true,
            bf16: false,
            workspace_mb: 2048,
        }
    }
}

impl EngineProfile {
    /// Short hash of the build options that affect the produced engine, so the
    /// cache key changes when the profile does (different precision or shape
    /// profile must NOT collide with a previously-built engine).
    fn cache_tag(&self) -> String {
        let mut h = Sha256::new();
        // Scalars keep their original textual form so that a static-shape profile
        // (`inputs: vec![]`) still hashes exactly as it did when this field was an
        // `Option`, and existing cached engines for those models stay valid.
        h.update(
            format!(
                "fp16={};bf16={};ws={};",
                self.fp16, self.bf16, self.workspace_mb
            )
            .as_bytes(),
        );
        // Every variable-length field is length-prefixed, making each entry
        // self-delimiting. Concatenating them into one `;`-delimited string instead
        // would let a tensor name containing the delimiter forge an entry boundary —
        // an input named `x;min=[1];opt=[1];max=[1];in=y` would hash identically to two
        // separate inputs `x` and `y`, and the cache would serve the wrong engine with
        // no error. Unreachable with torch-exported names, but the failure is silent,
        // so it is not worth relying on a naming convention we do not enforce.
        for (input, min, opt, max) in &self.inputs {
            h.update(format!("{}:", input.len()).as_bytes());
            h.update(input.as_bytes());
            for dims in [min, opt, max] {
                h.update(format!("{}:", dims.len()).as_bytes());
                for d in dims {
                    h.update(d.to_le_bytes());
                }
            }
        }
        format!("{:x}", h.finalize())[..8].to_string()
    }

    /// Canonical rendering of this profile's shape constraints, used to match a
    /// prebuilt [`EngineArtifact`] against what the caller actually needs.
    ///
    /// `"name:MINxDIMS|OPTxDIMS|MAXxDIMS"` per input, `;`-joined, sorted by input name
    /// so that declaration order cannot cause a spurious mismatch. Empty for a
    /// static-shape profile.
    ///
    /// Deliberately readable rather than a hash: it goes into the registry by hand, and
    /// a reviewer should be able to see which resolutions an artifact covers.
    pub fn shape_tag(&self) -> String {
        let dims = |d: &[i64]| {
            d.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("x")
        };
        let mut parts: Vec<String> = self
            .inputs
            .iter()
            .map(|(name, min, opt, max)| {
                format!("{name}:{}|{}|{}", dims(min), dims(opt), dims(max))
            })
            .collect();
        parts.sort();
        parts.join(";")
    }

    /// Reject build options that are individually legal but wrong together.
    ///
    /// `fp16` defaults to **true**, so `EngineProfile { bf16: true, ..default() }` —
    /// the natural way to ask for bf16 — silently sets both flags.  TensorRT then
    /// picks precision per layer on speed alone, with no knowledge of dynamic range,
    /// which reintroduces exactly the fp16 overflow bf16 was chosen to avoid (all-NaN
    /// on DINOv3).  Fail here rather than after a multi-minute build.
    fn validate(&self) -> Result<(), HubError> {
        if self.fp16 && self.bf16 {
            return Err(HubError::Build(
                "EngineProfile sets both fp16 and bf16 — pick one (fp16 defaults to \
                 true, so bf16 profiles must set `fp16: false`)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The precision this profile asks for — what a prebuilt [`EngineArtifact`] must
    /// have been built at to be usable here.
    ///
    /// Call [`validate`](Self::validate) first: with both flags set (which validate
    /// rejects) the answer would be arbitrary.
    pub fn precision(&self) -> Precision {
        match (self.fp16, self.bf16) {
            (_, true) => Precision::Bf16,
            (true, _) => Precision::Fp16,
            _ => Precision::Fp32,
        }
    }
}

/// On-device engine cache keyed by ONNX content + TRT version + GPU arch.
///
/// Key: `<name>-<onnx_sha8>-trt<version>-sm<cc>.engine` under
/// `~/.cache/vision-rt/engines/`.  Any change to the weights, the installed
/// TensorRT, or the GPU produces a different key → automatic rebuild.
/// Writes are atomic (tmp file + rename) so concurrent first-runs can't
/// corrupt the cache.
pub struct EngineCache {
    dir: PathBuf,
}

impl Default for EngineCache {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Self {
            dir: PathBuf::from(home).join(".cache/vision-rt/engines"),
        }
    }
}

impl EngineCache {
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Cache path for a model under a given build profile — exists or not.
    ///
    /// The key folds in the build profile (precision + shape profile) so a
    /// different profile can't be served a previously-built incompatible engine.
    pub fn key_path(
        &self,
        name: &str,
        onnx: &Path,
        profile: &EngineProfile,
    ) -> Result<PathBuf, HubError> {
        // `name` becomes a path component — reject anything that could escape dir.
        if name.is_empty()
            || name.contains('/')
            || name.contains('\\')
            || name.split(['/', '\\']).any(|c| c == "..")
        {
            return Err(HubError::InvalidName(name.into()));
        }
        let onnx_sha8 = &sha256_file(onnx)?[..8];
        let cfg = profile.cache_tag();
        let trt_ver = vrt::TENSORRT_VERSION;
        let sm = compute_capability()?;
        Ok(self.dir.join(format!(
            "{name}-{onnx_sha8}-{cfg}-trt{trt_ver}-sm{sm}.engine"
        )))
    }

    /// Return the cached engine for (`name`, `onnx`), building it on-device
    /// on a miss.  The build takes minutes (one-time per key).
    ///
    /// Build path: in-process `vrt::builder::EngineBuilder` with feature
    /// `builder`; otherwise a `trtexec` subprocess.
    pub fn get_or_build(
        &self,
        name: &str,
        onnx: &Path,
        profile: &EngineProfile,
    ) -> Result<PathBuf, HubError> {
        profile.validate()?;
        let path = self.key_path(name, onnx, profile)?;
        if path.exists() {
            return Ok(path);
        }

        fs::create_dir_all(&self.dir)?;
        eprintln!(
            "[vision-rt] building engine for '{name}' (one-time, ~1-5 min): {}",
            path.display()
        );

        let blob = build_engine(onnx, profile)?;

        // Atomic publish: unique tmp in the same dir, then rename.
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        fs::write(&tmp, &blob)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    /// Resolve a model path to a usable engine path: pass `.engine` files
    /// through unchanged, or build `.onnx` into the cache (see [`get_or_build`]).
    ///
    /// [`get_or_build`]: EngineCache::get_or_build
    pub fn resolve(
        &self,
        name: &str,
        model_path: &str,
        profile: &EngineProfile,
    ) -> Result<String, HubError> {
        if model_path.ends_with(".onnx") {
            Ok(self
                .get_or_build(name, Path::new(model_path), profile)?
                .to_string_lossy()
                .into_owned())
        } else {
            Ok(model_path.to_string())
        }
    }
}

/// GPU compute capability as e.g. "87".
fn compute_capability() -> Result<String, HubError> {
    let ctx = vrt::cudarc::driver::CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    Ok(format!("{major}{minor}"))
}

// ── Engine building ───────────────────────────────────────────────────────────

#[cfg(feature = "builder")]
fn build_engine(onnx: &Path, profile: &EngineProfile) -> Result<Vec<u8>, HubError> {
    use vrt::logger::Severity;
    let logger = vrt::Logger::new(Severity::Warning)?;
    let mut b = vrt::builder::EngineBuilder::from_onnx(onnx.to_string_lossy())
        .fp16(profile.fp16)
        .bf16(profile.bf16)
        .workspace_mb(profile.workspace_mb);
    // The C shim binds a single optimization profile, so multi-input models must go
    // through the trtexec path (which is the default — `builder` is opt-in).
    if profile.inputs.len() > 1 {
        return Err(HubError::Build(format!(
            "the in-process 'builder' path supports one shape profile, got {} — \
             build this model through the default trtexec path instead",
            profile.inputs.len()
        )));
    }
    if let Some((input, min, opt, max)) = profile.inputs.first() {
        b = b.shape_profile(input.clone(), min, opt, max);
    }
    Ok(b.build_serialized(&logger)?)
}

#[cfg(not(feature = "builder"))]
fn build_engine(onnx: &Path, profile: &EngineProfile) -> Result<Vec<u8>, HubError> {
    // trtexec subprocess fallback (JetPack ships it outside PATH).
    let trtexec = ["/usr/src/tensorrt/bin/trtexec", "trtexec"]
        .iter()
        .find(|p| Path::new(p).exists() || which(p))
        .ok_or_else(|| {
            HubError::Build("trtexec not found and 'builder' feature is disabled".into())
        })?;

    let out = tempfile_path("engine")?;
    let mut cmd = Command::new(trtexec);
    cmd.arg(format!("--onnx={}", onnx.display()))
        .arg(format!("--saveEngine={}", out.display()))
        .arg(format!("--memPoolSize=workspace:{}", profile.workspace_mb));
    if profile.fp16 {
        cmd.arg("--fp16");
    }
    if profile.bf16 {
        cmd.arg("--bf16");
    }
    // trtexec takes all inputs in one comma-separated flag per bound:
    //   --minShapes=images:1x3x256x256,mask:1x1x256x256
    if !profile.inputs.is_empty() {
        let join = |pick: fn(&ShapeProfile) -> &Vec<i64>| {
            profile
                .inputs
                .iter()
                .map(|p| format!("{}:{}", p.0, dims_x(pick(p))))
                .collect::<Vec<_>>()
                .join(",")
        };
        cmd.arg(format!("--minShapes={}", join(|p| &p.1)))
            .arg(format!("--optShapes={}", join(|p| &p.2)))
            .arg(format!("--maxShapes={}", join(|p| &p.3)));
    }

    let status = cmd.status()?;
    if !status.success() {
        return Err(HubError::Build(format!("trtexec failed with {status}")));
    }
    let blob = fs::read(&out)?;
    let _ = fs::remove_file(&out);
    Ok(blob)
}

#[cfg(not(feature = "builder"))]
fn dims_x(dims: &[i64]) -> String {
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

#[cfg(not(feature = "builder"))]
fn which(bin: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|p| p.join(bin).exists()))
}

#[cfg(not(feature = "builder"))]
fn tempfile_path(ext: &str) -> Result<PathBuf, HubError> {
    // PID + a process-unique counter: two concurrent get_or_build calls for
    // DIFFERENT models in one process must not write to the same trtexec target.
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    Ok(std::env::temp_dir().join(format!("vrt-hub-{}-{n}.{ext}", std::process::id())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lookup() {
        assert!(spec("xfeat-backbone").is_some());
        assert!(spec("nope").is_none());
    }

    #[test]
    fn sha256_of_known_bytes() {
        let tmp = std::env::temp_dir().join("vrt-hub-test-sha");
        fs::write(&tmp, b"hello").unwrap();
        assert_eq!(
            sha256_file(&tmp).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn profile_changes_cache_tag() {
        // Different build options must hash to different tags, so the engine
        // cache key never serves an incompatible engine as a hit.
        let base = EngineProfile::default();
        let diff_prec = EngineProfile {
            fp16: !base.fp16,
            ..EngineProfile::default()
        };
        // bf16 must be in the tag too: without it a bf16 profile would hit an engine
        // cached from an fp32/fp16 build of the same ONNX — for a ViT that means
        // silently loading the all-NaN fp16 engine.
        let diff_bf16 = EngineProfile {
            fp16: false,
            bf16: true,
            ..EngineProfile::default()
        };
        let fp32 = EngineProfile {
            fp16: false,
            ..EngineProfile::default()
        };
        let shaped = EngineProfile {
            inputs: vec![(
                "x".into(),
                vec![1, 3, 64, 64],
                vec![1, 3, 64, 64],
                vec![1, 3, 64, 64],
            )],
            ..EngineProfile::default()
        };
        assert_ne!(base.cache_tag(), diff_prec.cache_tag());
        assert_ne!(base.cache_tag(), shaped.cache_tag());
        assert_ne!(fp32.cache_tag(), diff_bf16.cache_tag());
        assert_ne!(base.cache_tag(), diff_bf16.cache_tag());
        // Deterministic.
        assert_eq!(base.cache_tag(), EngineProfile::default().cache_tag());
    }

    /// Two *different* input lists must never hash alike, or the cache silently serves
    /// an engine built for other shapes — a wrong-answer failure, not a crash.
    ///
    /// The dangerous case is delimiter forgery. Concatenating entries into one
    /// `;`-delimited string made an input named `x;min=[1];opt=[1];max=[1];in=y`
    /// indistinguishable from two separate inputs `x` and `y`. Length-prefixing every
    /// variable-length field closes it.
    #[test]
    fn cache_tag_cannot_be_forged_by_a_delimiter_in_a_tensor_name() {
        let one = |name: &str| EngineProfile {
            inputs: vec![(name.into(), vec![1], vec![1], vec![1])],
            ..EngineProfile::default()
        };
        let forged = one("x;min=[1];opt=[1];max=[1];in=y");
        let honest = EngineProfile {
            inputs: vec![
                ("x".into(), vec![1], vec![1], vec![1]),
                ("y".into(), vec![1], vec![1], vec![1]),
            ],
            ..EngineProfile::default()
        };
        assert_ne!(forged.cache_tag(), honest.cache_tag());

        // Adjacent fields must not run together either: a name absorbing the next
        // field's digits, or dims of different lengths concatenating to the same bytes.
        assert_ne!(one("ab").cache_tag(), one("a").cache_tag());
        let split_dims = EngineProfile {
            inputs: vec![("x".into(), vec![1, 1], vec![1], vec![1])],
            ..EngineProfile::default()
        };
        assert_ne!(one("x").cache_tag(), split_dims.cache_tag());

        // Order is part of the identity (a reordered profile is a different build).
        let swapped = EngineProfile {
            inputs: honest.inputs.iter().rev().cloned().collect(),
            ..EngineProfile::default()
        };
        assert_ne!(honest.cache_tag(), swapped.cache_tag());
    }

    /// A static-shape profile must hash exactly as it did before `input: Option<_>`
    /// became `inputs: Vec<_>`, so upgrading does not invalidate every cached engine
    /// on a box where each rebuild costs minutes.
    #[test]
    fn static_profile_cache_tag_is_unchanged_by_the_vec_migration() {
        let mut h = Sha256::new();
        h.update(b"fp16=true;bf16=false;ws=2048;"); // the pre-migration string for `None`
        let legacy = format!("{:x}", h.finalize())[..8].to_string();
        assert_eq!(EngineProfile::default().cache_tag(), legacy);
    }

    /// A prebuilt is only served when it covers the SHAPES the caller needs.
    ///
    /// A TensorRT engine accepts only shapes inside the optimization profile it was built
    /// with. Matching on trt+sm+precision alone hands a caller an engine built for another
    /// resolution range; it deserializes happily and then rejects perfectly ordinary
    /// frames at setInputShape — with an empty TensorRT message, so it reads as a bug in
    /// the caller's code rather than a mismatched download. Falling back to an on-device
    /// build is always correct, so a mismatch must simply miss.
    #[test]
    fn prebuilt_must_cover_the_requested_shape_profile() {
        const NARROW: &str = "images:1x3x640x640|1x3x640x640|1x3x640x640";
        const WIDE: &str = "images:1x3x256x256|2x3x512x512|2x3x640x640";
        const ARTS: &[EngineArtifact] = &[EngineArtifact {
            filename: "narrow-trt10.3.0.30-sm87-fp16.engine",
            sha256: "00",
            trt_version: "10.3.0.30",
            sm: "87",
            precision: Precision::Fp16,
            shape_profile: NARROW,
        }];

        // Same env and precision, different shapes → must NOT be served.
        assert!(pick_artifact(ARTS, "10.3.0.30", "87", Precision::Fp16, WIDE).is_none());
        // Exact match → served.
        assert!(pick_artifact(ARTS, "10.3.0.30", "87", Precision::Fp16, NARROW).is_some());
        // A static-shape request must not pick up a dynamic-profile engine either.
        assert!(pick_artifact(ARTS, "10.3.0.30", "87", Precision::Fp16, "").is_none());
    }

    /// `shape_tag` must be stable and order-independent, or a registry pin written by hand
    /// silently stops matching when someone reorders a profile's inputs.
    #[test]
    fn shape_tag_is_canonical_and_order_independent() {
        let mk = |inputs: Vec<ShapeProfile>| EngineProfile {
            inputs,
            ..EngineProfile::default()
        };
        let a = ("a".to_string(), vec![1, 2], vec![3, 4], vec![5, 6]);
        let b = ("b".to_string(), vec![7], vec![8], vec![9]);

        assert_eq!(
            mk(vec![a.clone(), b.clone()]).shape_tag(),
            mk(vec![b, a.clone()]).shape_tag()
        );
        assert_eq!(mk(vec![a]).shape_tag(), "a:1x2|3x4|5x6");
        assert_eq!(mk(vec![]).shape_tag(), "");
    }

    /// A prebuilt is only served when its precision matches the request. Without this
    /// an fp16 artifact answers a bf16 request — for a ViT that is the all-NaN engine,
    /// delivered silently in preference to a correct on-device build.
    #[test]
    fn prebuilt_must_match_requested_precision() {
        const ARTS: &[EngineArtifact] = &[
            EngineArtifact {
                filename: "m-fp16.engine",
                sha256: "aa",
                trt_version: "10.3.0.30",
                sm: "87",
                precision: Precision::Fp16,
                shape_profile: "",
            },
            EngineArtifact {
                filename: "m-bf16.engine",
                sha256: "bb",
                trt_version: "10.3.0.30",
                sm: "87",
                precision: Precision::Bf16,
                shape_profile: "",
            },
        ];
        let pick = |p| pick_artifact(ARTS, "10.3.0.30", "87", p, "").map(|a| a.filename);
        assert_eq!(pick(Precision::Fp16), Some("m-fp16.engine"));
        assert_eq!(pick(Precision::Bf16), Some("m-bf16.engine"));
        // No fp32 artifact listed → build on-device rather than serving either of these.
        assert_eq!(pick(Precision::Fp32), None);
        // trt/sm still gate independently.
        assert_eq!(
            pick_artifact(ARTS, "10.4.0.0", "87", Precision::Fp16, "").map(|a| a.filename),
            None
        );
        assert_eq!(
            pick_artifact(ARTS, "10.3.0.30", "86", Precision::Fp16, "").map(|a| a.filename),
            None
        );
    }

    /// The flags a profile carries must map to the precision a prebuilt is matched on.
    #[test]
    fn profile_precision_maps_from_flags() {
        let p = |fp16, bf16| {
            EngineProfile {
                fp16,
                bf16,
                ..EngineProfile::default()
            }
            .precision()
        };
        assert_eq!(p(true, false), Precision::Fp16);
        assert_eq!(p(false, true), Precision::Bf16);
        assert_eq!(p(false, false), Precision::Fp32);
    }

    /// Every registry artifact must be reachable: its precision has to equal what the
    /// owning crate's `engine_profile()` asks for, or the prebuilt is dead weight and
    /// every user silently pays for an on-device build instead.
    #[test]
    fn registry_engine_precisions_match_their_filenames() {
        for spec in REGISTRY {
            for e in spec.engines {
                let want = if e.filename.contains("-bf16.") {
                    Precision::Bf16
                } else if e.filename.contains("-fp16.") {
                    Precision::Fp16
                } else {
                    Precision::Fp32
                };
                assert_eq!(
                    e.precision, want,
                    "{}: precision {:?} disagrees with filename",
                    e.filename, e.precision
                );
            }
        }
    }

    #[test]
    fn rejects_fp16_and_bf16_together() {
        // `fp16` defaults to true, so `EngineProfile { bf16: true, ..default() }` — the
        // natural way to ask for bf16 — sets both. TensorRT would then pick per layer
        // on speed alone and can reintroduce the fp16 overflow bf16 exists to avoid.
        let both = EngineProfile {
            bf16: true,
            ..EngineProfile::default()
        };
        assert!(
            both.fp16,
            "fp16 still defaults to true — this test's premise"
        );
        assert!(matches!(both.validate(), Err(HubError::Build(_))));
        assert!(EngineProfile::default().validate().is_ok());
        assert!(EngineProfile {
            fp16: false,
            bf16: true,
            ..EngineProfile::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn key_path_rejects_unsafe_names() {
        // The name check runs before any CUDA call, so this is host-only.
        let cache = EngineCache::at("/tmp/vrt-hub-test-cache");
        let onnx = Path::new("/nonexistent.onnx");
        let prof = EngineProfile::default();
        for bad in ["../escape", "a/b", "..", "a\\b", ""] {
            assert!(
                matches!(
                    cache.key_path(bad, onnx, &prof),
                    Err(HubError::InvalidName(_))
                ),
                "expected InvalidName for {bad:?}"
            );
        }
    }
}

#[cfg(test)]
mod integration {
    use super::*;

    /// End-to-end on this Jetson: ONNX -> in-process build -> cache hit.
    /// Slow (engine build); run explicitly:
    ///   cargo test -p trt-hub --features builder -- --ignored
    #[test]
    #[ignore]
    fn build_xfeat_engine_via_cache() {
        let onnx_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/xfeat/xfeat_backbone.onnx"
        );
        let onnx = Path::new(onnx_path);
        assert!(onnx.exists(), "test needs the local xfeat ONNX");

        let profile = EngineProfile {
            inputs: vec![(
                "image".into(),
                vec![1, 3, 240, 320],
                vec![1, 3, 240, 320],
                vec![1, 3, 240, 320],
            )],
            fp16: true,
            bf16: false,
            workspace_mb: 1024,
        };
        let cache = EngineCache::at(std::env::temp_dir().join("vrt-hub-it"));
        let path = cache.get_or_build("xfeat-it", onnx, &profile).unwrap();
        let len = fs::metadata(&path).unwrap().len();
        assert!(len > 100_000, "engine suspiciously small: {len} bytes");

        // Second call must be a pure cache hit (same path, no rebuild).
        let again = cache.get_or_build("xfeat-it", onnx, &profile).unwrap();
        assert_eq!(path, again);
    }
}
