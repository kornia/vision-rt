//! Model weights distribution + on-device TensorRT engine cache.
//!
//! ## The shipping model
//! - **ONNX weights** are the portable artifact, hosted on Hugging Face Hub
//!   and pinned by sha256 — never committed to the repo (feature `hub`).
//! - **Engines** are machine-locked (TRT version + GPU arch). They are normally
//!   built **on-device** from ONNX into a versioned cache. A registry MAY also
//!   list prebuilt engines for exact-match environments; [`ModelHub::get_engine`]
//!   downloads one only when its `trt_version` + `sm` match the local box, so a
//!   mismatched engine is never fetched — it falls back to an on-device build.
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
//!     input:  Some(("image".into(),
//!                   vec![1,3,240,320], vec![1,3,640,640], vec![1,3,1088,1920])),
//!     fp16: true,
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
/// GPU `sm` (compute capability); it is downloaded only when both match the
/// local box, otherwise the engine is built on-device from the ONNX instead.
pub struct EngineArtifact {
    pub filename: &'static str, // e.g. "xfeat_backbone-trt10.3.0.30-sm87-fp16.engine"
    pub sha256: &'static str,
    pub trt_version: &'static str, // must equal `vrt::TENSORRT_VERSION`, e.g. "10.3.0.30"
    pub sm: &'static str,          // GPU compute capability, e.g. "87"
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
    pub fn get_engine(name: &str) -> Result<Option<PathBuf>, HubError> {
        let spec = spec(name).ok_or_else(|| HubError::UnknownModel(name.into()))?;
        if spec.engines.is_empty() {
            return Ok(None);
        }
        let (local_trt, local_sm) = (vrt::TENSORRT_VERSION, compute_capability()?);
        let art = match spec
            .engines
            .iter()
            .find(|e| e.trt_version == local_trt && e.sm == local_sm)
        {
            Some(a) => a,
            None => return Ok(None), // no prebuilt for this env → build from ONNX
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
    pub fn get_engine(_name: &str) -> Result<Option<PathBuf>, HubError> {
        Ok(None)
    }
}

/// Resolve a registry model to a usable engine path (feature `hub`): a matching
/// prebuilt engine if the registry lists one for this box's TRT+SM, otherwise the
/// pinned ONNX downloaded and built/cached on-device. This is the one call behind
/// each model crate's `from_hub` constructor.
#[cfg(feature = "hub")]
pub fn resolve_engine(name: &str, profile: &EngineProfile) -> Result<String, HubError> {
    if let Some(engine) = ModelHub::get_engine(name)? {
        return Ok(engine.to_string_lossy().into_owned());
    }
    let onnx = ModelHub::get(name)?;
    EngineCache::default().resolve(name, &onnx.to_string_lossy(), profile)
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
    /// Profile for dynamic-shape models; None = static shapes.
    pub input: Option<ShapeProfile>,
    pub fp16: bool,
    pub workspace_mb: i64,
}

impl Default for EngineProfile {
    fn default() -> Self {
        Self {
            input: None,
            fp16: true,
            workspace_mb: 2048,
        }
    }
}

impl EngineProfile {
    /// Short hash of the build options that affect the produced engine, so the
    /// cache key changes when the profile does (different precision or shape
    /// profile must NOT collide with a previously-built engine).
    fn cache_tag(&self) -> String {
        let mut s = format!("fp16={};ws={};", self.fp16, self.workspace_mb);
        if let Some((input, min, opt, max)) = &self.input {
            s.push_str(&format!("in={input};min={min:?};opt={opt:?};max={max:?}"));
        }
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        format!("{:x}", h.finalize())[..8].to_string()
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
        .workspace_mb(profile.workspace_mb);
    if let Some((input, min, opt, max)) = &profile.input {
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
    if let Some((input, min, opt, max)) = &profile.input {
        cmd.arg(format!("--minShapes={input}:{}", dims_x(min)))
            .arg(format!("--optShapes={input}:{}", dims_x(opt)))
            .arg(format!("--maxShapes={input}:{}", dims_x(max)));
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
        let shaped = EngineProfile {
            input: Some((
                "x".into(),
                vec![1, 3, 64, 64],
                vec![1, 3, 64, 64],
                vec![1, 3, 64, 64],
            )),
            ..EngineProfile::default()
        };
        assert_ne!(base.cache_tag(), diff_prec.cache_tag());
        assert_ne!(base.cache_tag(), shaped.cache_tag());
        // Deterministic.
        assert_eq!(base.cache_tag(), EngineProfile::default().cache_tag());
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
            input: Some((
                "image".into(),
                vec![1, 3, 240, 320],
                vec![1, 3, 240, 320],
                vec![1, 3, 240, 320],
            )),
            fp16: true,
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
