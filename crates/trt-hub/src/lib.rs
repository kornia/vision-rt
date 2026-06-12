//! Model weights distribution + on-device TensorRT engine cache.
//!
//! ## The shipping model
//! - **ONNX weights** are the portable artifact, hosted on Hugging Face Hub
//!   and pinned by sha256 — never committed to the repo (feature `hub`).
//! - **Engines** are machine-locked (TRT version + GPU arch) and are built
//!   **on-device** from ONNX into a versioned cache, never distributed.
//!
//! ```no_run
//! use trt_hub::{ModelHub, EngineCache, EngineProfile};
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
//! # Ok::<(), trt::BoxError>(())
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
    #[error("unknown model '{0}' — see trt_hub::REGISTRY for known names")]
    UnknownModel(String),
    #[error("model '{0}' has no files in the registry")]
    EmptyModel(String),
    #[error("sha256 mismatch for {path}: expected {expected}, got {actual} (corrupted download? delete and retry)")]
    Sha256Mismatch { path: PathBuf, expected: String, actual: String },
    #[cfg(feature = "hub")]
    #[error("Hugging Face Hub: {0}")]
    Hf(#[from] hf_hub::api::sync::ApiError),
    #[cfg(not(feature = "hub"))]
    #[error("trt-hub built without the 'hub' feature — enable it to download '{0}', or pass an explicit ONNX path")]
    HubFeatureDisabled(String),
    #[error(transparent)]
    Trt(#[from] trt::TrtError),
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("engine build: {0}")]
    Build(String),
}



// ── Model registry ────────────────────────────────────────────────────────────

/// One file inside a hub model: name + sha256 pin.
pub struct ModelFile {
    pub filename: &'static str,
    pub sha256:   &'static str,
}

/// A distributable model: where it lives on the Hub and what it contains.
///
/// `files[0]` is the entry-point .onnx; the rest are sidecars (e.g.
/// `.onnx.data` external weights) that must land in the same directory.
pub struct ModelSpec {
    pub name:     &'static str,
    pub hf_repo:  &'static str,
    pub revision: &'static str,
    pub files:    &'static [ModelFile],
}

/// Static registry of known models.
///
/// To add a model: export ONNX (scripts/), upload to the HF repo, add an
/// entry here with `sha256sum` pins.
pub static REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        name:     "xfeat-backbone",
        hf_repo:  "edgarriba/trt-rs-models",
        revision: "main",
        files: &[
            ModelFile {
                filename: "xfeat_backbone.onnx",
                sha256:   "86d7d549b380405f208933efb5202e1584d9762f3a72e06e7ed81ca1436972e0",
            },
            ModelFile {
                filename: "xfeat_backbone.onnx.data",
                sha256:   "d4498528d37bf7c737cce9c135f9b0340d828bab7dc808339e50553ac8c1b7d9",
            },
        ],
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
            verify_sha256(&path, f.sha256)?;
            if entry.is_none() { entry = Some(path); }
        }
        entry.ok_or_else(|| HubError::EmptyModel(name.into()))
    }

    #[cfg(not(feature = "hub"))]
    pub fn get(name: &str) -> Result<PathBuf, HubError> {
        Err(HubError::HubFeatureDisabled(name.into()))
    }
}

/// Verify a file against an expected sha256 hex digest.
pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), HubError> {
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(HubError::Sha256Mismatch {
            path:     path.to_path_buf(),
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
        if n == 0 { break; }
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
    pub input:        Option<ShapeProfile>,
    pub fp16:         bool,
    pub workspace_mb: i64,
}

impl Default for EngineProfile {
    fn default() -> Self {
        Self { input: None, fp16: true, workspace_mb: 2048 }
    }
}

/// On-device engine cache keyed by ONNX content + TRT version + GPU arch.
///
/// Key: `<name>-<onnx_sha8>-trt<version>-sm<cc>.engine` under
/// `~/.cache/trt-rs/engines/`.  Any change to the weights, the installed
/// TensorRT, or the GPU produces a different key → automatic rebuild.
/// Writes are atomic (tmp file + rename) so concurrent first-runs can't
/// corrupt the cache.
pub struct EngineCache {
    dir: PathBuf,
}

impl Default for EngineCache {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Self { dir: PathBuf::from(home).join(".cache/trt-rs/engines") }
    }
}

impl EngineCache {
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Cache path for a model — exists or not.
    pub fn key_path(&self, name: &str, onnx: &Path) -> Result<PathBuf, HubError> {
        let onnx_sha8 = &sha256_file(onnx)?[..8];
        let trt_ver = trt_sys_version();
        let sm = compute_capability()?;
        Ok(self.dir.join(format!("{name}-{onnx_sha8}-trt{trt_ver}-sm{sm}.engine")))
    }

    /// Return the cached engine for (`name`, `onnx`), building it on-device
    /// on a miss.  The build takes minutes (one-time per key).
    ///
    /// Build path: in-process `trt::builder::EngineBuilder` with feature
    /// `builder`; otherwise a `trtexec` subprocess.
    pub fn get_or_build(
        &self,
        name: &str,
        onnx: &Path,
        profile: &EngineProfile,
    ) -> Result<PathBuf, HubError> {
        let path = self.key_path(name, onnx)?;
        if path.exists() {
            return Ok(path);
        }

        fs::create_dir_all(&self.dir)?;
        eprintln!(
            "[trt-hub] building engine for '{name}' (one-time, ~1-5 min): {}",
            path.display()
        );

        let blob = build_engine(onnx, profile)?;

        // Atomic publish: unique tmp in the same dir, then rename.
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        fs::write(&tmp, &blob)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }
}

/// TRT version string for cache keys (from NvInferVersion.h at build time).
fn trt_sys_version() -> &'static str {
    trt::TENSORRT_VERSION
}

/// GPU compute capability as e.g. "87".
fn compute_capability() -> Result<String, HubError> {
    let ctx = cudarc_context()?;
    let (major, minor) = ctx.compute_capability()?;
    Ok(format!("{major}{minor}"))
}

fn cudarc_context() -> Result<std::sync::Arc<trt::cudarc::driver::CudaContext>, HubError> {
    Ok(trt::cudarc::driver::CudaContext::new(0)?)
}

// ── Engine building ───────────────────────────────────────────────────────────

#[cfg(feature = "builder")]
fn build_engine(onnx: &Path, profile: &EngineProfile) -> Result<Vec<u8>, HubError> {
    use trt::logger::Severity;
    let logger = trt::Logger::new(Severity::Warning)?;
    let mut b = trt::builder::EngineBuilder::from_onnx(onnx.to_string_lossy())
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
        .ok_or_else(|| HubError::Build("trtexec not found and 'builder' feature is disabled".into()))?;

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
    dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")
}

#[cfg(not(feature = "builder"))]
fn which(bin: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|p| p.join(bin).exists())
    })
}

#[cfg(not(feature = "builder"))]
fn tempfile_path(ext: &str) -> Result<PathBuf, HubError> {
    Ok(std::env::temp_dir().join(format!("trt-hub-{}.{ext}", std::process::id())))
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
        let tmp = std::env::temp_dir().join("trt-hub-test-sha");
        fs::write(&tmp, b"hello").unwrap();
        assert_eq!(
            sha256_file(&tmp).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        let _ = fs::remove_file(&tmp);
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
        let onnx = Path::new("/home/nvidia/trt-rs/models/xfeat/xfeat_backbone.onnx");
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
        let cache = EngineCache::at(std::env::temp_dir().join("trt-hub-it"));
        let path = cache.get_or_build("xfeat-it", onnx, &profile).unwrap();
        let len = fs::metadata(&path).unwrap().len();
        assert!(len > 100_000, "engine suspiciously small: {len} bytes");

        // Second call must be a pure cache hit (same path, no rebuild).
        let again = cache.get_or_build("xfeat-it", onnx, &profile).unwrap();
        assert_eq!(path, again);
    }
}
