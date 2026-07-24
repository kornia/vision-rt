use std::{env, path::Path, path::PathBuf};

include!("trt_version.rs");

/// Version from the installed NvInferVersion.h — engine-cache keys depend on it.
fn parse_trt_version(trt_inc: &str) -> Option<String> {
    let text = std::fs::read_to_string(format!("{trt_inc}/NvInferVersion.h")).ok()?;
    parse_trt_version_text(&text)
}

/// Exact runtime version from a wheel install: `tensorrt*_libs-<v>.dist-info`
/// sits next to the lib dir in site-packages. None for system installs
/// (JetPack apt/tarball) — those carry no dist-info and need no check.
fn wheel_dist_version(trt_lib: &Path) -> Option<String> {
    let site_packages = trt_lib.parent()?;
    for entry in site_packages.read_dir().ok()?.flatten() {
        let name = entry.file_name().into_string().ok()?;
        if let Some(stem) = name.strip_suffix(".dist-info") {
            if stem.starts_with("tensorrt") && stem.contains("_libs-") {
                return stem.rsplit('-').next().map(str::to_owned);
            }
        }
    }
    None
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(trt_stub)");

    // Stub mode (docs.rs / hosted CI without TRT headers): skip the C++
    // shims, bindgen, and link directives entirely; lib.rs falls back to
    // src/pregenerated_bindings.rs.  `cargo check`/`clippy`/`doc` work;
    // anything that links or runs requires a real TRT install.
    if env::var("DOCS_RS").is_ok() || env::var("TRT_STUB").is_ok() {
        println!("cargo:rustc-cfg=trt_stub");
        println!("cargo:rustc-env=TENSORRT_VERSION=0.0.0.0-stub");
        println!("cargo:rerun-if-env-changed=TRT_STUB");
        return;
    }
    println!("cargo:rerun-if-env-changed=TRT_STUB");

    let trt_inc =
        env::var("TRT_INCLUDE_DIR").unwrap_or_else(|_| "/usr/include/aarch64-linux-gnu".into());
    let trt_lib = env::var("TRT_LIB_DIR").unwrap_or_else(|_| "/usr/lib/aarch64-linux-gnu".into());
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let cuda_inc = format!("{cuda_home}/include");
    let cuda_lib64 = format!("{cuda_home}/lib64");
    let cuda_lib = if Path::new(&cuda_lib64).is_dir() {
        cuda_lib64
    } else {
        format!("{cuda_home}/lib")
    };

    // Parse before any link directives: the installed major selects the only
    // versioned soname fallback that can safely satisfy this exact build.
    let version = parse_trt_version(&trt_inc).unwrap_or_else(|| {
        panic!(
            "failed to parse TensorRT version from {trt_inc}/NvInferVersion.h; \
             check TRT_INCLUDE_DIR"
        )
    });
    let major = version
        .split('.')
        .next()
        .expect("parsed TensorRT version always has a major component");
    println!("cargo:rustc-env=TENSORRT_VERSION={version}");
    println!("cargo:rerun-if-changed={trt_inc}/NvInferVersion.h");

    // Wheel installs state their exact runtime version in dist-info; when
    // present it must equal the header version — headers and libs are pinned
    // independently (recipes/tensorrt-dev vs the tensorrt-cu13 pin in
    // pixi.toml), and a bump to one without the other would compile against
    // skewed headers while TENSORRT_VERSION mis-keys every engine.
    if let Some(wheel_version) = wheel_dist_version(Path::new(&trt_lib)) {
        assert_eq!(
            wheel_version, version,
            "TensorRT header/runtime skew: headers at {trt_inc} are {version} but the \
             installed wheel is {wheel_version}; bump recipes/tensorrt-dev and the \
             tensorrt-cu13 pin in pixi.toml together"
        );
    }

    // ── 1. Compile logger shim (ILogger subclass — only thing needing C++ subclassing) ──
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/logger_shim.cpp")
        .include(&trt_inc)
        .include(&cuda_inc)
        .include("include")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("btrt_logger_shim");

    // ── 2. Compile TRT bridge (runtime/engine/context/CUDA — direct TRT header calls) ──
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/trt_bridge.cpp")
        .include(&trt_inc)
        .include(&cuda_inc)
        .include("include")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("btrt_trt_bridge");

    // ── 2b. Builder shim (feature = "builder"): ONNX -> engine via nvonnxparser ────────
    let builder_feature = env::var("CARGO_FEATURE_BUILDER").is_ok();
    if builder_feature {
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("src/builder_shim.cpp")
            .include(&trt_inc)
            .include(&cuda_inc)
            .include("include")
            .flag_if_supported("-Wno-deprecated-declarations")
            .compile("btrt_builder_shim");
        println!("cargo:rerun-if-changed=src/builder_shim.cpp");
        println!("cargo:rerun-if-changed=include/builder_shim.h");
    }

    // ── 3. bindgen for the btrt_* C bridge (logger + runtime/engine/context + CUDA) ────
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // All btrt_* symbols from both logger_shim.h and trt_bridge.h.
    // trt_bridge.h includes logger_shim.h, so one header covers everything.
    let bridge_bindings = bindgen::Builder::default()
        .header("include/trt_bridge.h")
        .allowlist_function("btrt_.*")
        .allowlist_type("btrt_.*")
        .generate()
        .expect("bindgen failed on trt_bridge.h");
    bridge_bindings
        .write_to_file(out_dir.join("bridge_bindings.rs"))
        .expect("failed to write bridge_bindings.rs");

    if builder_feature {
        let builder_bindings = bindgen::Builder::default()
            .header("include/builder_shim.h")
            .allowlist_function("btrt_build_engine_from_onnx")
            .allowlist_function("btrt_blob_free")
            // btrt_logger_t already comes from bridge_bindings.rs
            .blocklist_type("btrt_logger_.*")
            .blocklist_function("btrt_logger_.*")
            .generate()
            .expect("bindgen failed on builder_shim.h");
        builder_bindings
            .write_to_file(out_dir.join("builder_bindings.rs"))
            .expect("failed to write builder_bindings.rs");
    }

    // ── 4. Link directives ──────────────────────────────────────────────────────────────
    // Per library: prefer the unversioned linker name (JetPack / dev installs), else
    // the versioned soname matching the parsed header major (pip wheels ship only
    // *.so.10) — so a mixed dir still links, and only a truly absent lib panics.
    let mut libraries = vec!["nvinfer", "nvinfer_plugin"];
    if builder_feature {
        libraries.push("nvonnxparser");
    }
    let trt_lib_path = Path::new(&trt_lib);
    println!("cargo:rustc-link-search=native={trt_lib}");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    for library in &libraries {
        let unversioned = format!("lib{library}.so");
        let versioned = format!("lib{library}.so.{major}");
        if trt_lib_path.join(&unversioned).is_file() {
            println!("cargo:rustc-link-lib=dylib={library}");
        } else if trt_lib_path.join(&versioned).is_file() {
            println!("cargo:rustc-link-lib=dylib:+verbatim={versioned}");
        } else {
            let resolved = trt_lib_path
                .canonicalize()
                .unwrap_or_else(|_| trt_lib_path.to_path_buf());
            panic!(
                "TensorRT library not found in {}: expected {unversioned} or {versioned}",
                resolved.display(),
            );
        }
    }
    // cudart/stdc++ stay unversioned-only: JetPack, conda (cuda-cudart-dev), and
    // the compiler runtime all provide lib*.so — a wheel-only layout without a
    // CUDA_HOME dev install fails here on cudart, not on the nvinfer trio above.
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // ── 5. Version support warning ─────────────────────────────────────────────────────
    // Warn (never fail) if the installed TRT is outside the tested range. The
    // version flows into engine-cache keys, so an off-version box simply gets its
    // own cache namespace rather than a miskeyed hit — but the shims are only
    // validated against the listed versions, so surface the mismatch to the builder.
    const SUPPORTED_TRT: [(&str, &str); 2] = [("10", "3"), ("10", "13")]; // major, minor
    let mut parts = version.split('.');
    let installed = (parts.next(), parts.next());
    if !SUPPORTED_TRT
        .iter()
        .any(|&(major, minor)| installed == (Some(major), Some(minor)))
    {
        let tested_versions = SUPPORTED_TRT
            .iter()
            .map(|(major, minor)| format!("{major}.{minor}.x"))
            .collect::<Vec<_>>()
            .join("/");
        println!(
            "cargo:warning=TensorRT {version} is outside the tested {tested_versions} ranges; \
             engines/cache-keys are version-specific and may need an on-device rebuild",
        );
    }

    // ── 6. Rebuild triggers ─────────────────────────────────────────────────────────────
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/logger_shim.cpp");
    println!("cargo:rerun-if-changed=src/trt_bridge.cpp");
    println!("cargo:rerun-if-changed=include/logger_shim.h");
    println!("cargo:rerun-if-changed=include/trt_bridge.h");
    println!("cargo:rerun-if-env-changed=TRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
