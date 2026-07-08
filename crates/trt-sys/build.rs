use std::{env, path::PathBuf};

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
    let cuda_lib = format!("{cuda_home}/lib64");
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
    println!("cargo:rustc-link-search=native={trt_lib}");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin");
    if builder_feature {
        println!("cargo:rustc-link-lib=dylib=nvonnxparser");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // ── 5. Version constants — parsed from NvInferVersion.h, not hardcoded ─────────────
    let version = parse_trt_version(&trt_inc).unwrap_or_else(|| "10.3.0.30".to_string());
    println!("cargo:rustc-env=TENSORRT_VERSION={version}");
    println!("cargo:rerun-if-changed={trt_inc}/NvInferVersion.h");

    // Warn (never fail) if the installed TRT is outside the tested range. The
    // version flows into engine-cache keys, so an off-version box simply gets its
    // own cache namespace rather than a miskeyed hit — but the shims are only
    // validated against 10.3.x, so surface the mismatch to the builder.
    const SUPPORTED_TRT: (&str, &str) = ("10", "3"); // major, minor
    let mut parts = version.split('.');
    if (parts.next(), parts.next()) != (Some(SUPPORTED_TRT.0), Some(SUPPORTED_TRT.1)) {
        println!(
            "cargo:warning=TensorRT {version} is outside the tested {}.{}.x range; \
             engines/cache-keys are version-specific and may need an on-device rebuild",
            SUPPORTED_TRT.0, SUPPORTED_TRT.1
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

/// Parse "MAJOR.MINOR.PATCH.BUILD" from NvInferVersion.h so the version
/// constant tracks the actually-installed TRT (engine-cache keys depend on it).
fn parse_trt_version(trt_inc: &str) -> Option<String> {
    let text = std::fs::read_to_string(format!("{trt_inc}/NvInferVersion.h")).ok()?;
    let grab = |name: &str| -> Option<u32> {
        text.lines()
            .find(|l| l.contains(&format!("#define {name} ")))
            // Take the token immediately AFTER the macro name, not the last
            // token — the headers carry trailing `//!< …` Doxygen comments, so
            // `.last()` would grab a comment word and the parse would always
            // fail (silently falling back to a hardcoded version → mis-keyed
            // engine cache on any non-default TRT).
            .and_then(|l| l.split_whitespace().skip_while(|t| *t != name).nth(1))
            .and_then(|v| v.parse().ok())
    };
    Some(format!(
        "{}.{}.{}.{}",
        grab("NV_TENSORRT_MAJOR")?,
        grab("NV_TENSORRT_MINOR")?,
        grab("NV_TENSORRT_PATCH")?,
        grab("NV_TENSORRT_BUILD")?,
    ))
}
