use std::{env, path::PathBuf};

fn main() {
    let trt_inc = env::var("TRT_INCLUDE_DIR")
        .unwrap_or_else(|_| "/usr/include/aarch64-linux-gnu".into());
    let trt_lib = env::var("TRT_LIB_DIR")
        .unwrap_or_else(|_| "/usr/lib/aarch64-linux-gnu".into());
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

    // ── 4. Link directives ──────────────────────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={trt_lib}");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // ── 5. Version constants ────────────────────────────────────────────────────────────
    println!("cargo:rustc-env=TENSORRT_VERSION_MAJOR=10");
    println!("cargo:rustc-env=TENSORRT_VERSION_MINOR=3");
    println!("cargo:rustc-env=TENSORRT_VERSION_PATCH=0");
    println!("cargo:rustc-env=TENSORRT_VERSION_BUILD=30");

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
