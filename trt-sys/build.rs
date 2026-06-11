use std::env;
use std::path::PathBuf;

fn main() {
    // ── Rerun triggers ────────────────────────────────────────────────────────
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shim.cpp");
    println!("cargo:rerun-if-changed=include/shim.h");

    // ── Path resolution ───────────────────────────────────────────────────────
    // Override via environment variables for cross-compilation or non-standard installs.
    let trt_inc = env::var("TRT_INCLUDE_DIR")
        .unwrap_or_else(|_| "/usr/include/aarch64-linux-gnu".into());
    let trt_lib = env::var("TRT_LIB_DIR")
        .unwrap_or_else(|_| "/usr/lib/aarch64-linux-gnu".into());
    let cuda_home = env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    let cuda_inc = format!("{}/include", cuda_home);
    let cuda_lib = format!("{}/lib64", cuda_home);

    // ── TensorRT version constants ────────────────────────────────────────────
    // Confirmed from NvInferVersion.h: NV_TENSORRT_MAJOR=10, MINOR=3, PATCH=0, BUILD=30
    // Update these when upgrading TensorRT (search for NV_TENSORRT_ in NvInferVersion.h).
    println!("cargo:rustc-env=TENSORRT_VERSION_MAJOR=10");
    println!("cargo:rustc-env=TENSORRT_VERSION_MINOR=3");
    println!("cargo:rustc-env=TENSORRT_VERSION_PATCH=0");
    println!("cargo:rustc-env=TENSORRT_VERSION_BUILD=30");

    // ── Compile the C++ shim ──────────────────────────────────────────────────
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/shim.cpp")
        .include(&trt_inc)
        .include(&cuda_inc)
        .include("include")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("btrt_shim");
    // cc::Build::compile() emits cargo:rustc-link-lib=static=btrt_shim automatically.

    // ── Link directives ───────────────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={}", trt_lib);
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // ── Run bindgen over the pure-C shim header ───────────────────────────────
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let bindings = bindgen::Builder::default()
        .header("include/shim.h")
        .allowlist_function("btrt_.*")
        .allowlist_type("btrt_.*")
        .generate()
        .expect("bindgen failed to generate bindings from include/shim.h");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
