fn main() {
    // Stub mode (docs.rs / hosted CI without CUDA): skip the C++ helper
    // compile and link directives. `cargo check`/`clippy` typecheck the FFI
    // without linking; anything that links or runs requires a real CUDA install.
    if std::env::var("DOCS_RS").is_ok() || std::env::var("TRT_STUB").is_ok() {
        println!("cargo:rerun-if-env-changed=TRT_STUB");
        return;
    }
    println!("cargo:rerun-if-env-changed=TRT_STUB");

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/preproc_helpers.cpp")
        .include(format!("{cuda_home}/include"))
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("preproc_helpers");

    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/preproc_helpers.cpp");
}
