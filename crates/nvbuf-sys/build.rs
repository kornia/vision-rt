fn main() {
    let cuda_inc = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/nvbuf_shim.cpp")
        .include("/usr/src/jetson_multimedia_api/include")
        .include(format!("{cuda_inc}/include"))
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("nvbuf_shim");

    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu/nvidia");
    println!("cargo:rustc-link-search=native={cuda_inc}/lib64");
    println!("cargo:rustc-link-lib=dylib=nvbufsurface");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/nvbuf_shim.cpp");
}
