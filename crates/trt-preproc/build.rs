fn main() {
    let cuda_home = std::env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/usr/local/cuda".into());

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
