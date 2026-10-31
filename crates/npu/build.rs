// Compiles csrc/xrt_shim.cpp (extern "C" wrapper over the XRT C++ API) with the system g++ and links libxrt_coreutil;
// compiles shaders/*.comp with glslc.
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=csrc/xrt_shim.cpp");
    println!("cargo:rustc-check-cfg=cfg(npu_stub)");
    let have_xrt = std::path::Path::new("/usr/include/xrt/xrt_bo.h").exists();
    if have_xrt {
        let obj = format!("{out}/xrt_shim.o");
        let s = std::process::Command::new("g++")
            .args(["-O2", "-std=c++17", "-fPIC", "-c", "csrc/xrt_shim.cpp", "-o", &obj, "-I/usr/include/xrt"])
            .status().expect("g++");
        assert!(s.success(), "g++ failed on xrt_shim.cpp");
        let s = std::process::Command::new("ar").args(["crs", &format!("{out}/libxrt_shim.a"), &obj]).status().expect("ar");
        assert!(s.success());
        println!("cargo:rustc-link-search=native={out}");
        println!("cargo:rustc-link-lib=static=xrt_shim");
        println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else {
        println!("cargo:warning=XRT headers not found; building npu as a no-op stub (no NPU offload)");
        println!("cargo:rustc-cfg=npu_stub");
    }
    println!("cargo:rerun-if-changed=shaders");
    for e in std::fs::read_dir("shaders").unwrap() {
        let p = e.unwrap().path();
        if p.extension().map(|e| e == "comp").unwrap_or(false) {
            let stem = p.file_stem().unwrap().to_str().unwrap();
            println!("cargo:rerun-if-changed={}", p.display());
            let s = std::process::Command::new("glslc").args(["--target-env=vulkan1.3", "-O", "-fshader-stage=compute"])
                .arg(&p).arg("-o").arg(format!("{out}/{stem}.spv")).status().expect("glslc");
            assert!(s.success(), "glslc failed on {}", p.display());
        }
    }
}
