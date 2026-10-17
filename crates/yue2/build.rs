// Compiles crates/yue2/shaders/ye_*.comp with glslc into OUT_DIR.
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=shaders");
    for e in std::fs::read_dir("shaders").unwrap() {
        let p = e.unwrap().path(); println!("cargo:rerun-if-changed={}", p.display());
        if p.extension().map(|e| e == "comp").unwrap_or(false) {
            let stem = p.file_stem().unwrap().to_str().unwrap();
            let s = std::process::Command::new("glslc").args(["--target-env=vulkan1.3", "-O", "-fshader-stage=compute"]).arg(&p).arg("-o").arg(format!("{out}/{stem}.spv")).status().expect("glslc");
            assert!(s.success(), "glslc failed on {}", p.display());
        }
    }
}
