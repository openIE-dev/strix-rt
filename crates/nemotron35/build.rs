// Compiles ../nemotron/shaders/*.comp and ../mcv/shaders/*.comp (reused unchanged; only mcv_vpad is used) and shaders/*.comp (Nemotron 3.5 additions).
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    for dir in ["../nemotron/shaders", "../mcv/shaders", "shaders"] {
        println!("cargo:rerun-if-changed={dir}");
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().map(|e| e == "comp").unwrap_or(false) && (dir != "../mcv/shaders" || p.ends_with("mcv_vpad.comp")) {
                let stem = p.file_stem().unwrap().to_str().unwrap();
                println!("cargo:rerun-if-changed={}", p.display());
                let s = std::process::Command::new("glslc").args(["--target-env=vulkan1.3", "-O", "-fshader-stage=compute", "-I../vkrt/shaders", "-I../nemotron/shaders"])
                    .arg(&p).arg("-o").arg(format!("{out}/{stem}.spv")).status().expect("glslc");
                assert!(s.success(), "glslc failed on {}", p.display());
            }
        }
    }
}
