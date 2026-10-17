// Compiles crates/mcv/shaders/*.comp (and the PRESEL gemv3 fork) with glslc into OUT_DIR.
fn glslc(src: &str, out: &str, defs: &[&str]) {
    println!("cargo:rerun-if-changed={src}");
    let mut c = std::process::Command::new("glslc");
    c.args(["--target-env=vulkan1.3", "-O", "-fshader-stage=compute", "-I", "../vkrt/shaders"]);
    for d in defs { c.arg(format!("-D{d}")); }
    let s = c.arg(src).arg("-o").arg(out).status().expect("glslc");
    assert!(s.success(), "glslc failed on {src} {defs:?}");
}
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=shaders");
    for e in std::fs::read_dir("shaders").unwrap() {
        let p = e.unwrap().path();
        if p.extension().map(|e| e == "comp").unwrap_or(false) {
            let stem = p.file_stem().unwrap().to_str().unwrap();
            glslc(p.to_str().unwrap(), &format!("{out}/{stem}.spv"), &[]);
        }
    }
    for l in [4u32, 8, 16, 32] {
        glslc("shaders/mcv_gemv3.inc", &format!("{out}/mcv_moe_l{l}.spv"), &[&format!("LPR={l}u")]);
        glslc("shaders/mcv_gemv3.inc", &format!("{out}/mcv_moe_swiglu_l{l}.spv"), &["SWIGLU", &format!("LPR={l}u")]);
    }
}
