// Compiles shaders/*.comp; gemv.comp is compiled into several variants via -D defines.
fn glslc(src: &str, out: &str, defs: &[&str]) {
    println!("cargo:rerun-if-changed={src}");
    let mut c = std::process::Command::new("glslc");
    c.args(["--target-env=vulkan1.3", "-O", "-fshader-stage=compute"]);
    for d in defs { c.arg(format!("-D{d}")); }
    let s = c.arg(src).arg("-o").arg(out).status().expect("glslc");
    assert!(s.success(), "glslc failed on {src} {defs:?}");
}
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=shaders");
    for e in std::fs::read_dir("shaders").unwrap() {
        let p = e.unwrap().path();
        let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
        if p.extension().map(|e| e == "comp").unwrap_or(false) && stem != "gemv" && stem != "gemvq" {
            glslc(p.to_str().unwrap(), &format!("{out}/{stem}.spv"), &[]);
        }
    }
    println!("cargo:rerun-if-changed=shaders/q4k6k.glsl");
    println!("cargo:rerun-if-changed=shaders/gemv3.inc");
    glslc("shaders/gemv3.inc", &format!("{out}/gemv3.spv"), &[]);
    glslc("shaders/gemv3.inc", &format!("{out}/gemv3_swiglu.spv"), &["SWIGLU"]);
    for l in [4, 8, 16] {
        glslc("shaders/gemv3.inc", &format!("{out}/gemv3_l{l}.spv"), &[&format!("LPR={l}u")]);
        glslc("shaders/gemv3.inc", &format!("{out}/gemv3_swiglu_l{l}.spv"), &["SWIGLU", &format!("LPR={l}u")]);
    }
    // wide aux layout (sq slots 0..128, key at word 128, pos 160, prompt len 161) for models with > 31 layers
    glslc("shaders/argmax.comp", &format!("{out}/argmax_w.spv"), &["AUX_KEY64=64u"]);
    glslc("shaders/sample.comp", &format!("{out}/sample_w.spv"), &["AUX_KEY=128u", "AUX_POS=160u", "AUX_PLEN=161u", "AUX_PRED=162u"]);
    println!("cargo:rerun-if-changed=shaders/q_any.glsl");
    for l in [4, 8, 16, 32] {
        glslc("shaders/gemvq.comp", &format!("{out}/gemvq_l{l}.spv"), &[&format!("LPR={l}u")]);
        glslc("shaders/gemvq.comp", &format!("{out}/gemvq_swiglu_l{l}.spv"), &["SWIGLU", &format!("LPR={l}u")]);
    }
    glslc("shaders/embed_q4k.comp", &format!("{out}/embed_q4k_xw.spv"), &["AUX_POS=160u", "AUX_KEY=128u", "AUX_XWIDE"]);
    glslc("shaders/gemv.comp", &format!("{out}/gemv2_q4k.spv"), &["Q4K"]);
    glslc("shaders/gemv.comp", &format!("{out}/gemv2_q6k.spv"), &["Q6K"]);
    glslc("shaders/gemv.comp", &format!("{out}/gemv2_q4k_swiglu.spv"), &["Q4K", "SWIGLU"]);
}
