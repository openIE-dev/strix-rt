// Compiles crates/ternary/shaders/tern_*.comp (+ variants) with glslc into OUT_DIR.
fn glslc(src: &str, out: &str, defs: &[String]) {
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
        println!("cargo:rerun-if-changed={}", p.display());
    }
    // GEMV variants: FMT x lanes-per-row (T) x rows-per-cluster (NR)
    for fmt in ["PQ2", "PTQ1", "TQ21", "TH5"] {
        for t in [4u32, 8, 16, 32] {
            for nr in [1u32, 2, 4] {
                glslc("shaders/tern_gemv.comp", &format!("{out}/tern_gemv_{}_t{t}_r{nr}.spv", fmt.to_lowercase()),
                      &[fmt.to_string(), format!("T={t}u"), format!("NR={nr}u")]);
            }
        }
    }
    glslc("shaders/tern_fwht.comp", &format!("{out}/tern_fwht.spv"), &[]);
    glslc("shaders/tern_quantx.comp", &format!("{out}/tern_quantx.spv"), &[]);
    {
        for fmt in ["PQ2", "PTQ1"] {
            glslc("shaders/tern_gemm.comp", &format!("{out}/tern_gemm_{}.spv", fmt.to_lowercase()), &[fmt.to_string()]);
        }
    }
}
