// Compiles crates/parakeet/shaders/pk_*.comp (+ -D variants) with glslc into OUT_DIR.
fn glslc(src: &str, out: &str, defs: &[&str]) {
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
        let p = e.unwrap().path(); println!("cargo:rerun-if-changed={}", p.display());
        let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
        if p.extension().map(|e| e == "comp").unwrap_or(false) { glslc(p.to_str().unwrap(), &format!("{out}/{stem}.spv"), &[]); }
    }
}
