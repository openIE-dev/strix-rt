// Compiles shaders/*.comp with glslc into OUT_DIR.
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=shaders");
    for e in std::fs::read_dir("shaders").unwrap() {
        let p = e.unwrap().path(); let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
        let s = std::process::Command::new("glslc").args(["--target-env=vulkan1.3", "-O", "-fshader-stage=compute"]).arg(&p).arg("-o").arg(format!("{out}/{stem}.spv")).status().expect("glslc");
        assert!(s.success(), "glslc {}", p.display());
    }
}
