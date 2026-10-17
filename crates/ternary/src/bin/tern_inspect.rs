//! tern_inspect <file.gguf|file.safetensors> [filter]: dump KV (arrays truncated) and tensors.
use ternary::gg::{Gg, Value};
fn short(v: &Value) -> String { match v { Value::Arr(a) => format!("[{} items: {}{}]", a.len(), a.iter().take(8).map(short).collect::<Vec<_>>().join(", "), if a.len() > 8 { ", ..." } else { "" }), Value::Str(s) => format!("{:?}", if s.len() > 200 { &s[..200] } else { s }), o => format!("{o:?}") } }
fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).unwrap(); let filt = std::env::args().nth(2).unwrap_or_default();
    if p.ends_with(".safetensors") {
        let s = ternary::st::St::open(&p)?; println!("meta: {}", &s.meta[..s.meta.len().min(2000)]);
        let mut hist = std::collections::BTreeMap::new();
        for t in &s.tensors { *hist.entry(t.dtype.clone()).or_insert(0usize) += t.end - t.start; if t.name.contains(&filt) { println!("{} {} {:?} {}", t.name, t.dtype, t.shape, t.end - t.start); } }
        println!("bytes by dtype {hist:?}"); return Ok(());
    }
    let g = Gg::open(&p)?;
    let mut keys: Vec<_> = g.kv.keys().collect(); keys.sort();
    for k in keys { println!("{k} = {}", short(&g.kv[k])); }
    let mut hist = std::collections::BTreeMap::new();
    for t in &g.tensors { let ty = t.ty;
        *hist.entry(ty).or_insert(0usize) += 1; if t.name.contains(&filt) { println!("{} ty={} dims={:?} off={}", t.name, ty, t.dims, t.offset); } }
    println!("ggml type histogram {hist:?}");
    Ok(())
}
