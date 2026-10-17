fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).unwrap();
    let all = std::env::var("ALL").is_ok();
    let g = gguf::Gguf::open(&p)?;
    let mut keys: Vec<_> = g.kv.iter().collect(); keys.sort_by_key(|k| k.0.clone());
    for (k, v) in keys { let s = match v { gguf::Value::Arr(a) if a.len() <= 64 => format!("{a:?}"), gguf::Value::Arr(a) => format!("[array len {}]", a.len()), v => format!("{v:?}") }; println!("{k} = {s}"); }
    for t in &g.tensors { if all || t.ty != gguf::GgmlType::F32 || !t.name.contains("norm") { println!("{} {:?} {:?} off={}", t.name, t.ty, t.dims, t.offset); } }
    Ok(())
}
