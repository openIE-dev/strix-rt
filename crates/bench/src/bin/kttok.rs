//! `kttok <model.gguf> <text>...` prints the Rust tokenizer ids for each argument as a JSON line
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = gguf::Gguf::open(&a[1])?; let t = kitten::tok::Tokenizer::from_gguf(&g)?;
    for s in &a[2..] { println!("{:?}", t.encode(s)) }
    Ok(())
}
