//! kev_tok <gguf> <file>: one string per line (\n escaped as \\n), prints token ids per line. Used by ref/tok_check.py.
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = gguf::Gguf::open(&a[1])?; let t = kev::tok::Tokenizer::from_gguf(&g)?;
    for l in std::fs::read_to_string(&a[2])?.lines() { let s = l.replace("\\n", "\n");
        println!("{}", t.encode_plain(&s).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")); }
    Ok(())
}
