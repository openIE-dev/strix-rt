//! hf_tok <tokenizer.json> <file>: one string per line (\n escaped as \\n), prints token ids per line (ref/hf_tok_check.py).
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect(); let t = kev::hftok::HfTok::load(&a[1])?;
    for l in std::fs::read_to_string(&a[2])?.lines() { println!("{}", t.encode(&l.replace("\\n", "\n")).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")); }
    Ok(())
}
