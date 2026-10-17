//! glm_tok <model.gguf> <prompt text | @file(json array of strings)>: print token ids (add_bos, parse_special) one list per prompt.
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = gguf::Gguf::open(&a[1])?; let t = glimmer::tok::Tokenizer::from_gguf(&g)?;
    let prompts: Vec<String> = if let Some(f) = a[2].strip_prefix('@') { let s = std::fs::read_to_string(f)?; let v: serde_json::Value = serde_json::from_str(&s)?; v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() } else { vec![a[2].clone()] };
    for p in prompts { println!("{}", t.encode(&p).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")); }
    Ok(())
}
