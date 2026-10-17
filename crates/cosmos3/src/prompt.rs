//! Cosmos3OmniPipeline prompt construction: duration / resolution templates (inverse ones for the negative prompt),
//! video system prompt, Qwen chat template with generation prompt, then [eos, <|vision_start|>] (start of generation).
//! Tokenized with qimage's byte-level BPE on text_tokenizer/tokenizer.json.
use anyhow::{anyhow, Result};
pub use qimage::tok::Tok;
const SYS_VIDEO: &str = "You are a helpful assistant who will generate videos from a give prompt.";
const SYS_IMAGE: &str = "You are a helpful assistant who will generate images from a give prompt.";
fn append(base: &str, add: &str) -> String { let b = base.trim_end_matches('.'); if b.is_empty() { add.to_string() } else { format!("{b}. {add}") } }
/// Python "{:.1f}" (round half to even on the exact binary value; Rust's formatter does the same).
pub fn templated(text: &str, neg: bool, frames: u32, h: u32, w: u32, fps: f64) -> String {
    let img = frames == 1; let not = if neg { "not " } else { "" };
    let mut t = text.to_string();
    if !img { t = append(&t, &format!("The video is {not}{:.1} seconds long and is {not}of {:.0} FPS.", frames as f64 / fps, fps)); }
    append(&t, &format!("This {} is {not}of {h}x{w} resolution.", if img { "image" } else { "video" }))
}
pub fn ids(tk: &Tok, text: &str, neg: bool, frames: u32, h: u32, w: u32, fps: f64) -> Result<Vec<u32>> {
    let sys = if frames == 1 { SYS_IMAGE } else { SYS_VIDEO };
    let s = format!("<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", templated(text, neg, frames, h, w, fps));
    let mut v = tk.encode(&s);
    v.push(tk.id("<|im_end|>").ok_or(anyhow!("no eos"))?);
    v.push(tk.id("<|vision_start|>").ok_or(anyhow!("no vision_start"))?);
    Ok(v)
}

/// Understanding prompt (Qwen3-VL chat template, one image then the question, generation prompt; no system turn).
pub fn und_ids(tk: &Tok, question: &str, n_img: usize) -> Result<(Vec<u32>, usize)> {
    let id = |s: &str| tk.id(s).ok_or(anyhow!("no {s}"));
    let mut v = tk.encode("<|im_start|>user\n"); v.push(id("<|vision_start|>")?);
    let r0 = v.len(); v.extend(std::iter::repeat(id("<|image_pad|>")?).take(n_img)); v.push(id("<|vision_end|>")?);
    v.extend(tk.encode(&format!("{question}<|im_end|>\n<|im_start|>assistant\n"))); Ok((v, r0))
}
/// Qwen3-VL text positions: text runs count up; an image (grid gh x gw patches, merge 2) at running position p gets
/// (p, p + row, p + col) and advances p by max(gh, gw) / 2.
pub fn und_positions(ids: &[u32], img_tok: u32, gh: usize, gw: usize) -> Vec<[f64; 3]> {
    let (mh, mw) = (gh / 2, gw / 2); let mut out = vec![]; let mut p = 0f64; let mut i = 0;
    while i < ids.len() { if ids[i] == img_tok { for k in 0..mh * mw { out.push([p, p + (k / mw) as f64, p + (k % mw) as f64]); } i += mh * mw; p += mh.max(mw) as f64; }
        else { out.push([p; 3]); p += 1.0; i += 1; } }
    out
}
/// Byte-level BPE detokenizer (inverse of the GPT-2 byte -> unicode map) from tokenizer.json.
pub struct Detok { tok: Vec<String>, u2b: std::collections::HashMap<char, u8> }
impl Detok {
    pub fn load(path: &str) -> Result<Self> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?; let mut tok = vec![String::new(); 160000];
        if let Some(v) = j["model"]["vocab"].as_object() { for (k, x) in v { let i = x.as_u64().unwrap_or(0) as usize; if i < tok.len() { tok[i] = k.clone() } } }
        if let Some(a) = j["added_tokens"].as_array() { for t in a { let i = t["id"].as_u64().unwrap_or(0) as usize; if i < tok.len() { tok[i] = t["content"].as_str().unwrap_or("").to_string() } } }
        let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect(); let mut cs = bs.clone(); let mut n = 0;
        for b in 0..256u32 { if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; } }
        let u2b = bs.iter().zip(&cs).map(|(b, c)| (char::from_u32(*c).unwrap(), *b as u8)).collect();
        Ok(Detok { tok, u2b })
    }
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = vec![]; for &i in ids { let t = &self.tok[i as usize]; if t.starts_with("<|") && t.ends_with("|>") { continue }
            for c in t.chars() { if let Some(x) = self.u2b.get(&c) { b.push(*x) } } }
        String::from_utf8_lossy(&b).into_owned()
    }
}
