//! GPT-2 byte-level BPE with the qwen2 pre-tokenizer, read from the GGUF vocab (tokenizer.ggml.*), matching
//! llama_tokenize(add_special=false, parse_special=true) as used by kitten-tts-2-cpp.
use anyhow::{anyhow, Result};
use std::collections::HashMap;
pub struct Tokenizer { ids: HashMap<String, u32>, ranks: HashMap<(String, String), usize>, specials: Vec<(String, u32)>, b2u: [char; 256] }
fn byte_map() -> [char; 256] {
    let mut m = ['\0'; 256]; let mut n = 0u32;
    for b in 0..256u32 { let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        m[b as usize] = if printable { char::from_u32(b).unwrap() } else { let c = char::from_u32(256 + n).unwrap(); n += 1; c }; }
    m
}
impl Tokenizer {
    pub fn from_gguf(g: &gguf::Gguf) -> Result<Self> {
        let arr = |k: &str| -> Result<&Vec<gguf::Value>> { match g.get(k)? { gguf::Value::Arr(v) => Ok(v), _ => Err(anyhow!("{k} not array")) } };
        let toks = arr("tokenizer.ggml.tokens")?; let types = arr("tokenizer.ggml.token_type")?; let merges = arr("tokenizer.ggml.merges")?;
        let mut ids = HashMap::new(); let mut specials = vec![];
        for (i, t) in toks.iter().enumerate() { let s = t.as_str().unwrap_or("").to_string();
            let ty = types.get(i).and_then(|v| v.as_u64()).unwrap_or(1); // 3 control, 4 user-defined
            if ty == 3 || ty == 4 { specials.push((s.clone(), i as u32)) } ids.insert(s, i as u32); }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut ranks = HashMap::new();
        for (r, m) in merges.iter().enumerate() { if let Some((a, b)) = m.as_str().and_then(|s| s.split_once(' ')) { ranks.insert((a.to_string(), b.to_string()), r); } }
        Ok(Tokenizer { ids, ranks, specials, b2u: byte_map() })
    }
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = vec![]; let mut rest = text;
        while !rest.is_empty() {
            let mut best: Option<(usize, usize, u32)> = None; // (pos, len, id): earliest, longest first on ties
            for (s, id) in &self.specials { if s.is_empty() { continue } if let Some(p) = rest.find(s.as_str()) { if best.map_or(true, |b| p < b.0) { best = Some((p, s.len(), *id)) } } }
            match best { Some((p, l, id)) => { self.plain(&rest[..p], &mut out); out.push(id); rest = &rest[p + l..]; } None => { self.plain(rest, &mut out); break } }
        }
        out
    }
    fn plain(&self, text: &str, out: &mut Vec<u32>) { for w in pretokenize(text) { self.bpe(&w, out) } }
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.bytes().map(|b| self.b2u[b as usize].to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) { if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) { if best.map_or(true, |b| r < b.0) { best = Some((r, i)) } } }
            let Some((_, i)) = best else { break }; let j = parts.remove(i + 1); parts[i].push_str(&j);
        }
        for p in parts { match self.ids.get(&p) { Some(&id) => out.push(id), None => for c in p.chars() { if let Some(&id) = self.ids.get(&c.to_string()) { out.push(id) } } } }
    }
}
fn is_l(c: char) -> bool { c.is_alphabetic() }
fn is_n(c: char) -> bool { c.is_numeric() }
fn is_nl(c: char) -> bool { c == '\r' || c == '\n' }
/// qwen2: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
pub fn pretokenize(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect(); let n = c.len(); let mut out = vec![]; let mut i = 0;
    while i < n {
        let s = i;
        if c[i] == '\'' && i + 1 < n {
            let l = |k: usize| c.get(i + k).map(|x| x.to_ascii_lowercase());
            let len = match (l(1), l(2)) { (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3, (Some('s' | 't' | 'm' | 'd'), _) => 2, _ => 0 };
            if len > 0 { out.push(c[i..i + len].iter().collect()); i += len; continue }
        }
        { let mut j = i; if !is_nl(c[j]) && !is_l(c[j]) && !is_n(c[j]) && j + 1 < n && is_l(c[j + 1]) { j += 1 }
          if is_l(c[j]) { while j < n && is_l(c[j]) { j += 1 } out.push(c[s..j].iter().collect()); i = j; continue } }
        if is_n(c[i]) { out.push(c[i].to_string()); i += 1; continue }
        { let mut j = i; if c[j] == ' ' { j += 1 } let st = j;
          while j < n && !c[j].is_whitespace() && !is_l(c[j]) && !is_n(c[j]) { j += 1 }
          if j > st { while j < n && is_nl(c[j]) { j += 1 } out.push(c[s..j].iter().collect()); i = j; continue } }
        let mut j = i; while j < n && c[j].is_whitespace() { j += 1 }
        if let Some(lastnl) = (i..j).rev().find(|&k| is_nl(c[k])) { out.push(c[i..=lastnl].iter().collect()); i = lastnl + 1; continue }
        if j < n && j - i > 1 { out.push(c[i..j - 1].iter().collect()); i = j - 1; continue }
        out.push(c[i..j].iter().collect()); i = j;
    }
    out
}
