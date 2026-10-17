//! Qwen2/Qwen3 byte-level BPE from tokenizer.json (NFC, added tokens split first, qwen2 pre-tokenizer from kev::tok).
use anyhow::{anyhow, Result};
use kev::json::{self, J};
use std::collections::HashMap;
pub struct Tok { ids: HashMap<String, u32>, ranks: HashMap<(String, String), usize>, added: Vec<(String, u32)>, b2u: [char; 256] }
impl Tok {
    pub fn load(path: &str) -> Result<Self> {
        let j = json::parse(&std::fs::read_to_string(path)?)?; let m = j.get("model").ok_or_else(|| anyhow!("no model"))?;
        let num = |x: &J| match x { J::Num(t) => t.parse().unwrap_or(0), _ => 0 };
        let mut ids = HashMap::new(); if let Some(J::Obj(v)) = m.get("vocab") { for (k, x) in v { ids.insert(k.clone(), num(x)); } }
        let mut ranks = HashMap::new();
        if let Some(J::Arr(v)) = m.get("merges") { for (r, x) in v.iter().enumerate() { let p = match x {
            J::Arr(ab) if ab.len() == 2 => (ab[0].as_str().unwrap_or("").to_string(), ab[1].as_str().unwrap_or("").to_string()),
            J::Str(s) => { let (a, b) = s.split_once(' ').ok_or_else(|| anyhow!("merge"))?; (a.to_string(), b.to_string()) } _ => continue };
            ranks.entry(p).or_insert(r); } }
        let mut added = vec![];
        if let Some(J::Arr(v)) = j.get("added_tokens") { for a in v { let s = a.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string(); let id = num(a.get("id").unwrap_or(&J::Null)); ids.insert(s.clone(), id); added.push((s, id)); } }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Tok { ids, ranks, added, b2u: kev::tok::byte_map() })
    }
    pub fn id(&self, s: &str) -> Option<u32> { self.ids.get(s).copied() }
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = vec![]; let mut rest = text;
        while !rest.is_empty() {
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.added { if let Some(p) = rest.find(s.as_str()) { if best.map_or(true, |b| p < b.0) { best = Some((p, s.len(), *id)) } } }
            match best { Some((p, l, id)) => { self.plain(&rest[..p], &mut out); out.push(id); rest = &rest[p + l..]; } None => { self.plain(rest, &mut out); break } }
        }
        out
    }
    fn plain(&self, text: &str, out: &mut Vec<u32>) {
        use unicode_normalization::UnicodeNormalization; let t: String = text.nfc().collect();
        for w in kev::tok::pretokenize(&t, false) {
            let mut parts: Vec<String> = w.bytes().map(|b| self.b2u[b as usize].to_string()).collect();
            loop { let mut best: Option<(usize, usize)> = None;
                for i in 0..parts.len().saturating_sub(1) { if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) { if best.map_or(true, |b| r < b.0) { best = Some((r, i)) } } }
                let Some((_, i)) = best else { break }; let j = parts.remove(i + 1); parts[i].push_str(&j); }
            for p in parts { if let Some(&id) = self.ids.get(&p) { out.push(id) } }
        }
    }
}
/// Qwen-Image-2.1 text-to-image prompt template; returns (ids, drop_idx = system-prompt token count).
pub fn prompt_ids(t: &Tok, prompt: &str) -> (Vec<u32>, usize) {
    let sys = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    let p = if prompt.is_empty() { " " } else { prompt };
    (t.encode(&format!("{sys}<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")), t.encode(sys).len())
}
