//! Glimmer tokenizer: GPT-2 byte-level BPE, llama.cpp "llama4"/GPT4O pre-tokenizer (the lookahead form llama.cpp actually evaluates), vocab+merges
//! from the GGUF (tokenizer.ggml.*). Matches llama-tokenize (add_bos=true, parse_special=true). No NFC, ignore_merges off.
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use crate::ucat::{is_l, is_n};
pub struct Tokenizer { toks: Vec<String>, ids: HashMap<String, u32>, ranks: HashMap<(String, String), usize>, specials: Vec<(String, u32)>, b2u: [char; 256], pub bos: Option<u32> }
/// image marker in prompt text
pub const MEDIA: &str = "<__media__>";
fn byte_map() -> [char; 256] {
    let mut m = [char::from(0u8); 256]; let mut n = 0u32;
    for b in 0..256u32 { let p = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        m[b as usize] = if p { char::from_u32(b).unwrap() } else { let c = char::from_u32(256 + n).unwrap(); n += 1; c }; }
    m
}
impl Tokenizer {
    pub fn from_gguf(g: &gguf::Gguf) -> Result<Self> {
        let arr = |k: &str| -> Result<&Vec<gguf::Value>> { match g.get(k)? { gguf::Value::Arr(v) => Ok(v), _ => Err(anyhow!("{k} not array")) } };
        let toks = arr("tokenizer.ggml.tokens")?; let types = arr("tokenizer.ggml.token_type")?; let merges = arr("tokenizer.ggml.merges")?;
        let mut ids = HashMap::new(); let mut specials = vec![];
        for (i, t) in toks.iter().enumerate() { let s = t.as_str().unwrap_or("").to_string();
            let ty = types.get(i).and_then(|v| v.as_u64()).unwrap_or(1); // 3 control, 4 user-defined
            if (ty == 3 || ty == 4) && !s.is_empty() { specials.push((s.clone(), i as u32)) } ids.insert(s, i as u32); }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut ranks = HashMap::new();
        for (r, m) in merges.iter().enumerate() { if let Some((a, b)) = m.as_str().and_then(|s| s.split_once(' ')) { ranks.insert((a.to_string(), b.to_string()), r); } }
        let bos = g.get("tokenizer.ggml.bos_token_id").ok().and_then(|v| v.as_u64()).map(|v| v as u32);
        let tv: Vec<String> = toks.iter().map(|t| t.as_str().unwrap_or("").to_string()).collect();
        Ok(Tokenizer { toks: tv, ids, ranks, specials, b2u: byte_map(), bos })
    }
    /// add_bos + parse_special, like llama-tokenize defaults
    pub fn encode(&self, text: &str) -> Vec<u32> { let mut o: Vec<u32> = self.bos.into_iter().collect(); o.extend(self.encode_raw(text)); o }
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let mut out = vec![]; let mut rest = text;
        while !rest.is_empty() {
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.specials { if let Some(p) = rest.find(s.as_str()) { if best.map_or(true, |b| p < b.0) { best = Some((p, s.len(), *id)) } } }
            match best { Some((p, l, id)) => { self.plain(&rest[..p], &mut out); out.push(id); rest = &rest[p + l..]; } None => { self.plain(rest, &mut out); break } }
        }
        out
    }
    /// Detokenize (byte-level BPE pieces back to UTF-8, lossy); special/control tokens are printed as their text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut u2b = HashMap::new(); for (b, c) in self.b2u.iter().enumerate() { u2b.insert(*c, b as u8); }
        let mut bytes = vec![];
        for &i in ids { if let Some(t) = self.toks.get(i as usize) { for c in t.chars() { match u2b.get(&c) { Some(&b) => bytes.push(b), None => bytes.extend(c.to_string().bytes()) } } } }
        String::from_utf8_lossy(&bytes).into_owned()
    }
    pub fn id(&self, s: &str) -> Option<u32> { self.ids.get(s).copied() }
    /// Build prompt ids. `media` marks where an image goes; it becomes <|image_start|> + n_img placeholder rows (id 0) + <|image_end|>
    /// (llama.cpp mtmd layout; the text after the marker, typically "\n...", tokenizes as usual). Returns (ids, image row start position).
    pub fn build_prompt(&self, text: &str, n_img: usize) -> (Vec<u32>, Option<usize>) {
        let mut out: Vec<u32> = self.bos.into_iter().collect(); let mut pos = None;
        let (is, ie) = (self.id("<|image_start|>"), self.id("<|image_end|>"));
        for (k, seg) in text.split(MEDIA).enumerate() {
            if k > 0 { if let (Some(a), Some(b)) = (is, ie) { out.push(a); pos = Some(out.len()); out.extend(std::iter::repeat(0).take(n_img)); out.push(b); } }
            out.extend(self.encode_raw(seg));
        }
        (out, pos)
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
fn is_nl(c: char) -> bool { c == '\r' || c == '\n' }
fn is_u(c: char) -> bool { is_l(c) && !c.is_ascii_lowercase() }   // (?=\p{L})[^a-z]
fn is_lw(c: char) -> bool { is_l(c) && !c.is_ascii_uppercase() } // (?=\p{L})[^A-Z]
fn contraction(c: &[char], i: usize) -> usize {
    if i < c.len() && c[i] == char::from(39u8) {
        let l = |k: usize| c.get(i + k).map(|x| x.to_ascii_lowercase());
        match (l(1), l(2)) { (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => return 3, (Some('s' | 't' | 'm' | 'd'), _) => return 2, _ => {} }
    }
    0
}
/// alt1: P? U* Lw+ C?   alt2: P? U+ Lw* C?   (P=[^\r\n\p{L}\p{N}]); tries with the prefix first, then without
fn word_alt(c: &[char], i: usize, alt: u8) -> Option<usize> {
    let n = c.len();
    let body = |j: usize| -> Option<usize> {
        let mut k = j; while k < n && is_u(c[k]) { k += 1 }
        if alt == 1 { // U* then Lw+ ; backtrack U*
            let mut u = k; loop {
                let mut m = u; while m < n && is_lw(c[m]) { m += 1 }
                if m > u { return Some(m + contraction(c, m)) }
                if u == j { return None } u -= 1;
            }
        } else { // U+ then Lw*
            if k == j { return None }
            let mut m = k; while m < n && is_lw(c[m]) { m += 1 }
            Some(m + contraction(c, m))
        }
    };
    if !is_nl(c[i]) && !is_l(c[i]) && !is_n(c[i]) { if let Some(e) = body(i + 1) { return Some(e) } }
    body(i)
}
pub fn pretokenize(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect(); let n = c.len(); let mut out = vec![]; let mut i = 0;
    while i < n {
        let s = i;
        if let Some(e) = word_alt(&c, i, 1).or_else(|| word_alt(&c, i, 2)) { out.push(c[s..e].iter().collect()); i = e; continue }
        if is_n(c[i]) { let mut j = i; while j < n && j - i < 3 && is_n(c[j]) { j += 1 } out.push(c[i..j].iter().collect()); i = j; continue }
        { let mut j = i; if c[j] == ' ' { j += 1 } let st = j;
          while j < n && !c[j].is_whitespace() && !is_l(c[j]) && !is_n(c[j]) { j += 1 }
          if j > st { while j < n && (is_nl(c[j]) || c[j] == '/') { j += 1 } out.push(c[s..j].iter().collect()); i = j; continue } }
        let mut j = i; while j < n && c[j].is_whitespace() { j += 1 }
        if let Some(lastnl) = (i..j).rev().find(|&k| is_nl(c[k])) { out.push(c[i..=lastnl].iter().collect()); i = lastnl + 1; continue }
        if j < n && j - i > 1 { out.push(c[i..j - 1].iter().collect()); i = j - 1; continue }
        out.push(c[i..j].iter().collect()); i = j;
    }
    out
}
