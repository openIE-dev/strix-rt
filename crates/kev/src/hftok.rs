//! HF tokenizer.json byte-level BPE (ModernBERT / OLMo vocab): NFC, added-token splitting (leftmost-longest, lstrip/rstrip,
//! non-normalized tokens first as HF's AddedVocabulary does), GPT-2 pre-tokenizer regex, ranked merges. From scratch.
use crate::json::{self, J};
use crate::ucat::{is_l, is_n};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

struct Added { s: Vec<char>, id: u32, lstrip: bool, rstrip: bool, normalized: bool }
pub struct HfTok { ids: HashMap<String, u32>, ranks: HashMap<(String, String), usize>, added: Vec<Added>, b2u: [char; 256], nfc: bool }
enum Piece { Text(String), Tok(u32) }

impl HfTok {
    pub fn load(path: &str) -> Result<Self> {
        let j = json::parse(&std::fs::read_to_string(path)?)?;
        let m = j.get("model").ok_or_else(|| anyhow!("no model"))?;
        anyhow::ensure!(m.get("type").and_then(|t| t.as_str()) == Some("BPE"), "only BPE tokenizer.json is supported");
        let num = |x: &J| -> u32 { match x { J::Num(t) => t.parse().unwrap_or(0), _ => 0 } };
        let mut ids = HashMap::new();
        if let Some(J::Obj(v)) = m.get("vocab") { for (k, x) in v { ids.insert(k.clone(), num(x)); } }
        let mut ranks = HashMap::new();
        if let Some(J::Arr(v)) = m.get("merges") { for (r, x) in v.iter().enumerate() {
            let p = match x { J::Arr(ab) if ab.len() == 2 => (ab[0].as_str().unwrap_or("").to_string(), ab[1].as_str().unwrap_or("").to_string()),
                J::Str(s) => { let (a, b) = s.split_once(' ').ok_or_else(|| anyhow!("bad merge"))?; (a.to_string(), b.to_string()) } _ => continue };
            ranks.entry(p).or_insert(r); } }
        let flag = |a: &J, k: &str| matches!(a.get(k), Some(J::Bool(true)));
        let mut added = vec![];
        if let Some(J::Arr(v)) = j.get("added_tokens") { for a in v { let s = a.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
            let id = num(a.get("id").unwrap_or(&J::Null)); ids.insert(s.clone(), id);
            added.push(Added { s: s.chars().collect(), id, lstrip: flag(a, "lstrip"), rstrip: flag(a, "rstrip"), normalized: flag(a, "normalized") && !flag(a, "special") }); } }
        let nfc = j.get("normalizer").and_then(|n| n.get("type")).and_then(|t| t.as_str()) == Some("NFC");
        Ok(HfTok { ids, ranks, added, b2u: crate::tok::byte_map(), nfc })
    }
    pub fn id(&self, s: &str) -> Option<u32> { self.ids.get(s).copied() }
    /// tokenizer(text, add_special_tokens=False)["input_ids"]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = vec![];
        for p in self.split(vec![Piece::Text(text.to_string())], false) { match p { Piece::Tok(id) => out.push(id), Piece::Text(t) => {
            let t: String = if self.nfc { t.nfc().collect() } else { t };
            for q in self.split(vec![Piece::Text(t)], true) { match q { Piece::Tok(id) => out.push(id), Piece::Text(t) => for w in gpt2_pretok(&t) { self.bpe(&w, &mut out) } } } } } }
        out
    }
    fn split(&self, pieces: Vec<Piece>, normalized: bool) -> Vec<Piece> {
        let set: Vec<&Added> = self.added.iter().filter(|a| a.normalized == normalized && !a.s.is_empty()).collect();
        let mut out = vec![];
        for p in pieces { let t = match p { Piece::Text(t) => t, x => { out.push(x); continue } };
            let c: Vec<char> = t.chars().collect(); let (mut i, mut last) = (0usize, 0usize);
            while i < c.len() {
                let best = set.iter().filter(|a| c[i..].starts_with(&a.s)).max_by_key(|a| a.s.len());
                let Some(a) = best else { i += 1; continue };
                let (mut s, mut e) = (i, i + a.s.len());
                if a.lstrip { while s > last && c[s - 1].is_whitespace() { s -= 1 } }
                if a.rstrip { while e < c.len() && c[e].is_whitespace() { e += 1 } }
                if s > last { out.push(Piece::Text(c[last..s].iter().collect())) }
                out.push(Piece::Tok(a.id)); i = e; last = e;
            }
            if last < c.len() { out.push(Piece::Text(c[last..].iter().collect())) } }
        out
    }
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.bytes().map(|b| self.b2u[b as usize].to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) { if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) { if best.map_or(true, |b| r < b.0) { best = Some((r, i)) } } }
            let Some((_, i)) = best else { break }; let j = parts.remove(i + 1); parts[i].push_str(&j);
        }
        for p in parts { if let Some(&id) = self.ids.get(&p) { out.push(id) } }
    }
}

/// GPT-2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
pub fn gpt2_pretok(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect(); let n = c.len(); let mut out = vec![]; let mut i = 0;
    let other = |x: char| !x.is_whitespace() && !is_l(x) && !is_n(x);
    while i < n {
        if c[i] == '\'' {
            let len = match (c.get(i + 1), c.get(i + 2)) { (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3, (Some('s' | 't' | 'm' | 'd'), _) => 2, _ => 0 };
            if len > 0 { out.push(c[i..i + len].iter().collect()); i += len; continue }
        }
        let st = if c[i] == ' ' { i + 1 } else { i };
        if st < n {
            let classes: [&dyn Fn(char) -> bool; 3] = [&is_l, &is_n, &other];
            if let Some(cl) = classes.iter().find(|cl| cl(c[st])) { let mut j = st; while j < n && cl(c[j]) { j += 1 } out.push(c[i..j].iter().collect()); i = j; continue }
        }
        let mut j = i; while j < n && c[j].is_whitespace() { j += 1 }
        let e = if j < n && j - i > 1 { j - 1 } else { j };
        out.push(c[i..e].iter().collect()); i = e;
    }
    out
}
