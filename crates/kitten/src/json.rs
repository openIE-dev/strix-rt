//! Minimal JSON reader (objects, arrays, numbers, strings, bools, null) for decoder IR, voices.json, tokenizer files.
use anyhow::{bail, Result};
use std::collections::BTreeMap;
#[derive(Debug, Clone)]
pub enum J { Null, B(bool), N(f64), S(String), A(Vec<J>), O(BTreeMap<String, J>) }
impl J {
    pub fn get(&self, k: &str) -> Option<&J> { if let J::O(m) = self { m.get(k) } else { None } }
    pub fn at(&self, k: &str) -> Result<&J> { self.get(k).ok_or_else(|| anyhow::anyhow!("missing key {k}")) }
    pub fn arr(&self) -> &[J] { if let J::A(v) = self { v } else { &[] } }
    pub fn num(&self) -> f64 { if let J::N(x) = self { *x } else { f64::NAN } }
    pub fn str(&self) -> &str { if let J::S(s) = self { s } else { "" } }
    pub fn obj(&self) -> Option<&BTreeMap<String, J>> { if let J::O(m) = self { Some(m) } else { None } }
    pub fn f32s(&self) -> Vec<f32> { self.arr().iter().map(|x| x.num() as f32).collect() }
    pub fn i64s(&self) -> Vec<i64> { self.arr().iter().map(|x| x.num() as i64).collect() }
}
pub fn parse(s: &str) -> Result<J> { let b = s.as_bytes(); let mut i = 0; let v = val(b, &mut i)?; Ok(v) }
fn ws(b: &[u8], i: &mut usize) { while *i < b.len() && (b[*i] as char).is_ascii_whitespace() { *i += 1 } }
fn val(b: &[u8], i: &mut usize) -> Result<J> {
    ws(b, i);
    if *i >= b.len() { bail!("eof") }
    Ok(match b[*i] {
        b'{' => { *i += 1; let mut m = BTreeMap::new(); loop { ws(b, i); if b[*i] == b'}' { *i += 1; break }
            let k = if let J::S(s) = val(b, i)? { s } else { bail!("key") }; ws(b, i); *i += 1; let v = val(b, i)?; m.insert(k, v); ws(b, i);
            if b[*i] == b',' { *i += 1 } } J::O(m) }
        b'[' => { *i += 1; let mut v = vec![]; loop { ws(b, i); if b[*i] == b']' { *i += 1; break } v.push(val(b, i)?); ws(b, i); if b[*i] == b',' { *i += 1 } } J::A(v) }
        b'"' => { *i += 1; let mut s: Vec<u8> = vec![]; loop { let c = b[*i]; *i += 1; match c {
            b'"' => break,
            b'\\' => { let e = b[*i]; *i += 1; match e { b'n' => s.push(b'\n'), b't' => s.push(b'\t'), b'r' => s.push(b'\r'), b'b' => s.push(8), b'f' => s.push(12),
                b'u' => { let mut cp = u32::from_str_radix(std::str::from_utf8(&b[*i..*i + 4])?, 16)?; *i += 4;
                    if (0xD800..0xDC00).contains(&cp) && b[*i] == b'\\' && b[*i + 1] == b'u' { let lo = u32::from_str_radix(std::str::from_utf8(&b[*i + 2..*i + 6])?, 16)?; *i += 6; cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
                    let ch = char::from_u32(cp).unwrap_or('\u{fffd}'); let mut t = [0u8; 4]; s.extend_from_slice(ch.encode_utf8(&mut t).as_bytes()); }
                o => s.push(o) } }
            o => s.push(o) } } J::S(String::from_utf8_lossy(&s).into_owned()) }
        b't' => { *i += 4; J::B(true) } b'f' => { *i += 5; J::B(false) } b'n' => { *i += 4; J::Null }
        _ if b[*i..].starts_with(b"-Infinity") => { *i += 9; J::N(f64::NEG_INFINITY) }
        b'I' => { *i += 8; J::N(f64::INFINITY) } b'N' => { *i += 3; J::N(f64::NAN) }
        _ => { let st = *i; while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { *i += 1 }
            let t = std::str::from_utf8(&b[st..*i])?; J::N(match t { "Infinity" => f64::INFINITY, _ => t.parse().map_err(|_| anyhow::anyhow!("bad number {t:?} at {st}"))? }) }
    })
}
