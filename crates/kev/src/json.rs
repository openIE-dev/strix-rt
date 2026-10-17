//! Minimal JSON (parse + serialize) for the System One server: objects keep key order, numbers keep their text.
use anyhow::{anyhow, bail, Result};
#[derive(Clone, Debug, PartialEq)]
pub enum J { Null, Bool(bool), Num(String), Str(String), Arr(Vec<J>), Obj(Vec<(String, J)>) }
impl J {
    pub fn get(&self, k: &str) -> Option<&J> { match self { J::Obj(v) => v.iter().find(|(a, _)| a == k).map(|x| &x.1), _ => None } }
    pub fn as_str(&self) -> Option<&str> { match self { J::Str(s) => Some(s), _ => None } }
    pub fn num(x: f64) -> J { J::Num(if x.is_finite() { format!("{x}") } else { "null".into() }) }
    pub fn obj(v: Vec<(&str, J)>) -> J { J::Obj(v.into_iter().map(|(k, x)| (k.to_string(), x)).collect()) }
    pub fn dump(&self) -> String { let mut s = String::new(); self.w(&mut s); s }
    fn w(&self, o: &mut String) {
        match self {
            J::Null => o.push_str("null"), J::Bool(b) => o.push_str(if *b { "true" } else { "false" }), J::Num(n) => o.push_str(n),
            J::Str(s) => esc(s, o),
            J::Arr(v) => { o.push('['); for (i, x) in v.iter().enumerate() { if i > 0 { o.push(',') } x.w(o) } o.push(']') }
            J::Obj(v) => { o.push('{'); for (i, (k, x)) in v.iter().enumerate() { if i > 0 { o.push(',') } esc(k, o); o.push(':'); x.w(o) } o.push('}') }
        }
    }
}
fn esc(s: &str, o: &mut String) {
    o.push('"');
    for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
        c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"');
}
pub fn parse(s: &str) -> Result<J> { let b = s.as_bytes(); let mut p = 0; let v = val(b, &mut p)?; ws(b, &mut p); if p != b.len() { bail!("trailing data at {p}") } Ok(v) }
fn ws(b: &[u8], p: &mut usize) { while *p < b.len() && matches!(b[*p], b' ' | b'\n' | b'\r' | b'\t') { *p += 1 } }
fn val(b: &[u8], p: &mut usize) -> Result<J> {
    ws(b, p); let c = *b.get(*p).ok_or_else(|| anyhow!("unexpected end"))?;
    match c {
        b'{' => { *p += 1; let mut v = vec![]; ws(b, p); if b.get(*p) == Some(&b'}') { *p += 1; return Ok(J::Obj(v)) }
            loop { ws(b, p); let k = match val(b, p)? { J::Str(s) => s, _ => bail!("object key must be a string") }; ws(b, p);
                if b.get(*p) != Some(&b':') { bail!("expected ':' at {}", *p) } *p += 1; let x = val(b, p)?; v.push((k, x)); ws(b, p);
                match b.get(*p) { Some(b',') => *p += 1, Some(b'}') => { *p += 1; return Ok(J::Obj(v)) } _ => bail!("expected ',' or '}}' at {}", *p) } } }
        b'[' => { *p += 1; let mut v = vec![]; ws(b, p); if b.get(*p) == Some(&b']') { *p += 1; return Ok(J::Arr(v)) }
            loop { v.push(val(b, p)?); ws(b, p); match b.get(*p) { Some(b',') => *p += 1, Some(b']') => { *p += 1; return Ok(J::Arr(v)) } _ => bail!("expected ',' or ']' at {}", *p) } } }
        b'"' => { *p += 1; let mut s = String::new();
            loop { let c = *b.get(*p).ok_or_else(|| anyhow!("unterminated string"))?; *p += 1;
                match c { b'"' => return Ok(J::Str(s)),
                    b'\\' => { let e = *b.get(*p).ok_or_else(|| anyhow!("bad escape"))?; *p += 1;
                        match e { b'n' => s.push('\n'), b't' => s.push('\t'), b'r' => s.push('\r'), b'b' => s.push('\u{8}'), b'f' => s.push('\u{c}'), b'/' => s.push('/'), b'\\' => s.push('\\'), b'"' => s.push('"'),
                            b'u' => { let h = |p: &mut usize| -> Result<u32> { let t = std::str::from_utf8(b.get(*p..*p + 4).ok_or_else(|| anyhow!("bad \\u"))?)?; *p += 4; Ok(u32::from_str_radix(t, 16)?) };
                                let mut u = h(p)?; if (0xD800..0xDC00).contains(&u) && b.get(*p) == Some(&b'\\') && b.get(*p + 1) == Some(&b'u') { *p += 2; let l = h(p)?; u = 0x10000 + ((u - 0xD800) << 10) + (l - 0xDC00); }
                                s.push(char::from_u32(u).unwrap_or('\u{FFFD}')) }
                            _ => bail!("bad escape") } }
                    _ => { let st = *p - 1; let mut e = *p; while e < b.len() && b[e] != b'"' && b[e] != b'\\' { e += 1 } s.push_str(std::str::from_utf8(&b[st..e])?); *p = e; } } } }
        b't' if b[*p..].starts_with(b"true") => { *p += 4; Ok(J::Bool(true)) }
        b'f' if b[*p..].starts_with(b"false") => { *p += 5; Ok(J::Bool(false)) }
        b'n' if b[*p..].starts_with(b"null") => { *p += 4; Ok(J::Null) }
        b'-' | b'0'..=b'9' => { let st = *p; while *p < b.len() && matches!(b[*p], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { *p += 1 }
            let t = std::str::from_utf8(&b[st..*p])?; t.parse::<f64>().map_err(|_| anyhow!("bad number {t}"))?; Ok(J::Num(t.to_string())) }
        _ => bail!("unexpected '{}' at {}", c as char, *p),
    }
}
