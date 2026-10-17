//! Minimal safetensors reader (mmap). Header: u64 LE length + JSON {name: {dtype, shape, data_offsets}}.
use anyhow::{anyhow, Result};
pub struct StTensor { pub name: String, pub dtype: String, pub shape: Vec<usize>, pub start: usize, pub end: usize }
pub struct St { pub mmap: memmap2::Mmap, pub base: usize, pub tensors: Vec<StTensor>, pub meta: String }
fn skip_ws(s: &[u8], p: &mut usize) { while *p < s.len() && (s[*p] as char).is_whitespace() { *p += 1; } }
fn parse_str(s: &[u8], p: &mut usize) -> String { skip_ws(s, p); assert_eq!(s[*p], b'"'); *p += 1; let st = *p; while s[*p] != b'"' { if s[*p] == b'\\' { *p += 1; } *p += 1; } let r = String::from_utf8_lossy(&s[st..*p]).into_owned(); *p += 1; r }
/// skip any JSON value, returning its raw text
fn raw_val(s: &[u8], p: &mut usize) -> String { skip_ws(s, p); let st = *p; let mut depth = 0i32; let mut ins = false;
    loop { let c = s[*p]; if ins { if c == b'\\' { *p += 1; } else if c == b'"' { ins = false; if depth == 0 { *p += 1; break; } } }
        else { match c { b'"' => ins = true, b'{' | b'[' => depth += 1, b'}' | b']' => { if depth == 0 { break; } depth -= 1; if depth == 0 { *p += 1; break; } } b',' if depth == 0 => break, _ => {} } }
        *p += 1; }
    String::from_utf8_lossy(&s[st..*p]).trim().to_string() }
fn field<'a>(obj: &'a str, key: &str) -> Option<&'a str> { let k = format!("\"{key}\""); let i = obj.find(&k)? + k.len(); let r = obj[i..].trim_start().strip_prefix(':')?.trim_start(); Some(r) }
fn nums(s: &str) -> Vec<usize> { let e = s.find(']').unwrap(); s[1..e].split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect() }
impl St {
    pub fn open(path: &str) -> Result<St> {
        let f = std::fs::File::open(path)?; let mmap = unsafe { memmap2::Mmap::map(&f)? };
        let n = u64::from_le_bytes(mmap[0..8].try_into()?) as usize; let h = &mmap[8..8 + n];
        let mut p = 0; skip_ws(h, &mut p); assert_eq!(h[p], b'{'); p += 1;
        let mut tensors = vec![]; let mut meta = String::new();
        loop { skip_ws(h, &mut p); if h[p] == b'}' { break; } if h[p] == b',' { p += 1; continue; }
            let name = parse_str(h, &mut p); skip_ws(h, &mut p); p += 1; let v = raw_val(h, &mut p);
            if name == "__metadata__" { meta = v; continue; }
            let dtype = field(&v, "dtype").map(|d| d.trim_start_matches('"').split('"').next().unwrap().to_string()).ok_or_else(|| anyhow!("dtype"))?;
            let shape = nums(field(&v, "shape").unwrap()); let off = nums(field(&v, "data_offsets").unwrap());
            tensors.push(StTensor { name, dtype, shape, start: off[0], end: off[1] }); }
        Ok(St { mmap, base: 8 + n, tensors, meta })
    }
    pub fn get(&self, name: &str) -> Option<&StTensor> { self.tensors.iter().find(|t| t.name == name) }
    pub fn data(&self, t: &StTensor) -> &[u8] { &self.mmap[self.base + t.start..self.base + t.end] }
    /// read as f32 (supports F32, F16, BF16)
    pub fn f32s(&self, t: &StTensor) -> Vec<f32> { let d = self.data(t); match t.dtype.as_str() {
        "F32" => d.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        "F16" => d.chunks(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        "BF16" => d.chunks(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        o => panic!("dtype {o}") } }
}
