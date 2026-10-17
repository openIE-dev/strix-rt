//! GGUF reader that tolerates the fork-private ternary types (crates/gguf asserts known block sizes).
use anyhow::{bail, Result};
pub use gguf::Value;
use std::collections::HashMap;
pub struct TInfo { pub name: String, pub dims: Vec<u64>, pub ty: u32, pub offset: u64 }
pub struct Gg { pub mmap: memmap2::Mmap, pub kv: HashMap<String, Value>, pub tensors: Vec<TInfo>, pub data_offset: usize }
struct Rd<'a> { b: &'a [u8], p: usize }
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> { if self.p + n > self.b.len() { bail!("gguf truncated") } let s = &self.b[self.p..self.p + n]; self.p += n; Ok(s) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> Result<String> { let n = self.u64()? as usize; Ok(String::from_utf8_lossy(self.take(n)?).into_owned()) }
    fn val(&mut self, t: u32) -> Result<Value> { Ok(match t {
        0 => Value::U8(self.take(1)?[0]), 1 => Value::I8(self.take(1)?[0] as i8), 2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)), 3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
        4 => Value::U32(self.u32()?), 5 => Value::I32(self.u32()? as i32), 6 => Value::F32(f32::from_bits(self.u32()?)), 7 => Value::Bool(self.take(1)?[0] != 0), 8 => Value::Str(self.str()?),
        9 => { let et = self.u32()?; let n = self.u64()? as usize; let mut v = Vec::with_capacity(n.min(1 << 20)); for _ in 0..n { v.push(self.val(et)?) } Value::Arr(v) }
        10 => Value::U64(self.u64()?), 11 => Value::I64(self.u64()? as i64), 12 => Value::F64(f64::from_bits(self.u64()?)), _ => bail!("bad value type {t}") }) }
}
/// (block elems, block bytes) for the ggml types we care about
pub fn type_block(t: u32) -> (usize, usize) { match t { 0 => (1, 4), 1 => (1, 2), 30 => (1, 2), 8 => (32, 34), 12 => (256, 144), 13 => (256, 176), 14 => (256, 210), 142 => (128, 34), 143 => (128, 28), 43 => (256, 68), 35 => (256, 66), 34 => (256, 54), _ => (0, 0) } }
impl TInfo { pub fn n_bytes(&self) -> usize { let (e, b) = type_block(self.ty); if e == 0 { 0 } else { self.dims.iter().product::<u64>() as usize / e * b } } }
impl Gg {
    pub fn open(path: &str) -> Result<Gg> {
        let f = std::fs::File::open(path)?; let mmap = unsafe { memmap2::Mmap::map(&f)? };
        let (kv, tensors, data_offset) = { let mut r = Rd { b: &mmap[..], p: 0 };
            if r.take(4)? != b"GGUF" { bail!("not gguf") } let _v = r.u32()?; let nt = r.u64()? as usize; let nkv = r.u64()? as usize;
            let mut kv = HashMap::new(); for _ in 0..nkv { let k = r.str()?; let t = r.u32()?; let v = r.val(t)?; kv.insert(k, v); }
            let mut ts = vec![]; for _ in 0..nt { let name = r.str()?; let nd = r.u32()? as usize; let mut dims = vec![]; for _ in 0..nd { dims.push(r.u64()?) } let ty = r.u32()?; let offset = r.u64()?; ts.push(TInfo { name, dims, ty, offset }); }
            let al = kv.get("general.alignment").and_then(|v: &Value| v.as_u64()).unwrap_or(32) as usize; (kv, ts, r.p.div_ceil(al) * al) };
        Ok(Gg { mmap, kv, tensors, data_offset })
    }
    pub fn tensor(&self, n: &str) -> Option<&TInfo> { self.tensors.iter().find(|t| t.name == n) }
    pub fn data(&self, t: &TInfo) -> &[u8] { let s = self.data_offset + t.offset as usize; &self.mmap[s..s + t.n_bytes()] }
}
