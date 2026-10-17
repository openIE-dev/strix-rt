//! Minimal GGUF v2/v3 reader with zero-copy mmap access to tensor data.
use anyhow::{anyhow, bail, Result};
use memmap2::Mmap;
use std::{collections::HashMap, fs::File, path::Path};
pub mod dequant;
pub mod iq3s_grid;

#[derive(Debug, Clone)]
pub enum Value { U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool), Str(String), Arr(Vec<Value>), U64(u64), I64(i64), F64(f64) }
impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self { Value::U8(v) => Some(v as u64), Value::U16(v) => Some(v as u64), Value::U32(v) => Some(v as u64), Value::U64(v) => Some(v),
            Value::I32(v) if v >= 0 => Some(v as u64), Value::I64(v) if v >= 0 => Some(v as u64), _ => None }
    }
    pub fn as_f32(&self) -> Option<f32> { match *self { Value::F32(v) => Some(v), Value::F64(v) => Some(v as f32), _ => None } }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType { F32, F16, Q4_0, Q8_0, Q4K, Q5K, Q6K, Other(u32) }
impl GgmlType {
    pub fn from_u32(v: u32) -> Self { use GgmlType::*; match v { 0 => F32, 1 => F16, 2 => Q4_0, 8 => Q8_0, 12 => Q4K, 13 => Q5K, 14 => Q6K, o => Other(o) } }
    /// (elements per block, bytes per block)
    pub fn block(&self) -> (usize, usize) { use GgmlType::*; match self { F32 => (1, 4), F16 => (1, 2), Q4_0 => (32, 18), Q8_0 => (32, 34), Q4K => (256, 144), Q5K => (256, 176), Q6K => (256, 210),
        // other ggml types (sizes only, for layout; no dequant yet): Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, IQ4_NL, IQ4_XS, BF16
        Other(3) => (32, 20), Other(6) => (32, 22), Other(7) => (32, 24), Other(10) => (256, 84), Other(11) => (256, 110), Other(20) => (32, 18), Other(21) => (256, 110), Other(23) => (256, 136), Other(30) => (1, 2), Other(142) => (128, 34), Other(143) => (128, 28), Other(43) => (256, 68), Other(39) => (32, 17), Other(_) => (0, 0) } }
}

#[derive(Debug, Clone)]
pub struct TensorInfo { pub name: String, pub dims: Vec<u64>, pub ty: GgmlType, pub offset: u64 }
impl TensorInfo {
    pub fn n_elems(&self) -> u64 { self.dims.iter().product() }
    pub fn n_bytes(&self) -> usize { let (b, s) = self.ty.block(); if b == 0 { return 0 } self.n_elems() as usize / b * s }
    /// bytes per row (dims[0] is the contiguous/inner dimension)
    pub fn row_bytes(&self) -> usize { let (b, s) = self.ty.block(); self.dims[0] as usize / b * s }
}

pub struct Gguf { pub mmap: Mmap, pub version: u32, pub kv: HashMap<String, Value>, pub tensors: Vec<TensorInfo>, pub data_offset: usize, pub alignment: usize }

struct Rd<'a> { b: &'a [u8], p: usize }
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> { if self.p + n > self.b.len() { bail!("gguf truncated") } let s = &self.b[self.p..self.p + n]; self.p += n; Ok(s) }
    fn u8(&mut self) -> Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into()?)) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> Result<String> { let n = self.u64()? as usize; Ok(String::from_utf8_lossy(self.take(n)?).into_owned()) }
    fn val(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U8(self.u8()?), 1 => Value::I8(self.u8()? as i8), 2 => Value::U16(self.u16()?), 3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?), 5 => Value::I32(self.u32()? as i32), 6 => Value::F32(f32::from_bits(self.u32()?)), 7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.str()?),
            9 => { let et = self.u32()?; let n = self.u64()? as usize; let mut v = Vec::with_capacity(n.min(1 << 20)); for _ in 0..n { v.push(self.val(et)?) } Value::Arr(v) }
            10 => Value::U64(self.u64()?), 11 => Value::I64(self.u64()? as i64), 12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => bail!("bad gguf value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let mut r = Rd { b: &mmap[..], p: 0 };
        if r.take(4)? != b"GGUF" { bail!("not a GGUF file") }
        let version = r.u32()?;
        if version < 2 { bail!("GGUF v{version} unsupported") }
        let nt = r.u64()? as usize; let nkv = r.u64()? as usize;
        let mut kv = HashMap::new();
        for _ in 0..nkv { let k = r.str()?; let t = r.u32()?; let v = r.val(t)?; kv.insert(k, v); }
        let mut tensors = Vec::with_capacity(nt);
        for _ in 0..nt {
            let name = r.str()?; let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd); for _ in 0..nd { dims.push(r.u64()?) }
            let ty = GgmlType::from_u32(r.u32()?); let offset = r.u64()?;
            tensors.push(TensorInfo { name, dims, ty, offset });
        }
        let alignment = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let data_offset = r.p.div_ceil(alignment) * alignment;
        let g = Gguf { version, kv, tensors, data_offset, alignment, mmap: unsafe { Mmap::map(&f)? } };
        drop(mmap);
        for t in &g.tensors { if g.data_offset + t.offset as usize + t.n_bytes() > g.mmap.len() { bail!("tensor {} out of bounds", t.name) } }
        Ok(g)
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> { self.tensors.iter().find(|t| t.name == name) }
    pub fn data(&self, t: &TensorInfo) -> &[u8] { let s = self.data_offset + t.offset as usize; &self.mmap[s..s + t.n_bytes()] }
    pub fn get(&self, k: &str) -> Result<&Value> { self.kv.get(k).ok_or_else(|| anyhow!("missing key {k}")) }
    pub fn u64(&self, k: &str) -> Result<u64> { self.get(k)?.as_u64().ok_or_else(|| anyhow!("{k} not int")) }
    pub fn f32(&self, k: &str) -> Result<f32> { self.get(k)?.as_f32().ok_or_else(|| anyhow!("{k} not float")) }
}
