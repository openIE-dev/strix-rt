//! Multi-shard safetensors through the strix-rt memory tier: shards are mmapped read-only, warmed into the page cache
//! (MADV_HUGEPAGE + POPULATE_READ) and their page-cache folios are exported by /dev/strixpc as dma-bufs and imported
//! as GPU buffers (zero copy, 2 MiB VA/file alignment). Imports are chunked (< 3.5 GiB) so every weight offset fits the
//! GEMM's 32-bit byte offset. Without /dev/strixpc (or QI_ZC=0) the chunks are copied into device buffers instead.
use anyhow::{anyhow, bail, Context, Result};
use gguf::GgmlType;
use half::{bf16, f16};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{Buffer, Ctx};

pub struct Tensor { pub shard: usize, pub dtype: String, pub shape: Vec<usize>, pub start: usize, pub end: usize }
pub struct Shard { pub mm: memmap2::Mmap }
pub struct Model { pub shards: Vec<Shard>, pub t: HashMap<String, Tensor>, pub bufs: Vec<Buffer>, pub chunk_of: HashMap<String, (usize, u64)>, pub zero_copy: bool, pub load_s: f64, pub f16w: bool }

/// Group key: the name up to and including its first numeric component (one transformer block / layer).
fn group(k: &str) -> &str { let mut e = 0; for part in k.split('.') { e += part.len() + 1; if part.bytes().all(|b| b.is_ascii_digit()) { return &k[..e - 1] } } k }
fn header(mm: &[u8]) -> Result<(usize, serde_json::Map<String, serde_json::Value>)> {
    let n = u64::from_le_bytes(mm[0..8].try_into()?) as usize;
    let h: serde_json::Value = serde_json::from_slice(&mm[8..8 + n])?;
    Ok((8 + n, h.as_object().ok_or(anyhow!("st header"))?.clone()))
}

impl Model {
    /// Open every *.safetensors in `dir`. Tensors whose name passes `want` are made GPU-visible (zero copy when possible).
    pub fn open(ctx: &Ctx, dir: &str, want: &dyn Fn(&str) -> bool) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let mut files: Vec<_> = std::fs::read_dir(dir).with_context(|| dir.to_string())?.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false)).collect();
        files.sort();
        let mut shards = vec![]; let mut t = HashMap::new();
        for (si, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f)?; let mm = unsafe { memmap2::Mmap::map(&file)? };
            let (base, h) = header(&mm)?;
            for (k, v) in h { if k == "__metadata__" { continue }
                let o = &v["data_offsets"];
                t.insert(k, Tensor { shard: si, dtype: v["dtype"].as_str().unwrap_or("").to_string(), shape: v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                    start: base + o[0].as_u64().unwrap() as usize, end: base + o[1].as_u64().unwrap() as usize }); }
            shards.push(Shard { mm });
        }
        let f16w = std::env::var("QI_F16W").map(|v| v == "1").unwrap_or(false) && dir.contains("transformer");
        // QI_WMEM=<memory type index>: experimental weight placement (copies into that type; implies no zero-copy). Off by default.
        let wmem: Option<u32> = std::env::var("QI_WMEM").ok().and_then(|v| v.parse().ok());
        // QI_WALIGN=1: copy every tensor to a 4 KB aligned offset (safetensors data offsets are often only 8/16 B aligned, which
        // slows the coop-matrix GEMM weight loads ~2x on some shards). Implies no zero-copy. Off by default.
        let walign = std::env::var("QI_WALIGN").map(|v| v == "1").unwrap_or(false);
        let zc = !f16w && !walign && wmem.is_none() && vkrt::pagecache::available() && std::env::var("QI_ZC").map(|v| v != "0").unwrap_or(true);
        // chunk the wanted tensors of each shard by file offset
        const MAXC: usize = 3584 << 20; const AL: usize = 2 << 20;
        let mut bufs = vec![]; let mut chunk_of = HashMap::new();
        for si in 0..shards.len() {
            let mut ts: Vec<(&String, &Tensor)> = t.iter().filter(|(k, v)| v.shard == si && want(k)).collect();
            if ts.is_empty() { continue }
            ts.sort_by_key(|(_, v)| v.start);
            let mm = &shards[si].mm;
            if zc { rtcore::warm_page_cache(&mm[..])?; }
            let mut i = 0;
            while i < ts.len() {
                let cs = ts[i].1.start / AL * AL; let mut j = i; let mut ce = ts[i].1.end;
                while j + 1 < ts.len() && ts[j + 1].1.end - cs <= MAXC { j += 1; }
                // never split a layer/block across chunks (fused gate|up GEMMs need both in one arena)
                if j + 1 < ts.len() { let j0 = j; while j > i && group(ts[j].0) == group(ts[j + 1].0) { j -= 1; } if j == i && group(ts[i].0) == group(ts[i + 1].0) { j = j0; } }
                for x in &ts[i..=j] { ce = ce.max(x.1.end); }
                let pe = ((ce + 4095) / 4096 * 4096).min(mm.len().div_ceil(4096) * 4096);
                let mut aligned: Option<Vec<u64>> = None;
                let (buf, base) = if walign && !f16w {
                    let mut offs = vec![]; let mut tot = 0u64; for (_, v) in &ts[i..=j] { offs.push(tot); tot += ((v.end - v.start) as u64 + 4095) / 4096 * 4096; }
                    let b = match wmem { Some(ty) => ctx.buffer_in_type(tot, ty)?, None => ctx.buffer(tot)? };
                    let mut host = vec![0u8; tot as usize];
                    for ((_, v), o) in ts[i..=j].iter().zip(&offs) { host[*o as usize..*o as usize + (v.end - v.start)].copy_from_slice(&mm[v.start..v.end]); }
                    ctx.upload(&b, &host)?; aligned = Some(offs); (b, cs)
                } else if f16w {   // experiment: BF16 -> F16 copies (not zero-copy)
                    let n = ce - cs; let b = ctx.buffer(n as u64)?; let src = &mm[cs..ce];
                    let dst = unsafe { std::slice::from_raw_parts_mut(b.ptr as *mut u16, n / 2) };
                    std::thread::scope(|sc| { for (d, s2) in dst.chunks_mut(1 << 22).zip(src.chunks(2 << 22)) { sc.spawn(move || {
                        for (o, c) in d.iter_mut().zip(s2.chunks_exact(2)) { *o = f16::from_f32(bf16::from_le_bytes([c[0], c[1]]).to_f32()).to_bits() } }); } });
                    (b, cs)
                } else if zc {
                    match rtcore::import_pagecache_aligned(ctx, mm.as_ptr(), cs, pe) {
                        Ok(v) => v,
                        Err(_) => { let fd = unsafe { vkrt::pagecache::export(mm.as_ptr().add(cs), pe - cs)? }; (unsafe { ctx.import_dmabuf(fd, (pe - cs) as u64)? }, cs) }
                    }
                } else if let Some(ty) = wmem {
                    let b = ctx.buffer_in_type((ce - cs) as u64, ty)?; ctx.upload(&b, &mm[cs..ce])?; (b, cs)
                } else {
                    let b = ctx.buffer((ce - cs) as u64)?; let n = ce - cs; b.bytes_mut()[..n].copy_from_slice(&mm[cs..ce]); (b, cs)
                };
                for (n, (k, v)) in ts[i..=j].iter().enumerate() { let off = match &aligned { Some(o) => o[n], None => (v.start - base) as u64 }; chunk_of.insert((*k).clone(), (bufs.len(), off)); }
                bufs.push(buf); i = j + 1;
            }
        }
        Ok(Model { shards, t, bufs, chunk_of, zero_copy: zc, load_s: t0.elapsed().as_secs_f64(), f16w })
    }
    pub fn get(&self, k: &str) -> Result<&Tensor> { self.t.get(k).ok_or_else(|| anyhow!("missing tensor {k}")) }
    pub fn bytes(&self, k: &str) -> Result<&[u8]> { let x = self.get(k)?; Ok(&self.shards[x.shard].mm[x.start..x.end]) }
    /// GEMM weight reference ([n][k] row-major, BF16 or F16); `arena` indexes `bufs`.
    pub fn wref(&self, k: &str) -> Result<WRef> {
        let x = self.get(k)?; let (ci, off) = *self.chunk_of.get(k).ok_or_else(|| anyhow!("{k} not GPU-mapped"))?;
        let ty = match x.dtype.as_str() { "BF16" if self.f16w => GgmlType::F16, "BF16" => GgmlType::Other(30), "F16" => GgmlType::F16, d => bail!("{k}: dtype {d} not a GEMM weight") };
        let n = x.shape[0] as u32; let kk = x.shape[1..].iter().product::<usize>() as u32;
        Ok(WRef { arena: ci, off, ty, k: kk, n, e: 1, bytes: (x.end - x.start) as u64 })
    }
    pub fn f32(&self, k: &str) -> Result<Vec<f32>> {
        let x = self.get(k)?; let b = self.bytes(k)?;
        Ok(match x.dtype.as_str() { "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            "BF16" => b.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(), d => bail!("{k}: dtype {d}") })
    }
}
