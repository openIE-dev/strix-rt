//! Model-agnostic runtime pieces shared by the model crates (qwen2, lfm2, next: Bonsai/Gated-DeltaNet):
//! GGUF weight placement on the GPU (copied arenas or zero-copy page-cache dma-buf imports), the F32 side buffer,
//! a descriptor-set cache, and the fused-GEMV (gemv3) launch description.
use anyhow::{anyhow, bail, Result};
use ash::vk;
use gguf::{dequant::dequant_row, GgmlType, Gguf};
use std::{cell::RefCell, collections::HashMap};
use vkrt::{push_u32, Buffer, Ctx, Kernel};

/// A quantized tensor on the GPU: arena index, byte offset in that arena, type, dims (k = row length, n = rows, e = experts).
#[derive(Clone, Copy, Debug)]
pub struct WRef { pub arena: usize, pub off: u64, pub ty: GgmlType, pub k: u32, pub n: u32, pub e: u32, pub bytes: u64 }
impl WRef {
    /// Offset in the shader's addressing unit (u32 for Q4_K, u16 for Q6_K).
    pub fn unit_off(&self) -> Result<u32> { Ok((self.off / unit(self.ty)?) as u32) }
    /// Bytes of one expert slice (n rows), in shader units.
    pub fn expert_stride_units(&self) -> Result<u32> { Ok((self.bytes / self.e.max(1) as u64 / unit(self.ty)?) as u32) }
}
/// Format code for the generic GEMV (shaders/q_any.glsl).
pub fn qtype(t: GgmlType) -> Result<u32> { Ok(match t { GgmlType::Q4K => 0, GgmlType::Q6K => 1, GgmlType::Q5K => 2, GgmlType::Q8_0 => 3, GgmlType::Other(20) => 4,
    GgmlType::Other(23) => 5, GgmlType::Other(11) => 6, GgmlType::Other(21) => 7,
    GgmlType::Other(142) => 8, GgmlType::Other(143) => 9, GgmlType::Other(43) => 12, GgmlType::Other(30) => 10, GgmlType::F16 => 11, GgmlType::Other(PAD_BF16) => 13, GgmlType::Other(39) => 14 /* MXFP4: gptoss kernels only */, GgmlType::Other(6) => 15 /* Q5_0: single-column gemvq / embed_any only */, GgmlType::Q4_0 => 16 /* Q4_0: nemotron35 n35_q.glsl kernels only (does not fit gemvq's 4-bit type field) */, t => bail!("no generic GEMV format for {t:?}") }) }

/// gemvq push constants (shaders/gemvq.comp). Offsets are BYTE offsets into the weight binding.
#[derive(Default, Clone, Copy)]
pub struct GQ { pub n: u32, pub k: u32, pub flags: u32, pub y_off: u32, pub norm_off: u32, pub sq_in: u32, pub sq_out: u32, pub r: [u32; 3], pub types: u32,
    pub w: [u32; 4], pub wup: u32, pub b: [u32; 4], pub eps: f32, pub x_off4: u32, pub x_cs: u32, pub y_cs: u32 }
impl GQ {
    pub fn bytes(&self) -> Vec<u8> { push_u32(&[self.n, self.k, self.flags, self.y_off, self.norm_off, self.sq_in, self.sq_out, self.r[0], self.r[1], self.r[2], self.types,
        self.w[0], self.w[1], self.w[2], self.w[3], self.wup, self.b[0], self.b[1], self.b[2], self.b[3], self.eps.to_bits(), self.x_off4, self.x_cs, self.y_cs]) }
    /// Rows from up to 4 segments (same K, any formats), optional SwiGLU up-projection, optional per-segment f32 bias offsets.
    pub fn segs(segs: &[WRef], up: Option<WRef>, bias: &[Option<u32>], eps: f32) -> Result<(Self, usize)> {
        if segs.is_empty() || segs.len() > 4 { bail!("1..4 segments") }
        let arena = segs[0].arena;
        if segs.iter().any(|s| s.arena != arena || s.k != segs[0].k) || up.map(|u| u.arena != arena || u.k != segs[0].k).unwrap_or(false) { bail!("segments must share arena and K") }
        let mut g = GQ { k: segs[0].k, eps, b: [u32::MAX; 4], r: [u32::MAX; 3], ..Default::default() };
        let mut rows = 0u32;
        for (i, w) in segs.iter().enumerate() {
            if w.off > u32::MAX as u64 { bail!("weight offset beyond 4 GiB") }
            g.w[i] = w.off as u32; g.types |= qtype(w.ty)? << (4 * i); rows += w.n * w.e.max(1);
            if i < 3 { g.r[i] = rows; }
            if let Some(Some(b)) = bias.get(i) { g.b[i] = *b; }
        }
        if let Some(u) = up { g.wup = u.off as u32; g.types |= qtype(u.ty)? << 16; }
        if segs[0].k % 32 != 0 { bail!("K must be a multiple of 32") }
        for w in segs.iter().chain(up.iter()) { if w.off % 4 != 0 || (w.bytes / (w.n as u64 * w.e.max(1) as u64)) % 4 != 0 { bail!("gemvq needs 4-byte aligned rows") } }
        g.n = rows; Ok((g, arena))
    }
}
/// gemvq pipelines specialized per weight format on demand (uniform-format calls get the format switch folded away).
pub struct GkQ { lpr: u32, ls: u32, tern: (u32, u32), swiglu: bool, sg: u32, cache: std::cell::RefCell<HashMap<(u32, u32, u32, u32), Box<Gk>>> }
impl GkQ {
    pub fn new(var: &str, def: (u32, u32), swiglu: bool, sg: u32) -> Self {
        let (lpr, ls) = std::env::var(var).ok().and_then(|v| { let p: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 2).then(|| (p[0], p[1])) }).unwrap_or(def);
        // Prism ternary formats (128-weight items) prefer fewer lanes per row; env GEMV_TERN="lpr,ls" overrides (8,64).
        let tern = std::env::var("GEMV_TERN").ok().and_then(|v| { let p: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 2).then(|| (p[0], p[1])) }).unwrap_or((8, 64));
        GkQ { lpr: lpr.min(sg), ls, tern: (tern.0.min(sg), tern.1), swiglu, sg, cache: Default::default() }
    }
    pub fn get(&self, ctx: &Ctx, g: &GQ, nseg: usize) -> Result<&Gk> { self.get_nb(ctx, g, nseg, 1) }
    /// `nb` token columns per dispatch (1..=4).
    pub fn get_nb(&self, ctx: &Ctx, g: &GQ, nseg: usize, nb: u32) -> Result<&Gk> {
        let t0 = g.types & 0xF;
        let uni = (0..nseg).all(|i| (g.types >> (4 * i)) & 0xF == t0) && std::env::var("GEMVQ_GENERIC").is_err();
        let key = if uni { (t0, if self.swiglu { (g.types >> 16) & 0xF } else { 255 }) } else { (255, 255) };
        // NR rows per lane cluster for the low-bit formats (x-load bound); needs NR-aligned segment boundaries.
        let nr_env = std::env::var("GEMVQ_NR").ok().and_then(|v| v.parse::<u32>().ok());
        let mut nr = if uni && ((t0 == 8 || t0 == 9 || t0 == 12)) { nr_env.unwrap_or(if std::env::var("TERN_Q8").map(|v| v == "0").unwrap_or(false) { 2 } else { 1 }).clamp(1, 4)  /* int8 path (default) prefers 1 */ } else { 1 };
        if (0..nseg.saturating_sub(1)).any(|i| g.r[i] % nr != 0) { nr = 1; }
        // multi-column: ternary keeps NR rows (env GEMVQ_NBNR, default 2) so x loads are shared across rows
        if nb > 1 { nr = if uni && ((t0 == 8 || t0 == 9 || t0 == 12)) { std::env::var("GEMVQ_NBNR").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1).clamp(1, 4) } else { std::env::var("GEMVQ_NBNR_Q").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1).clamp(1, 4) };
            if (0..nseg.saturating_sub(1)).any(|i| g.r[i] % nr != 0) { nr = 1; } }
        // TERN_LDS=1: single-column ternary also uses the LDS-tiled path (nb | 16 marks forced)
        let fnb = nb == 1 && uni && ((t0 == 8 || t0 == 9 || t0 == 12)) && std::env::var("TERN_LDS").map(|v| v == "1").unwrap_or(false);
        if fnb { nr = std::env::var("GEMVQ_NBNR").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1).clamp(1, 4); if (0..nseg.saturating_sub(1)).any(|i| g.r[i] % nr != 0) { nr = 1; } }
        let key = (key.0, key.1, nr, nb | if fnb { 16 } else { 0 });
        if !self.cache.borrow().contains_key(&key) {
            let (lpr, ls) = if (key.0 == 8 || key.0 == 9 || key.0 == 12) { self.tern } else { (self.lpr, self.ls) };
            let k = ctx.kernel(vkrt::shaders::gemvq(lpr, self.swiglu), 6, 96, &[ls, key.0, key.1, key.2, key.3 & 15, key.3 >> 4, std::env::var("GEMVQ_Q4P").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0)], Some(self.sg))?;
            self.cache.borrow_mut().insert(key, Box::new(Gk { k, rows_per_wg: ls / lpr * key.2 }));
        }
        let p: *const Gk = &**self.cache.borrow().get(&key).unwrap();
        Ok(unsafe { &*p }) // boxed entries are never removed while self lives
    }
}
/// A gemvq pipeline variant plus its rows-per-workgroup. Env `var` = "lpr,ls" overrides the default.
pub fn gkq(ctx: &Ctx, var: &str, def: (u32, u32), swiglu: bool, sg: u32) -> Result<Gk> {
    let (lpr, ls) = std::env::var(var).ok().and_then(|v| { let p: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 2).then(|| (p[0], p[1])) }).unwrap_or(def);
    let lpr = lpr.min(sg);
    Ok(Gk { k: ctx.kernel(vkrt::shaders::gemvq(lpr, swiglu), 6, 96, &[ls], Some(sg))?, rows_per_wg: ls / lpr })
}
pub fn unit(t: GgmlType) -> Result<u64> { Ok(match t { GgmlType::Q4K => 4, GgmlType::Q6K => 2, GgmlType::Q8_0 => 2, t => bail!("no GPU kernel for {t:?}") }) }

/// Codebook floats reserved at the start of `Weights::f32b`.
pub const TBL_FLOATS: usize = 528;
/// Private type code: BF16 rows padded by 256 bytes (prefill GEMM only, qtype 13). RT_PAD16=1 makes Weights::load
/// store blk.* BF16 matrices whose row is a multiple of 4 KiB this way: power-of-two row strides put every lane of a
/// wave (one row each) on the same memory channel and the K=4096 GEMMs ran at 1/3 speed. Decode GEMVs do not know 13.
pub const PAD_BF16: u32 = 130;
pub struct Weights { pub g: Gguf, pub arenas: Vec<Buffer>, pub sizes: Vec<u64>, pub w: HashMap<String, WRef>, pub f32b: Buffer, pub fo: HashMap<String, u32>, pub mode: String }

impl Weights {
    /// ZEROCOPY=dmabuf: one 2 MiB-aligned page-cache import per layer group via /dev/strixpc (weights never copied).
    /// ZEROCOPY=userptr|1: VK_EXT_external_memory_host. Otherwise: copied into <=2 GiB HOST_VISIBLE|DEVICE_LOCAL arenas.
    pub fn load(ctx: &Ctx, path: &str) -> Result<Self> {
        let g = Gguf::open(path)?;
        let mut groups: Vec<(String, Vec<usize>)> = vec![];
        let mut f32_total = 0usize;
        for (i, t) in g.tensors.iter().enumerate() {
            match t.ty {
                GgmlType::F32 => f32_total += t.n_elems() as usize,
                ty if qtype(ty).is_ok() => {
                    let key = if t.name.starts_with("blk.") { t.name.split('.').take(2).collect::<Vec<_>>().join(".") } else { t.name.clone() };
                    match groups.iter_mut().find(|x| x.0 == key) { Some(x) => x.1.push(i), None => groups.push((key, vec![i])) }
                }
                t2 => bail!("unsupported tensor type {t2:?} for {}", t.name),
            }
        }
        let mk = |t: &gguf::TensorInfo, arena: usize, off: u64| WRef { arena, off, ty: t.ty, k: t.dims[0] as u32, n: t.dims.get(1).copied().unwrap_or(1) as u32,
            e: t.dims.get(2).copied().unwrap_or(1) as u32, bytes: t.n_bytes() as u64 };
        let mode = std::env::var("ZEROCOPY").unwrap_or_default();
        let (mut arenas, mut sizes, mut w) = (vec![], vec![], HashMap::new());
        let t0 = std::time::Instant::now();
        if !mode.is_empty() && mode != "0" {
            const AL2: usize = 2 << 20;
            let flen = g.mmap.len().div_ceil(4096) * 4096;
            if std::env::var("ZC_WARM").map(|v| v != "0").unwrap_or(true) { warm_page_cache(&g.mmap[..])?; }
            for (_, idx) in &groups {
                let s0 = idx.iter().map(|&i| g.data_offset + g.tensors[i].offset as usize).min().unwrap();
                let e0 = idx.iter().map(|&i| g.data_offset + g.tensors[i].offset as usize + g.tensors[i].n_bytes()).max().unwrap();
                let (ps, pe) = (s0 / AL2 * AL2, (e0.div_ceil(AL2) * AL2).min(flen));
                let (buf, start) = if mode == "dmabuf" { import_pagecache_aligned(ctx, g.mmap.as_ptr(), ps, pe)? }
                    else { (unsafe { ctx.import_host(g.mmap.as_ptr().add(ps) as *mut u8, (pe - ps) as u64)? }, ps) };
                let ai = arenas.len(); arenas.push(buf); sizes.push((pe - start) as u64);
                for &i in idx { let t = &g.tensors[i]; w.insert(t.name.clone(), mk(t, ai, (g.data_offset + t.offset as usize - start) as u64)); }
            }
        } else {
            const ARENA: u64 = 2 << 30; const AL: u64 = 256;
            sizes.push(0);
            let pad16 = std::env::var("RT_PAD16").map_or(false, |v| v != "0");
            let padded = |t: &gguf::TensorInfo| pad16 && t.ty == GgmlType::Other(30) && t.name.starts_with("blk.") && t.dims.len() == 2 && (t.dims[0] * 2) % 4096 == 0;
            let sz = |t: &gguf::TensorInfo| if padded(t) { t.dims[1] as u64 * (t.dims[0] as u64 * 2 + 256) } else { t.n_bytes() as u64 };
            for (_, idx) in &groups {
                let gsz: u64 = idx.iter().map(|&i| sz(&g.tensors[i]).div_ceil(AL) * AL).sum();
                if *sizes.last().unwrap() + gsz > ARENA { sizes.push(0); }
                let ai = sizes.len() - 1;
                for &i in idx { let t = &g.tensors[i]; let mut r = mk(t, ai, sizes[ai]); if padded(t) { r.ty = GgmlType::Other(PAD_BF16); r.bytes = sz(t) }
                    w.insert(t.name.clone(), r); sizes[ai] += sz(t).div_ceil(AL) * AL; }
            }
            arenas = sizes.iter().map(|&s| ctx.buffer(s.max(256))).collect::<Result<_>>()?;
            for (name, r) in &w { let t = g.tensor(name).unwrap(); let dst = &mut arenas[r.arena].bytes_mut()[r.off as usize..(r.off + r.bytes) as usize];
                if r.ty == GgmlType::Other(PAD_BF16) { let (rb, n) = (t.dims[0] as usize * 2, t.dims[1] as usize); let src = g.data(t);
                    for i in 0..n { dst[i * (rb + 256)..i * (rb + 256) + rb].copy_from_slice(&src[i * rb..(i + 1) * rb]) } }
                else { dst.copy_from_slice(g.data(t)) } }
        }
        // Floats [0, 528) of f32b hold the codebooks for shaders/q_any.glsl (bit patterns): iq3s_grid[512], then kvalues_iq4nl[16].
        let f32b = ctx.buffer(((f32_total + TBL_FLOATS).max(64) * 4) as u64)?;
        { let f = f32b.as_f32_mut(); for (i, &v) in gguf::iq3s_grid::IQ3S_GRID.iter().enumerate() { f[i] = f32::from_bits(v); }
          for (i, &v) in gguf::dequant::KV_IQ4NL.iter().enumerate() { f[512 + i] = v as f32; } }
        let mut fo = HashMap::new(); let mut o = TBL_FLOATS;
        for t in g.tensors.iter().filter(|t| t.ty == GgmlType::F32) {
            let n = t.n_elems() as usize; dequant_row(GgmlType::F32, g.data(t), &mut f32b.as_f32_mut()[o..o + n]); fo.insert(t.name.clone(), o as u32); o += n;
        }
        eprintln!("weights: {} in {:.3}s, {} arenas/imports ({} MiB), {} quant tensors, {} f32 floats", if mode.is_empty() || mode == "0" { "copied".to_string() } else { format!("zero-copy {mode}") },
            t0.elapsed().as_secs_f64(), arenas.len(), sizes.iter().sum::<u64>() >> 20, w.len(), f32_total);
        Ok(Weights { g, arenas, sizes, w, f32b, fo, mode })
    }
    pub fn wref(&self, name: &str) -> Result<WRef> { self.w.get(name).copied().ok_or_else(|| anyhow!("missing tensor {name}")) }
    pub fn f(&self, name: &str) -> Result<u32> { self.fo.get(name).copied().ok_or_else(|| anyhow!("missing f32 tensor {name}")) }
    pub fn vocab_strings(&self) -> Vec<String> {
        match self.g.kv.get("tokenizer.ggml.tokens") { Some(gguf::Value::Arr(a)) => a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect(), _ => vec![] }
    }
}

/// Descriptor-set cache keyed on pipeline + full buffer list.
#[derive(Default)]
pub struct DsCache(RefCell<HashMap<u64, vk::DescriptorSet>>);
impl DsCache {
    /// Forget every cached set (call after freeing buffers: a recycled VkBuffer handle would otherwise hit a stale descriptor set -> GPU page fault).
    pub fn clear(&self) { self.0.borrow_mut().clear() }
    pub fn get(&self, ctx: &Ctx, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> {
        use ash::vk::Handle;
        let mut hsh = k.pipeline.as_raw();
        for b in bufs { hsh = hsh.rotate_left(13) ^ b.buf.as_raw().wrapping_mul(0x9E3779B97F4A7C15); }
        if let Some(ds) = self.0.borrow().get(&hsh) { return Ok(*ds) }
        let ds = ctx.bind(k, bufs)?; self.0.borrow_mut().insert(hsh, ds); Ok(ds)
    }
}

/// gemv3 push constants (see shaders/gemv3.inc). flags: 1 accumulate, 4 fused-norm input scale (sq_in),
/// 8|16 produce sum-of-squares into sq_out and xw = y * norm(norm_off), 32 MoE rows, 64 MoE down.
#[derive(Default, Clone, Copy)]
pub struct G { pub n: u32, pub k: u32, pub flags: u32, pub y_off: u32, pub norm_off: u32, pub sq_in: u32, pub sq_out: u32, pub r1: u32, pub r2: u32, pub types: u32, pub w: [u32; 3], pub b: [u32; 3], pub wup: u32, pub eps: f32, pub bsel: u32, pub misc: u32 }
impl G { pub fn bytes(&self) -> Vec<u8> { push_u32(&[self.n, self.k, self.flags, self.y_off, self.norm_off, self.sq_in, self.sq_out, self.r1, self.r2, self.types,
    self.w[0], self.w[1], self.w[2], self.b[0], self.b[1], self.b[2], self.wup, self.eps.to_bits(), self.bsel, self.misc]) } }

/// A gemv3 pipeline variant plus its rows-per-workgroup.
pub struct Gk { pub k: Kernel, pub rows_per_wg: u32 }
/// Build a gemv3 variant; env var `var` = "lpr,ls" overrides the default.
pub fn gk(ctx: &Ctx, var: &str, def: (u32, u32), swiglu: bool, sg: u32) -> Result<Gk> {
    let (lpr, ls) = std::env::var(var).ok().and_then(|v| { let p: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 2).then(|| (p[0], p[1])) }).unwrap_or(def);
    let lpr = lpr.min(sg);
    Ok(Gk { k: ctx.kernel(vkrt::shaders::gemv3(lpr, swiglu), 6, 80, &[ls], Some(sg))?, rows_per_wg: ls / lpr })
}

/// Make sure a mapped file range is resident with large folios (MADV_HUGEPAGE + MADV_POPULATE_READ).
pub fn warm_page_cache(m: &[u8]) -> Result<()> {
    let n = m.len().div_ceil(4096); let mut vec = vec![0u8; n];
    unsafe { libc::mincore(m.as_ptr() as *mut _, m.len(), vec.as_mut_ptr()); }
    let resident = vec.iter().filter(|v| **v & 1 != 0).count();
    if resident == n { return Ok(()) }
    let t = std::time::Instant::now();
    unsafe { libc::madvise(m.as_ptr() as *mut _, m.len(), libc::MADV_HUGEPAGE); libc::madvise(m.as_ptr() as *mut _, m.len(), libc::MADV_SEQUENTIAL); }
    let r = unsafe { libc::madvise(m.as_ptr() as *mut _, m.len(), 22 /* MADV_POPULATE_READ */) };
    eprintln!("warm: {resident}/{n} pages were resident; MADV_POPULATE_READ ret {r} in {:.3}s", t.elapsed().as_secs_f64());
    Ok(())
}

/// Export [ps,pe) of the mapping through /dev/strixpc and import it so GPU VA and file offset agree mod 2 MiB.
pub fn import_pagecache_aligned(ctx: &Ctx, base: *const u8, ps: usize, pe: usize) -> Result<(Buffer, usize)> {
    const AL2: u64 = 2 << 20;
    let mut pad = 0usize;
    for _ in 0..4 {
        let start = ps - pad;
        let fd = unsafe { vkrt::pagecache::export(base.add(start), pe - start)? };
        let b = unsafe { ctx.import_dmabuf(fd, (pe - start) as u64)? };
        let va = ctx.device_address(&b);
        if (va + pad as u64) % AL2 == 0 { return Ok((b, start)) }
        ctx.free_buffer(b);
        let need = ((AL2 - va % AL2) % AL2) as usize;
        if need > ps { bail!("cannot align dma-buf import at file offset {ps}") }
        pad = need;
    }
    bail!("dma-buf VA alignment did not converge")
}

/// GPT-2 byte-level BPE token string -> raw bytes.
pub fn gpt2_bytes(s: &str) -> Vec<u8> {
    let mut map = HashMap::new(); let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable { b } else { n += 1; 255 + n };
        map.insert(char::from_u32(c).unwrap(), b as u8);
    }
    s.chars().filter_map(|c| map.get(&c).copied()).collect()
}
