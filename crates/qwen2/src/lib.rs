//! Qwen2 decode forward pass on vkrt. Weights live in a few HOST_VISIBLE|DEVICE_LOCAL arenas
//! (<=2 GiB each, whole layers per arena so gate/up share one binding). The whole token is
//! recorded into one command buffer that is resubmitted every step (pos is read from a buffer).
use anyhow::{anyhow, bail, Result};
use ash::vk;
use gguf::{dequant::dequant_row, GgmlType, Gguf};
use std::{cell::RefCell, collections::HashMap};
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel, Recorded};
pub mod fast;

#[derive(Clone, Copy, Debug)]
pub struct WRef { pub arena: usize, pub off: u64, pub ty: GgmlType, pub k: u32, pub n: u32, pub bytes: u64 }

pub struct Cfg { pub n_layer: u32, pub dim: u32, pub ffn: u32, pub n_head: u32, pub n_kv: u32, pub vocab: u32, pub eps: f32, pub theta: f32, pub max_ctx: u32 }

pub struct Gemvs { pub ls: u32, pub sg: u32, pub q4k_s: Kernel, pub q4k_g: Kernel, pub q6k_s: Kernel, pub q6k_g: Kernel, pub swi_s: Kernel }
impl Gemvs {
    pub fn new(ctx: &Ctx, ls: u32, sg: u32, dim: u32) -> Result<Self> {
        let xs = if std::env::var("SHARED_X").map(|v| v == "1").unwrap_or(false) { dim / 4 } else { 1 };
        let mk = |spv: &[u8], x: u32| ctx.kernel(spv, 4, 28, &[ls, x], Some(sg));
        Ok(Gemvs { ls, sg, q4k_s: mk(shaders::GEMV2_Q4K, xs)?, q4k_g: mk(shaders::GEMV2_Q4K, 1)?, q6k_s: mk(shaders::GEMV2_Q6K, xs)?, q6k_g: mk(shaders::GEMV2_Q6K, 1)?, swi_s: mk(shaders::GEMV2_Q4K_SWIGLU, xs)? })
    }
}

pub struct Bufs { pub x: Buffer, pub xn: Buffer, pub q: Buffer, pub k: Buffer, pub v: Buffer, pub ao: Buffer, pub hb: Buffer, pub logits: Buffer, pub kc: Buffer, pub vc: Buffer, pub params: Buffer }

pub struct Model {
    pub ctx: Ctx, pub g: Gguf, pub cfg: Cfg, pub arenas: Vec<Buffer>, pub f32b: Buffer,
    pub w: HashMap<String, WRef>, pub fo: HashMap<String, u32>, pub b: Bufs, pub gv: Gemvs,
    rms: Kernel, rope: Kernel, attn: Kernel, ds: RefCell<HashMap<(u64, u64, u64, u64), vk::DescriptorSet>>,
}

fn h(b: &Buffer) -> u64 { use ash::vk::Handle; b.buf.as_raw() }

impl Model {
    pub fn load(path: &str, max_ctx: u32, ls: u32, sg: u32) -> Result<Self> {
        let g = Gguf::open(path)?;
        let a = "qwen2";
        let cfg = Cfg { n_layer: g.u64(&format!("{a}.block_count"))? as u32, dim: g.u64(&format!("{a}.embedding_length"))? as u32, ffn: g.u64(&format!("{a}.feed_forward_length"))? as u32,
            n_head: g.u64(&format!("{a}.attention.head_count"))? as u32, n_kv: g.u64(&format!("{a}.attention.head_count_kv"))? as u32,
            vocab: g.tensor("output.weight").ok_or_else(|| anyhow!("no output.weight"))?.dims[1] as u32,
            eps: g.f32(&format!("{a}.attention.layer_norm_rms_epsilon"))?, theta: g.f32(&format!("{a}.rope.freq_base"))?, max_ctx: max_ctx.min(4096) };
        if cfg.dim / cfg.n_head != 128 { bail!("head_dim must be 128") }
        let ctx = Ctx::new()?;
        // plan: group quant tensors by layer ("blk.N." prefix, else own group)
        let mut groups: Vec<(String, Vec<usize>)> = vec![];
        let mut f32_total = 0usize;
        for (i, t) in g.tensors.iter().enumerate() {
            match t.ty {
                GgmlType::F32 => f32_total += t.n_elems() as usize,
                GgmlType::Q4K | GgmlType::Q6K => {
                    let key = if t.name.starts_with("blk.") { t.name.split('.').take(2).collect::<Vec<_>>().join(".") } else { t.name.clone() };
                    match groups.iter_mut().find(|x| x.0 == key) { Some(x) => x.1.push(i), None => groups.push((key, vec![i])) }
                }
                t2 => bail!("unsupported tensor type {t2:?} for {}", t.name),
            }
        }
        const ARENA: u64 = 2 << 30; const AL: u64 = 256;
        let zerocopy = std::env::var("ZEROCOPY").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
        let mut plan: Vec<u64> = vec![0]; let mut w = HashMap::new();
        let mut arenas: Vec<Buffer> = vec![];
        let t_up = std::time::Instant::now();
        if zerocopy {
            // Zero-copy straight over the mmap'd GGUF pages (page cache = weight cache), one import per layer group.
            // ZEROCOPY=1|userptr: VK_EXT_external_memory_host (needs tools/libuserptr_shim.so on stock RADV; per-submit userptr tax)
            // ZEROCOPY=dmabuf: kmod/strixpc pins the page-cache folios and exports a dma-buf (no per-submit tax).
            // Ranges are 2 MiB aligned in the file so large page-cache folios line up with 2 MiB GPU fragments.
            let mode = std::env::var("ZEROCOPY").unwrap();
            let dmabuf = mode == "dmabuf";
            const AL2: usize = 2 << 20;
            let flen = g.mmap.len().div_ceil(4096) * 4096;
            if std::env::var("ZC_WARM").map(|v| v != "0").unwrap_or(true) { warm_page_cache(&g.mmap[..])?; }
            plan.clear();
            for (_, idx) in &groups {
                let s0 = idx.iter().map(|&i| g.data_offset + g.tensors[i].offset as usize).min().unwrap();
                let e0 = idx.iter().map(|&i| g.data_offset + g.tensors[i].offset as usize + g.tensors[i].n_bytes()).max().unwrap();
                let (ps, pe) = (s0 / AL2 * AL2, (e0.div_ceil(AL2) * AL2).min(flen));
                let (buf, start) = if dmabuf { import_pagecache_aligned(&ctx, g.mmap.as_ptr(), ps, pe)? }
                    else { (unsafe { ctx.import_host(g.mmap.as_ptr().add(ps) as *mut u8, (pe - ps) as u64)? }, ps) };
                let ai = arenas.len(); arenas.push(buf); plan.push((pe - start) as u64);
                for &i in idx { let t = &g.tensors[i];
                    w.insert(t.name.clone(), WRef { arena: ai, off: (g.data_offset + t.offset as usize - start) as u64, ty: t.ty, k: t.dims[0] as u32, n: t.dims.get(1).copied().unwrap_or(1) as u32, bytes: t.n_bytes() as u64 }); }
            }
        } else {
            for (_, idx) in &groups {
                let gsz: u64 = idx.iter().map(|&i| (g.tensors[i].n_bytes() as u64).div_ceil(AL) * AL).sum();
                if *plan.last().unwrap() + gsz > ARENA { plan.push(0); }
                let ai = plan.len() - 1;
                for &i in idx { let t = &g.tensors[i]; let off = plan[ai];
                    w.insert(t.name.clone(), WRef { arena: ai, off, ty: t.ty, k: t.dims[0] as u32, n: t.dims.get(1).copied().unwrap_or(1) as u32, bytes: t.n_bytes() as u64 });
                    plan[ai] += (t.n_bytes() as u64).div_ceil(AL) * AL; }
            }
            arenas = plan.iter().map(|&s| ctx.buffer(s.max(256))).collect::<Result<_>>()?;
            for (name, r) in &w { let t = g.tensor(name).unwrap(); arenas[r.arena].bytes_mut()[r.off as usize..(r.off + r.bytes) as usize].copy_from_slice(g.data(t)); }
        }
        eprintln!("weights {} in {:.3}s", if zerocopy { "imported zero-copy" } else { "copied" }, t_up.elapsed().as_secs_f64());
        let f32b = ctx.buffer((f32_total * 4) as u64)?;
        let mut fo = HashMap::new(); let mut o = 0usize;
        for t in g.tensors.iter().filter(|t| t.ty == GgmlType::F32) {
            let n = t.n_elems() as usize; dequant_row(GgmlType::F32, g.data(t), &mut f32b.as_f32_mut()[o..o + n]); fo.insert(t.name.clone(), o as u32); o += n;
        }
        let f = |n: u32| ctx.buffer(n as u64 * 4);
        let kvsz = cfg.n_layer * cfg.max_ctx * cfg.n_kv * 128;
        let b = Bufs { x: f(cfg.dim)?, xn: f(cfg.dim)?, q: f(cfg.dim)?, k: f(cfg.n_kv * 128)?, v: f(cfg.n_kv * 128)?, ao: f(cfg.dim)?, hb: f(cfg.ffn)?, logits: ctx.buffer_readback(cfg.vocab as u64 * 4)?, kc: f(kvsz)?, vc: f(kvsz)?, params: f(4)? };
        let gv = Gemvs::new(&ctx, ls, sg, cfg.dim)?;
        let rms = ctx.kernel(shaders::RMSNORM, 3, 12, &[], None)?;
        let rope = ctx.kernel(shaders::ROPE_KV, 6, 16, &[], None)?;
        let attn = ctx.kernel(shaders::ATTN, 5, 16, &[], None)?;
        eprintln!("loaded: {} arenas/imports, sizes {:?} MiB, f32 {} floats, {} quant tensors on GPU", arenas.len(), plan.iter().map(|s| s >> 20).collect::<Vec<_>>(), f32_total, w.len());
        Ok(Model { ctx, g, cfg, arenas, f32b, w, fo, b, gv, rms, rope, attn, ds: RefCell::new(HashMap::new()) })
    }

    pub fn set(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> {
        use ash::vk::Handle;
        let mut key = [k.pipeline.as_raw(), 0, 0, 0];
        // key on pipeline + first three buffers (enough to disambiguate our uses)
        for (i, b) in bufs.iter().take(3).enumerate() { key[i + 1] = h(b); }
        let key = (key[0], key[1], key[2], key[3] ^ bufs.get(3).map(|b| h(b).rotate_left(17)).unwrap_or(0) ^ bufs.get(4).map(|b| h(b).rotate_left(29)).unwrap_or(0));
        if let Some(ds) = self.ds.borrow().get(&key) { return Ok(*ds) }
        let ds = self.ctx.bind(k, bufs)?; self.ds.borrow_mut().insert(key, ds); Ok(ds)
    }

    /// Descriptor-set cache keyed on the full buffer list (any length).
    pub fn set6(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> {
        use ash::vk::Handle;
        let mut hsh = k.pipeline.as_raw();
        for b in bufs { hsh = hsh.rotate_left(13) ^ h(b).wrapping_mul(0x9E3779B97F4A7C15); }
        let key = (u64::MAX, hsh, 0, 0);
        if let Some(ds) = self.ds.borrow().get(&key) { return Ok(*ds) }
        let ds = self.ctx.bind(k, bufs)?; self.ds.borrow_mut().insert(key, ds); Ok(ds)
    }
    pub fn wref(&self, name: &str) -> Result<WRef> { self.w.get(name).copied().ok_or_else(|| anyhow!("missing {name}")) }

    /// Record a GEMV y (+)= W x (+bias). `gate_up`: Some(up) -> fused SwiGLU with W as gate.
    pub fn gemv(&self, gv: &Gemvs, cb: vk::CommandBuffer, wr: WRef, up: Option<WRef>, x: &Buffer, y: &Buffer, bias: Option<&str>, acc: bool) -> Result<()> {
        let shared = wr.k == self.cfg.dim;
        let (kern, unit) = match (wr.ty, up.is_some(), shared) {
            (GgmlType::Q4K, true, true) => (&gv.swi_s, 4),
            (GgmlType::Q4K, false, true) => (&gv.q4k_s, 4), (GgmlType::Q4K, false, false) => (&gv.q4k_g, 4),
            (GgmlType::Q6K, false, true) => (&gv.q6k_s, 2), (GgmlType::Q6K, false, false) => (&gv.q6k_g, 2),
            c => bail!("no gemv kernel for {c:?}"),
        };
        if let Some(u) = up { if u.arena != wr.arena || u.ty != wr.ty { bail!("gate/up must share arena+type") } }
        let ds = self.set(kern, &[&self.arenas[wr.arena], x, y, &self.f32b])?;
        let bias_off = match bias { Some(n) => *self.fo.get(n).ok_or_else(|| anyhow!("no bias {n}"))?, None => u32::MAX };
        let p = push_u32(&[wr.n, wr.k, (wr.off / unit) as u32, up.map(|u| (u.off / unit) as u32).unwrap_or(0), bias_off, acc as u32, 0]);
        self.ctx.dispatch(cb, kern, ds, &p, wr.n.div_ceil(gv.ls / gv.sg));
        Ok(())
    }

    fn rmsnorm(&self, cb: vk::CommandBuffer, x: &Buffer, y: &Buffer, w: &str) -> Result<()> {
        let ds = self.set(&self.rms, &[x, y, &self.f32b])?;
        let p = push_u32(&[self.cfg.dim, self.fo[w], self.cfg.eps.to_bits()]);
        self.ctx.dispatch(cb, &self.rms, ds, &p, 1); Ok(())
    }

    /// Record the full single-token forward pass (embedding already in b.x, pos in params[0]).
    pub fn record_token(&self) -> Result<Recorded> {
        let mut err = Ok(());
        let r = self.ctx.record(|cb| { err = self.record_into(cb); })?;
        err?; Ok(r)
    }
    fn record_into(&self, cb: vk::CommandBuffer) -> Result<()> {
        let (c, b, gv) = (&self.cfg, &self.b, &self.gv);
        let bar = || self.ctx.barrier(cb);
        for l in 0..c.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let cache_off = l * c.max_ctx * c.n_kv * 128;
            self.rmsnorm(cb, &b.x, &b.xn, &n("attn_norm.weight"))?; bar();
            self.gemv(gv, cb, self.wref(&n("attn_q.weight"))?, None, &b.xn, &b.q, Some(&n("attn_q.bias")), false)?;
            self.gemv(gv, cb, self.wref(&n("attn_k.weight"))?, None, &b.xn, &b.k, Some(&n("attn_k.bias")), false)?;
            self.gemv(gv, cb, self.wref(&n("attn_v.weight"))?, None, &b.xn, &b.v, Some(&n("attn_v.bias")), false)?; bar();
            let ds = self.set(&self.rope, &[&b.q, &b.k, &b.v, &b.kc, &b.vc, &b.params])?;
            self.ctx.dispatch(cb, &self.rope, ds, &push_u32(&[c.n_head, c.n_kv, cache_off, c.theta.to_bits()]), c.n_head + c.n_kv); bar();
            let ds = self.set(&self.attn, &[&b.q, &b.kc, &b.vc, &b.ao, &b.params])?;
            self.ctx.dispatch(cb, &self.attn, ds, &push_u32(&[c.n_head, c.n_kv, cache_off, c.max_ctx]), c.n_head); bar();
            self.gemv(gv, cb, self.wref(&n("attn_output.weight"))?, None, &b.ao, &b.x, None, true)?; bar();
            self.rmsnorm(cb, &b.x, &b.xn, &n("ffn_norm.weight"))?; bar();
            self.gemv(gv, cb, self.wref(&n("ffn_gate.weight"))?, Some(self.wref(&n("ffn_up.weight"))?), &b.xn, &b.hb, None, false)?; bar();
            self.gemv(gv, cb, self.wref(&n("ffn_down.weight"))?, None, &b.hb, &b.x, None, true)?; bar();
        }
        self.rmsnorm(cb, &b.x, &b.xn, "output_norm.weight")?; bar();
        self.gemv(gv, cb, self.wref("output.weight")?, None, &b.xn, &b.logits, None, false)?;
        Ok(())
    }

    pub fn embed(&self, tok: u32) {
        let t = self.g.tensor("token_embd.weight").unwrap();
        let rb = t.row_bytes(); let d = self.g.data(t);
        dequant_row(t.ty, &d[tok as usize * rb..(tok as usize + 1) * rb], self.b.x.as_f32_mut());
    }
    pub fn set_pos(&self, pos: u32) { unsafe { *(self.b.params.ptr as *mut u32) = pos } }
    pub fn argmax(&self) -> u32 { let l = self.b.logits.as_f32(); let mut bi = 0; for i in 1..l.len() { if l[i] > l[bi] { bi = i } } bi as u32 }

    pub fn vocab_strings(&self) -> Vec<String> {
        match self.g.kv.get("tokenizer.ggml.tokens") { Some(gguf::Value::Arr(a)) => a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect(), _ => vec![] }
    }
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

/// Make sure a mapped file range is resident, populated by sequential buffered reads (readahead builds large,
/// physically contiguous folios; faulting through the mapping or GUP tends to create small ones).
pub fn warm_page_cache(m: &[u8]) -> Result<()> {
    let n = m.len().div_ceil(4096); let mut vec = vec![0u8; n];
    unsafe { libc::mincore(m.as_ptr() as *mut _, m.len(), vec.as_mut_ptr()); }
    let resident = vec.iter().filter(|v| **v & 1 != 0).count();
    if resident == n { return Ok(()) }
    // touch via read(2) on /proc/self/fd is not possible from a slice; caller passes the mmap of the file, so read the file by path-less fd:
    // use madvise(MADV_POPULATE_READ) after MADV_HUGEPAGE (PMD-order readahead for VM_HUGEPAGE file mappings).
    let t = std::time::Instant::now();
    unsafe { libc::madvise(m.as_ptr() as *mut _, m.len(), libc::MADV_HUGEPAGE); libc::madvise(m.as_ptr() as *mut _, m.len(), libc::MADV_SEQUENTIAL); }
    let r = unsafe { libc::madvise(m.as_ptr() as *mut _, m.len(), 22 /* MADV_POPULATE_READ */) };
    eprintln!("warm: {resident}/{n} pages were resident; MADV_POPULATE_READ ret {r} in {:.3}s", t.elapsed().as_secs_f64());
    Ok(())
}

/// Export [ps,pe) of the mapping through /dev/strixpc and import it so that GPU VA and file offsets agree mod 2 MiB
/// (RADV places imported BOs at 1 MiB alignment; we shift the export start back by the VA misalignment).
/// Returns the buffer and the file offset that buffer byte 0 corresponds to.
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
        let need = ((AL2 - va % AL2) % AL2) as usize; // VA misalignment we must absorb
        if need > ps { bail!("cannot align dma-buf import at file offset {ps}") }
        pad = need;
    }
    bail!("dma-buf VA alignment did not converge")
}
