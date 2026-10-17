//! Batched prefill for Muse Glimmer on crates/prefill (coop-matrix GEMMs over f16 activations) plus glimmer kernels batched over token
//! columns (norms: workgroup y; q/k/v prep + RoPE: y; split-K attention: z = query; combine: y). Chunks of <= `chunk` tokens; local layers use a
//! ring of win + chunk slots so a chunk never overwrites keys its own queries still need. Leaves KV cache and aux[160] as the token path would.
use crate::{Model, NONE, POST_EPS};
use anyhow::{bail, Result};
use ash::vk;
use prefill::{GemmOp, Pf};
use rtcore::{qtype, DsCache, WRef};
use vkrt::{push_u32, Buffer, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }

pub struct Prefill { pub pf: Pf, pub chunk: u32, ds: DsCache, h: Kernel, x: Buffer, a: Buffer, al: u32, y: Buffer, t: Buffer, pr: Buffer, ldp: u32, ao: Buffer, part: Buffer, copy: Kernel,
    /// multimodal: (embeddings [cnt][d] f32, first position, cnt); rows replace the token embeddings (before the weightless RMSNorm, as llama.cpp raw-embd inputs)
    pub img: Option<(Buffer, u32, u32)> }

impl Prefill {
    pub fn new(m: &Model, chunk: u32) -> Result<Self> {
        let (c, ctx) = (&m.cfg, &m.ctx);
        if c.layers.iter().any(|l| l.swa && l.cap < (c.win + chunk).min(c.max_ctx)) { bail!("local ring too small for chunk {chunk} (Model::load with pf_chunk)") }
        let (qn, kn) = (c.n_head * c.hd, c.n_kv * c.hd);
        let ldp = 2 * qn + 2 * kn; let al = c.dim.max(c.ff).max(qn).next_multiple_of(64);
        let mm = chunk as u64; let f = |n: u64| ctx.buffer(n.max(64) * 4);
        Ok(Prefill { pf: Pf::new(ctx)?, chunk, ds: DsCache::default(), h: ctx.kernel(spv!("glm_h"), 3, 28, &[], None)?,
            x: f(mm * c.dim as u64)?, a: ctx.buffer((mm + 256) * al as u64 * 2)?, al, y: f(mm * 2 * c.ff as u64)?, t: f(mm * c.dim as u64)?,
            pr: f(mm * ldp as u64)?, ldp, ao: f(mm * qn as u64)?, part: f(mm * (c.n_head * (2 + c.hd)) as u64)?, copy: ctx.kernel(spv!("glm_copy"), 2, 12, &[], None)?, img: None })
    }
    fn set(&self, m: &Model, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&m.ctx, k, bufs) }
    fn h(&self, m: &Model, cb: vk::CommandBuffer, x: &Buffer, b: &Buffer, k: u32, xcs: u32, mode: u32, f: f32, b_off: u32, bcs: u32, n: u32) -> Result<()> {
        let ds = self.set(m, &self.h, &[x, b, &self.a])?;
        m.ctx.dispatch3(cb, &self.h, ds, &push_u32(&[k, self.al, xcs, mode, f.to_bits(), b_off, bcs]), k.div_ceil(256), n, 1);
        Ok(())
    }
    fn gemm(&self, m: &Model, cb: vk::CommandBuffer, w: WRef, n: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        self.pf.gemm(&m.ctx, cb, &m.wt.arenas[w.arena], &m.wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &self.a, lda: self.al, a_row0: 0, m: n, y, ldy, y_col, acc: false })
    }
    fn rmsb(&self, m: &Model, cb: vk::CommandBuffer, inp: &Buffer, out: &Buffer, w: u32, mode: u32, eps: f32, n: u32) -> Result<()> {
        let d = m.cfg.dim;
        let ds = self.set(m, &m.norm, &[inp, out, &m.aux, &m.wt.f32b])?;
        m.ctx.dispatch3(cb, &m.norm, ds, &push_u32(&[d, w, eps.to_bits(), mode, NONE, 0, d, d]), 1, n, 1);
        Ok(())
    }
    pub fn record(&self, m: &Model, cb: vk::CommandBuffer, n: u32) -> Result<()> {
        let (c, cfg, wt) = (&m.ctx, &m.cfg, &m.wt);
        let bar = || c.barrier(cb);
        let d = cfg.dim; let (qn, kn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
        bar();
        let te = wt.wref("token_embd.weight")?;
        for j in 0..n {
            let ds = self.set(m, &m.embed, &[&wt.arenas[te.arena], &self.x, &m.aux, &m.toks, &wt.f32b])?;
            c.dispatch(cb, &m.embed, ds, &push_u32(&[te.off as u32, d, qtype(te.ty)?, 160, j, j * d]), 1);
        }
        bar();
        if let Some((eb, ip, cnt)) = &self.img {
            let p0 = unsafe { *(m.aux.ptr as *const u32).add(160) };
            let (lo, hi) = (p0.max(*ip), (p0 + n).min(ip + cnt));
            if lo < hi {
                let ds = self.set(m, &self.copy, &[eb, &self.x])?;
                c.dispatch(cb, &self.copy, ds, &push_u32(&[(hi - lo) * d, (lo - ip) * d, (lo - p0) * d]), ((hi - lo) * d).div_ceil(256)); bar();
            }
        }
        self.rmsb(m, cb, &self.x, &self.x, 0, 2, cfg.eps, n)?; bar();
        for l in 0..cfg.n_layer {
            let nm = |s: &str| format!("blk.{l}.{s}");
            let ly = cfg.layers[l as usize];
            self.pf.run(c, "rmsnorm", &[&self.x, &self.a, &wt.f32b], &[d, self.al, wt.f(&nm("attn_norm.weight"))?, cfg.eps.to_bits(), 0], n, cb)?; bar();
            self.gemm(m, cb, wt.wref(&nm("attn_q.weight"))?, n, &self.pr, self.ldp, 0)?;
            self.gemm(m, cb, wt.wref(&nm("attn_k.weight"))?, n, &self.pr, self.ldp, qn)?;
            self.gemm(m, cb, wt.wref(&nm("attn_v.weight"))?, n, &self.pr, self.ldp, qn + kn)?;
            self.gemm(m, cb, wt.wref(&nm("attn_gate.weight"))?, n, &self.pr, self.ldp, qn + 2 * kn)?; bar();
            let ds = self.set(m, &m.qkr, &[&self.pr, &m.kc, &m.vc, &m.aux, &wt.f32b])?;
            c.dispatch3(cb, &m.qkr, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cfg.hd, wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?, ly.theta.to_bits(),
                if ly.swa { 0 } else { 2 }, ly.cache_off, ly.cap, qn, qn + kn, 160, cfg.eps.to_bits(), 0, 0, self.ldp]), cfg.n_head + cfg.n_kv, n, 1); bar();
            let ds = self.set(m, &m.attn, &[&self.pr, &m.kc, &m.vc, &self.part, &m.aux])?;
            c.dispatch3(cb, &m.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, ly.cache_off, 1, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0,
                if ly.swa { cfg.win } else { 0 }, ly.cap, self.ldp]), cfg.n_head, 1, n); bar();
            let ds = self.set(m, &m.comb, &[&self.part, &self.ao])?;
            c.dispatch3(cb, &m.comb, ds, &push_u32(&[1, 0, qn, cfg.n_head]), cfg.n_head, n, 1); bar();
            self.h(m, cb, &self.ao, &self.pr, qn, qn, 4, 1.0, qn + 2 * kn, self.ldp, n)?; bar();
            self.gemm(m, cb, wt.wref(&nm("attn_output.weight"))?, n, &self.t, d, 0)?; bar();
            self.rmsb(m, cb, &self.t, &self.x, wt.f(&nm("post_attention_norm.weight"))?, 1, POST_EPS, n)?; bar();
            self.pf.run(c, "rmsnorm", &[&self.x, &self.a, &wt.f32b], &[d, self.al, wt.f(&nm("ffn_norm.weight"))?, cfg.eps.to_bits(), 0], n, cb)?; bar();
            self.gemm(m, cb, wt.wref(&nm("ffn_gate.weight"))?, n, &self.y, 2 * cfg.ff, 0)?;
            self.gemm(m, cb, wt.wref(&nm("ffn_up.weight"))?, n, &self.y, 2 * cfg.ff, cfg.ff)?; bar();
            self.h(m, cb, &self.y, &self.y, cfg.ff, 2 * cfg.ff, 1, 1.0, 0, 0, n)?; bar();
            self.gemm(m, cb, wt.wref(&nm("ffn_down.weight"))?, n, &self.t, d, 0)?; bar();
            self.rmsb(m, cb, &self.t, &self.x, wt.f(&nm("post_ffw_norm.weight"))?, 1, POST_EPS, n)?; bar();
        }
        bar();
        Ok(())
    }
    /// Prefill prompt[..np-1] in chunks (toks/au set by the caller); returns positions done (au[160] = that).
    pub fn run(&self, m: &Model, np: usize) -> Result<usize> {
        if np < 2 { return Ok(0) }
        let au = unsafe { std::slice::from_raw_parts_mut(m.aux.ptr as *mut u32, 1024) };
        let (mut p, end) = (0usize, np - 1);
        while p < end {
            let n = (end - p).min(self.chunk as usize) as u32;
            au[160] = p as u32;
            let mut err = Ok(());
            let rec = m.ctx.record(|cb| { err = self.record(m, cb, n); })?; err?;
            m.ctx.submit(&rec)?;
            p += n as usize;
        }
        au[160] = p as u32;
        Ok(p)
    }
}
