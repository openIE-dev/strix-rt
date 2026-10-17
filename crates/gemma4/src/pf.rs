//! Batched prefill for Gemma 4 on crates/prefill (coop-matrix GEMMs over f16 activations) plus gemma4 kernels batched
//! over token columns: norms (workgroup y), q/k/v prep + RoPE (y), split-K attention (z = query), combine (y).
//! Chunks of <= `chunk` tokens; SWA layers use a ring of win + chunk slots so a chunk never overwrites keys its own
//! queries still need. Leaves the KV cache and aux[160] where the token-by-token path would.
use crate::{Model, NONE};
use anyhow::{bail, Result};
use ash::vk;
use prefill::{GemmOp, Pf};
use rtcore::{qtype, DsCache, WRef};
use vkrt::{push_u32, Buffer, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }

/// Upload image embeddings for positions [pos0, pos0 + rows) (token ids there should be 0: PLE uses the padding row).
pub fn image_buf(m: &Model, emb: &[f32]) -> Result<Buffer> { let b = m.ctx.buffer((emb.len() * 4) as u64)?; b.as_f32_mut()[..emb.len()].copy_from_slice(emb); Ok(b) }

thread_local! { static PQ: std::cell::Cell<Option<vk::QueryPool>> = std::cell::Cell::new(None); static PL: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(vec![]); static CUR: std::cell::Cell<&'static str> = std::cell::Cell::new("start"); }
fn lab(x: &'static str) { CUR.with(|c| c.set(x)) }
fn tick(ctx: &vkrt::Ctx, cb: vk::CommandBuffer) { if let Some(qp) = PQ.with(|q| q.get()) { PL.with(|l| { let mut l = l.borrow_mut(); let i = l.len() as u32; unsafe { ctx.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::COMPUTE_SHADER, qp, i) } l.push(CUR.with(|c| c.get()).to_string()); }) } }

pub struct Prefill { pub pf: Pf, pub chunk: u32, ds: DsCache, h: Kernel, x: Buffer, a: Buffer, al: u32, y: Buffer, ldy: u32, t: Buffer, pr: Buffer, ldp: u32, ao: Buffer, lda_o: u32,
    praw: Buffer, plei: Buffer, part: Buffer, xs: f32, m1: Buffer, hm: Buffer, route: Buffer,
    gm: Option<Gm>, fa: Option<Kernel>, fa5: Option<(Kernel, Buffer)>, copy: Kernel,
    /// multimodal: (embeddings [cnt][d] f32, first position, cnt); rows replace the scaled token embeddings (llama.cpp raw-embd path)
    pub img: Option<(Buffer, u32, u32)> }
/// grouped-MoE prefill state (crate::gmoe)
struct Gm { dq: Vec<Option<(Buffer, WRef)>>, sort: Kernel, comb: Kernel, g: Buffer, gt: Buffer, ag: Buffer, yg: Buffer, hg: Buffer, yd: Buffer, bm: u32, qh: bool }

impl Prefill {
    pub fn new(m: &Model, chunk: u32) -> Result<Self> {
        let (c, ctx) = (&m.cfg, &m.ctx);
        if c.layers.iter().any(|l| l.swa && l.has_kv && l.cap < (c.win + chunk).min(c.max_ctx)) { bail!("SWA ring too small for chunk {chunk} (Model::load with pf_chunk)") }
        let hmax = c.layers.iter().map(|l| l.hd).max().unwrap();
        let kvmax = c.layers.iter().map(|l| l.hd * l.n_kv).max().unwrap();
        let ldp = c.n_head * hmax + 2 * kvmax;
        let pw = c.ple * c.n_layer;
        let al = c.dim.max(c.ff).max(c.n_head * hmax).max(c.ple).next_multiple_of(64);
        let ldy = (2 * c.ff).max(pw).max(c.ple).max(c.dim);
        // Gemma 4 prefill tuning of the shared pf_qgemmh (measured logs/g4/pp_sweep2.log: 1546 -> 1621 tok/s): 1x8 waves, k-step 128 for Q8_0.
        // Process-wide env defaults (set only if unset, so PF_QH_WR=2 / PF_QH_KD=8 still reproduce the old config); affects only this process.
        for (k, v) in [("PF_QH_WR", "1"), ("PF_QH_KD", "4"), ("PF_QMM_MI", "1")] { if std::env::var_os(k).is_none() { unsafe { std::env::set_var(k, v); } } }
        let mm = chunk as u64;
        let f = |n: u64| ctx.buffer(n.max(64) * 4);
        Ok(Prefill { pf: Pf::new(ctx)?, chunk, ds: DsCache::default(), h: ctx.kernel(spv!("g4_h"), 3, 36, &[], None)?,
            x: f(mm * c.dim as u64)?, a: ctx.buffer((mm + 256) * al as u64 * 2)?, al, y: f(mm * ldy as u64)?, ldy, t: f(mm * c.dim as u64)?,
            pr: f(mm * ldp as u64)?, ldp, ao: f(mm * (c.n_head * hmax) as u64)?, lda_o: c.n_head * hmax,
            praw: f(mm * pw as u64)?, plei: f(mm * pw as u64)?, part: f(mm * (c.n_head * (2 + hmax)) as u64)?,
            m1: f(mm * c.dim as u64)?, hm: f(mm * (c.top_k * c.ff_exp) as u64)?, route: f(mm * 16 + 1024)?,
            gm: if c.n_exp > 0 && std::env::var("G4_PF_GMOE").map(|v| v != "0").unwrap_or(true) {
                let nk = mm * c.top_k as u64; let gu = m.wt.wref("blk.0.ffn_gate_up_exps.weight")?.n as u64; let qh = std::env::var("G4_QMMH").map(|v| v != "0").unwrap_or(true); let bm = if qh { Pf::qgemm_rows() } else { 32u32 };
                Some(Gm { dq: crate::gmoe::down_q8(ctx, m)?, sort: ctx.kernel(spv!("g4_msort"), 3, 16, &[], None)?, comb: ctx.kernel(spv!("g4_mcomb"), 5, 16, &[], None)?,
                    g: f(2 * nk)?, gt: f(4 + 3 * (nk / bm as u64 + c.n_exp as u64 + 2))?, ag: ctx.buffer((nk + 256) * c.dim as u64 * 2)?, yg: f(nk * gu)?,
                    hg: ctx.buffer((nk + 256) * c.ff_exp as u64 * 2)?, yd: f(nk * c.dim as u64)?, bm, qh })
            } else { None },
            copy: ctx.kernel(spv!("g4_copy"), 2, 12, &[], None)?, img: None,
            fa5: if std::env::var("G4_PF_FA5").map(|v| v != "0").unwrap_or(true) && hmax == 512 {
                Some((ctx.kernel(spv!("g4_attnfa512"), 5, 24, &[], Some(32))?, ctx.buffer((mm + 64) * (c.n_head * 512) as u64 * 2)?)) } else { None },
            fa: if std::env::var("G4_PF_FA").map(|v| v != "0").unwrap_or(true) { Some(ctx.kernel(spv!("g4_attnfa"), 5, 44, &[std::env::var("G4_PF_FA_QLO").ok().and_then(|v| v.parse().ok()).unwrap_or(1)], Some(32))?) } else { None },
            xs: std::env::var("G4_PF_XS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0 / 16.0) })
    }
    fn set(&self, m: &Model, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&m.ctx, k, bufs) }
    fn h(&self, m: &Model, cb: vk::CommandBuffer, x: &Buffer, b: &Buffer, k: u32, xcs: u32, mode: u32, f: f32, b_off: u32, bcs: u32, n: u32) -> Result<()> {
        let ds = self.set(m, &self.h, &[x, b, &self.a])?;
        m.ctx.dispatch3(cb, &self.h, ds, &push_u32(&[k, self.al, xcs, mode, f.to_bits(), b_off, bcs, (-65504f32).to_bits(), 65504f32.to_bits()]), k.div_ceil(256), n, 1);
        Ok(())
    }
    fn gemm(&self, m: &Model, cb: vk::CommandBuffer, w: WRef, n: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        self.pf.gemm(&m.ctx, cb, &m.wt.arenas[w.arena], &m.wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &self.a, lda: self.al, a_row0: 0, m: n, y, ldy, y_col, acc: false })
    }
    /// grouped expert FFN for n columns: input y (pre_ffw_norm_2 output, f32 [n][d]), output t (f32 [n][d])
    fn gmoe(&self, m: &Model, cb: vk::CommandBuffer, g: &Gm, l: u32, n: u32) -> Result<()> {
        let (c, cfg, wt) = (&m.ctx, &m.cfg, &m.wt); let bar = || { c.barrier(cb); tick(c, cb); };
        let (d, fe, tk) = (cfg.dim, cfg.ff_exp, cfg.top_k); let nk = n * tk;
        let gu = wt.wref(&format!("blk.{l}.ffn_gate_up_exps.weight"))?;
        let maxt = nk.div_ceil(g.bm) + cfg.n_exp;
        let ds = self.set(m, &g.sort, &[&self.route, &g.g, &g.gt])?;
        c.dispatch(cb, &g.sort, ds, &push_u32(&[n, tk, cfg.n_exp, g.bm]), 1);
        lab("moe_gather"); self.hx(m, cb, &self.y, &self.y, &self.a, self.al, d, d, 0, 1.0, n)?; bar();
        self.pf.run(c, "gather", &[&self.a, &g.ag, &g.g], &[d / 2, self.al / 2, d / 2, nk], (d / 2 * nk).div_ceil(256), cb)?; bar();
        // G4_QMMH (default on, 0 = old pf gemm_grouped): shared packed-LDS expert kernel (prefill pf_qgemmh: Q4_K gate_up, Q8_0 down)
        lab("moe_up"); if g.qh { self.pf.qgemmh_grouped(c, cb, &wt.arenas[gu.arena], gu, &g.gt, maxt, &g.ag, d, &g.yg, gu.n, 0)?; }
        else { self.pf.gemm_grouped(c, cb, &wt.arenas[gu.arena], &wt.f32b, gu, &g.gt, g.bm, maxt, &g.ag, d, &g.yg, gu.n, 0)?; } bar();
        lab("moe_act"); self.hx(m, cb, &g.yg, &g.yg, &g.hg, fe, fe, gu.n, 1, 1.0, nk)?; bar(); lab("moe_down");
        let dn = wt.wref(&format!("blk.{l}.ffn_down_exps.weight"))?;
        match &g.dq[l as usize] {
            Some((b, w)) if g.qh => self.pf.qgemmh_grouped(c, cb, b, *w, &g.gt, maxt, &g.hg, fe, &g.yd, d, 0)?,
            Some((b, w)) => self.pf.gemm_grouped(c, cb, b, &wt.f32b, *w, &g.gt, g.bm, maxt, &g.hg, fe, &g.yd, d, 0)?,
            None => self.pf.gemm_grouped(c, cb, &wt.arenas[dn.arena], &wt.f32b, dn, &g.gt, g.bm, maxt, &g.hg, fe, &g.yd, d, 0)?,
        }
        bar();
        lab("moe_comb"); let ds = self.set(m, &g.comb, &[&self.t, &g.yd, &self.route, &g.g, &wt.f32b])?;
        c.dispatch3(cb, &g.comb, ds, &push_u32(&[d, n, tk, wt.f(&format!("blk.{l}.ffn_down_exps.scale")).unwrap_or(NONE)]), d.div_ceil(256), n, 1); bar();
        Ok(())
    }
    /// g4_h into an arbitrary f16 buffer
    fn hx(&self, m: &Model, cb: vk::CommandBuffer, x: &Buffer, b: &Buffer, out: &Buffer, lda: u32, k: u32, xcs: u32, mode: u32, f: f32, n: u32) -> Result<()> {
        let ds = self.set(m, &self.h, &[x, b, out])?;
        m.ctx.dispatch3(cb, &self.h, ds, &push_u32(&[k, lda, xcs, mode, f.to_bits(), 0, 0, (-65504f32).to_bits(), 65504f32.to_bits()]), k.div_ceil(256), n, 1);
        Ok(())
    }
    /// batched g4_norm over n columns (stride d)
    fn rmsb(&self, m: &Model, cb: vk::CommandBuffer, inp: &Buffer, out: &Buffer, w: u32, mode: u32, s: u32, n: u32) -> Result<()> {
        let d = m.cfg.dim;
        let ds = self.set(m, &m.norm, &[inp, out, &m.aux, &m.wt.f32b])?;
        m.ctx.dispatch3(cb, &m.norm, ds, &push_u32(&[d, w, m.cfg.eps.to_bits(), mode, s, 0, d, d]), 1, n, 1);
        Ok(())
    }
    /// Record one chunk of n tokens at positions au[160]..+n (toks already in m.toks).
    pub fn record(&self, m: &Model, cb: vk::CommandBuffer, n: u32) -> Result<()> {
        let (c, cfg, wt) = (&m.ctx, &m.cfg, &m.wt);
        let bar = || { c.barrier(cb); tick(c, cb); };
        let d = cfg.dim; let pw = cfg.ple * cfg.n_layer;
        bar();
        let te = wt.wref("token_embd.weight")?;
        let pe = if cfg.ple > 0 { Some(wt.wref("per_layer_token_embd.weight")?) } else { None };
        for j in 0..n {
            let ds = self.set(m, &m.embed, &[&wt.arenas[te.arena], &self.x, &m.aux, &m.toks, &wt.f32b])?;
            c.dispatch(cb, &m.embed, ds, &push_u32(&[te.off as u32, d, qtype(te.ty)?, 160, j, j * d]), 1);
            if let Some(pe) = pe {
                let ds = self.set(m, &m.embed, &[&wt.arenas[pe.arena], &self.praw, &m.aux, &m.toks, &wt.f32b])?;
                c.dispatch(cb, &m.embed, ds, &push_u32(&[pe.off as u32, pw, qtype(pe.ty)?, 160, j, j * pw]), 1);
            }
        }
        bar();
        let ds = self.set(m, &m.misc, &[&self.x, &self.x, &self.x])?;
        c.dispatch(cb, &m.misc, ds, &push_u32(&[n * d, 2, 0, (d as f32).sqrt().to_bits(), 0, 0, 0]), (n * d).div_ceil(256)); bar();
        if let Some((eb, ip, cnt)) = &self.img {
            let p0 = unsafe { *(m.aux.ptr as *const u32).add(160) };
            let (lo, hi) = (p0.max(*ip), (p0 + n).min(ip + cnt));
            if lo < hi {
                let ds = self.set(m, &self.copy, &[eb, &self.x])?;
                c.dispatch(cb, &self.copy, ds, &push_u32(&[(hi - lo) * d, (lo - ip) * d, (lo - p0) * d]), ((hi - lo) * d).div_ceil(256)); bar();
            }
        }
        let xs = self.xs;
        if cfg.ple > 0 {
            self.h(m, cb, &self.x, &self.x, d, d, 0, xs, 0, 0, n)?; bar();
            self.gemm(m, cb, wt.wref("per_layer_model_proj.weight")?, n, &self.y, pw, 0)?; bar();
            let ds = self.set(m, &m.ple, &[&self.y, &self.praw, &self.plei, &wt.f32b])?;
            // proj computed on x * xs: fold 1/xs into the projection scale (the RMSNorm only sees the product)
            c.dispatch3(cb, &m.ple, ds, &push_u32(&[cfg.ple, wt.f("per_layer_proj_norm.weight")?, cfg.eps.to_bits(), (1.0 / (d as f32).sqrt() / xs).to_bits(),
                (cfg.ple as f32).sqrt().to_bits(), std::f32::consts::FRAC_1_SQRT_2.to_bits(), pw]), cfg.n_layer, n, 1); bar();
        }
        for l in 0..cfg.n_layer {
            let nm = |s: &str| format!("blk.{l}.{s}");
            let ly = cfg.layers[l as usize];
            let (hd, qn, kn) = (ly.hd, cfg.n_head * ly.hd, ly.n_kv * ly.hd);
            lab("norm"); self.pf.run(c, "rmsnorm", &[&self.x, &self.a, &wt.f32b], &[d, self.al, wt.f(&nm("attn_norm.weight"))?, cfg.eps.to_bits(), 0], n, cb)?; bar();
            lab("qkv"); self.gemm(m, cb, wt.wref(&nm("attn_q.weight"))?, n, &self.pr, self.ldp, 0)?;
            let mut v_base = qn + kn;
            if ly.has_kv {
                self.gemm(m, cb, wt.wref(&nm("attn_k.weight"))?, n, &self.pr, self.ldp, qn)?;
                match wt.wref(&nm("attn_v.weight")) { Ok(v) => self.gemm(m, cb, v, n, &self.pr, self.ldp, qn + kn)?, Err(_) => v_base = qn }
            }
            bar();
            lab("qkr"); let ds = self.set(m, &m.qkr, &[&self.pr, &m.kc, &m.vc, &m.aux, &wt.f32b])?;
            let kn_off = if ly.has_kv { wt.f(&nm("attn_k_norm.weight"))? } else { 0 };
            c.dispatch3(cb, &m.qkr, ds, &push_u32(&[cfg.n_head, ly.n_kv, hd, wt.f(&nm("attn_q_norm.weight"))?, kn_off, ly.theta.to_bits(), ly.ff_off, ly.cache_off, ly.cap,
                qn, v_base, 160, cfg.eps.to_bits(), ly.has_kv as u32, 0, 0, self.ldp]), cfg.n_head + if ly.has_kv { ly.n_kv } else { 0 }, n, 1); bar();
            lab(if ly.swa { "attn_swa" } else { "attn_glob" }); let ki = (hd == 512) as usize;
            if hd == 512 && self.fa5.is_some() && !ly.swa {
                let (k5, q16) = self.fa5.as_ref().unwrap();
                self.hx(m, cb, &self.pr, &self.pr, q16, qn, qn, self.ldp, 0, 1.0, n)?; bar();
                let ds = self.set(m, k5, &[q16, &m.kc, &m.vc, &self.a, &m.aux])?;
                c.dispatch(cb, k5, ds, &push_u32(&[cfg.n_head, ly.n_kv, ly.cache_off, qn, self.al, n]), n.div_ceil(32) * cfg.n_head); bar();
            } else if hd == 256 && self.fa.is_some() {
                // coopmat flash attention (g4_attnfa) -> f16 A directly
                let fa = self.fa.as_ref().unwrap();
                let ds = self.set(m, fa, &[&self.pr, &m.kc, &m.vc, &self.a, &m.aux])?;
                c.dispatch(cb, fa, ds, &push_u32(&[cfg.n_head, ly.n_kv, ly.cache_off, hd, 0, 1.0f32.to_bits(), self.ldp, self.al, n, if ly.swa { cfg.win } else { 0 }, ly.cap]), n.div_ceil(64) * cfg.n_head); bar();
            } else {
            let ds = self.set(m, &m.attn[ki], &[&self.pr, &m.kc, &m.vc, &self.part, &m.aux])?;
            if std::env::var("G4_PF_SKIP").map(|v| v == "4").unwrap_or(false) { bar(); } else { c.dispatch3(cb, &m.attn[ki], ds, &push_u32(&[cfg.n_head, ly.n_kv, ly.cache_off, 1, hd, 160, 1.0f32.to_bits(), 0, 0, if ly.swa { cfg.win } else { 0 }, ly.cap, self.ldp]),
                cfg.n_head, 1, n); } bar();
            let ds = self.set(m, &m.comb[ki], &[&self.part, &self.ao])?;
            c.dispatch3(cb, &m.comb[ki], ds, &push_u32(&[1, 0, self.lda_o, cfg.n_head]), cfg.n_head, n, 1); bar();
            self.h(m, cb, &self.ao, &self.ao, qn, self.lda_o, 0, 1.0, 0, 0, n)?; bar();
            }
            lab("attn_o"); self.gemm(m, cb, wt.wref(&nm("attn_output.weight"))?, n, &self.t, d, 0)?; bar();
            lab("post_attn"); self.rmsb(m, cb, &self.t, &self.x, wt.f(&nm("post_attention_norm.weight"))?, 1, NONE, n)?; bar();
            lab("ffn_norm"); self.pf.run(c, "rmsnorm", &[&self.x, &self.a, &wt.f32b], &[d, self.al, wt.f(&nm("ffn_norm.weight"))?, cfg.eps.to_bits(), 0], n, cb)?; bar();
            lab("ffn_gu"); self.gemm(m, cb, wt.wref(&nm("ffn_gate.weight"))?, n, &self.y, 2 * cfg.ff, 0)?;
            self.gemm(m, cb, wt.wref(&nm("ffn_up.weight"))?, n, &self.y, 2 * cfg.ff, cfg.ff)?; bar();
            lab("ffn_act"); self.h(m, cb, &self.y, &self.y, cfg.ff, 2 * cfg.ff, 1, 1.0, 0, 0, n)?; bar();
            lab("ffn_dn"); self.gemm(m, cb, wt.wref(&nm("ffn_down.weight"))?, n, &self.t, d, 0)?; bar();
            let sc = wt.f(&nm("layer_output_scale.weight")).unwrap_or(NONE);
            if ly.moe {
                // routed experts per column (GEMV kernels, workgroup y = column; routing slots in self.route)
                lab("moe_n1"); self.rmsb(m, cb, &self.t, &self.m1, wt.f(&nm("post_ffw_norm_1.weight"))?, 0, NONE, n)?;
                if PQ.with(|q| q.get()).is_some() { bar(); } lab("moe_rt"); m.route(cb, l, &self.x, &self.route, n)?; if PQ.with(|q| q.get()).is_some() { bar(); }
                lab("moe_n2"); self.rmsb(m, cb, &self.x, &self.y, wt.f(&nm("pre_ffw_norm_2.weight"))?, 0, NONE, n)?; bar();
                lab("moe_sort"); if let Some(g) = &self.gm { self.gmoe(m, cb, g, l, n)?; } else { m.moe(cb, l, &self.y, &self.hm, &self.t, n, &self.route)?; }
                lab("moe_post"); self.rmsb(m, cb, &self.t, &self.m1, wt.f(&nm("post_ffw_norm_2.weight"))?, 1, NONE, n)?; bar();
                self.rmsb(m, cb, &self.m1, &self.x, wt.f(&nm("post_ffw_norm.weight"))?, 1, sc, n)?; bar();
                continue;
            }
            if cfg.ple == 0 { self.rmsb(m, cb, &self.t, &self.x, wt.f(&nm("post_ffw_norm.weight"))?, 1, sc, n)?; bar(); continue }
            self.rmsb(m, cb, &self.t, &self.x, wt.f(&nm("post_ffw_norm.weight"))?, 1, NONE, n)?; bar();
            self.h(m, cb, &self.x, &self.x, d, d, 0, xs, 0, 0, n)?; bar();
            self.gemm(m, cb, wt.wref(&nm("inp_gate.weight"))?, n, &self.y, cfg.ple, 0)?; bar();
            self.h(m, cb, &self.y, &self.plei, cfg.ple, cfg.ple, 2, 1.0 / xs, l * cfg.ple, pw, n)?; bar();
            self.gemm(m, cb, wt.wref(&nm("proj.weight"))?, n, &self.t, d, 0)?; bar();
            self.rmsb(m, cb, &self.t, &self.x, wt.f(&nm("post_norm.weight"))?, 1, sc, n)?; bar();
        }
        bar();
        Ok(())
    }
    /// G4_PFPROF=1: one profiled pass over the first chunk (timestamp after every barrier, summed per label), printed to stderr.
    fn profile(&self, m: &Model, n: u32) -> Result<()> {
        let d = &m.ctx.device; let nq = 4096u32;
        let qp = unsafe { d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(nq), None)? };
        PL.with(|l| l.borrow_mut().clear()); lab("start"); let mut err = Ok(());
        let tot = m.ctx.run(|cb| { unsafe { d.cmd_reset_query_pool(cb, qp, 0, nq) }; PQ.with(|q| q.set(Some(qp))); tick(&m.ctx, cb); err = self.record(m, cb, n); PQ.with(|q| q.set(None)); })?; err?;
        let labs = PL.with(|l| l.borrow().clone()); let mut ts = vec![0u64; labs.len()];
        unsafe { d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; d.destroy_query_pool(qp, None); }
        let pr = m.ctx.info.ts_period_ns as f64 * 1e-9; let mut agg: Vec<(String, f64, u32)> = vec![];
        for i in 1..labs.len() { let t = ts[i].wrapping_sub(ts[i - 1]) as f64 * pr; if let Some(a) = agg.iter_mut().find(|a| a.0 == labs[i]) { a.1 += t; a.2 += 1 } else { agg.push((labs[i].clone(), t, 1)) } }
        let sum: f64 = agg.iter().map(|a| a.1).sum(); agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("G4_PFPROF chunk {n} tok: timestamps {:.1} ms, run() GPU {:.1} ms", sum * 1e3, tot * 1e3);
        for (l, t, c) in agg { eprintln!("{:>12} {:8.2} ms {:5.1}%  x{}", l, t * 1e3, 100.0 * t / sum, c) }
        Ok(())
    }
    /// Prefill prompt[..np-1] in chunks (toks/au already set by the caller); returns positions done (au[160] = that).
    pub fn run(&self, m: &Model, np: usize) -> Result<usize> {
        if np < 2 { return Ok(0) }
        let au = unsafe { std::slice::from_raw_parts_mut(m.aux.ptr as *mut u32, 1024) };
        let (mut p, end) = (0usize, np - 1);
        if std::env::var("G4_PFPROF").map(|v| v == "1").unwrap_or(false) { au[160] = 0; self.profile(m, end.min(self.chunk as usize) as u32)?; }
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
