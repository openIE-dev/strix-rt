//! Meta Muse Glimmer 30B (GGUF arch "muse-glimmer"): dense decoder, GPU-driven greedy decode (structure of crates/gemma4).
//! Per layer: attn_norm (RMS, weight+1 folded by the converter) -> q/k/v/gate GEMVs -> q/k RMSNorm (q weight carries qk_scale_factor, k = 1)
//! -> interleaved (NORM) RoPE on local layers only (global layers NoPE) -> GQA attention, scale 1/sqrt(128), sliding window (ring KV, slot = pos % cap)
//! on local layers (pattern from GGUF, [L,L,L,G] x 13) -> * sigmoid(gate) -> o_proj -> post_attention_norm (eps 1e-8) + residual -> ffn_norm ->
//! SwiGLU (silu(gate) * up) -> down -> post_ffw_norm (eps 1e-8) + residual. Embedding row is RMS-normalised (no weight), the head output is
//! scaled by logit_scale and optionally tanh-softcapped. aux words (as gemma4/gptoss): argmax key 128..129; pos 160; prompt len 161; pred 162.
use anyhow::{bail, Result};
use ash::vk;
use gguf::Value;
use rtcore::{qtype, DsCache, GkQ, Weights, WRef, GQ};
pub mod pf;
pub mod vision;
pub mod tok;
mod ucat;
pub use pf::Prefill;
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
pub(crate) const NONE: u32 = u32::MAX;
pub const POST_EPS: f32 = 1e-8;

#[derive(Clone, Copy, Debug)]
pub struct Layer { pub swa: bool, pub cache_off: u32, pub cap: u32, pub theta: f32 }
pub struct Cfg { pub n_layer: u32, pub dim: u32, pub n_head: u32, pub n_kv: u32, pub hd: u32, pub vocab: u32, pub eps: f32, pub max_ctx: u32, pub win: u32, pub ff: u32,
    pub softcap: f32, pub logit_scale: f32, pub layers: Vec<Layer> }

fn arr(g: &gguf::Gguf, k: &str, n: usize) -> Option<Vec<u64>> {
    match g.get(k).ok()? {
        Value::Arr(a) => Some(a.iter().map(|v| match v { Value::Bool(b) => *b as u64, v => v.as_u64().unwrap_or(0) }).collect()),
        v => v.as_u64().map(|x| vec![x; n]),
    }
}
impl Cfg {
    pub fn from(wt: &Weights, max_ctx: u32, pf_chunk: u32) -> Result<Self> {
        let g = &wt.g; let a = "muse-glimmer";
        let arch = g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default();
        if arch != a { bail!("expected {a}, got {arch}") }
        let u = |k: &str| g.u64(&format!("{a}.{k}")).map(|v| v as u32);
        let fl = |k: &str| g.f32(&format!("{a}.{k}"));
        let nl = u("block_count")?; let dim = u("embedding_length")?; let n_head = u("attention.head_count")?;
        let hd = u("attention.key_length").unwrap_or(dim / n_head);
        let n_kv = arr(g, &format!("{a}.attention.head_count_kv"), nl as usize).map(|v| v[0] as u32).ok_or_else(|| anyhow::anyhow!("no head_count_kv"))?;
        let win = u("attention.sliding_window")?;
        let swa: Vec<u64> = arr(g, &format!("{a}.attention.sliding_window_pattern"), nl as usize).unwrap_or_else(|| (0..nl as u64).map(|l| (l % 4 != 3) as u64).collect());
        let th_g = fl("rope.freq_base")?; let th_s = fl("rope.freq_base_swa").unwrap_or(th_g);
        let max_ctx = max_ctx.min(u("context_length").unwrap_or(131072));
        let mut layers = vec![]; let mut off = 0u32;
        for l in 0..nl { let s = swa[l as usize] != 0;
            let cap = if s { (win + pf_chunk).min(max_ctx) } else { max_ctx };
            layers.push(Layer { swa: s, cache_off: off, cap, theta: if s { th_s } else { th_g } }); off += cap * n_kv * hd; }
        Ok(Cfg { n_layer: nl, dim, n_head, n_kv, hd, vocab: wt.wref("token_embd.weight")?.n, eps: fl("attention.layer_norm_rms_epsilon")?, max_ctx, win, ff: u("feed_forward_length")?,
            softcap: fl("final_logit_softcapping").unwrap_or(0.0), logit_scale: fl("logit_scale").unwrap_or(1.0), layers })
    }
    pub fn kv_elems(&self) -> u64 { self.layers.iter().map(|l| (l.cap * self.n_kv * self.hd) as u64).sum() }
}

pub struct Model {
    pub ctx: Ctx, pub wt: Weights, pub cfg: Cfg, ds: DsCache,
    g_proj: GkQ, g_ffn: GkQ, fuse_ffn: bool, g_head: GkQ,
    pub(crate) embed: Kernel, q4v: u32, xsk: Kernel, xs: Buffer, use_xs: bool, xqk: Kernel, qgv: Kernel, qgv_rows: u32, xqb: Buffer, idot: bool, pub(crate) norm: Kernel, pub(crate) misc: Kernel, pub(crate) qkr: Kernel, pub(crate) attn: Kernel, pub(crate) comb: Kernel, argmax: Kernel, sample: Kernel, nc: u32,
    pub x: Buffer, xw: Buffer, t: Buffer, pr: Buffer, ao: Buffer, hb: Buffer, hg: Buffer,
    pub logits: Buffer, pub(crate) kc: Buffer, pub(crate) vc: Buffer, part: Buffer, pub toks: Buffer, pub aux: Buffer, pub step: vk::CommandBuffer,
}

impl Model {
    pub fn load(path: &str, max_ctx: u32, pf_chunk: u32) -> Result<Self> {
        let ctx = Ctx::new()?;
        let wt = Weights::load(&ctx, path)?;
        let cfg = Cfg::from(&wt, max_ctx, pf_chunk)?;
        let sg = env("GEMV_SG", 32);
        let f = |n: u64| ctx.buffer(n.max(16) * 4);
        let nc = env("ATTN_NC", 4);
        // Q4_K item-pair GEMV is the default for Glimmer (GEMVQ_Q4P=0 restores the previous kernel); read by rtcore::GkQ at pipeline creation
        if std::env::var("GEMVQ_Q4P").is_err() { std::env::set_var("GEMVQ_Q4P", "5"); }
        let q4v: u32 = std::env::var("GEMVQ_Q4P").ok().and_then(|v| v.parse().ok()).unwrap_or(0); let q4p = q4v != 0;
        let (qlpr, qls) = { let v = std::env::var("GLM_IDOT_LPR").unwrap_or_default(); let p: Vec<u32> = v.split(",").filter_map(|x| x.parse().ok()).collect(); if p.len() == 2 { (p[0], p[1]) } else { (16, 128) } };
        let kvn = cfg.kv_elems();
        let (qn, kn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
        let m = Model {
            q4v, xsk: ctx.kernel(if q4v & 8 != 0 { spv!("glm_xs2") } else { spv!("glm_xs") }, 2, 12, &[], None)?, xs: f(16384)?, use_xs: q4v & 4 != 0, xqk: ctx.kernel(spv!("glm_xq"), 2, 8, &[], None)?, qgv: ctx.kernel(match qlpr.min(sg) { 4 => spv!("glm_qgv_l4"), 8 => spv!("glm_qgv_l8"), 32 => spv!("glm_qgv_l32"), _ => spv!("glm_qgv_l16") }, 3, 96, &[qls], Some(sg))?, qgv_rows: qls / qlpr.min(sg), xqb: ctx.buffer(4 * 1024 * 32)?, idot: env("GLM_IDOT", 0) != 0,
            g_ffn: GkQ::new("GEMV_FFN", (16, 256), true, sg), fuse_ffn: q4p && q4v & 5 == 5 && env("GLM_FUSE_FFN", 0) != 0,
            g_proj: GkQ::new("GEMV_PROJ", if q4p { (16, 256) } else { (32, 128) }, false, sg), g_head: GkQ::new("GEMV_HEAD", (8, 256), false, sg),
            embed: ctx.kernel(shaders::EMBED_ANY, 5, 24, &[], None)?, norm: ctx.kernel(spv!("glm_norm"), 4, 32, &[env("GLM_NORM_WG", 1024)], None)?, misc: ctx.kernel(spv!("glm_misc"), 3, 28, &[], None)?,
            qkr: ctx.kernel(spv!("glm_qkr"), 5, 64, &[], None)?, attn: ctx.kernel(spv!("glm_attn"), 5, 48, &[cfg.hd], None)?, comb: ctx.kernel(spv!("glm_comb"), 2, 16, &[cfg.hd], None)?,
            argmax: ctx.kernel(shaders::ARGMAX_W, 2, 4, &[], None)?, sample: ctx.kernel(shaders::SAMPLE_W, 2, 0, &[], None)?, nc,
            x: f(cfg.dim as u64)?, xw: f(cfg.dim as u64)?, t: f(cfg.dim as u64)?, pr: f((2 * qn + 2 * kn) as u64)?, ao: f(qn as u64)?,
            hb: f(2 * cfg.ff as u64)?, hg: f(cfg.ff as u64)?,
            logits: f(cfg.vocab as u64)?, kc: ctx.buffer((kvn * 2).max(256))?, vc: ctx.buffer((kvn * 2).max(256))?, part: f((cfg.n_head * nc * (2 + cfg.hd)) as u64)?,
            toks: f(2 * cfg.max_ctx as u64 + 2)?, aux: f(4096)?, step: vk::CommandBuffer::null(), ds: DsCache::default(), cfg, wt, ctx };
        let mut m = m;
        let mut err = Ok(());
        let step = m.ctx.record_raw(|cb| { err = m.record(cb); })?; err?;
        m.step = step;
        Ok(m)
    }
    fn set(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&self.ctx, k, bufs) }
    fn gq(&self, cb: vk::CommandBuffer, gq: &GkQ, segs: &[WRef], x: &Buffer, y: &Buffer, acc: bool, grp: u32) -> Result<()> {
        if env("GLM_SKIP", 0) & grp != 0 { return Ok(()) }
        let (mut g, arena) = GQ::segs(segs, None, &[], self.cfg.eps)?;
        if self.idot && !acc && segs.iter().all(|w| matches!(w.ty, gguf::GgmlType::Q4K | gguf::GgmlType::Q6K)) {
            let ds = self.set(&self.xqk, &[x, &self.xqb])?;
            self.ctx.dispatch(cb, &self.xqk, ds, &push_u32(&[g.k, 0]), (g.k / 32).div_ceil(64));
            self.ctx.barrier(cb);
            let ds = self.set(&self.qgv, &[&self.wt.arenas[arena], &self.xqb, y])?;
            self.ctx.dispatch(cb, &self.qgv, ds, &g.bytes(), g.n.div_ceil(self.qgv_rows));
            return Ok(());
        }
        if self.use_xs && (segs.iter().all(|w| matches!(w.ty, gguf::GgmlType::Q4K)) && g.k % 64 == 0 || self.q4v & 64 != 0 && segs.iter().all(|w| matches!(w.ty, gguf::GgmlType::Q6K)) && g.k % 128 == 0) {
            let ds = self.set(&self.xsk, &[x, &self.xs])?;
            self.ctx.dispatch(cb, &self.xsk, ds, &push_u32(&[g.k, g.x_off4, 1 << ((self.q4v >> 4) & 3)]), (g.k / 32).div_ceil(64));
            self.ctx.barrier(cb);
        }
        let gk = gq.get(&self.ctx, &g, segs.len())?;
        g.flags |= 64; if acc { g.flags |= 1; }
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xs])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    /// Fused ffn gate+up+SwiGLU (opt-in GLM_FUSE_FFN=1, needs GEMVQ_Q4P with bits 0 and 2): one dispatch, silu(gate)*up written straight to y. Falls back to the two-step path unless both are Q4_K.
    fn ffn_fused(&self, cb: vk::CommandBuffer, gate: WRef, up: WRef, x: &Buffer, y: &Buffer) -> Result<bool> {
        if !self.fuse_ffn || !matches!(gate.ty, gguf::GgmlType::Q4K) || !matches!(up.ty, gguf::GgmlType::Q4K) || gate.k % 64 != 0 { return Ok(false) }
        if env("GLM_SKIP", 0) & 1 != 0 { return Ok(true) }
        let (mut g, arena) = GQ::segs(&[gate], Some(up), &[], self.cfg.eps)?;
        let ds = self.set(&self.xsk, &[x, &self.xs])?;
        self.ctx.dispatch(cb, &self.xsk, ds, &push_u32(&[g.k, g.x_off4, 1]), (g.k / 32).div_ceil(64));
        self.ctx.barrier(cb);
        let gk = self.g_ffn.get(&self.ctx, &g, 1)?;
        g.flags |= 64;
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xs])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(true)
    }
    /// mode 0: out = norm(inp)*w; mode 1: out += norm(inp)*w; mode 2: out = norm(inp)
    fn rms(&self, cb: vk::CommandBuffer, inp: &Buffer, out: &Buffer, k: u32, w: u32, mode: u32, eps: f32, key: u32) -> Result<()> {
        let ds = self.set(&self.norm, &[inp, out, &self.aux, &self.wt.f32b])?;
        self.ctx.dispatch(cb, &self.norm, ds, &push_u32(&[k, w, eps.to_bits(), mode, NONE, key, k, k]), 1);
        Ok(())
    }
    fn misc(&self, cb: vk::CommandBuffer, a: &Buffer, b: &Buffer, o: &Buffer, n: u32, mode: u32, b_off: u32, f: f32) -> Result<()> {
        let ds = self.set(&self.misc, &[a, b, o])?;
        self.ctx.dispatch(cb, &self.misc, ds, &push_u32(&[n, mode, b_off, f.to_bits(), 0, 0, 0]), n.div_ceil(256));
        Ok(())
    }
    pub fn record(&self, cb: vk::CommandBuffer) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let bar = || c.barrier(cb);
        let d = cfg.dim; let (qn, kn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
        bar();
        let te = wt.wref("token_embd.weight")?;
        let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &self.x, &self.aux, &self.toks, &wt.f32b])?;
        c.dispatch(cb, &self.embed, ds, &push_u32(&[te.off as u32, d, qtype(te.ty)?, 160, 0, 0]), 1);
        bar();
        self.rms(cb, &self.x, &self.x, d, 0, 2, cfg.eps, 0)?; bar();
        let nl_dbg = env("DBG_LAYERS", cfg.n_layer);
        for l in 0..nl_dbg {
            let n = |s: &str| format!("blk.{l}.{s}");
            let ly = cfg.layers[l as usize];
            self.rms(cb, &self.x, &self.xw, d, wt.f(&n("attn_norm.weight"))?, 0, cfg.eps, 0)?; bar();
            self.gq(cb, &self.g_proj, &[wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?, wt.wref(&n("attn_gate.weight"))?], &self.xw, &self.pr, false, 16)?; bar();
            if env("GLM_SKIP", 0) & 4 == 0 {
            let ds = self.set(&self.qkr, &[&self.pr, &self.kc, &self.vc, &self.aux, &wt.f32b])?;
            c.dispatch(cb, &self.qkr, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cfg.hd, wt.f(&n("attn_q_norm.weight"))?, wt.f(&n("attn_k_norm.weight"))?, ly.theta.to_bits(),
                if ly.swa { 0 } else { 2 }, ly.cache_off, ly.cap, qn, qn + kn, 160, cfg.eps.to_bits(), 0, 0, 0]), cfg.n_head + cfg.n_kv); bar();
            let ds = self.set(&self.attn, &[&self.pr, &self.kc, &self.vc, &self.part, &self.aux])?;
            c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, ly.cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0,
                if ly.swa { cfg.win } else { 0 }, ly.cap, 0]), cfg.n_head, self.nc, 1); bar();
            let ds = self.set(&self.comb, &[&self.part, &self.ao])?;
            c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, 0, 0, cfg.n_head]), cfg.n_head); bar();
            }
            self.misc(cb, &self.ao, &self.pr, &self.ao, qn, 4, qn + 2 * kn, 0.0)?; bar();
            self.gq(cb, &self.g_proj, &[wt.wref(&n("attn_output.weight"))?], &self.ao, &self.t, false, 32)?; bar();
            self.rms(cb, &self.t, &self.x, d, wt.f(&n("post_attention_norm.weight"))?, 1, POST_EPS, 0)?; bar();
            self.rms(cb, &self.x, &self.xw, d, wt.f(&n("ffn_norm.weight"))?, 0, cfg.eps, 0)?; bar();
            if self.ffn_fused(cb, wt.wref(&n("ffn_gate.weight"))?, wt.wref(&n("ffn_up.weight"))?, &self.xw, &self.hg)? { bar(); } else {
            self.gq(cb, &self.g_proj, &[wt.wref(&n("ffn_gate.weight"))?, wt.wref(&n("ffn_up.weight"))?], &self.xw, &self.hb, false, 1)?; bar();
            self.misc(cb, &self.hb, &self.hb, &self.hg, cfg.ff, 0, 0, 0.0)?; bar(); }
            self.gq(cb, &self.g_proj, &[wt.wref(&n("ffn_down.weight"))?], &self.hg, &self.t, false, 2)?; bar();
            self.rms(cb, &self.t, &self.x, d, wt.f(&n("post_ffw_norm.weight"))?, 1, POST_EPS, 0)?; bar();
        }
        self.rms(cb, &self.x, &self.xw, d, wt.f("output_norm.weight")?, 0, cfg.eps, 128)?; bar();
        let head = wt.wref("output.weight").or_else(|_| wt.wref("token_embd.weight"))?;
        self.gq(cb, &self.g_head, &[head], &self.xw, &self.logits, false, 8)?; bar();
        if cfg.logit_scale != 1.0 { self.misc(cb, &self.logits, &self.logits, &self.logits, cfg.vocab, 2, 0, cfg.logit_scale)?; bar(); }
        if cfg.softcap > 0.0 { self.misc(cb, &self.logits, &self.logits, &self.logits, cfg.vocab, 3, 0, cfg.softcap)?; bar(); }
        let ds = self.set(&self.argmax, &[&self.logits, &self.aux])?;
        c.dispatch(cb, &self.argmax, ds, &push_u32(&[cfg.vocab]), 64); bar();
        let ds = self.set(&self.sample, &[&self.aux, &self.toks])?;
        c.dispatch(cb, &self.sample, ds, &[], 1); bar();
        Ok(())
    }
    fn au(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.aux.ptr as *mut u32, 1024) } }
    fn tk(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.toks.ptr as *mut u32, self.toks.size as usize / 4) } }
    pub fn logits(&self) -> &[f32] { unsafe { std::slice::from_raw_parts(self.logits.ptr as *const f32, self.cfg.vocab as usize) } }
    /// Teacher-forced: feed prompt then cont one position per step; for each j, (argmax, top1-top2 margin, top1 - logit[cont[j]]).
    pub fn forced(&self, pf: Option<&Prefill>, prompt: &[u32], cont: &[u32]) -> Result<Vec<(u32, f32, f32)>> {
        let (np, nc) = (prompt.len(), cont.len());
        if np + nc + 2 > self.cfg.max_ctx as usize { bail!("context too long") }
        let (au, tk) = (self.au(), self.tk());
        au[160] = 0; au[161] = (np + nc) as u32 + 1; au[162] = 0;
        tk[..np].copy_from_slice(prompt); tk[np..np + nc].copy_from_slice(cont);
        let mut i = match pf { Some(p) => p.run(self, np)?, None => 0 };
        while i + 1 < np { let k = (np - 1 - i).min(32); self.ctx.submit_repeated(self.step, k)?; i += k; }
        let mut out = vec![];
        for j in 0..nc {
            self.ctx.submit_repeated(self.step, 1)?;
            let lg = self.logits();
            let (mut b, mut s2) = ((0usize, f32::MIN), f32::MIN);
            for (v, &x) in lg.iter().enumerate() { if x > b.1 { s2 = b.1; b = (v, x) } else if x > s2 { s2 = x } }
            out.push((b.0 as u32, b.1 - s2, b.1 - lg[cont[j] as usize]));
        }
        Ok(out)
    }
    /// Greedy: prompt one position per step, then n_new-1 decode steps. Returns (ids, prompt s, decode GPU s, decode wall s).
    pub fn generate(&self, pf: Option<&Prefill>, prompt: &[u32], n_new: usize) -> Result<(Vec<u32>, f64, f64, f64)> {
        let n = prompt.len();
        if n + n_new + 2 > self.cfg.max_ctx as usize { bail!("context too long") }
        let (au, tk) = (self.au(), self.tk());
        au[160] = 0; au[161] = n as u32; au[162] = 0;
        tk[..n].copy_from_slice(prompt);
        let t = std::time::Instant::now();
        let mut i = match pf { Some(p) => p.run(self, n)?, None => 0 };
        while i < n { let k = (n - i).min(32); self.ctx.submit_repeated(self.step, k)?; i += k; }
        let tp = t.elapsed().as_secs_f64();
        let tw = std::time::Instant::now(); let mut dg = 0.0; let mut done = 1;
        while done < n_new { let k = (n_new - done).min(64); dg += self.ctx.submit_repeated(self.step, k)?; done += k; }
        Ok((tk[n..n + n_new].to_vec(), tp, dg, tw.elapsed().as_secs_f64()))
    }
}
