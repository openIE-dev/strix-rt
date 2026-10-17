//! gpt-oss (GGUF arch "gpt-oss"): GPU-driven greedy decode. Q8_0 attention/embedding/head through gemvq, MXFP4 routed
//! experts (32 experts, top-4, softmax over the selected logits, biases, clamped SwiGLU alpha 1.702 / limit 7),
//! attention sinks (one logit per head in the softmax denominator), sliding window 128 on even layers, YaRN RoPE.
//! aux words: argmax key 128..129; pos 160 (KV index); prompt len 161; pred 162; expert ids 440..; weights 448...
use anyhow::{bail, Result};
use ash::vk;
use rtcore::{qtype, DsCache, GkQ, Weights, WRef, GQ};
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

pub struct Cfg { pub n_layer: u32, pub dim: u32, pub n_head: u32, pub n_kv: u32, pub hd: u32, pub vocab: u32, pub eps: f32, pub theta: f32, pub max_ctx: u32,
    pub n_exp: u32, pub top_k: u32, pub ff: u32, pub win: u32, pub yarn: (f32, f32, f32, f32) /* freq_scale, c0, c1, mscale */ }

pub struct Model {
    pub ctx: Ctx, pub wt: Weights, pub cfg: Cfg, ds: DsCache,
    g_proj: GkQ, g_out: GkQ, g_head: GkQ,
    embed: Kernel, norm: Kernel, rope: Kernel, attn: Kernel, comb: Kernel, router: Kernel, nrouter: Kernel, gu: Kernel, dn: Kernel, argmax: Kernel, sample: Kernel, nc: u32, lpr: u32,
    pub x: Buffer, xw: Buffer, pr: Buffer, ao: Buffer, hb: Buffer, pub logits: Buffer, kc: Buffer, vc: Buffer, part: Buffer, pub toks: Buffer, pub aux: Buffer, pub step: vk::CommandBuffer,
    // batched prefill: nb token columns per pass (GO_PF_NB, 1..=4); column buffers X/XW/PR/AO
    pub pnb: u32, xs: Buffer, xws: Buffer, prs: Buffer, aos: Buffer, pub pf: vk::CommandBuffer,
}

impl Model {
    pub fn load(path: &str, max_ctx: u32) -> Result<Self> {
        let ctx = Ctx::new()?;
        let wt = Weights::load(&ctx, path)?;
        let g = &wt.g; let a = "gpt-oss";
        let arch = g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default();
        if arch != a { bail!("expected gpt-oss, got {arch}") }
        let u = |k: &str| g.u64(&format!("{a}.{k}")).map(|v| v as u32);
        let fl = |k: &str| g.f32(&format!("{a}.{k}"));
        let theta = fl("rope.freq_base")?;
        let factor = fl("rope.scaling.factor").unwrap_or(1.0);
        let n_orig = u("rope.scaling.original_context_length").unwrap_or(4096) as f32;
        let (bf, bs) = (fl("rope.scaling.yarn_beta_fast").unwrap_or(32.0), fl("rope.scaling.yarn_beta_slow").unwrap_or(1.0));
        let hd = u("attention.key_length")?;
        // ggml_rope_yarn_corr_dims
        let cd = |nr: f32| hd as f32 * (n_orig / (nr * 2.0 * std::f32::consts::PI)).ln() / (2.0 * theta.ln());
        let (c0, c1) = (cd(bf).floor().max(0.0), cd(bs).ceil().min(hd as f32 - 1.0));
        let fs = 1.0 / factor;
        let mscale = if factor != 1.0 { 1.0 + 0.1 * (1.0 / fs).ln() } else { 1.0 };
        let cfg = Cfg { n_layer: u("block_count")?, dim: u("embedding_length")?, n_head: u("attention.head_count")?, n_kv: u("attention.head_count_kv")?, hd,
            vocab: wt.wref("output.weight")?.n, eps: fl("attention.layer_norm_rms_epsilon")?, theta, max_ctx: max_ctx.min(u("context_length").unwrap_or(4096)),
            n_exp: u("expert_count")?, top_k: u("expert_used_count")?, ff: u("expert_feed_forward_length")?, win: u("attention.sliding_window").unwrap_or(0),
            yarn: (fs, c0, c1, mscale) };
        if cfg.hd != 64 || cfg.top_k > 8 || cfg.n_exp > 64 || cfg.dim % 32 != 0 || cfg.ff % 32 != 0 { bail!("unsupported shape") }
        let sg = env("GEMV_SG", 32);
        let f = |n: u64| ctx.buffer(n.max(16) * 4);
        let nc = env("ATTN_NC", 4);
        let kvn = (cfg.n_layer * cfg.max_ctx * cfg.n_kv * cfg.hd) as u64;
        let qkv = cfg.n_head * cfg.hd + 2 * cfg.n_kv * cfg.hd;
        let lpr = match env("GO_LPR", 32) { 8 => 8u32, 32 => 32, _ => 16 };
        let (gu_spv, dn_spv): (&[u8], &[u8]) = match lpr { 8 => (spv!("go_moe_gu_l8"), spv!("go_moe_dn_l8")), 32 => (spv!("go_moe_gu_l32"), spv!("go_moe_dn_l32")), _ => (spv!("go_moe_gu"), spv!("go_moe_dn")) };
        let pnb = env("GO_PF_NB", 4).clamp(1, 4);
        let mut m = Model {
            g_proj: GkQ::new("GEMV_PROJ", (32, 128), false, sg), g_out: GkQ::new("GEMV_OUT", (32, 128), false, sg), g_head: GkQ::new("GEMV_HEAD", (8, 256), false, sg),
            embed: ctx.kernel(shaders::EMBED_ANY, 5, 24, &[], None)?, norm: ctx.kernel(spv!("go_norm"), 4, 16, &[], None)?,
            rope: ctx.kernel(spv!("go_rope"), 4, 44, &[], None)?, attn: ctx.kernel(spv!("go_attn"), 5, 40, &[cfg.hd], None)?, comb: ctx.kernel(spv!("go_comb"), 3, 8, &[], None)?,
            router: ctx.kernel(spv!("go_router"), 3, 20, &[], Some(32))?, nrouter: ctx.kernel(spv!("go_normrouter"), 4, 28, &[], Some(32))?,
            gu: ctx.kernel(gu_spv, 5, 40, &[], Some(sg))?, dn: ctx.kernel(dn_spv, 5, 24, &[], Some(sg))?,
            argmax: ctx.kernel(shaders::ARGMAX_W, 2, 4, &[], None)?, sample: ctx.kernel(shaders::SAMPLE_W, 2, 0, &[], None)?, nc, lpr,
            x: f(cfg.dim as u64)?, xw: f(cfg.dim as u64)?, pr: f(qkv as u64)?, ao: f((cfg.n_head * cfg.hd) as u64)?, hb: f((pnb * cfg.top_k * cfg.ff) as u64)?,
            logits: f(cfg.vocab as u64)?, kc: ctx.buffer((kvn * 2).max(256))?, vc: ctx.buffer((kvn * 2).max(256))?, part: f((cfg.n_head * nc * (2 + cfg.hd)) as u64)?,
            toks: f(2 * cfg.max_ctx as u64 + 2)?, aux: f(4096)?, step: vk::CommandBuffer::null(), pf: vk::CommandBuffer::null(), pnb,
            xs: f((pnb * cfg.dim) as u64)?, xws: f((pnb * cfg.dim) as u64)?, prs: f((pnb * qkv) as u64)?, aos: f((pnb * cfg.n_head * cfg.hd) as u64)?, ds: DsCache::default(), cfg, wt, ctx };
        let mut err = Ok(());
        let step = m.ctx.record_raw(|cb| { err = m.record(cb); })?; err?;
        m.step = step;
        let mut err = Ok(());
        if env("GO_PF", 1) == 1 { let pf = m.ctx.record_raw(|cb| { err = m.record_pf(cb); })?; err?;
        m.pf = pf; }
        Ok(m)
    }
    fn set(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&self.ctx, k, bufs) }
    /// gemvq over Q8_0 segments with optional biases; input xw (already normalized), optional accumulate into y
    fn gq(&self, cb: vk::CommandBuffer, gq: &GkQ, segs: &[WRef], bias: &[Option<u32>], x: &Buffer, y: &Buffer, acc: bool) -> Result<()> {
        let (mut g, arena) = GQ::segs(segs, None, bias, self.cfg.eps)?;
        let gk = gq.get(&self.ctx, &g, segs.len())?;
        g.flags |= 64; if acc { g.flags |= 1; }
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xw])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    fn rms(&self, cb: vk::CommandBuffer, w: u32, key: bool) -> Result<()> {
        let ds = self.set(&self.norm, &[&self.x, &self.xw, &self.aux, &self.wt.f32b])?;
        self.ctx.dispatch(cb, &self.norm, ds, &push_u32(&[self.cfg.dim, w, self.cfg.eps.to_bits(), if key { 128 } else { 0 }]), 1);
        Ok(())
    }
    pub fn record(&self, cb: vk::CommandBuffer) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let bar = || c.barrier(cb);
        bar();
        let te = wt.wref("token_embd.weight")?;
        if te.off > u32::MAX as u64 { bail!("token_embd offset beyond 4 GiB") }
        let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &self.x, &self.aux, &self.toks, &wt.f32b])?;
        c.dispatch(cb, &self.embed, ds, &push_u32(&[te.off as u32, cfg.dim, qtype(te.ty)?, 160, 0, 0]), 1); bar();
        let (qn, kn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
        for l in 0..env("DBG_LAYERS", cfg.n_layer) {
            let n = |s: &str| format!("blk.{l}.{s}");
            self.rms(cb, wt.f(&n("attn_norm.weight"))?, false)?; bar();
            let segs = [wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?];
            let bias = [Some(wt.f(&n("attn_q.bias"))?), Some(wt.f(&n("attn_k.bias"))?), Some(wt.f(&n("attn_v.bias"))?)];
            self.gq(cb, &self.g_proj, &segs, &bias, &self.xw, &self.pr, false)?; bar();
            let cache_off = l * cfg.max_ctx * cfg.n_kv * cfg.hd;
            let (fs, c0, c1, ms) = cfg.yarn;
            let ds = self.set(&self.rope, &[&self.pr, &self.kc, &self.vc, &self.aux])?;
            c.dispatch(cb, &self.rope, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, cfg.theta.to_bits(), fs.to_bits(), c0.to_bits(), c1.to_bits(), ms.to_bits(), qn, qn + kn, 160]), cfg.n_head + cfg.n_kv); bar();
            let win = if l % 2 == 0 { cfg.win } else { 0 };
            let ds = self.set(&self.attn, &[&self.pr, &self.kc, &self.vc, &self.part, &self.aux])?;
            c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0, win]), cfg.n_head, self.nc, 1); bar();
            let ds = self.set(&self.comb, &[&self.part, &self.ao, &wt.f32b])?;
            c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, wt.f(&n("attn_sinks.weight"))?]), cfg.n_head); bar();
            self.gq(cb, &self.g_out, &[wt.wref(&n("attn_output.weight"))?], &[Some(wt.f(&n("attn_output.bias"))?)], &self.ao, &self.x, true)?; bar();
            if env("GO_NR", 1) == 1 {
                let ds = self.set(&self.nrouter, &[&wt.f32b, &self.x, &self.aux, &self.xw])?;
                c.dispatch(cb, &self.nrouter, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, wt.f(&n("ffn_gate_inp.bias"))?, cfg.dim, cfg.n_exp, cfg.top_k,
                    wt.f(&n("post_attention_norm.weight"))?, cfg.eps.to_bits()]), 1); bar();
            } else {
            self.rms(cb, wt.f(&n("post_attention_norm.weight"))?, false)?; bar();
            let ds = self.set(&self.router, &[&wt.f32b, &self.xw, &self.aux])?;
            c.dispatch(cb, &self.router, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, wt.f(&n("ffn_gate_inp.bias"))?, cfg.dim, cfg.n_exp, cfg.top_k]), 1); bar();
            }
            let (gw, uw, dw) = (wt.wref(&n("ffn_gate_exps.weight"))?, wt.wref(&n("ffn_up_exps.weight"))?, wt.wref(&n("ffn_down_exps.weight"))?);
            if gw.arena != uw.arena || uw.off + uw.bytes > u32::MAX as u64 || dw.off + dw.bytes > u32::MAX as u64 { bail!("expert layout") }
            let es = (gw.bytes / gw.e as u64) as u32;
            let rows_wg = 256 / self.lpr;
            let ds = self.set(&self.gu, &[&wt.arenas[gw.arena], &self.xw, &self.hb, &wt.f32b, &self.aux])?;
            c.dispatch(cb, &self.gu, ds, &push_u32(&[gw.off as u32, uw.off as u32, es, cfg.ff, cfg.dim, wt.f(&n("ffn_gate_exps.bias"))?, wt.f(&n("ffn_up_exps.bias"))?, cfg.top_k,
                1.702f32.to_bits(), 7.0f32.to_bits()]), (cfg.top_k * cfg.ff).div_ceil(rows_wg)); bar();
            let ds = self.set(&self.dn, &[&wt.arenas[dw.arena], &self.hb, &self.x, &wt.f32b, &self.aux])?;
            c.dispatch(cb, &self.dn, ds, &push_u32(&[dw.off as u32, (dw.bytes / dw.e as u64) as u32, cfg.dim, cfg.ff, wt.f(&n("ffn_down_exps.bias"))?, cfg.top_k]), cfg.dim.div_ceil(rows_wg)); bar();
        }
        self.rms(cb, wt.f("output_norm.weight")?, true)?; bar();
        self.gq(cb, &self.g_head, &[wt.wref("output.weight")?], &[], &self.xw, &self.logits, false)?; bar();
        let ds = self.set(&self.argmax, &[&self.logits, &self.aux])?;
        c.dispatch(cb, &self.argmax, ds, &push_u32(&[cfg.vocab]), 64); bar();
        let ds = self.set(&self.sample, &[&self.aux, &self.toks])?;
        c.dispatch(cb, &self.sample, ds, &[], 1); bar();
        Ok(())
    }
    fn copy(&self, cb: vk::CommandBuffer, a: &Buffer, ao: u32, b: &Buffer, bo: u32, n: u32) {
        let d = &self.ctx.device;
        let fb = |cb| { let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
            unsafe { d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &mb, &[], &[]) } };
        fb(cb);
        unsafe { d.cmd_copy_buffer(cb, a.buf, b.buf, &[vk::BufferCopy { src_offset: ao as u64 * 4, dst_offset: bo as u64 * 4, size: n as u64 * 4 }]) };
        fb(cb);
    }
    fn gqn(&self, cb: vk::CommandBuffer, gq: &GkQ, segs: &[WRef], bias: &[Option<u32>], x: &Buffer, y: &Buffer, acc: bool, xcs: u32, ycs: u32) -> Result<()> {
        let (mut g, arena) = GQ::segs(segs, None, bias, self.cfg.eps)?;
        let gk = gq.get_nb(&self.ctx, &g, segs.len(), self.pnb)?;
        g.flags |= 64; if acc { g.flags |= 1; } g.x_cs = xcs; g.y_cs = ycs;
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xw])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    /// Batched prefill pass over pnb positions au[160]..+pnb (au[170+j] = au[160]+j, set by the CPU).
    /// QKV and output projections read each weight once for all columns (gemvq nb); rope/attention/MoE run per column
    /// in order, so column j attends to the K/V written by columns < j. No LM head: state only.
    fn rmsn(&self, cb: vk::CommandBuffer, w: u32) -> Result<()> {
        let ds = self.set(&self.norm, &[&self.xs, &self.xws, &self.aux, &self.wt.f32b])?;
        self.ctx.dispatch3(cb, &self.norm, ds, &push_u32(&[self.cfg.dim, w, self.cfg.eps.to_bits(), 0]), 1, self.pnb, 1);
        Ok(())
    }
    pub fn record_pf(&self, cb: vk::CommandBuffer) -> Result<()> {
        let (c, cfg, wt, nb) = (&self.ctx, &self.cfg, &self.wt, self.pnb);
        let bar = || c.barrier(cb);
        let (qn, kn, d) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd, cfg.dim);
        let qkv = qn + 2 * kn;
        bar();
        let te = wt.wref("token_embd.weight")?;
        for j in 0..nb {
            let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &self.xs, &self.aux, &self.toks, &wt.f32b])?;
            c.dispatch(cb, &self.embed, ds, &push_u32(&[te.off as u32, d, qtype(te.ty)?, 160, j, j * d]), 1);
        }
        bar();
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            self.rmsn(cb, wt.f(&n("attn_norm.weight"))?)?; bar();
            let segs = [wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?];
            let bias = [Some(wt.f(&n("attn_q.bias"))?), Some(wt.f(&n("attn_k.bias"))?), Some(wt.f(&n("attn_v.bias"))?)];
            self.gqn(cb, &self.g_proj, &segs, &bias, &self.xws, &self.prs, false, d, qkv)?;
            let cache_off = l * cfg.max_ctx * cfg.n_kv * cfg.hd;
            let (fs, c0, c1, ms) = cfg.yarn;
            let win = if l % 2 == 0 { cfg.win } else { 0 };
            for j in 0..nb {
                self.copy(cb, &self.prs, j * qkv, &self.pr, 0, qkv);
                let ds = self.set(&self.rope, &[&self.pr, &self.kc, &self.vc, &self.aux])?;
                c.dispatch(cb, &self.rope, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, cfg.theta.to_bits(), fs.to_bits(), c0.to_bits(), c1.to_bits(), ms.to_bits(), qn, qn + kn, 170 + j]), cfg.n_head + cfg.n_kv); bar();
                let ds = self.set(&self.attn, &[&self.pr, &self.kc, &self.vc, &self.part, &self.aux])?;
                c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, self.nc, cfg.hd, 170 + j, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0, win]), cfg.n_head, self.nc, 1); bar();
                let ds = self.set(&self.comb, &[&self.part, &self.ao, &wt.f32b])?;
                c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, wt.f(&n("attn_sinks.weight"))?]), cfg.n_head);
                self.copy(cb, &self.ao, 0, &self.aos, j * qn, qn);
            }
            self.gqn(cb, &self.g_out, &[wt.wref(&n("attn_output.weight"))?], &[Some(wt.f(&n("attn_output.bias"))?)], &self.aos, &self.xs, true, qn, d)?;
            let (gw, uw, dw) = (wt.wref(&n("ffn_gate_exps.weight"))?, wt.wref(&n("ffn_up_exps.weight"))?, wt.wref(&n("ffn_down_exps.weight"))?);
            let es = (gw.bytes / gw.e as u64) as u32;
            let rows_wg = 256 / self.lpr;
            // MoE batched over columns (workgroup y = column): per-column router slots au[440+16j], hb column j
            bar();
            if env("GO_NR", 1) == 1 {
                // same fused norm+router kernel as decode, so batched prefill stays bitwise identical to it
                let ds = self.set(&self.nrouter, &[&wt.f32b, &self.xs, &self.aux, &self.xws])?;
                c.dispatch3(cb, &self.nrouter, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, wt.f(&n("ffn_gate_inp.bias"))?, d, cfg.n_exp, cfg.top_k,
                    wt.f(&n("post_attention_norm.weight"))?, cfg.eps.to_bits()]), 1, nb, 1); bar();
            } else {
            self.rmsn(cb, wt.f(&n("post_attention_norm.weight"))?)?; bar();
            let ds = self.set(&self.router, &[&wt.f32b, &self.xws, &self.aux])?;
            c.dispatch3(cb, &self.router, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, wt.f(&n("ffn_gate_inp.bias"))?, d, cfg.n_exp, cfg.top_k]), 1, nb, 1); bar();
            }
            let ds = self.set(&self.gu, &[&wt.arenas[gw.arena], &self.xws, &self.hb, &wt.f32b, &self.aux])?;
            c.dispatch3(cb, &self.gu, ds, &push_u32(&[gw.off as u32, uw.off as u32, es, cfg.ff, d, wt.f(&n("ffn_gate_exps.bias"))?, wt.f(&n("ffn_up_exps.bias"))?, cfg.top_k,
                1.702f32.to_bits(), 7.0f32.to_bits()]), (cfg.top_k * cfg.ff).div_ceil(rows_wg), nb, 1); bar();
            let ds = self.set(&self.dn, &[&wt.arenas[dw.arena], &self.hb, &self.xs, &wt.f32b, &self.aux])?;
            c.dispatch3(cb, &self.dn, ds, &push_u32(&[dw.off as u32, (dw.bytes / dw.e as u64) as u32, d, cfg.ff, wt.f(&n("ffn_down_exps.bias"))?, cfg.top_k]), d.div_ceil(rows_wg), nb, 1); bar();
        }
        bar();
        Ok(())
    }
    /// Batched prefill of prompt[..n-1] (pnb positions per pass, GO_PF=1); leaves au[160] = n-1 so the next
    /// single-token step processes the last prompt position and samples the first new token.
    fn prefill(&self, n: usize) -> Result<usize> {
        if env("GO_PF", 1) == 0 || n < 2 { return Ok(0) }
        let (au, nb) = (self.au(), self.pnb as usize);
        let full = (n - 1) / nb * nb;
        let mut p = 0;
        while p < full {
            let k = ((full - p) / nb).min(8);
            for _ in 0..k { au[160] = p as u32; for j in 0..nb { au[170 + j] = (p + j) as u32; } self.ctx.submit_repeated(self.pf, 1)?; p += nb; }
        }
        au[160] = p as u32;
        Ok(p)
    }
    fn au(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.aux.ptr as *mut u32, 1024) } }
    fn tk(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.toks.ptr as *mut u32, self.toks.size as usize / 4) } }
    /// Teacher-forced after the (GO_PF) prompt prefill: for each j, the prediction for cont[j] given prompt + cont[..j],
    /// as (argmax, top1 - top2 margin, top1 - logit[cont[j]]).
    pub fn pf_forced(&self, prompt: &[u32], cont: &[u32]) -> Result<Vec<(u32, f32, f32)>> {
        let (np, nc) = (prompt.len(), cont.len());
        if np + nc + 2 > self.cfg.max_ctx as usize { bail!("context too long") }
        let (au, tk) = (self.au(), self.tk());
        au[160] = 0; au[161] = (np + nc) as u32 + 1; au[162] = 0;
        tk[..np].copy_from_slice(prompt); tk[np..np + nc].copy_from_slice(cont);
        let mut i = self.prefill(np)?;
        while i + 1 < np { let k = (np - 1 - i).min(32); self.ctx.submit_repeated(self.step, k)?; i += k; }
        let lg = unsafe { std::slice::from_raw_parts(self.logits.ptr as *const f32, self.cfg.vocab as usize) };
        let mut out = vec![];
        for j in 0..nc {
            self.ctx.submit_repeated(self.step, 1)?;
            let (mut b, mut s2) = ((0usize, f32::MIN), f32::MIN);
            for (v, &x) in lg.iter().enumerate() { if x > b.1 { s2 = b.1; b = (v, x) } else if x > s2 { s2 = x } }
            out.push((b.0 as u32, b.1 - s2, b.1 - lg[cont[j] as usize]));
        }
        Ok(out)
    }
    /// Teacher-forced: feed seq one position per step; returns the greedy prediction after each position.
    pub fn forced(&self, seq: &[u32]) -> Result<Vec<u32>> {
        let n = seq.len(); let mc = self.cfg.max_ctx as usize;
        if n + 2 > mc { bail!("context too long") }
        let (au, tk) = (self.au(), self.tk());
        au[160] = 0; au[161] = n as u32 + 1; au[162] = mc as u32;
        tk[..n].copy_from_slice(seq);
        let mut i = 0;
        while i < n { let k = (n - i).min(32); self.ctx.submit_repeated(self.step, k)?; i += k; }
        au[162] = 0;
        Ok(tk[mc + 1..mc + 1 + n].to_vec())
    }
    /// Greedy: prompt fed one position per step (one submission per position), then n_new-1 decode steps in one batch.
    /// Returns (generated ids, prompt seconds, decode GPU seconds).
    pub fn generate(&self, prompt: &[u32], n_new: usize) -> Result<(Vec<u32>, f64, f64, f64)> {
        let n = prompt.len();
        if n + n_new + 2 > self.cfg.max_ctx as usize { bail!("context too long") }
        let (au, tk) = (self.au(), self.tk());
        au[160] = 0; au[161] = n as u32; au[162] = 0;
        tk[..n].copy_from_slice(prompt);
        let t = std::time::Instant::now();
        let mut i = self.prefill(n)?;
        while i < n { let k = (n - i).min(32); self.ctx.submit_repeated(self.step, k)?; i += k; }
        let tp = t.elapsed().as_secs_f64();
        let tw = std::time::Instant::now(); let mut dg = 0.0; let mut done = 1;
        while done < n_new { let k = (n_new - done).min(64); dg += self.ctx.submit_repeated(self.step, k)?; done += k; }
        Ok((tk[n..n + n_new].to_vec(), tp, dg, tw.elapsed().as_secs_f64()))
    }
}
