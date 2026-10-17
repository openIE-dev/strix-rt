//! Gemma 4 (GGUF arch "gemma4"): GPU-driven greedy decode for E2B/E4B (dense + per-layer embeddings) and 26B-A4B (MoE).
//! Per layer: sandwich RMSNorms (attn_norm / post_attention_norm, ffn_norm / post_ffw_norm), q/k RMSNorm, v RMSNorm
//! without weight, NeoX RoPE with per-layer base (swa vs global, proportional rope_freqs on global layers), attention
//! scale 1.0, sliding window on SWA layers over a ring-buffer KV cache (slot = pos % window), late layers sharing the KV
//! of the last earlier layer of the same kind, GeGLU (gelu_tanh(gate) * up), layer_output_scale, final logit softcap.
//! aux words (as gptoss): argmax key 128..129; pos 160; prompt len 161; pred 162; column positions 170...
use anyhow::{bail, Result};
use ash::vk;
use gguf::Value;
use rtcore::{qtype, DsCache, GkQ, Weights, WRef, GQ};
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel};
pub mod pf;
pub mod vision;
pub mod audio;
pub mod gmoe;
pub use pf::Prefill;

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
pub(crate) const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Layer { pub swa: bool, pub hd: u32, pub n_kv: u32, pub has_kv: bool, pub kv_src: u32, pub cache_off: u32, pub cap: u32, pub theta: f32, pub ff_off: u32, pub moe: bool }

pub struct Cfg { pub n_layer: u32, pub dim: u32, pub n_head: u32, pub vocab: u32, pub eps: f32, pub max_ctx: u32, pub win: u32, pub ff: u32,
    pub ple: u32 /* per-layer embedding width, 0 = none */, pub softcap: f32, pub n_exp: u32, pub top_k: u32, pub ff_exp: u32, pub layers: Vec<Layer> }

fn arr(g: &gguf::Gguf, k: &str, n: usize) -> Option<Vec<u64>> {
    match g.get(k).ok()? {
        Value::Arr(a) => Some(a.iter().map(|v| match v { Value::Bool(b) => *b as u64, v => v.as_u64().unwrap_or(0) }).collect()),
        v => v.as_u64().map(|x| vec![x; n]),
    }
}

impl Cfg {
    pub fn from(wt: &Weights, max_ctx: u32) -> Result<Self> {
        let g = &wt.g; let a = "gemma4";
        let arch = g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default();
        if arch != a { bail!("expected gemma4, got {arch}") }
        let u = |k: &str| g.u64(&format!("{a}.{k}")).map(|v| v as u32);
        let fl = |k: &str| g.f32(&format!("{a}.{k}"));
        let nl = u("block_count")?;
        let swa = arr(g, "gemma4.attention.sliding_window_pattern", nl as usize).ok_or_else(|| anyhow::anyhow!("no swa pattern"))?;
        let nkv = arr(g, "gemma4.attention.head_count_kv", nl as usize).ok_or_else(|| anyhow::anyhow!("no head_count_kv"))?;
        let shared = u("attention.shared_kv_layers").unwrap_or(0);
        let kv_from = nl - shared;
        let (hd_g, hd_s) = (u("attention.key_length")?, u("attention.key_length_swa")?);
        let (th_g, th_s) = (fl("rope.freq_base")?, fl("rope.freq_base_swa")?);
        let win = u("attention.sliding_window")?;
        let max_ctx = max_ctx.min(u("context_length").unwrap_or(8192));
        let ff_off = wt.f("rope_freqs.weight").unwrap_or(NONE);
        let mut layers = vec![]; let mut off = 0u32;
        for l in 0..nl {
            let s = swa[l as usize] != 0;
            let (hd, cap) = if s { (hd_s, (win + env("G4_PF_CHUNK", 512)).min(max_ctx)) } else { (hd_g, max_ctx) };
            let has_kv = l < kv_from;
            // llama.cpp: shared layers reuse kv_from-2 (swa) / kv_from-1 (global)
            let kv_src = if has_kv { l } else { kv_from - if s { 2 } else { 1 } };
            let n_kv = nkv[l as usize] as u32;
            let mut ly = Layer { swa: s, hd, n_kv, has_kv, kv_src, cache_off: 0, cap, theta: if s { th_s } else { th_g }, ff_off: if s { NONE } else { ff_off },
                moe: wt.w.contains_key(&format!("blk.{l}.ffn_gate_inp.weight")) || wt.fo.contains_key(&format!("blk.{l}.ffn_gate_inp.weight")) };
            if has_kv { ly.cache_off = off; off += cap * n_kv * hd; } else {
                let src: Layer = layers[kv_src as usize];
                if src.swa != s || src.hd != hd || src.n_kv != n_kv { bail!("shared kv layer {l} shape mismatch") }
                ly.cache_off = src.cache_off; ly.cap = src.cap;
            }
            layers.push(ly);
        }
        let cfg = Cfg { n_layer: nl, dim: u("embedding_length")?, n_head: u("attention.head_count")?, vocab: wt.wref("token_embd.weight")?.n, eps: fl("attention.layer_norm_rms_epsilon")?,
            max_ctx, win, ff: u("feed_forward_length")?, ple: u("embedding_length_per_layer_input").unwrap_or(0), softcap: fl("final_logit_softcapping").unwrap_or(0.0),
            n_exp: u("expert_count").unwrap_or(0), top_k: u("expert_used_count").unwrap_or(0), ff_exp: u("expert_feed_forward_length").unwrap_or(0), layers };
        Ok(cfg)
    }
    pub fn kv_elems(&self) -> u64 { self.layers.iter().filter(|l| l.has_kv).map(|l| (l.cap * l.n_kv * l.hd) as u64).sum() }
}

/// rtcore::Weights::load (copy mode) that also accepts Q5_1 (ggml type 7, gemma4 MoE down experts; decoded by
/// g4_moe_dn only). Used when the file has a type the shared loader rejects.
pub fn load_weights(ctx: &Ctx, path: &str) -> Result<Weights> {
    use gguf::GgmlType;
    let g = gguf::Gguf::open(path)?;
    let ok = |t: GgmlType| qtype(t).is_ok() || t == GgmlType::Other(7);
    if !g.tensors.iter().any(|t| t.ty == GgmlType::Other(7)) { return Weights::load(ctx, path) }
    let t0 = std::time::Instant::now();
    let mut groups: Vec<(String, Vec<usize>)> = vec![]; let mut f32_total = 0usize;
    for (i, t) in g.tensors.iter().enumerate() {
        if t.ty == GgmlType::F32 { f32_total += t.n_elems() as usize; continue }
        if !ok(t.ty) { bail!("unsupported tensor type {:?} for {}", t.ty, t.name) }
        let key = if t.name.starts_with("blk.") { t.name.split('.').take(2).collect::<Vec<_>>().join(".") } else { t.name.clone() };
        match groups.iter_mut().find(|x| x.0 == key) { Some(x) => x.1.push(i), None => groups.push((key, vec![i])) }
    }
    const ARENA: u64 = 2 << 30; const AL: u64 = 256;
    let mut sizes = vec![0u64]; let mut w = std::collections::HashMap::new();
    for (_, idx) in &groups {
        let gsz: u64 = idx.iter().map(|&i| (g.tensors[i].n_bytes() as u64).div_ceil(AL) * AL).sum();
        if *sizes.last().unwrap() + gsz > ARENA { sizes.push(0); }
        let ai = sizes.len() - 1;
        for &i in idx { let t = &g.tensors[i];
            w.insert(t.name.clone(), WRef { arena: ai, off: sizes[ai], ty: t.ty, k: t.dims[0] as u32, n: t.dims.get(1).copied().unwrap_or(1) as u32,
                e: t.dims.get(2).copied().unwrap_or(1) as u32, bytes: t.n_bytes() as u64 });
            sizes[ai] += (t.n_bytes() as u64).div_ceil(AL) * AL; }
    }
    let arenas: Vec<Buffer> = sizes.iter().map(|&s| ctx.buffer(s.max(256))).collect::<Result<_>>()?;
    for (name, r) in &w { let t = g.tensor(name).unwrap(); arenas[r.arena].bytes_mut()[r.off as usize..(r.off + r.bytes) as usize].copy_from_slice(g.data(t)); }
    let f32b = ctx.buffer(((f32_total + rtcore::TBL_FLOATS).max(64) * 4) as u64)?;
    { let f = f32b.as_f32_mut(); for (i, &v) in gguf::iq3s_grid::IQ3S_GRID.iter().enumerate() { f[i] = f32::from_bits(v); }
      for (i, &v) in gguf::dequant::KV_IQ4NL.iter().enumerate() { f[512 + i] = v as f32; } }
    let mut fo = std::collections::HashMap::new(); let mut o = rtcore::TBL_FLOATS;
    for t in g.tensors.iter().filter(|t| t.ty == GgmlType::F32) {
        let n = t.n_elems() as usize; gguf::dequant::dequant_row(GgmlType::F32, g.data(t), &mut f32b.as_f32_mut()[o..o + n]); fo.insert(t.name.clone(), o as u32); o += n; }
    eprintln!("weights (gemma4 loader, Q5_1 ok): copied in {:.3}s, {} arenas ({} MiB), {} quant tensors, {} f32 floats", t0.elapsed().as_secs_f64(), arenas.len(), sizes.iter().sum::<u64>() >> 20, w.len(), f32_total);
    Ok(Weights { g, arenas, sizes, w, f32b, fo, mode: String::new() })
}

/// Columns per workgroup of the prefill router-logits kernel (G4_RLOG_TB, default 1 = old single-column grid; 2-5 tested, no gain). Shared memory: TB * dim floats.
fn rlog_tb() -> u32 { { let t = env("G4_RLOG_TB", 1) as u32; if (1..=8).contains(&t) { t } else { 1 } } }

pub struct Model {
    pub ctx: Ctx, pub wt: Weights, pub cfg: Cfg, ds: DsCache,
    g_proj: GkQ, g_head: GkQ,
    embed: Kernel, norm: Kernel, misc: Kernel, qkr: Kernel, attn: [Kernel; 2], comb: [Kernel; 2], ple: Kernel, router: Kernel, pub(crate) rlog: Kernel, pub(crate) rlog4: Kernel, fnorm: Kernel, t2: Buffer, xw2: Buffer, pub(crate) rl: Buffer, mgu: Kernel, mdn: Kernel, argmax: Kernel, sample: Kernel, nc: u32,
    pub x: Buffer, xw: Buffer, t: Buffer, pr: Buffer, ao: Buffer, hb: Buffer, hg: Buffer, pr_raw: Buffer, pr_proj: Buffer, plei: Buffer, pg: Buffer, pub m1: Buffer, hm: Buffer,
    pub logits: Buffer, kc: Buffer, vc: Buffer, part: Buffer, pub toks: Buffer, pub aux: Buffer, pub step: vk::CommandBuffer,
}

impl Model {
    pub fn load(path: &str, max_ctx: u32) -> Result<Self> {
        let ctx = Ctx::new()?;
        let wt = load_weights(&ctx, path)?;
        let cfg = Cfg::from(&wt, max_ctx)?;
        let sg = env("GEMV_SG", 32);
        let f = |n: u64| ctx.buffer(n.max(16) * 4);
        let nc = env("ATTN_NC", 4);
        let kvn = cfg.kv_elems();
        let hmax = cfg.layers.iter().map(|l| l.hd).max().unwrap();
        let kvmax = cfg.layers.iter().map(|l| l.hd * l.n_kv).max().unwrap();
        let qkv = cfg.n_head * hmax + 2 * kvmax;
        let nl = cfg.n_layer;
        let pw = (cfg.ple * nl) as u64;
        // Same expert format in every layer -> specialize the MoE GEMVs on it (G4_SPEC=0 disables)
        let uty = |n: &str| -> u32 {
            let t: Vec<_> = (0..cfg.n_layer).filter_map(|l| wt.wref(&format!("blk.{l}.{n}.weight")).ok().map(|w| w.ty)).collect();
            if env("G4_SPEC", 1) == 0 || t.is_empty() || t.iter().any(|x| *x != t[0]) { 255 } else { qtype(t[0]).unwrap_or(255) }
        };
        let (gty, dty) = (uty("ffn_gate_up_exps"), uty("ffn_down_exps"));
        let m = Model {
            g_proj: GkQ::new("GEMV_PROJ", (32, 128), false, sg), g_head: GkQ::new("GEMV_HEAD", (8, 256), false, sg),
            embed: ctx.kernel(shaders::EMBED_ANY, 5, 24, &[], None)?, norm: ctx.kernel(spv!("g4_norm"), 4, 32, &[env("G4_NORM_WG", 1024)], None)?, misc: ctx.kernel(spv!("g4_misc"), 3, 28, &[], None)?,
            qkr: ctx.kernel(spv!("g4_qkr"), 5, 68, &[], None)?,
            attn: [ctx.kernel(spv!("g4_attn"), 5, 48, &[256], None)?, ctx.kernel(spv!("g4_attn"), 5, 48, &[512], None)?],
            comb: [ctx.kernel(spv!("g4_comb"), 2, 16, &[256], None)?, ctx.kernel(spv!("g4_comb"), 2, 16, &[512], None)?],
            ple: ctx.kernel(spv!("g4_ple"), 4, 28, &[], None)?, router: ctx.kernel(spv!("g4_router"), 4, 32, &[], Some(32))?, rlog: ctx.kernel(spv!("g4_rlog"), 3, 28, &[1, 4096], Some(32))?, rlog4: ctx.kernel(spv!("g4_rlog"), 3, 28, &[rlog_tb(), cfg.dim.next_multiple_of(64)], Some(32))?, fnorm: ctx.kernel(spv!("g4_fnorm"), 7, 36, &[], None)?, t2: f(cfg.dim as u64)?, xw2: f(cfg.dim as u64)?, rl: ctx.buffer(512 * 128 * 4)?,
            mgu: ctx.kernel(spv!("g4_moe_gu"), 5, 24, &[gty, (gty == 0 && cfg.dim <= 8192 && env("G4_IDOT", 1) != 0) as u32], Some(sg))?, mdn: ctx.kernel(if env("G4_DQ2", 1) != 0 { spv!("g4_moe_dn_v2") } else { spv!("g4_moe_dn") }, 5, 28, &[dty], Some(sg))?,
            argmax: ctx.kernel(shaders::ARGMAX_W, 2, 4, &[], None)?, sample: ctx.kernel(shaders::SAMPLE_W, 2, 0, &[], None)?, nc,
            x: f(cfg.dim as u64)?, xw: f(cfg.dim as u64)?, t: f(cfg.dim as u64)?, pr: f(qkv as u64)?, ao: f((cfg.n_head * hmax) as u64)?,
            hb: f(2 * cfg.ff.max(cfg.top_k * cfg.ff_exp) as u64)?, hg: f(cfg.ff.max(cfg.top_k * cfg.ff_exp) as u64)?,
            pr_raw: f(pw)?, pr_proj: f(pw)?, plei: f(pw)?, pg: f(cfg.ple as u64)?, m1: f(cfg.dim as u64)?, hm: f((cfg.top_k * cfg.ff_exp) as u64)?,
            logits: f(cfg.vocab as u64)?, kc: ctx.buffer((kvn * 2).max(256))?, vc: ctx.buffer((kvn * 2).max(256))?, part: f((cfg.n_head * nc * (2 + hmax)) as u64)?,
            toks: f(2 * cfg.max_ctx as u64 + 2)?, aux: f(4096)?, step: vk::CommandBuffer::null(), ds: DsCache::default(), cfg, wt, ctx };
        if m.cfg.top_k > 8 || m.cfg.n_exp > 128 || m.cfg.dim > 4096 { bail!("MoE shape") }
        let mut m = m;
        let mut err = Ok(());
        let step = m.ctx.record_raw(|cb| { err = m.record(cb); })?; err?;
        m.step = step;
        Ok(m)
    }
    fn set(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&self.ctx, k, bufs) }
    fn gq(&self, cb: vk::CommandBuffer, gq: &GkQ, segs: &[WRef], x: &Buffer, y: &Buffer, acc: bool) -> Result<()> {
        let (mut g, arena) = GQ::segs(segs, None, &[], self.cfg.eps)?;
        let gk = gq.get(&self.ctx, &g, segs.len())?;
        g.flags |= 64; if acc { g.flags |= 1; }
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xw])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    /// mode 0: out = norm(inp)*w; mode 1: out = (out + norm(inp)*w) * s
    fn rms(&self, cb: vk::CommandBuffer, inp: &Buffer, out: &Buffer, k: u32, w: u32, mode: u32, s: u32, key: u32) -> Result<()> {
        let ds = self.set(&self.norm, &[inp, out, &self.aux, &self.wt.f32b])?;
        if key != 0 || env("G4_SKIP", 0) & (if mode == 1 { 64 } else { 1024 }) == 0 { self.ctx.dispatch(cb, &self.norm, ds, &push_u32(&[k, w, self.cfg.eps.to_bits(), mode, s, key, k, k]), 1); }
        Ok(())
    }
    fn misc(&self, cb: vk::CommandBuffer, a: &Buffer, b: &Buffer, o: &Buffer, n: u32, mode: u32, b_off: u32, f: f32) -> Result<()> {
        let ds = self.set(&self.misc, &[a, b, o])?;
        if env("G4_SKIP", 0) & 512 == 0 { self.ctx.dispatch(cb, &self.misc, ds, &push_u32(&[n, mode, b_off, f.to_bits(), 0, 0, 0]), n.div_ceil(256)); }
        Ok(())
    }
    pub fn record(&self, cb: vk::CommandBuffer) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let bar = || c.barrier(cb);
        let d = cfg.dim;
        bar();
        let te = wt.wref("token_embd.weight")?;
        let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &self.x, &self.aux, &self.toks, &wt.f32b])?;
        c.dispatch(cb, &self.embed, ds, &push_u32(&[te.off as u32, d, qtype(te.ty)?, 160, 0, 0]), 1);
        if cfg.ple > 0 {
            let pe = wt.wref("per_layer_token_embd.weight")?;
            if pe.off + pe.bytes > u32::MAX as u64 { bail!("per_layer_token_embd beyond 4 GiB") }
            let ds = self.set(&self.embed, &[&wt.arenas[pe.arena], &self.pr_raw, &self.aux, &self.toks, &wt.f32b])?;
            c.dispatch(cb, &self.embed, ds, &push_u32(&[pe.off as u32, cfg.ple * cfg.n_layer, qtype(pe.ty)?, 160, 0, 0]), 1);
        }
        bar();
        self.misc(cb, &self.x, &self.x, &self.x, d, 2, 0, (d as f32).sqrt())?; bar();
        if cfg.ple > 0 {
            self.gq(cb, &self.g_proj, &[wt.wref("per_layer_model_proj.weight")?], &self.x, &self.pr_proj, false)?; bar();
            let ds = self.set(&self.ple, &[&self.pr_proj, &self.pr_raw, &self.plei, &wt.f32b])?;
            c.dispatch3(cb, &self.ple, ds, &push_u32(&[cfg.ple, wt.f("per_layer_proj_norm.weight")?, cfg.eps.to_bits(), (1.0 / (d as f32).sqrt()).to_bits(),
                (cfg.ple as f32).sqrt().to_bits(), std::f32::consts::FRAC_1_SQRT_2.to_bits(), 0]), cfg.n_layer, 1, 1); bar();
        }
        let (nl_dbg, part) = (env("DBG_LAYERS", cfg.n_layer), env("DBG_PART", 0));
        // G4_SKIP (profiling only, wrong output): 1 routed experts, 2 dense FFN GEMVs, 4 attention+combine, 8 lm_head, 16 q/k/v GEMV
        let skip = env("G4_SKIP", 0);
        for l in 0..nl_dbg {
            let stop = |k: u32| part == k && l + 1 == nl_dbg;
            let n = |s: &str| format!("blk.{l}.{s}");
            let ly = cfg.layers[l as usize];
            let (hd, qn, kn) = (ly.hd, cfg.n_head * ly.hd, ly.n_kv * ly.hd);
            // G4_FUSE (default on, MoE models): attn_norm was produced by the previous layer's fused end-of-layer norm
            let fuse = env("G4_FUSE", 1) != 0 && ly.moe && cfg.ple == 0 && part == 0;
            if !(fuse && l > 0 && cfg.layers[l as usize - 1].moe) { self.rms(cb, &self.x, &self.xw, d, wt.f(&n("attn_norm.weight"))?, 0, NONE, 0)?; bar(); }
            let mut segs = vec![wt.wref(&n("attn_q.weight"))?];
            let mut v_base = qn + kn;
            if ly.has_kv { segs.push(wt.wref(&n("attn_k.weight"))?); match wt.wref(&n("attn_v.weight")) { Ok(v) => segs.push(v), Err(_) => v_base = qn } }
            if skip & 16 == 0 { self.gq(cb, &self.g_proj, &segs, &self.xw, &self.pr, false)?; } bar();
            let ds = self.set(&self.qkr, &[&self.pr, &self.kc, &self.vc, &self.aux, &wt.f32b])?;
            let kn_off = if ly.has_kv { wt.f(&n("attn_k_norm.weight"))? } else { 0 };
            if skip & 128 == 0 { c.dispatch(cb, &self.qkr, ds, &push_u32(&[cfg.n_head, ly.n_kv, hd, wt.f(&n("attn_q_norm.weight"))?, kn_off, ly.theta.to_bits(), ly.ff_off, ly.cache_off, ly.cap,
                qn, v_base, 160, cfg.eps.to_bits(), ly.has_kv as u32, 0, 0, 0]), cfg.n_head + if ly.has_kv { ly.n_kv } else { 0 }); } bar();
            let ki = (hd == 512) as usize;
            if hd != 256 && hd != 512 { bail!("head dim {hd}") }
            let ds = self.set(&self.attn[ki], &[&self.pr, &self.kc, &self.vc, &self.part, &self.aux])?;
            if skip & 4 == 0 { c.dispatch3(cb, &self.attn[ki], ds, &push_u32(&[cfg.n_head, ly.n_kv, ly.cache_off, self.nc, hd, 160, 1.0f32.to_bits(), 0, 0, if ly.swa { cfg.win } else { 0 }, ly.cap, 0]),
                cfg.n_head, self.nc, 1); } bar();
            let ds = self.set(&self.comb[ki], &[&self.part, &self.ao])?;
            if skip & 256 == 0 { c.dispatch(cb, &self.comb[ki], ds, &push_u32(&[self.nc, 0, 0, cfg.n_head]), cfg.n_head); } bar();
            self.gq(cb, &self.g_proj, &[wt.wref(&n("attn_output.weight"))?], &self.ao, &self.t, false)?; bar();
            if fuse {
                // x += n(t) post_attn; xw = n(x) ffn_norm; xw2 = n(x) pre_ffw_norm_2 | dense gate_up + router logits | geglu + top-k
                // | dense down + expert gate_up | expert down | fused end: m = n(t) w1 + n(t2) w2; x = (x + n(m) w) s; xw = n(x) next
                self.fnorm(cb, 0, [wt.f(&n("post_attention_norm.weight"))?, wt.f(&n("ffn_norm.weight"))?, wt.f(&n("pre_ffw_norm_2.weight"))?, NONE], NONE, 0)?; bar();
                if skip & 2 == 0 { self.gq(cb, &self.g_proj, &[wt.wref(&n("ffn_gate.weight"))?, wt.wref(&n("ffn_up.weight"))?], &self.xw, &self.hb, false)?; }
                self.route_logits(cb, l, &self.x, 1)?; bar();
                self.misc(cb, &self.hb, &self.hb, &self.hg, cfg.ff, 0, 0, 0.0)?;
                self.route_topk(cb, l, &self.x, &self.aux, 1)?; bar();
                if skip & 2 == 0 { self.gq(cb, &self.g_proj, &[wt.wref(&n("ffn_down.weight"))?], &self.hg, &self.t, false)?; }
                if skip & 1 == 0 { self.moe_gu(cb, l, &self.xw2, &self.hm, 1, &self.aux)?; } bar();
                if skip & 1 == 0 { self.moe_dn(cb, l, &self.hm, &self.t2, 1, &self.aux)?; } bar();
                let last = l + 1 == cfg.n_layer;
                let next = if last { wt.f("output_norm.weight")? } else { wt.f(&format!("blk.{}.attn_norm.weight", l + 1))? };
                self.fnorm(cb, 1, [wt.f(&n("post_ffw_norm_1.weight"))?, wt.f(&n("post_ffw_norm_2.weight"))?, wt.f(&n("post_ffw_norm.weight"))?, next],
                    wt.f(&n("layer_output_scale.weight")).unwrap_or(NONE), if last { 128 } else { 0 })?; bar();
                continue;
            }
            self.rms(cb, &self.t, &self.x, d, wt.f(&n("post_attention_norm.weight"))?, 1, NONE, 0)?; bar();
            if stop(1) { break }  // DBG_PART=1: x = attn_out
            // dense GeGLU FFN
            self.rms(cb, &self.x, &self.xw, d, wt.f(&n("ffn_norm.weight"))?, 0, NONE, 0)?; bar();
            if skip & 2 == 0 { self.gq(cb, &self.g_proj, &[wt.wref(&n("ffn_gate.weight"))?, wt.wref(&n("ffn_up.weight"))?], &self.xw, &self.hb, false)?; } bar();
            self.misc(cb, &self.hb, &self.hb, &self.hg, cfg.ff, 0, 0, 0.0)?; bar();
            if skip & 2 == 0 { self.gq(cb, &self.g_proj, &[wt.wref(&n("ffn_down.weight"))?], &self.hg, &self.t, false)?; } bar();
            let sc = wt.f(&n("layer_output_scale.weight")).unwrap_or(NONE);
            if ly.moe {
                // dense branch -> m1 = norm1(mlp); routed branch on attn_out (x); m1 += norm2(moe); x = (x + norm(m1)) * sc
                self.rms(cb, &self.t, &self.m1, d, wt.f(&n("post_ffw_norm_1.weight"))?, 0, NONE, 0)?;
                if stop(2) { bar(); break }  // DBG_PART=2: m1 = ffn_mlp
                self.route(cb, l, &self.x, &self.aux, 1)?;
                self.rms(cb, &self.x, &self.xw, d, wt.f(&n("pre_ffw_norm_2.weight"))?, 0, NONE, 0)?; bar();
                if skip & 1 == 0 { self.moe(cb, l, &self.xw, &self.hm, &self.t, 1, &self.aux)?; }
                self.rms(cb, &self.t, &self.m1, d, wt.f(&n("post_ffw_norm_2.weight"))?, 1, NONE, 0)?; bar();
                if stop(3) { break }  // DBG_PART=3: m1 = ffn_mlp + ffn_moe, aux = routes
                self.rms(cb, &self.m1, &self.x, d, wt.f(&n("post_ffw_norm.weight"))?, 1, sc, 0)?; bar();
                continue;
            }
            if cfg.ple == 0 {
                self.rms(cb, &self.t, &self.x, d, wt.f(&n("post_ffw_norm.weight"))?, 1, sc, 0)?; bar();
            } else {
                self.rms(cb, &self.t, &self.x, d, wt.f(&n("post_ffw_norm.weight"))?, 1, NONE, 0)?; bar();
                self.gq(cb, &self.g_proj, &[wt.wref(&n("inp_gate.weight"))?], &self.x, &self.pg, false)?; bar();
                self.misc(cb, &self.pg, &self.plei, &self.pg, cfg.ple, 1, l * cfg.ple, 0.0)?; bar();
                self.gq(cb, &self.g_proj, &[wt.wref(&n("proj.weight"))?], &self.pg, &self.t, false)?; bar();
                self.rms(cb, &self.t, &self.x, d, wt.f(&n("post_norm.weight"))?, 1, sc, 0)?; bar();
            }
        }
        let fused_end = env("G4_FUSE", 1) != 0 && cfg.layers[cfg.n_layer as usize - 1].moe && cfg.ple == 0 && part == 0 && nl_dbg == cfg.n_layer;
        if !fused_end { self.rms(cb, &self.x, &self.xw, d, wt.f("output_norm.weight")?, 0, NONE, 128)?; bar(); }
        let head = wt.wref("output.weight").or_else(|_| wt.wref("token_embd.weight"))?;
        if skip & 8 == 0 { self.gq(cb, &self.g_head, &[head], &self.xw, &self.logits, false)?; } bar();
        if cfg.softcap > 0.0 { self.misc(cb, &self.logits, &self.logits, &self.logits, cfg.vocab, 3, 0, cfg.softcap)?; bar(); }
        let ds = self.set(&self.argmax, &[&self.logits, &self.aux])?;
        c.dispatch(cb, &self.argmax, ds, &push_u32(&[cfg.vocab]), 64); bar();
        let ds = self.set(&self.sample, &[&self.aux, &self.toks])?;
        c.dispatch(cb, &self.sample, ds, &[], 1); bar();
        Ok(())
    }
    /// routed experts of layer l for nb columns: xw -> h (gelu gate * up) -> y (weighted down, overwrite)
    fn fnorm(&self, cb: vk::CommandBuffer, mode: u32, w: [u32; 4], s: u32, key: u32) -> Result<()> {
        let ds = self.set(&self.fnorm, &[&self.x, &self.t, &self.t2, &self.xw, &self.xw2, &self.aux, &self.wt.f32b])?;
        self.ctx.dispatch(cb, &self.fnorm, ds, &push_u32(&[self.cfg.dim, mode, w[0], w[1], w[2], w[3], s, key, self.cfg.eps.to_bits()]), 1);
        Ok(())
    }
    fn moe_gu(&self, cb: vk::CommandBuffer, l: u32, xw: &Buffer, h: &Buffer, nb: u32, au: &Buffer) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let gu = wt.wref(&format!("blk.{l}.ffn_gate_up_exps.weight"))?;
        if gu.off + gu.bytes > u32::MAX as u64 { bail!("expert offset") }
        let ds = self.set(&self.mgu, &[&wt.arenas[gu.arena], xw, h, &wt.f32b, au])?;
        c.dispatch3(cb, &self.mgu, ds, &push_u32(&[gu.off as u32, (gu.bytes / gu.e as u64) as u32, cfg.ff_exp, cfg.dim, cfg.top_k, qtype(gu.ty)?]),
            (cfg.top_k * cfg.ff_exp).div_ceil(256 / 16), nb, 1);
        Ok(())
    }
    fn moe_dn(&self, cb: vk::CommandBuffer, l: u32, h: &Buffer, y: &Buffer, nb: u32, au: &Buffer) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let dw = wt.wref(&format!("blk.{l}.ffn_down_exps.weight"))?;
        if dw.off + dw.bytes > u32::MAX as u64 { bail!("expert offset") }
        let dty = if dw.ty == gguf::GgmlType::Other(7) { 255 } else { qtype(dw.ty)? };
        let ds = self.set(&self.mdn, &[&wt.arenas[dw.arena], h, y, &wt.f32b, au])?;
        c.dispatch3(cb, &self.mdn, ds, &push_u32(&[dw.off as u32, (dw.bytes / dw.e as u64) as u32, cfg.dim, cfg.ff_exp, cfg.top_k, dty, wt.f(&format!("blk.{l}.ffn_down_exps.scale")).unwrap_or(NONE)]),
            cfg.dim.div_ceil(256 / 8), nb, 1);
        Ok(())
    }
    fn route_logits(&self, cb: vk::CommandBuffer, l: u32, x: &Buffer, nb: u32) -> Result<()> {
        let (cfg, wt) = (&self.cfg, &self.wt);
        let ds = self.set(&self.rlog, &[&wt.f32b, x, &self.rl])?;
        self.ctx.dispatch3(cb, &self.rlog, ds, &push_u32(&[wt.f(&format!("blk.{l}.ffn_gate_inp.weight"))?, wt.f(&format!("blk.{l}.ffn_gate_inp.scale"))?, cfg.dim, cfg.n_exp, cfg.eps.to_bits(), cfg.dim, nb]), cfg.n_exp.div_ceil(8), nb, 1);
        Ok(())
    }
    fn route_topk(&self, cb: vk::CommandBuffer, l: u32, x: &Buffer, au: &Buffer, nb: u32) -> Result<()> {
        let (cfg, wt) = (&self.cfg, &self.wt);
        let ds = self.set(&self.router, &[&wt.f32b, x, au, &self.rl])?;
        self.ctx.dispatch3(cb, &self.router, ds, &push_u32(&[wt.f(&format!("blk.{l}.ffn_gate_inp.weight"))?, wt.f(&format!("blk.{l}.ffn_gate_inp.scale"))?, cfg.dim, cfg.n_exp, cfg.top_k, cfg.eps.to_bits(), cfg.dim, 1]), 1, nb, 1);
        Ok(())
    }
    pub(crate) fn moe(&self, cb: vk::CommandBuffer, l: u32, xw: &Buffer, h: &Buffer, y: &Buffer, nb: u32, au: &Buffer) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let n = |s: &str| format!("blk.{l}.{s}");
        let gu = wt.wref(&n("ffn_gate_up_exps.weight"))?; let dw = wt.wref(&n("ffn_down_exps.weight"))?;
        if gu.off + gu.bytes > u32::MAX as u64 || dw.off + dw.bytes > u32::MAX as u64 { bail!("expert offset") }
        let (lg, ld) = (16u32, 8u32);
        let ds = self.set(&self.mgu, &[&wt.arenas[gu.arena], xw, h, &wt.f32b, au])?;
        c.dispatch3(cb, &self.mgu, ds, &push_u32(&[gu.off as u32, (gu.bytes / gu.e as u64) as u32, cfg.ff_exp, cfg.dim, cfg.top_k, qtype(gu.ty)?]),
            (cfg.top_k * cfg.ff_exp).div_ceil(256 / lg), nb, 1); c.barrier(cb);
        let dty = if dw.ty == gguf::GgmlType::Other(7) { 255 } else { qtype(dw.ty)? };
        let ds = self.set(&self.mdn, &[&wt.arenas[dw.arena], h, y, &wt.f32b, au])?;
        c.dispatch3(cb, &self.mdn, ds, &push_u32(&[dw.off as u32, (dw.bytes / dw.e as u64) as u32, cfg.dim, cfg.ff_exp, cfg.top_k, dty, wt.f(&n("ffn_down_exps.scale")).unwrap_or(NONE)]),
            cfg.dim.div_ceil(256 / ld), nb, 1); c.barrier(cb);
        Ok(())
    }
    /// MoE routing for nb columns of x (stride d): g4_rlog logits over workgroups (G4_RLOG=0: single-workgroup router), then
    /// g4_router top-k into route slots of `au`. Leaves a barrier pending only for the caller's next dependent dispatch.
    pub(crate) fn route(&self, cb: vk::CommandBuffer, l: u32, x: &Buffer, au: &Buffer, nb: u32) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt); let d = cfg.dim;
        let (wo, so) = (wt.f(&format!("blk.{l}.ffn_gate_inp.weight"))?, wt.f(&format!("blk.{l}.ffn_gate_inp.scale"))?);
        let pre = env("G4_RLOG", 1) != 0 && nb <= 512;
        if pre {
            let tb = if nb >= 8 && d <= 4096 { rlog_tb() } else { 1 }; let k = if tb > 1 { &self.rlog4 } else { &self.rlog };
            let ds = self.set(k, &[&wt.f32b, x, &self.rl])?;
            c.dispatch3(cb, k, ds, &push_u32(&[wo, so, d, cfg.n_exp, cfg.eps.to_bits(), d, nb]), cfg.n_exp.div_ceil(8), nb.div_ceil(tb), 1); c.barrier(cb);
        }
        let ds = self.set(&self.router, &[&wt.f32b, x, au, &self.rl])?;
        c.dispatch3(cb, &self.router, ds, &push_u32(&[wo, so, d, cfg.n_exp, cfg.top_k, cfg.eps.to_bits(), d, pre as u32]), 1, nb, 1);
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
