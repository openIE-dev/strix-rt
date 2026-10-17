//! LFM2 / LFM2-MoE (lfm2moe) GPU-driven decode on vkrt: double-gated short-conv blocks, GQA blocks with per-head
//! q/k RMSNorm (head_dim 64), leading dense SwiGLU FFNs, then sigmoid-routed top-k MoE FFNs with expert GEMVs that read
//! the selected experts straight out of the stacked [k, n, n_exp] GGUF tensors (no gather copy).
//! One reusable command buffer = one token step (embed -> 40 layers -> lm_head -> argmax -> token feedback).
//! aux layout (u32 words, "wide"): [0..128) f32 sum-of-squares slots (slot 2l = attn/conv input, 2l+1 = FFN input,
//! 2*n_layer = output norm), [128..130) u64 argmax key, [160] pos, [161] prompt length, [168..176) MoE ids / weights.
use anyhow::{bail, Result};
use ash::vk;
use gguf::GgmlType;
use rtcore::{gk, DsCache, Gk, Weights, WRef, G};
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel};

pub struct Cfg { pub n_layer: u32, pub dim: u32, pub n_head: u32, pub n_kv: Vec<u32>, pub hd: u32, pub ffn: u32, pub ff_exp: u32, pub n_exp: u32, pub top_k: u32,
    pub dense_lead: u32, pub vocab: u32, pub eps: f32, pub theta: f32, pub max_ctx: u32, pub w_scale: f32 }

pub struct Model {
    pub ctx: Ctx, pub wt: Weights, pub cfg: Cfg, ds: DsCache,
    gemv_small: Gk, gemv_inproj: Gk, swi: Gk, down: Gk, moe_gu: Gk, moe_down: Gk, head: Gk,
    q8: Kernel, q8emu: bool, embed: Kernel, router: Kernel, conv: Kernel, rope: Kernel, attn: Kernel, comb: Kernel, nc: u32, pub part: Buffer, argmax: Kernel, sample: Kernel,
    pub x: Buffer, pub xw: Buffer, pub bcx: Buffer, pub ao: Buffer, pub hb: Buffer, pub logits: Buffer, pub kc: Buffer, pub vc: Buffer, pub cst: Buffer, pub toks: Buffer, pub aux: Buffer,
    attn_idx: Vec<u32>, conv_idx: Vec<u32>, pub step: vk::CommandBuffer,
    /// LFM_DET=1: deterministic norm sums (gemv3 flag 128) for reproducible greedy serving; default off (float atomics)
    pub det: bool,
}

fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

impl Model {
    pub fn load(path: &str, max_ctx: u32) -> Result<Self> {
        let ctx = Ctx::new()?;
        let wt = Weights::load(&ctx, path)?;
        let g = &wt.g; let a = "lfm2moe";
        let arch = g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default();
        if arch != a { bail!("expected lfm2moe, got {arch}") }
        let u = |k: &str| g.u64(&format!("{a}.{k}")).map(|v| v as u32);
        let n_layer = u("block_count")?;
        let n_kv = match g.kv.get(&format!("{a}.attention.head_count_kv")) { Some(gguf::Value::Arr(v)) => v.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect(), _ => bail!("head_count_kv") };
        let dim = u("embedding_length")?; let n_head = u("attention.head_count")?;
        let cfg = Cfg { n_layer, dim, n_head, n_kv, hd: dim / n_head, ffn: u("feed_forward_length")?, ff_exp: u("expert_feed_forward_length")?, n_exp: u("expert_count")?,
            top_k: u("expert_used_count")?, dense_lead: u("leading_dense_block_count").unwrap_or(0), vocab: u("vocab_size")?, eps: g.f32(&format!("{a}.attention.layer_norm_rms_epsilon"))?,
            theta: g.f32(&format!("{a}.rope.freq_base"))?, max_ctx: max_ctx.min(u("context_length").unwrap_or(4096)), w_scale: g.f32(&format!("{a}.expert_weights_scale")).unwrap_or(1.0) };
        if cfg.n_exp > 64 || cfg.top_k > 8 || cfg.n_exp % 2 != 0 { bail!("MoE select supports <= 64 experts, top-k <= 8") }
        if cfg.w_scale != 1.0 { bail!("expert_weights_scale != 1 not wired into the GEMV select yet") }
        if cfg.hd != 64 { bail!("head_dim {} (kernels are written for 64)", cfg.hd) }
        if u("expert_gating_func").unwrap_or(2) != 2 { bail!("only sigmoid expert gating is implemented") }
        if 2 * cfg.n_layer + 1 > 128 { bail!("too many layers for the aux slot layout") }
        let sg = env("GEMV_SG", 32);
        let (mut attn_idx, mut conv_idx) = (vec![], vec![]); let (mut na, mut nc) = (0, 0);
        for l in 0..n_layer as usize { if cfg.n_kv[l] > 0 { attn_idx.push(na); conv_idx.push(u32::MAX); na += 1 } else { attn_idx.push(u32::MAX); conv_idx.push(nc); nc += 1 } }
        let nkv = cfg.n_kv.iter().copied().max().unwrap_or(0);
        let f = |n: u64| ctx.buffer(n * 4);
        let kvn = (na * cfg.max_ctx * nkv * cfg.hd) as u64;
        let hbn = cfg.ffn.max(cfg.top_k * cfg.ff_exp) as u64;
        let mut m = Model {
            gemv_small: gk(&ctx, "GEMV_SMALL", (8, 64), false, sg)?, gemv_inproj: gk(&ctx, "GEMV_INPROJ", (8, 64), false, sg)?, swi: gk(&ctx, "GEMV_SWIGLU", (16, 64), true, sg)?,
            down: gk(&ctx, "GEMV_DOWN", (8, 64), false, sg)?, moe_gu: gk(&ctx, "GEMV_MOE_GU", (8, 64), true, sg)?, moe_down: gk(&ctx, "GEMV_MOE_DOWN", (8, 64), false, sg)?, head: gk(&ctx, "GEMV_HEAD", (32, 128), false, sg)?,
            q8: ctx.kernel(shaders::Q8K_ROUND, 1, 4, &[], None)?, q8emu: std::env::var("Q8EMU").map(|v| v == "1").unwrap_or(false),
            embed: ctx.kernel(shaders::EMBED_Q6K, 6, 12, &[], None)?, router: ctx.kernel(shaders::ROUTER, 3, 32, &[], Some(32))?, conv: ctx.kernel(shaders::SHORTCONV, 4, 12, &[], None)?,
            rope: ctx.kernel(shaders::ROPE_QK64, 5, 28, &[], None)?, attn: ctx.kernel(shaders::ATTN_SPLIT, 5, 36, &[64], None)?, comb: ctx.kernel(shaders::ATTN_COMBINE, 3, 20, &[64], None)?, nc: env("ATTN_NC", 8),
            part: f((n_head * env("ATTN_NC", 8) * (2 + 64)) as u64)?,
            argmax: ctx.kernel(shaders::ARGMAX_W, 2, 4, &[], None)?, sample: ctx.kernel(shaders::SAMPLE_W, 2, 0, &[], None)?,
            x: f(dim as u64)?, xw: f(dim as u64)?, bcx: f(3 * dim as u64)?, ao: f(dim as u64)?, hb: f(hbn)?, logits: f(cfg.vocab as u64)?,
            kc: ctx.buffer((kvn * 2).max(256))?, vc: ctx.buffer((kvn * 2).max(256))?, cst: f((nc * 2 * dim).max(64) as u64)?, toks: f(2 * cfg.max_ctx as u64 + 2)?, aux: f(8192)?, // words 1020.. : deterministic norm-sum scratch (gemv3 flag 128)
            attn_idx, conv_idx, step: vk::CommandBuffer::null(), det: std::env::var("LFM_DET").map_or(false, |v| v == "1"), ds: DsCache::default(), cfg, wt, ctx };
        let mut err = Ok(());
        let step = m.ctx.record_raw(|cb| { err = m.record_prof(cb, None); })?; err?;
        m.step = step;
        Ok(m)
    }

    /// Q8EMU=1: round a GEMV input to the q8_K grid first (matches llama.cpp CPU arithmetic; costs a barrier).
    fn q8r(&self, cb: vk::CommandBuffer, b: &Buffer, n: u32) -> Result<()> {
        if !self.q8emu { return Ok(()) }
        let ds = self.set(&self.q8, &[b])?; self.ctx.dispatch(cb, &self.q8, ds, &push_u32(&[0]), n / 256); self.ctx.barrier(cb); Ok(())
    }
    fn set(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&self.ctx, k, bufs) }

    /// Plain / fused-segment GEMV (up to 3 row segments sharing an arena).
    fn gemv(&self, cb: vk::CommandBuffer, gk: &Gk, segs: &[WRef], up: Option<WRef>, x: &Buffer, y: &Buffer, acc: bool, scale_slot: Option<u32>, sq_out: Option<(u32, u32)>) -> Result<()> {
        let arena = segs[0].arena;
        if segs.iter().any(|s| s.arena != arena) || up.map(|u| u.arena != arena).unwrap_or(false) { bail!("fused segments must share an arena") }
        let mut g = G { k: segs[0].k, eps: self.cfg.eps, b: [u32::MAX; 3], ..Default::default() };
        let mut r = 0; let mut bounds = [0u32; 3];
        for (i, w) in segs.iter().enumerate() { g.w[i] = w.unit_off()?; if w.ty == GgmlType::Q6K { g.types |= 1 << i; } r += w.n; bounds[i] = r; }
        g.n = r; g.r1 = bounds[0]; g.r2 = if segs.len() > 1 { bounds[1] } else { r };
        if let Some(u) = up { if u.ty != GgmlType::Q4K || segs[0].ty != GgmlType::Q4K { bail!("swiglu gemv is Q4_K only") } g.wup = u.unit_off()?; }
        self.launch(cb, gk, g, arena, x, y, acc, scale_slot, sq_out)
    }
    fn launch(&self, cb: vk::CommandBuffer, gk: &Gk, mut g: G, arena: usize, x: &Buffer, y: &Buffer, acc: bool, scale_slot: Option<u32>, sq_out: Option<(u32, u32)>) -> Result<()> {
        if acc { g.flags |= 1; }
        if let Some(s) = scale_slot { g.flags |= 4; g.sq_in = s; }
        if let Some((s, nw)) = sq_out { g.flags |= 8 | 16; if self.det { g.flags |= 128; } g.sq_out = s; g.norm_off = nw; }
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xw])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }

    pub fn record_prof(&self, cb: vk::CommandBuffer, prof: Option<(vk::QueryPool, &std::cell::RefCell<Vec<&'static str>>)>) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let cur = std::cell::Cell::new("start");
        if let Some((qp, labels)) = prof { unsafe { c.device.cmd_reset_query_pool(cb, qp, 0, 1024); c.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 0); } labels.borrow_mut().push("start"); }
        let bar = || { c.barrier(cb); if let Some((qp, labels)) = prof { let mut l = labels.borrow_mut(); let i = l.len() as u32; l.push(cur.get());
            unsafe { c.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, i); } } };
        bar();
        let te = wt.wref("token_embd.weight")?;
        if te.ty != GgmlType::Q6K { bail!("embed kernel expects Q6_K token_embd") }
        let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &self.x, &self.aux, &self.toks, &wt.f32b, &self.xw])?;
        cur.set("embed"); c.dispatch(cb, &self.embed, ds, &push_u32(&[te.unit_off()?, cfg.dim, wt.f("blk.0.attn_norm.weight")?]), 1); bar();
        let nkv = cfg.n_kv.iter().copied().max().unwrap_or(0);
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let li = l as usize;
            let ffn_norm = wt.f(&n("ffn_norm.weight"))?;
            if cfg.n_kv[li] > 0 {
                let cache_off = self.attn_idx[li] * cfg.max_ctx * nkv * cfg.hd;
                cur.set("qkv"); self.q8r(cb, &self.xw, cfg.dim)?; self.gemv(cb, &self.gemv_small, &[wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?], None, &self.xw, &self.bcx, false, Some(2 * l), None)?; bar();
                cur.set("rope"); let ds = self.set(&self.rope, &[&self.bcx, &self.kc, &self.vc, &self.aux, &wt.f32b])?;
                c.dispatch(cb, &self.rope, ds, &push_u32(&[cfg.n_head, cfg.n_kv[li], cache_off, cfg.theta.to_bits(), wt.f(&n("attn_q_norm.weight"))?, wt.f(&n("attn_k_norm.weight"))?, cfg.eps.to_bits()]), cfg.n_head + cfg.n_kv[li]); bar();
                cur.set("attn"); let ds = self.set(&self.attn, &[&self.bcx, &self.kc, &self.vc, &self.part, &self.aux])?;
                c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv[li], cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0]), cfg.n_head, self.nc, 1); bar();
                let ds = self.set(&self.comb, &[&self.part, &self.ao, &self.bcx])?;
                c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, 0, 0, 0, 0]), cfg.n_head); bar();
                cur.set("o_proj"); self.q8r(cb, &self.ao, cfg.dim)?; self.gemv(cb, &self.gemv_small, &[wt.wref(&n("attn_output.weight"))?], None, &self.ao, &self.x, true, None, Some((2 * l + 1, ffn_norm)))?; bar();
            } else {
                cur.set("in_proj"); self.q8r(cb, &self.xw, cfg.dim)?; self.gemv(cb, &self.gemv_inproj, &[wt.wref(&n("shortconv.in_proj.weight"))?], None, &self.xw, &self.bcx, false, Some(2 * l), None)?; bar();
                cur.set("conv"); let ds = self.set(&self.conv, &[&self.bcx, &self.cst, &self.ao, &wt.f32b])?;
                c.dispatch(cb, &self.conv, ds, &push_u32(&[cfg.dim, self.conv_idx[li] * 2 * cfg.dim, wt.f(&n("shortconv.conv.weight"))?]), cfg.dim.div_ceil(64)); bar();
                cur.set("out_proj"); self.q8r(cb, &self.ao, cfg.dim)?; self.gemv(cb, &self.gemv_small, &[wt.wref(&n("shortconv.out_proj.weight"))?], None, &self.ao, &self.x, true, None, Some((2 * l + 1, ffn_norm)))?; bar();
            }
            let next_norm = if l + 1 < cfg.n_layer { wt.f(&format!("blk.{}.attn_norm.weight", l + 1))? } else { wt.f("token_embd_norm.weight")? };
            if l < cfg.dense_lead {
                cur.set("dense_gu"); self.q8r(cb, &self.xw, cfg.dim)?; self.gemv(cb, &self.swi, &[wt.wref(&n("ffn_gate.weight"))?], Some(wt.wref(&n("ffn_up.weight"))?), &self.xw, &self.hb, false, Some(2 * l + 1), None)?; bar();
                cur.set("dense_dn"); self.q8r(cb, &self.hb, cfg.ffn)?; self.gemv(cb, &self.down, &[wt.wref(&n("ffn_down.weight"))?], None, &self.hb, &self.x, true, None, Some((2 * l + 2, next_norm)))?; bar();
            } else {
                cur.set("router"); let ds = self.set(&self.router, &[&wt.f32b, &self.xw, &self.aux])?;
                c.dispatch(cb, &self.router, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, 0, cfg.dim, cfg.n_exp, 2 * l + 1, cfg.eps.to_bits(), cfg.top_k, cfg.w_scale.to_bits()]), cfg.n_exp / 2); bar();
                let (bsel, misc) = (wt.f(&n("exp_probs_b.bias"))?, cfg.n_exp | (cfg.top_k << 16));
                let (gw, uw, dw) = (wt.wref(&n("ffn_gate_exps.weight"))?, wt.wref(&n("ffn_up_exps.weight"))?, wt.wref(&n("ffn_down_exps.weight"))?);
                if gw.ty != GgmlType::Q4K || uw.ty != GgmlType::Q4K { bail!("MoE gate/up must be Q4_K") }
                if gw.arena != uw.arena || gw.arena != dw.arena { bail!("experts must share an arena") }
                // gate|up for the top-k experts: rows = top_k * ff_exp, output hb[slot * ff_exp + r] = silu(g) * u
                cur.set("moe_gu"); self.q8r(cb, &self.xw, cfg.dim)?;
                let g = G { n: cfg.top_k * gw.n, k: gw.k, flags: 32, r1: gw.n, r2: gw.expert_stride_units()?, w: [gw.unit_off()?, 0, 0], b: [u32::MAX, uw.expert_stride_units()?, u32::MAX], wup: uw.unit_off()?, eps: cfg.eps, bsel, misc, ..Default::default() };
                self.launch(cb, &self.moe_gu, g, gw.arena, &self.xw, &self.hb, false, Some(2 * l + 1), None)?; bar();
                // down: x[row] += sum_s w_s * Wd_{e_s}[row] . hb[s]
                cur.set("moe_dn"); self.q8r(cb, &self.hb, cfg.top_k * cfg.ff_exp)?;
                let g = G { n: dw.n, k: dw.k, flags: 64, r2: dw.expert_stride_units()?, types: (dw.ty == GgmlType::Q6K) as u32, w: [dw.unit_off()?, 0, 0], b: [cfg.top_k, u32::MAX, u32::MAX], eps: cfg.eps, bsel, misc, ..Default::default() };
                self.launch(cb, &self.moe_down, g, dw.arena, &self.hb, &self.x, true, None, Some((2 * l + 2, next_norm)))?; bar();
            }
        }
        cur.set("lm_head"); self.q8r(cb, &self.xw, cfg.dim)?; self.gemv(cb, &self.head, &[te], None, &self.xw, &self.logits, false, Some(2 * cfg.n_layer), None)?; bar();
        cur.set("argmax"); let ds = self.set(&self.argmax, &[&self.logits, &self.aux])?;
        c.dispatch(cb, &self.argmax, ds, &push_u32(&[cfg.vocab]), 64); bar();
        let ds = self.set(&self.sample, &[&self.aux, &self.toks])?;
        cur.set("sample"); c.dispatch(cb, &self.sample, ds, &[], 1); bar();
        Ok(())
    }

    /// Greedy generation fully on the GPU. Returns (ids, prompt GPU s, prompt wall s, decode GPU s, decode wall s, decode steps).
    pub fn generate(&self, prompt: &[u32], n_new: usize) -> Result<(Vec<u32>, f64, f64, f64, f64, usize)> {
        if prompt.len() + n_new > self.cfg.max_ctx as usize { bail!("context too long") }
        let toks = unsafe { std::slice::from_raw_parts_mut(self.toks.ptr as *mut u32, self.toks.size as usize / 4) };
        let au = unsafe { std::slice::from_raw_parts_mut(self.aux.ptr as *mut u32, 256) };
        self.cst.bytes_mut().fill(0);
        toks[..prompt.len()].copy_from_slice(prompt);
        au[160] = 0; au[161] = prompt.len() as u32; au[162] = 0;
        let t = std::time::Instant::now();
        let pg = self.ctx.submit_repeated(self.step, prompt.len())?;
        let pw = t.elapsed().as_secs_f64();
        let steps = n_new.saturating_sub(1);
        let t = std::time::Instant::now();
        let dg = if steps > 0 { self.ctx.submit_repeated(self.step, steps)? } else { 0.0 };
        let dw = t.elapsed().as_secs_f64();
        Ok((toks[prompt.len()..prompt.len() + n_new].to_vec(), pg, pw, dg, dw, steps))
    }

    /// Teacher-forced: run `seq` through the model; returns the model's argmax prediction after each position.
    pub fn predict_all(&self, seq: &[u32]) -> Result<Vec<u32>> {
        let mc = self.cfg.max_ctx as usize;
        if seq.len() > mc { bail!("sequence too long") }
        let toks = unsafe { std::slice::from_raw_parts_mut(self.toks.ptr as *mut u32, self.toks.size as usize / 4) };
        let au = unsafe { std::slice::from_raw_parts_mut(self.aux.ptr as *mut u32, 256) };
        self.cst.bytes_mut().fill(0);
        toks[..seq.len()].copy_from_slice(seq);
        au[160] = 0; au[161] = seq.len() as u32 + 1; au[162] = (mc + 1) as u32;
        self.ctx.submit_repeated(self.step, seq.len())?;
        au[162] = 0;
        Ok(toks[mc + 2..mc + 2 + seq.len()].to_vec())
    }
    /// One profiled step at the current state; per-stage GPU microseconds summed over layers.
    pub fn profile(&self) -> Result<Vec<(&'static str, f64)>> {
        let c = &self.ctx;
        let qp = unsafe { c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(1024), None)? };
        let labels = std::cell::RefCell::new(vec![]);
        let mut err = Ok(());
        let cb = c.record_raw(|cb| { err = self.record_prof(cb, Some((qp, &labels))); })?; err?;
        c.submit_repeated(cb, 1)?;
        let l = labels.borrow(); let mut ts = vec![0u64; l.len()];
        unsafe { c.device.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; }
        let mut agg: Vec<(&'static str, f64)> = vec![];
        for i in 1..l.len() { let us = (ts[i] - ts[i - 1]) as f64 * c.info.ts_period_ns as f64 / 1e3;
            match agg.iter_mut().find(|a| a.0 == l[i]) { Some(a) => a.1 += us, None => agg.push((l[i], us)) } }
        Ok(agg)
    }
}

/// Per-sequence decode state for multi-sequence serving (strix-serve1): own KV, conv state, token ring and aux words,
/// plus a step command buffer recorded against them. `with_st` swaps the state into the model so prefill / record /
/// any other entry point runs on that sequence unchanged (bitwise identical to single-stream).
pub struct St { pub kc: Buffer, pub vc: Buffer, pub cst: Buffer, pub toks: Buffer, pub aux: Buffer, pub step: vk::CommandBuffer }
impl Model {
    fn st_swap(&mut self, s: &mut St) {
        std::mem::swap(&mut self.kc, &mut s.kc); std::mem::swap(&mut self.vc, &mut s.vc); std::mem::swap(&mut self.cst, &mut s.cst);
        std::mem::swap(&mut self.toks, &mut s.toks); std::mem::swap(&mut self.aux, &mut s.aux); std::mem::swap(&mut self.step, &mut s.step);
    }
    pub fn new_st(&mut self) -> Result<St> {
        let c = &self.ctx;
        let mut s = St { kc: c.buffer(self.kc.size)?, vc: c.buffer(self.vc.size)?, cst: c.buffer(self.cst.size)?, toks: c.buffer(self.toks.size)?, aux: c.buffer(self.aux.size)?, step: vk::CommandBuffer::null() };
        s.aux.bytes_mut().copy_from_slice(self.aux.bytes_mut());
        self.st_swap(&mut s);
        let mut err = Ok(());
        let r = self.ctx.record_raw(|cb| { err = self.record_prof(cb, None); });
        if let Ok(cb) = r { self.step = cb; }
        self.st_swap(&mut s);
        r?; err?; Ok(s)
    }
    pub fn with_st<R>(&mut self, s: &mut St, f: impl FnOnce(&mut Self) -> R) -> R { self.st_swap(s); let r = f(self); self.st_swap(s); r }
}

// ---------------------------------------------------------------------------------------------------------------
// Batched multi-sequence decode (strix-serve1). Every dense projection (conv in/out, attention qkv/o, dense FFN,
// lm_head) runs as one gemvq over the m token columns (<= 8 columns per weight read). Kernels with fixed buffer
// layout (embed, shortconv, rope, attention, router, MoE gemv3) run per column on small per-column scratch, with
// copies between the wide batch buffers and the scratch; all columns share each barrier stage. The m = 1 case is
// the single-stream reference (server greedy equality). Wide aux `ba`: column j norm slots at s + 1024 j,
// greedy argmax of column j -> ba[600 + j]. Per-column aux `a[j]`: word 160 = KV position (host-written).
pub struct Multi { pub m: u32, x: Buffer, xw: Buffer, bcx: Buffer, ao: Buffer, hb: Buffer, pub logits: Buffer, pub ba: Buffer,
    pub a: Vec<Buffer>, xe: Vec<Buffer>, xwj: Vec<Buffer>, bcj: Vec<Buffer>, aoj: Vec<Buffer>, hbj: Vec<Buffer>, parts: Vec<Buffer>,
    g_proj: rtcore::GkQ, g_swi: rtcore::GkQ, g_head: rtcore::GkQ, argb: Kernel }
impl Model {
    pub fn new_multi(&self, m: u32) -> Result<Multi> {
        let (c, cfg) = (&self.ctx, &self.cfg); let f = |n: u64| c.buffer(n.max(16) * 4); let mm = m as u64; let d = cfg.dim as u64;
        let hbn = cfg.ffn.max(cfg.top_k * cfg.ff_exp) as u64; let sg = env("GEMV_SG", 32);
        let mut mu = Multi { m, x: f(mm * d)?, xw: f(mm * d)?, bcx: f(mm * 3 * d)?, ao: f(mm * d)?, hb: f(mm * hbn)?, logits: f(mm * cfg.vocab as u64)?, ba: f(57344)?,
            a: vec![], xe: vec![], xwj: vec![], bcj: vec![], aoj: vec![], hbj: vec![], parts: vec![],
            g_proj: rtcore::GkQ::new("MGEMV_PROJ", (16, 64), false, sg), g_swi: rtcore::GkQ::new("MGEMV_SWIGLU", (16, 64), true, sg), g_head: rtcore::GkQ::new("MGEMV_HEAD", (8, 256), false, sg),
            argb: c.kernel(shaders::ARGMAX_B, 2, 16, &[], None)? };
        for _ in 0..m { mu.a.push(f(8192)?); mu.xe.push(f(d)?); mu.xwj.push(f(d)?); mu.bcj.push(f(3 * d)?); mu.aoj.push(f(d)?); mu.hbj.push(f(hbn)?); mu.parts.push(c.buffer(self.part.size)?); }
        Ok(mu)
    }
    fn xbar(&self, cb: vk::CommandBuffer) {
        let a = vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ;
        let st = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let mb = [vk::MemoryBarrier::default().src_access_mask(a).dst_access_mask(a)];
        unsafe { self.ctx.device.cmd_pipeline_barrier(cb, st, st, vk::DependencyFlags::empty(), &mb, &[], &[]) };
    }
    fn cp(&self, cb: vk::CommandBuffer, a: &Buffer, b: &Buffer, ao: u64, bo: u64, n: u64) {
        unsafe { self.ctx.device.cmd_copy_buffer(cb, a.buf, b.buf, &[vk::BufferCopy { src_offset: ao, dst_offset: bo, size: n }]) };
    }
    /// gemvq over m columns (column strides x_cs / y_cs; norm slots of column g at s + 1024 g in mu.ba; xw output -> mu.xw)
    fn gq_m(&self, cb: vk::CommandBuffer, mu: &Multi, gq: &rtcore::GkQ, segs: &[WRef], up: Option<WRef>, x: &Buffer, x_cs: u32, y: &Buffer, y_cs: u32,
            acc: bool, sq_in: Option<u32>, sq_out: Option<(u32, u32)>, m: u32) -> Result<()> {
        let cap = std::env::var("SERVE_NB").ok().and_then(|v| v.parse().ok()).unwrap_or(8u32).clamp(1, 8);
        let mut g0 = 0;
        while g0 < m { let left = (m - g0).div_ceil(cap); let nb = (m - g0).div_ceil(left);
            let (mut g, arena) = rtcore::GQ::segs(segs, up, &[], self.cfg.eps)?;
            g.y_off = g0 * y_cs; g.x_off4 = g0 * x_cs / 4; g.x_cs = x_cs; g.y_cs = y_cs;
            let gk = gq.get_nb(&self.ctx, &g, segs.len(), nb)?;
            g.flags |= 64; if acc { g.flags |= 1; }
            if let Some(s) = sq_in { g.flags |= 4; g.sq_in = s + 1024 * g0; }
            if let Some((s, nw)) = sq_out { g.flags |= 8 | 16 | 64; g.sq_out = s + 1024 * g0; g.norm_off = nw; }
            let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &mu.ba, &mu.xw])?;
            self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
            g0 += nb; if g0 < m { self.ctx.barrier(cb); } }
        Ok(())
    }
    fn g3j(&self, cb: vk::CommandBuffer, gk: &Gk, mut g: G, arena: usize, x: &Buffer, y: &Buffer, acc: bool, sq_in: Option<u32>, sq_out: Option<(u32, u32)>, aux: &Buffer, xw: &Buffer) -> Result<()> {
        if acc { g.flags |= 1; }
        if let Some(s) = sq_in { g.flags |= 4; g.sq_in = s; }
        if let Some((s, nw)) = sq_out { g.flags |= 8 | 16 | 128; g.sq_out = s; g.norm_off = nw; }
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, aux, xw])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    /// One decode step for m = sts.len() <= mu.m sequences: column j embeds sts[j].toks[a[j][160]] at KV position a[j][160]
    /// (host-written); logits column j at mu.logits[j * vocab]; greedy argmax -> mu.ba[600 + j].
    pub fn record_multi(&self, cb: vk::CommandBuffer, sts: &[&St], mu: &Multi) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let m = sts.len() as u32; if m == 0 || m > mu.m { bail!("record_multi: {m} slots for {} columns", mu.m) }
        let (d, d4) = (cfg.dim, cfg.dim as u64 * 4); let hbn = cfg.ffn.max(cfg.top_k * cfg.ff_exp);
        let te = wt.wref("token_embd.weight")?;
        if te.ty != GgmlType::Q6K { bail!("embed kernel expects Q6_K token_embd") }
        c.barrier(cb);
        for j in 0..m as usize {
            let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &mu.xe[j], &mu.a[j], &sts[j].toks, &wt.f32b, &mu.xwj[j]])?;
            c.dispatch(cb, &self.embed, ds, &push_u32(&[te.unit_off()?, d, wt.f("blk.0.attn_norm.weight")?]), 1); }
        self.xbar(cb);
        for j in 0..m as usize { let jj = j as u64;
            self.cp(cb, &mu.xe[j], &mu.x, 0, jj * d4, d4); self.cp(cb, &mu.xwj[j], &mu.xw, 0, jj * d4, d4); self.cp(cb, &mu.a[j], &mu.ba, 0, jj * 4096, 512); }
        self.xbar(cb);
        let nkv = cfg.n_kv.iter().copied().max().unwrap_or(0);
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let li = l as usize;
            let ffn_norm = wt.f(&n("ffn_norm.weight"))?;
            if cfg.n_kv[li] > 0 {
                let cache_off = self.attn_idx[li] * cfg.max_ctx * nkv * cfg.hd;
                self.gq_m(cb, mu, &mu.g_proj, &[wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?], None, &mu.xw, d, &mu.bcx, 3 * d, false, Some(2 * l), None, m)?;
                self.xbar(cb);
                for j in 0..m as usize { self.cp(cb, &mu.bcx, &mu.bcj[j], j as u64 * 3 * d4, 0, 3 * d4); }
                self.xbar(cb);
                for j in 0..m as usize { let ds = self.set(&self.rope, &[&mu.bcj[j], &sts[j].kc, &sts[j].vc, &mu.a[j], &wt.f32b])?;
                    c.dispatch(cb, &self.rope, ds, &push_u32(&[cfg.n_head, cfg.n_kv[li], cache_off, cfg.theta.to_bits(), wt.f(&n("attn_q_norm.weight"))?, wt.f(&n("attn_k_norm.weight"))?, cfg.eps.to_bits()]), cfg.n_head + cfg.n_kv[li]); }
                c.barrier(cb);
                for j in 0..m as usize { let ds = self.set(&self.attn, &[&mu.bcj[j], &sts[j].kc, &sts[j].vc, &mu.parts[j], &mu.a[j]])?;
                    c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv[li], cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0]), cfg.n_head, self.nc, 1); }
                c.barrier(cb);
                for j in 0..m as usize { let ds = self.set(&self.comb, &[&mu.parts[j], &mu.aoj[j], &mu.bcj[j]])?; c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, 0, 0, 0, 0]), cfg.n_head); }
                self.xbar(cb);
                for j in 0..m as usize { self.cp(cb, &mu.aoj[j], &mu.ao, 0, j as u64 * d4, d4); }
                self.xbar(cb);
                self.gq_m(cb, mu, &mu.g_proj, &[wt.wref(&n("attn_output.weight"))?], None, &mu.ao, d, &mu.x, d, true, None, Some((2 * l + 1, ffn_norm)), m)?; c.barrier(cb);
            } else {
                self.gq_m(cb, mu, &mu.g_proj, &[wt.wref(&n("shortconv.in_proj.weight"))?], None, &mu.xw, d, &mu.bcx, 3 * d, false, Some(2 * l), None, m)?;
                self.xbar(cb);
                for j in 0..m as usize { self.cp(cb, &mu.bcx, &mu.bcj[j], j as u64 * 3 * d4, 0, 3 * d4); }
                self.xbar(cb);
                for j in 0..m as usize { let ds = self.set(&self.conv, &[&mu.bcj[j], &sts[j].cst, &mu.aoj[j], &wt.f32b])?;
                    c.dispatch(cb, &self.conv, ds, &push_u32(&[d, self.conv_idx[li] * 2 * d, wt.f(&n("shortconv.conv.weight"))?]), d.div_ceil(64)); }
                self.xbar(cb);
                for j in 0..m as usize { self.cp(cb, &mu.aoj[j], &mu.ao, 0, j as u64 * d4, d4); }
                self.xbar(cb);
                self.gq_m(cb, mu, &mu.g_proj, &[wt.wref(&n("shortconv.out_proj.weight"))?], None, &mu.ao, d, &mu.x, d, true, None, Some((2 * l + 1, ffn_norm)), m)?; c.barrier(cb);
            }
            let next_norm = if l + 1 < cfg.n_layer { wt.f(&format!("blk.{}.attn_norm.weight", l + 1))? } else { wt.f("token_embd_norm.weight")? };
            if l < cfg.dense_lead {
                self.gq_m(cb, mu, &mu.g_swi, &[wt.wref(&n("ffn_gate.weight"))?], Some(wt.wref(&n("ffn_up.weight"))?), &mu.xw, d, &mu.hb, hbn, false, Some(2 * l + 1), None, m)?; c.barrier(cb);
                self.gq_m(cb, mu, &mu.g_proj, &[wt.wref(&n("ffn_down.weight"))?], None, &mu.hb, hbn, &mu.x, d, true, None, Some((2 * l + 2, next_norm)), m)?; c.barrier(cb);
            } else {
                let s1 = 2 * l + 1; let s2 = 2 * l + 2;
                self.xbar(cb);
                for j in 0..m as usize { let jj = j as u64;
                    self.cp(cb, &mu.xw, &mu.xwj[j], jj * d4, 0, d4); self.cp(cb, &mu.ba, &mu.a[j], (s1 as u64 + 1024 * jj) * 4, s1 as u64 * 4, 4); }
                self.xbar(cb);
                for j in 0..m as usize { let ds = self.set(&self.router, &[&wt.f32b, &mu.xwj[j], &mu.a[j]])?;
                    c.dispatch(cb, &self.router, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, 0, d, cfg.n_exp, s1, cfg.eps.to_bits(), cfg.top_k, cfg.w_scale.to_bits()]), cfg.n_exp / 2); }
                c.barrier(cb);
                let (bsel, misc) = (wt.f(&n("exp_probs_b.bias"))?, cfg.n_exp | (cfg.top_k << 16));
                let (gw, uw, dw) = (wt.wref(&n("ffn_gate_exps.weight"))?, wt.wref(&n("ffn_up_exps.weight"))?, wt.wref(&n("ffn_down_exps.weight"))?);
                for j in 0..m as usize {
                    let g = G { n: cfg.top_k * gw.n, k: gw.k, flags: 32, r1: gw.n, r2: gw.expert_stride_units()?, w: [gw.unit_off()?, 0, 0], b: [u32::MAX, uw.expert_stride_units()?, u32::MAX], wup: uw.unit_off()?, eps: cfg.eps, bsel, misc, ..Default::default() };
                    self.g3j(cb, &self.moe_gu, g, gw.arena, &mu.xwj[j], &mu.hbj[j], false, Some(s1), None, &mu.a[j], &mu.xwj[j])?; }
                c.barrier(cb);
                for j in 0..m as usize {
                    let g = G { n: dw.n, k: dw.k, flags: 64, y_off: j as u32 * d, r2: dw.expert_stride_units()?, types: (dw.ty == GgmlType::Q6K) as u32, w: [dw.unit_off()?, 0, 0], b: [cfg.top_k, u32::MAX, u32::MAX], eps: cfg.eps, bsel, misc, ..Default::default() };
                    self.g3j(cb, &self.moe_down, g, dw.arena, &mu.hbj[j], &mu.x, true, None, Some((s2, next_norm)), &mu.a[j], &mu.xw)?; }
                self.xbar(cb);
                for j in 0..m as usize { self.cp(cb, &mu.a[j], &mu.ba, s2 as u64 * 4, (s2 as u64 + 1024 * j as u64) * 4, 4); }
                self.xbar(cb);
            }
        }
        self.gq_m(cb, mu, &mu.g_head, &[te], None, &mu.xw, d, &mu.logits, cfg.vocab, false, Some(2 * cfg.n_layer), None, m)?; c.barrier(cb);
        let ds = self.set(&mu.argb, &[&mu.logits, &mu.ba])?;
        c.dispatch(cb, &mu.argb, ds, &push_u32(&[cfg.vocab, cfg.vocab, 600, 0]), m); c.barrier(cb);
        Ok(())
    }
}
