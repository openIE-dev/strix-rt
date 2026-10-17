//! Nemotron-H (nemotron_h GGUF: Nemotron 3 Nano 4B) GPU-driven decode on vkrt.
//! Each block is exactly one of: Mamba-2 mixer, GQA attention (no RoPE), or squared-ReLU MLP, wrapped as
//! x += mixer(rmsnorm(x)). One reusable command buffer = one token step (embed -> blocks -> lm_head -> argmax -> feedback).
//! aux words: [128..130) u64 argmax key, [160] pos, [161] prompt length, [162] teacher-forced prediction base.
pub mod prefill;
use anyhow::{bail, Result};
use ash::vk;
use gguf::GgmlType;
use rtcore::{qtype, DsCache, GkQ, Weights, WRef, GQ};
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind { Mamba, Attn, Ffn, Moe }

pub struct Cfg { pub n_layer: u32, pub dim: u32, pub n_head: u32, pub n_kv: u32, pub hd: u32, pub ffn: u32, pub vocab: u32, pub eps: f32,
    pub d_conv: u32, pub d_state: u32, pub n_group: u32, pub d_inner: u32, pub ssm_heads: u32, pub max_ctx: u32, pub kinds: Vec<Kind>,
    pub n_exp: u32, pub top_k: u32, pub ff_exp: u32, pub w_scale: f32, pub w_norm: bool }
impl Cfg {
    pub fn ssm_hd(&self) -> u32 { self.d_inner / self.ssm_heads }
    pub fn conv_ch(&self) -> u32 { self.d_inner + 2 * self.n_group * self.d_state }
    pub fn in_proj(&self) -> u32 { 2 * self.d_inner + 2 * self.n_group * self.d_state + self.ssm_heads }
}

pub struct Model {
    pub ctx: Ctx, pub wt: Weights, pub cfg: Cfg, ds: DsCache, gq: GkQ,
    embed: Kernel, router: Kernel, rlog: Kernel, rp: Buffer, moe_up: Kernel, moe_dn: Kernel, pub hm: Buffer, moe_rpw: u32, rms: Kernel, conv: Kernel, scan: Kernel, gnorm: Kernel, relu2: Kernel, kv: Kernel, prep: Kernel, attn: Kernel, comb: Kernel, argmax: Kernel, sample: Kernel, nc: u32,
    pub x: Buffer, pub xn: Buffer, pub zx: Buffer, pub xbc: Buffer, pub yb: Buffer, pub ao: Buffer, pub part: Buffer, pub logits: Buffer,
    pub kc: Buffer, pub vc: Buffer, pub cst: Buffer, pub sst: Buffer, pub toks: Buffer, pub aux: Buffer,
    pub(crate) idx: Vec<u32>, pub step: vk::CommandBuffer, prof: std::cell::Cell<Option<vk::QueryPool>>, plab: std::cell::RefCell<Vec<String>>, lab: std::cell::Cell<&'static str>, kl: std::cell::Cell<&'static str>,
}

/// Like rtcore::GQ::segs (no up/bias) but accepts 2-byte aligned rows for formats whose single-column dot32x path
/// only does 16-bit-safe reads (Q6_K, Q8_0, Q5_0): K = 12544 / 3136 rows are not 4-byte multiples there.
fn segs_q(segs: &[WRef], eps: f32) -> Result<(GQ, usize)> {
    let arena = segs[0].arena;
    if segs.len() > 4 || segs.iter().any(|s| s.arena != arena || s.k != segs[0].k) { bail!("segments must share arena and K") }
    let mut g = GQ { k: segs[0].k, eps, b: [u32::MAX; 4], r: [u32::MAX; 3], ..Default::default() };
    let mut rows = 0u32;
    for (i, w) in segs.iter().enumerate() {
        if w.off > u32::MAX as u64 || w.off % 4 != 0 { bail!("weight offset") }
        let q = qtype(w.ty)?; let rb = w.bytes / (w.n as u64 * w.e.max(1) as u64);
        if rb % 4 != 0 && !(rb % 2 == 0 && (q == 1 || q == 3 || q == 15)) { bail!("row alignment for {:?}", w.ty) }
        g.w[i] = w.off as u32; g.types |= q << (4 * i); rows += w.n * w.e.max(1); if i < 3 { g.r[i] = rows; }
    }
    g.n = rows; Ok((g, arena))
}
fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn arr(g: &gguf::Gguf, k: &str) -> Result<Vec<u32>> {
    match g.kv.get(k) { Some(gguf::Value::Arr(v)) => Ok(v.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()), _ => bail!("missing array {k}") }
}

impl Model {
    pub fn load(path: &str, max_ctx: u32) -> Result<Self> {
        let ctx = Ctx::new()?;
        let wt = Weights::load(&ctx, path)?;
        let g = &wt.g;
        let arch = g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default();
        if arch != "nemotron_h" && arch != "nemotron_h_moe" { bail!("expected nemotron_h(_moe), got {arch}") }
        let a = arch.as_str();
        let u = |k: &str| g.u64(&format!("{a}.{k}")).map(|v| v as u32);
        let n_layer = u("block_count")?;
        let n_kv = arr(g, &format!("{a}.attention.head_count_kv"))?; let ffn = arr(g, &format!("{a}.feed_forward_length"))?;
        let kinds: Vec<Kind> = (0..n_layer as usize).map(|l| if n_kv[l] > 0 { Kind::Attn } else if ffn[l] > 0 { if wt.w.contains_key(&format!("blk.{l}.ffn_up_exps.weight")) { Kind::Moe } else { Kind::Ffn } } else { Kind::Mamba }).collect();
        let cfg = Cfg { n_layer, dim: u("embedding_length")?, n_head: u("attention.head_count")?, n_kv: n_kv.iter().copied().max().unwrap_or(0), hd: u("attention.key_length")?,
            ffn: ffn.iter().copied().max().unwrap_or(0), vocab: u("vocab_size")?, eps: g.f32(&format!("{a}.attention.layer_norm_rms_epsilon"))?,
            d_conv: u("ssm.conv_kernel")?, d_state: u("ssm.state_size")?, n_group: u("ssm.group_count")?, d_inner: u("ssm.inner_size")?, ssm_heads: u("ssm.time_step_rank")?,
            max_ctx, kinds, n_exp: u("expert_count").unwrap_or(0), top_k: u("expert_used_count").unwrap_or(0), ff_exp: u("expert_feed_forward_length").unwrap_or(0),
            w_scale: g.f32(&format!("{a}.expert_weights_scale")).unwrap_or(1.0), w_norm: g.get(&format!("{a}.expert_weights_norm")).ok().map(|v| matches!(v, gguf::Value::Bool(true))).unwrap_or(false) };
        if cfg.n_exp > 256 || cfg.top_k > 8 { bail!("router supports <= 256 experts, top-k <= 8") }
        if cfg.d_conv != 4 || cfg.d_state != 128 { bail!("kernels assume d_conv 4, d_state 128") }
        if cfg.ssm_hd() > 128 || cfg.ssm_heads % cfg.n_group != 0 { bail!("ssm head layout") }
        if n_kv.iter().any(|&v| v != 0 && v != cfg.n_kv) { bail!("per-layer kv head counts differ") }
        let (mut idx, mut cnt) = (vec![], [0u32; 4]);
        for k in &cfg.kinds { let i = *k as usize; idx.push(cnt[i]); cnt[i] += 1; }
        let f = |n: u64| ctx.buffer(n.max(64) * 4);
        let nc = env("ATTN_NC", 8);
        let kvn = (cnt[1] * cfg.max_ctx * cfg.n_kv * cfg.hd) as u64;
        let zxn = cfg.in_proj().max(cfg.ffn).max((cfg.n_head + 2 * cfg.n_kv) * cfg.hd) as u64;
        let sg = env("GEMV_SG", 32);
        let mut m = Model {
            gq: GkQ::new("GEMVQ", (8, 64), false, sg),
            embed: ctx.kernel(shaders::EMBED_ANY, 5, 24, &[], None)?, rms: ctx.kernel(shaders::RMSNORM, 3, 12, &[], None)?,
            conv: ctx.kernel(spv!("nh_conv"), 4, 20, &[], None)?, scan: ctx.kernel(spv!("nh_scan"), 5, 32, &[cfg.ssm_hd()], None)?,
            gnorm: ctx.kernel(spv!("nh_gnorm"), 2, 12, &[], None)?, relu2: ctx.kernel(spv!("nh_relu2"), 1, 4, &[], None)?, kv: ctx.kernel(spv!("nh_kv"), 4, 20, &[], None)?,
            prep: ctx.kernel(spv!("nh_prep"), 1, 0, &[], None)?,
            router: ctx.kernel(spv!("nh_router"), 4, 28, &[], Some(32))?, rlog: ctx.kernel(spv!("nh_rlog"), 3, 12, &[], Some(32))?, rp: f(256)?, moe_up: ctx.kernel(if env("MOE_LPR", 32) == 8 { spv!("nh_moe8") } else { spv!("nh_moe") }, 5, 24, &[0], None)?,
            moe_dn: ctx.kernel(if env("MOE_LPR", 32) == 8 { spv!("nh_moe8") } else { spv!("nh_moe") }, 5, 24, &[1], None)?, moe_rpw: 256 / env("MOE_LPR", 32),
            hm: f((cfg.top_k * cfg.ff_exp) as u64)?,
            attn: ctx.kernel(shaders::ATTN_SPLIT, 5, 36, &[cfg.hd], None)?, comb: ctx.kernel(shaders::ATTN_COMBINE, 3, 20, &[cfg.hd], None)?, nc,
            argmax: ctx.kernel(shaders::ARGMAX_W, 2, 4, &[], None)?, sample: ctx.kernel(shaders::SAMPLE_W, 2, 0, &[], None)?,
            x: f(cfg.dim as u64)?, xn: f(cfg.dim as u64)?, zx: f(zxn)?, xbc: f(cfg.conv_ch() as u64)?, yb: f(cfg.d_inner as u64)?, ao: f((cfg.n_head * cfg.hd) as u64)?,
            part: f((cfg.n_head * nc * (2 + cfg.hd)) as u64)?, logits: f(cfg.vocab as u64)?,
            kc: ctx.buffer((kvn * 2).max(256))?, vc: ctx.buffer((kvn * 2).max(256))?,
            cst: f((cnt[0] * cfg.conv_ch() * 3) as u64)?, sst: f(cnt[0] as u64 * (cfg.d_inner * cfg.d_state) as u64)?,
            toks: f(2 * cfg.max_ctx as u64 + 4)?, aux: f(256)?,
            idx, step: vk::CommandBuffer::null(), prof: Default::default(), plab: Default::default(), lab: std::cell::Cell::new(""), kl: std::cell::Cell::new(""), ds: DsCache::default(), cfg, wt, ctx };
        let mut err = Ok(());
        let step = m.ctx.record_raw(|cb| { err = m.record(cb); })?; err?;
        m.step = step;
        Ok(m)
    }
    fn set(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&self.ctx, k, bufs) }
    fn gemv(&self, cb: vk::CommandBuffer, segs: &[WRef], x: &Buffer, y: &Buffer, acc: bool) -> Result<()> {
        let (mut g, arena) = segs_q(segs, self.cfg.eps)?;
        if acc { g.flags |= 1; }
        let gk = self.gq.get(&self.ctx, &g, segs.len())?;
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xn])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    fn rmsnorm(&self, cb: vk::CommandBuffer, x: &Buffer, y: &Buffer, w: u32) -> Result<()> {
        let ds = self.set(&self.rms, &[x, y, &self.wt.f32b])?;
        self.ctx.dispatch(cb, &self.rms, ds, &push_u32(&[self.cfg.dim, w, self.cfg.eps.to_bits()]), 1); Ok(())
    }

    fn record(&self, cb: vk::CommandBuffer) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let bar = || { c.barrier(cb); self.tick(cb); };
        bar();
        self.lab.set("prep"); let ds = self.set(&self.prep, &[&self.aux])?; c.dispatch(cb, &self.prep, ds, &[], 1);
        let te = wt.wref("token_embd.weight")?;
        self.lab.set("embed"); let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &self.x, &self.aux, &self.toks, &wt.f32b])?;
        c.dispatch(cb, &self.embed, ds, &push_u32(&[te.off as u32, cfg.dim, qtype(te.ty)?, 160, 0, 0]), 1); bar();
        let (hd, ch) = (cfg.ssm_hd(), cfg.conv_ch());
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let i = self.idx[l as usize]; self.kl.set(match cfg.kinds[l as usize] { Kind::Mamba => "mamba", Kind::Attn => "attn", Kind::Moe => "moe", Kind::Ffn => "ffn" });
            self.lab.set("rms"); self.rmsnorm(cb, &self.x, &self.xn, wt.f(&n("attn_norm.weight"))?)?; bar();
            match cfg.kinds[l as usize] {
                Kind::Mamba => { self.kl.set("mamba");
                    self.lab.set("gemv"); self.gemv(cb, &[wt.wref(&n("ssm_in.weight"))?], &self.xn, &self.zx, false)?; bar();
                    self.lab.set("conv"); let ds = self.set(&self.conv, &[&self.zx, &self.cst, &self.xbc, &wt.f32b])?;
                    c.dispatch(cb, &self.conv, ds, &push_u32(&[ch, cfg.d_inner, i * ch * 3, wt.f(&n("ssm_conv1d.weight"))?, wt.f(&n("ssm_conv1d.bias"))?]), ch.div_ceil(64)); bar();
                    self.lab.set("scan"); let ds = self.set(&self.scan, &[&self.zx, &self.xbc, &self.sst, &self.yb, &wt.f32b])?;
                    c.dispatch(cb, &self.scan, ds, &push_u32(&[cfg.d_inner, cfg.n_group, cfg.ssm_heads, i * cfg.d_inner * cfg.d_state, wt.f(&n("ssm_a"))?, wt.f(&n("ssm_d"))?,
                        wt.f(&n("ssm_dt.bias"))?, 2 * cfg.d_inner + 2 * cfg.n_group * cfg.d_state]), cfg.ssm_heads); bar();
                    let _ = hd;
                    self.lab.set("gnorm"); let ds = self.set(&self.gnorm, &[&self.yb, &wt.f32b])?;
                    c.dispatch(cb, &self.gnorm, ds, &push_u32(&[cfg.d_inner / cfg.n_group, wt.f(&n("ssm_norm.weight"))?, cfg.eps.to_bits()]), cfg.n_group); bar();
                    self.lab.set("gemv"); self.gemv(cb, &[wt.wref(&n("ssm_out.weight"))?], &self.yb, &self.x, true)?; bar();
                }
                Kind::Attn => { self.kl.set("attn");
                    let (qn, kvn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
                    let cache_off = i * cfg.max_ctx * kvn;
                    self.lab.set("gemv"); self.gemv(cb, &[wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?], &self.xn, &self.zx, false)?; bar();
                    self.lab.set("kv"); let ds = self.set(&self.kv, &[&self.zx, &self.kc, &self.vc, &self.aux])?;
                    c.dispatch(cb, &self.kv, ds, &push_u32(&[kvn, qn, qn + kvn, cache_off, 160]), kvn.div_ceil(256)); bar();
                    self.lab.set("attn"); let ds = self.set(&self.attn, &[&self.zx, &self.kc, &self.vc, &self.part, &self.aux])?;
                    c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0]), cfg.n_head, self.nc, 1); bar();
                    self.lab.set("comb"); let ds = self.set(&self.comb, &[&self.part, &self.ao, &self.zx])?;
                    c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, 0, 0, 0, 0]), cfg.n_head); bar();
                    self.lab.set("gemv"); self.gemv(cb, &[wt.wref(&n("attn_output.weight"))?], &self.ao, &self.x, true)?; bar();
                }
                Kind::Moe => { self.kl.set("moe");
                    let ds = self.set(&self.rlog, &[&wt.f32b, &self.xn, &self.rp])?;
                    c.dispatch(cb, &self.rlog, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, cfg.dim, cfg.n_exp]), cfg.n_exp.div_ceil(8));
                    // shared expert up runs alongside the router logits
                    let up = wt.wref(&n("ffn_up_shexp.weight"))?;
                    self.lab.set("rlog+shup"); self.gemv(cb, &[up], &self.xn, &self.zx, false)?; bar();
                    let ds = self.set(&self.router, &[&wt.f32b, &self.xn, &self.aux, &self.rp])?;
                    c.dispatch(cb, &self.router, ds, &push_u32(&[0, wt.f(&n("exp_probs_b.bias"))?, cfg.dim, cfg.n_exp, cfg.top_k, cfg.w_norm as u32, cfg.w_scale.to_bits()]), 1);
                    self.lab.set("topk+relu2"); let ds = self.set(&self.relu2, &[&self.zx])?; c.dispatch(cb, &self.relu2, ds, &push_u32(&[up.n]), up.n.div_ceil(256)); bar();
                    let (ue, de) = (wt.wref(&n("ffn_up_exps.weight"))?, wt.wref(&n("ffn_down_exps.weight"))?);
                    self.lab.set("moe_up"); let ds = self.set(&self.moe_up, &[&wt.arenas[ue.arena], &self.xn, &self.hm, &wt.f32b, &self.aux])?;
                    c.dispatch3(cb, &self.moe_up, ds, &push_u32(&[ue.off as u32, (ue.bytes / ue.e as u64) as u32, ue.n, ue.k, qtype(ue.ty)?, cfg.top_k]), ue.n.div_ceil(self.moe_rpw), cfg.top_k, 1); bar();
                    self.lab.set("shdown"); self.gemv(cb, &[wt.wref(&n("ffn_down_shexp.weight"))?], &self.zx, &self.x, true)?; bar();
                    self.lab.set("moe_dn"); let ds = self.set(&self.moe_dn, &[&wt.arenas[de.arena], &self.hm, &self.x, &wt.f32b, &self.aux])?;
                    c.dispatch(cb, &self.moe_dn, ds, &push_u32(&[de.off as u32, (de.bytes / de.e as u64) as u32, de.n, de.k, qtype(de.ty)?, cfg.top_k]), de.n.div_ceil(self.moe_rpw)); bar();
                }
                Kind::Ffn => { self.kl.set("ffn");
                    let up = wt.wref(&n("ffn_up.weight"))?;
                    self.lab.set("gemv"); self.gemv(cb, &[up], &self.xn, &self.zx, false)?; bar();
                    self.lab.set("relu2"); let ds = self.set(&self.relu2, &[&self.zx])?; c.dispatch(cb, &self.relu2, ds, &push_u32(&[up.n]), up.n.div_ceil(256)); bar();
                    self.lab.set("gemv"); self.gemv(cb, &[wt.wref(&n("ffn_down.weight"))?], &self.zx, &self.x, true)?; bar();
                }
            }
        }
        self.lab.set("rms"); self.kl.set("head"); self.rmsnorm(cb, &self.x, &self.xn, wt.f("output_norm.weight")?)?; bar();
        let out = wt.wref("output.weight").or_else(|_| wt.wref("token_embd.weight"))?;
        self.lab.set("gemv"); self.gemv(cb, &[out], &self.xn, &self.logits, false)?; bar();
        self.lab.set("argmax"); let ds = self.set(&self.argmax, &[&self.logits, &self.aux])?; c.dispatch(cb, &self.argmax, ds, &push_u32(&[cfg.vocab]), 64); bar();
        self.lab.set("sample"); let ds = self.set(&self.sample, &[&self.aux, &self.toks])?; c.dispatch(cb, &self.sample, ds, &[], 1); bar();
        let _ = GgmlType::F32;
        Ok(())
    }

    fn tick(&self, cb: vk::CommandBuffer) {
        if let Some(qp) = self.prof.get() { let mut v = self.plab.borrow_mut(); let i = v.len() as u32;
            unsafe { self.ctx.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::COMPUTE_SHADER, qp, i) }
            v.push(format!("{}.{}", self.kl.get(), self.lab.get())); }
    }
    /// One decode step with a timestamp after every barrier; returns (label, seconds) per segment (state advances one token).
    pub fn profile(&self) -> Result<Vec<(String, f64)>> {
        let d = &self.ctx.device; let nq = 2048u32;
        let qp = unsafe { d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(nq), None)? };
        self.plab.borrow_mut().clear(); self.kl.set("embed"); self.lab.set("start");
        let mut err = Ok(());
        self.ctx.run(|cb| { unsafe { d.cmd_reset_query_pool(cb, qp, 0, nq) }; self.prof.set(Some(qp)); self.tick(cb); err = self.record(cb); self.prof.set(None); })?; err?;
        let labs = self.plab.borrow().clone(); let mut ts = vec![0u64; labs.len()];
        unsafe { d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; d.destroy_query_pool(qp, None); }
        let p = self.ctx.info.ts_period_ns as f64 * 1e-9;
        Ok((1..labs.len()).map(|i| (labs[i].clone(), ts[i].wrapping_sub(ts[i - 1]) as f64 * p)).collect())
    }
    pub(crate) fn reset(&self) { self.cst.bytes_mut().fill(0); self.sst.bytes_mut().fill(0); }
    pub(crate) fn au(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.aux.ptr as *mut u32, 256) } }
    pub(crate) fn tk(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.toks.ptr as *mut u32, self.toks.size as usize / 4) } }
    /// Run `n` steps, at most `chunk` per submit batch (keeps every GPU submission short). Returns GPU seconds.
    fn steps(&self, n: usize) -> Result<f64> {
        let chunk = env("STEP_CHUNK", 64) as usize; let mut t = 0.0; let mut left = n;
        while left > 0 { let k = left.min(chunk); t += self.ctx.submit_repeated(self.step, k)?; left -= k; }
        Ok(t)
    }

    /// Greedy generation fully on the GPU (prompt processed token by token). Returns (ids, prompt GPU s, prompt wall s, decode GPU s, decode wall s, steps).
    pub fn generate(&self, prompt: &[u32], n_new: usize) -> Result<(Vec<u32>, f64, f64, f64, f64, usize)> {
        if prompt.len() + n_new > self.cfg.max_ctx as usize { bail!("context too long") }
        self.reset();
        let (toks, au) = (self.tk(), self.au());
        toks[..prompt.len()].copy_from_slice(prompt);
        au[160] = 0; au[161] = prompt.len() as u32; au[162] = 0;
        let t = std::time::Instant::now(); let pg = self.steps(prompt.len())?; let pw = t.elapsed().as_secs_f64();
        let st = n_new.saturating_sub(1);
        let t = std::time::Instant::now(); let dg = self.steps(st)?; let dw = t.elapsed().as_secs_f64();
        Ok((toks[prompt.len()..prompt.len() + n_new].to_vec(), pg, pw, dg, dw, st))
    }
    /// Batched prefill of prompt[..L-1] (crates/prefill GEMMs + chunked scan), then greedy decode on the GPU.
    /// Returns (ids, prefill GPU s, prefill wall s, decode GPU s, decode wall s, steps); prefill time covers L-1 tokens.
    pub fn generate_pf(&self, s: &prefill::Scratch, prompt: &[u32], n_new: usize) -> Result<(Vec<u32>, f64, f64, f64, f64, usize)> {
        if prompt.len() + n_new > self.cfg.max_ctx as usize { bail!("context too long") }
        let t = std::time::Instant::now(); let pg = prefill::prefill(s, self, prompt)?; let pw = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now(); let dg = self.steps(n_new)?; let dw = t.elapsed().as_secs_f64();
        let toks = self.tk(); let l = prompt.len();
        Ok((toks[l..l + n_new].to_vec(), pg, pw, dg, dw, n_new))
    }
    /// Teacher-forced: argmax prediction after each position of `seq`.
    pub fn predict_all(&self, seq: &[u32]) -> Result<Vec<u32>> {
        let mc = self.cfg.max_ctx as usize;
        if seq.len() > mc { bail!("sequence too long") }
        self.reset();
        let (toks, au) = (self.tk(), self.au());
        toks[..seq.len()].copy_from_slice(seq);
        au[160] = 0; au[161] = seq.len() as u32 + 1; au[162] = (mc + 1) as u32;
        self.steps(seq.len())?;
        au[162] = 0;
        Ok(toks[mc + 2..mc + 2 + seq.len()].to_vec())
    }
}
