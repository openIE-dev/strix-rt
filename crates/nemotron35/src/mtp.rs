//! Two-column verify step for Nemotron 3.5 Super (speculative decoding): tokens A (pos) and B (pos+1) go through the
//! whole trunk in one command buffer. Dense GEMVs run as 2-column gemvq (weights read once), the LatentMoE experts run
//! per column (union of both columns' routed experts), conv/scan run column after column. Column 0 updates the
//! Mamba state in place; column 1 reads the post-column-0 state and writes the *alternate* state buffers, so a rejected
//! draft needs no rollback and an accepted one just swaps the roles of the two state sets (`flip`).
//! aux words: col1 expert ids/weights at 384/448 (AOFF 128), col1 position word 163, argmax results at 500/501.
use crate::*;
use vkrt::Ctx;
macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }

pub struct Buf2 { pub full: Buffer, pub v: [Buffer; 2], pub cs: u32 }
fn view(ctx: &Ctx, b: &Buffer, off: u64, size: u64) -> Result<Buffer> {
    let d = &ctx.device;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let buf = unsafe { d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)? };
    let req = unsafe { d.get_buffer_memory_requirements(buf) };
    if off % req.alignment != 0 { bail!("view offset {off} not aligned to {}", req.alignment) }
    unsafe { d.bind_buffer_memory(buf, b.mem, off)? };
    Ok(Buffer { buf, mem: b.mem, size, ptr: unsafe { b.ptr.add(off as usize) }, flags: b.flags })
}
fn buf2(ctx: &Ctx, floats: u64) -> Result<Buf2> {
    let st = (floats * 4).max(256).next_multiple_of(65536);
    let full = ctx.buffer(2 * st)?;
    let v = [view(ctx, &full, 0, st)?, view(ctx, &full, st, st)?];
    Ok(Buf2 { full, v, cs: (st / 4) as u32 })
}

pub struct V2 { x: Buf2, xn: Buf2, zx: Buf2, xbc: Buf2, yb: Buf2, ao: Buf2, part: Buf2, hm: Buf2, lat: Buf2, lat2: Buf2, logits: Buf2, prb: Buf2,
    cst2: Buffer, sst2: Buffer, conv2: Kernel, scan2: Kernel, router1: Kernel, moe_up1: Kernel, moe_dn1: Kernel, argmax_b: Kernel, pub step: [vk::CommandBuffer; 2] }

impl Model {
    pub fn v2_new(&self) -> Result<V2> {
        let (ctx, cfg) = (&self.ctx, &self.cfg);
        let zxn = cfg.in_proj().max(cfg.ffn).max((cfg.n_head + 2 * cfg.n_kv) * cfg.hd) as u64;
        let nc = self.nc;
        let mut v = V2 { x: buf2(ctx, cfg.dim as u64)?, xn: buf2(ctx, cfg.dim as u64)?, zx: buf2(ctx, zxn)?, xbc: buf2(ctx, cfg.conv_ch() as u64)?, yb: buf2(ctx, cfg.d_inner as u64)?,
            ao: buf2(ctx, (cfg.n_head * cfg.hd) as u64)?, part: buf2(ctx, (cfg.n_head * nc * (2 + cfg.hd)) as u64)?, hm: buf2(ctx, (cfg.top_k * cfg.ff_exp) as u64)?,
            lat: buf2(ctx, cfg.latent as u64)?, lat2: buf2(ctx, cfg.latent as u64)?, logits: buf2(ctx, cfg.vocab as u64)?, prb: buf2(ctx, 512)?,
            cst2: ctx.buffer(self.cst.size)?, sst2: ctx.buffer(self.sst.size)?,
            conv2: ctx.kernel(spv!("n35_conv2"), 5, 20, &[], None)?, scan2: ctx.kernel(spv!("n35_scan2"), 6, 32, &[cfg.ssm_hd()], None)?,
            router1: ctx.kernel(spv!("n35_router"), 4, 28, &[1, 128], Some(32))?, moe_up1: ctx.kernel(spv!("n35_moe"), 5, 24, &[0, 128], None)?, moe_dn1: ctx.kernel(spv!("n35_moe"), 5, 24, &[1, 128], None)?,
            argmax_b: ctx.kernel(shaders::ARGMAX_B, 2, 16, &[], None)?, step: [vk::CommandBuffer::null(); 2] };
        for f in 0..2 { let mut err = Ok(()); v.step[f] = ctx.record_raw(|cb| { err = self.record_v2(&v, cb, f == 1); })?; err?; }
        Ok(v)
    }
    fn gemv2(&self, cb: vk::CommandBuffer, segs: &[WRef], x: &Buf2, y: &Buf2, acc: bool) -> Result<()> {
        let (mut g, arena) = segs_q(segs, self.cfg.eps)?;
        if acc { g.flags |= 1; }
        g.flags |= 64; g.x_cs = x.cs; g.y_cs = y.cs;
        let gk = self.gq.get_nb(&self.ctx, &g, segs.len(), 2)?;
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], &x.full, &y.full, &self.wt.f32b, &self.aux, &self.xn])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    fn record_v2(&self, v: &V2, cb: vk::CommandBuffer, flip: bool) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let bar = || c.barrier(cb);
        let (cst_b, sst_b, cst_a, sst_a) = if flip { (&v.cst2, &v.sst2, &self.cst, &self.sst) } else { (&self.cst, &self.sst, &v.cst2, &v.sst2) };
        let te = wt.wref("token_embd.weight")?;
        let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &v.x.full, &self.aux, &self.toks, &wt.f32b])?;
        for col in 0..2u32 { c.dispatch(cb, &self.embed, ds, &push_u32(&[te.off as u32, cfg.dim, qtype(te.ty)?, 160, col, col * v.x.cs]), 1); }
        bar();
        let ch = cfg.conv_ch();
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let i = self.idx[l as usize];
            for col in 0..2usize {
                let ds = self.set(&self.rms, &[&v.x.v[col], &v.xn.v[col], &wt.f32b])?;
                c.dispatch(cb, &self.rms, ds, &push_u32(&[cfg.dim, wt.f(&n("attn_norm.weight"))?, cfg.eps.to_bits()]), 1);
            }
            bar();
            match cfg.kinds[l as usize] {
                Kind::Mamba => {
                    self.gemv2(cb, &[wt.wref(&n("ssm_in.weight"))?], &v.xn, &v.zx, false)?; bar();
                    let cp = |col: usize| -> Result<Vec<u8>> { let _ = col; Ok(push_u32(&[ch, cfg.d_inner, i * ch * 3, wt.f(&n("ssm_conv1d.weight"))?, wt.f(&n("ssm_conv1d.bias"))?])) };
                    let sp = push_u32(&[cfg.d_inner, cfg.n_group, cfg.ssm_heads, i * cfg.d_inner * cfg.d_state, wt.f(&n("ssm_a"))?, wt.f(&n("ssm_d"))?, wt.f(&n("ssm_dt.bias"))?, 2 * cfg.d_inner + 2 * cfg.n_group * cfg.d_state]);
                    let gp = push_u32(&[cfg.d_inner / cfg.n_group, wt.f(&n("ssm_norm.weight"))?, cfg.eps.to_bits()]);
                    // col 0: in place on the base state
                    let ds = self.set(&self.conv, &[&v.zx.v[0], cst_b, &v.xbc.v[0], &wt.f32b])?; c.dispatch(cb, &self.conv, ds, &cp(0)?, ch.div_ceil(64)); bar();
                    let ds = self.set(&self.scan, &[&v.zx.v[0], &v.xbc.v[0], sst_b, &v.yb.v[0], &wt.f32b])?; c.dispatch(cb, &self.scan, ds, &sp, cfg.ssm_heads);
                    // col 1 conv reads the post-col-0 conv state (conv col 0 finished at the barrier above), writes the alternate set
                    let ds = self.set(&v.conv2, &[&v.zx.v[1], cst_b, &v.xbc.v[1], &wt.f32b, cst_a])?; c.dispatch(cb, &v.conv2, ds, &cp(1)?, ch.div_ceil(64)); bar();
                    let ds = self.set(&v.scan2, &[&v.zx.v[1], &v.xbc.v[1], sst_b, &v.yb.v[1], &wt.f32b, sst_a])?; c.dispatch(cb, &v.scan2, ds, &sp, cfg.ssm_heads);
                    for col in 0..2 { let ds = self.set(&self.gnorm, &[&v.yb.v[col], &wt.f32b])?; if col == 0 { c.dispatch(cb, &self.gnorm, ds, &gp, cfg.n_group); } }
                    bar();
                    let ds = self.set(&self.gnorm, &[&v.yb.v[1], &wt.f32b])?; c.dispatch(cb, &self.gnorm, ds, &gp, cfg.n_group); bar();
                    self.gemv2(cb, &[wt.wref(&n("ssm_out.weight"))?], &v.yb, &v.x, true)?; bar();
                }
                Kind::Attn => {
                    let (qn, kvn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
                    let cache_off = i * cfg.max_ctx * kvn;
                    self.gemv2(cb, &[wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?], &v.xn, &v.zx, false)?; bar();
                    for col in 0..2usize { let ds = self.set(&self.kv, &[&v.zx.v[col], &self.kc, &self.vc, &self.aux])?;
                        c.dispatch(cb, &self.kv, ds, &push_u32(&[kvn, qn, qn + kvn, cache_off, if col == 0 { 160 } else { 163 }]), kvn.div_ceil(256)); }
                    bar();
                    for col in 0..2usize { let ds = self.set(&self.attn, &[&v.zx.v[col], &self.kc, &self.vc, &v.part.v[col], &self.aux])?;
                        c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, self.nc, cfg.hd, if col == 0 { 160 } else { 163 }, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0]), cfg.n_head, self.nc, 1); }
                    bar();
                    for col in 0..2usize { let ds = self.set(&self.comb, &[&v.part.v[col], &v.ao.v[col], &v.zx.v[col]])?; c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, 0, 0, 0, 0]), cfg.n_head); }
                    bar();
                    self.gemv2(cb, &[wt.wref(&n("attn_output.weight"))?], &v.ao, &v.x, true)?; bar();
                }
                Kind::Moe => {
                    for col in 0..2usize {
                        let gw = wt.f(&n("ffn_gate_inp.weight"))?;
                        let ds = self.set(&self.rlog, &[&wt.f32b, &v.xn.v[col], &v.prb.v[col]])?;
                        c.dispatch(cb, &self.rlog, ds, &push_u32(&[gw, cfg.dim, cfg.n_exp]), cfg.n_exp.div_ceil(8)); }
                    bar();
                    for col in 0..2usize {
                        let k = if col == 0 { &self.router } else { &v.router1 };
                        let ds = self.set(k, &[&wt.f32b, &v.xn.v[col], &self.aux, &v.prb.v[col]])?;
                        c.dispatch(cb, k, ds, &push_u32(&[wt.f(&n("ffn_gate_inp.weight"))?, wt.f(&n("exp_probs_b.bias"))?, cfg.dim, cfg.n_exp, cfg.top_k, cfg.w_norm as u32, cfg.w_scale.to_bits()]), 1);
                    }
                    let up = wt.wref(&n("ffn_up_shexp.weight"))?;
                    self.gemv2(cb, &[up], &v.xn, &v.zx, false)?;
                    for col in 0..2usize { self.lgemv(cb, wt.wref(&n("ffn_latent_down.weight"))?, &v.xn.v[col], &v.lat.v[col], false)?; }
                    bar();
                    for col in 0..2usize { let ds = self.set(&self.relu2, &[&v.zx.v[col]])?; c.dispatch(cb, &self.relu2, ds, &push_u32(&[up.n]), up.n.div_ceil(256)); }
                    let (ue, de) = (wt.wref(&n("ffn_up_exps.weight"))?, wt.wref(&n("ffn_down_exps.weight"))?);
                    for col in 0..2usize { let k = if col == 0 { &self.moe_up } else { &v.moe_up1 };
                        let ds = self.set(k, &[&wt.arenas[ue.arena], &v.lat.v[col], &v.hm.v[col], &wt.f32b, &self.aux])?;
                        c.dispatch3(cb, k, ds, &push_u32(&[ue.off as u32, (ue.bytes / ue.e as u64) as u32, ue.n, ue.k, qtype(ue.ty)?, cfg.top_k]), ue.n.div_ceil(self.moe_rpw), cfg.top_k, 1); }
                    bar();
                    self.gemv2(cb, &[wt.wref(&n("ffn_down_shexp.weight"))?], &v.zx, &v.x, true)?;
                    for col in 0..2usize { let k = if col == 0 { &self.moe_dn } else { &v.moe_dn1 };
                        let ds = self.set(k, &[&wt.arenas[de.arena], &v.hm.v[col], &v.lat2.v[col], &wt.f32b, &self.aux])?;
                        c.dispatch(cb, k, ds, &push_u32(&[de.off as u32, (de.bytes / de.e as u64) as u32, de.n, de.k, qtype(de.ty)?, cfg.top_k]), de.n.div_ceil(self.moe_rpw)); }
                    bar();
                    for col in 0..2usize { self.lgemv(cb, wt.wref(&n("ffn_latent_up.weight"))?, &v.lat2.v[col], &v.x.v[col], true)?; }
                    bar();
                }
                Kind::Ffn => bail!("Ffn blocks not supported in the verify path"),
            }
        }
        for col in 0..2usize { let ds = self.set(&self.rms, &[&v.x.v[col], &v.xn.v[col], &wt.f32b])?; c.dispatch(cb, &self.rms, ds, &push_u32(&[cfg.dim, wt.f("output_norm.weight")?, cfg.eps.to_bits()]), 1); }
        bar();
        let out = wt.wref("output.weight").or_else(|_| wt.wref("token_embd.weight"))?;
        self.gemv2(cb, &[out], &v.xn, &v.logits, false)?; bar();
        let ds = self.set(&v.argmax_b, &[&v.logits.full, &self.aux])?; c.dispatch(cb, &v.argmax_b, ds, &push_u32(&[cfg.vocab, v.logits.cs, 500, 0]), 2);
        Ok(())
    }
    /// Run one verify cycle on tokens (a at pos, b at pos+1). Returns (argmax after a, argmax after b, GPU seconds).
    pub fn v2_step(&self, v: &V2, a: u32, b: u32, pos: u32, flip: bool) -> Result<(u32, u32, f64)> {
        let (toks, au) = (self.tk(), self.au());
        toks[pos as usize] = a; toks[pos as usize + 1] = b; au[160] = pos; au[163] = pos + 1;
        let t = self.ctx.submit_repeated(v.step[flip as usize], 1)?;
        let au = unsafe { std::slice::from_raw_parts(self.aux.ptr as *const u32, 512) };
        Ok((au[500], au[501], t))
    }
    pub fn v2_time(&self, v: &V2, reps: usize) -> Result<f64> { Ok(self.ctx.submit_repeated(v.step[0], reps)? / reps as f64) }
}

// ---------------------------------------------------------------------------------------------------------------
// MTP draft head (blk.88 of the mtp GGUF): c = [rms(emb(tok)) * enorm | rms(h) * hnorm] -> eh_proj -> NoPE attention
// (own KV cache, positions counted per head call) -> LatentMoE + shared expert -> LayerNorm -> lm_head (Q4_0).
// The draft only steers speculation: any numeric difference to llama.cpp costs acceptance, never exactness.
pub struct Head { wt: Weights, eb: Buffer, cat: Buffer, cat_v: [Buffer; 2], cur: Buffer, xn: Buffer, qkv: Buffer, qv: [Buffer; 3], ao: Buffer, part: Buffer, hm: Buffer, lat: Buffer, lat2: Buffer,
    prb: Buffer, gate32: Buffer, shz: Buffer, hn: Buffer, logits: Buffer, kc: Buffer, vc: Buffer, ln: Kernel, am: Kernel, step: [vk::CommandBuffer; 8], hbuf: Buffer, hrow: Kernel, pub hpos: u32 }

impl Model {
    fn lgemv_w(&self, wt: &Weights, cb: vk::CommandBuffer, w: WRef, x: &Buffer, y: &Buffer, acc: bool) -> Result<()> {
        if w.off > u32::MAX as u64 { bail!("weight offset") }
        let ds = self.set(&self.lgemv, &[&wt.arenas[w.arena], x, y, &wt.f32b, &self.aux])?;
        self.ctx.dispatch(cb, &self.lgemv, ds, &push_u32(&[w.off as u32, 0, w.n, w.k, qtype(w.ty)?, acc as u32]), w.n.div_ceil(self.moe_rpw)); Ok(())
    }
    pub fn head_new(&self, v: &V2, s: &prefill::Scratch, mtp_path: &str) -> Result<Head> {
        let (ctx, cfg) = (&self.ctx, &self.cfg);
        let wt = Weights::load(ctx, mtp_path)?;
        let f = |n: u64| ctx.buffer((n * 4).max(256));
        let cat = f(2 * cfg.dim as u64)?; let qkv = f((cfg.n_head * cfg.hd + 2 * cfg.n_kv * cfg.hd) as u64)?;
        let (qn, kvn) = ((cfg.n_head * cfg.hd) as u64, (cfg.n_kv * cfg.hd) as u64);
        let cat_v = [view(ctx, &cat, 0, cfg.dim as u64 * 4)?, view(ctx, &cat, cfg.dim as u64 * 4, cfg.dim as u64 * 4)?];
        let qv = [view(ctx, &qkv, 0, qn * 4)?, view(ctx, &qkv, qn * 4, kvn * 4)?, view(ctx, &qkv, (qn + kvn) * 4, kvn * 4)?];
        let kvb = (cfg.max_ctx * cfg.n_kv * cfg.hd) as u64 * 2;
        let mut hd = Head { wt, eb: f(cfg.dim as u64)?, cat, cat_v, cur: f(cfg.dim as u64)?, xn: f(cfg.dim as u64)?, qkv, qv, ao: f(qn)?, part: f((cfg.n_head * self.nc * (2 + cfg.hd)) as u64)?,
            hm: f((cfg.top_k * cfg.ff_exp) as u64)?, lat: f(cfg.latent as u64)?, lat2: f(cfg.latent as u64)?, prb: f(512)?, gate32: ctx.buffer((cfg.n_exp * cfg.dim) as u64 * 4)?, shz: f(8192)?, hn: f(cfg.dim as u64)?, logits: f(cfg.vocab as u64)?,
            kc: ctx.buffer(kvb.max(256))?, vc: ctx.buffer(kvb.max(256))?, ln: ctx.kernel(spv!("n35_ln"), 3, 12, &[], None)?, am: ctx.kernel(shaders::ARGMAX_B, 2, 16, &[], None)?, step: [vk::CommandBuffer::null(); 8], hbuf: f(cfg.dim as u64)?, hrow: ctx.kernel(spv!("n35_hrow"), 3, 4, &[], None)?, hpos: 0 };
        hd.kc.bytes_mut().fill(0); hd.vc.bytes_mut().fill(0);
        { // router weights are BF16 in the MTP GGUF: expand to f32 once (rlog reads f32)
            let t = hd.wt.g.tensor("blk.88.ffn_gate_inp.weight").ok_or_else(|| anyhow::anyhow!("no blk.88.ffn_gate_inp.weight"))?; let raw = hd.wt.g.data(t);
            if raw.len() != (cfg.n_exp * cfg.dim * 2) as usize { bail!("unexpected router tensor size {}", raw.len()) }
            let dst = hd.gate32.as_f32_mut(); for (i, c) in raw.chunks_exact(2).enumerate() { dst[i] = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16); } }
        // h sources: 0 = model.x (plain decode), 1/2 = verify columns 0/1 (pre-norm hidden) ; 3 = model.xn (post-norm, H_MODE=1 probe)
        let hm = env("H_MODE", 1);
        let srcs: [&Buffer; 4] = if hm == 0 { [&self.x, &v.x.v[0], &v.x.v[1], &self.x] } else { [&self.xn, &v.xn.v[0], &v.xn.v[1], &self.xn] };
        for i in 0..3 { let mut err = Ok(()); hd.step[i] = ctx.record_raw(|cb| { err = self.record_head(&hd, cb, srcs[i], false, None); })?; err?; }
        for i in 0..3 { let mut err = Ok(()); hd.step[4 + i] = ctx.record_raw(|cb| { err = self.record_head(&hd, cb, srcs[i], true, None); })?; err?; }
        { let mut err = Ok(()); hd.step[7] = ctx.record_raw(|cb| { err = self.record_head(&hd, cb, &hd.hbuf, true, s.hseed.as_ref()); })?; err?; }
        Ok(hd)
    }
    fn record_head(&self, hd: &Head, cb: vk::CommandBuffer, hsrc: &Buffer, fill: bool, seed: Option<&Buffer>) -> Result<()> {
        let (c, cfg, wt, w2) = (&self.ctx, &self.cfg, &self.wt, &hd.wt);
        let bar = || c.barrier(cb);
        let nm = |s: &str| format!("blk.88.{s}");
        let (qn, kvn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
        if let Some(sb) = seed { let ds = self.set(&hd.hrow, &[sb, &hd.hbuf, &self.aux])?; c.dispatch(cb, &hd.hrow, ds, &push_u32(&[cfg.dim]), 1); bar(); }
        let te = wt.wref("token_embd.weight")?;
        let ds = self.set(&self.embed, &[&wt.arenas[te.arena], &hd.eb, &self.aux, &self.toks, &wt.f32b])?;
        c.dispatch(cb, &self.embed, ds, &push_u32(&[te.off as u32, cfg.dim, qtype(te.ty)?, 165, 1, 0]), 1); bar();
        let rms = |x: &Buffer, y: &Buffer, w: u32| -> Result<()> { let ds = self.set(&self.rms, &[x, y, &w2.f32b])?; c.dispatch(cb, &self.rms, ds, &push_u32(&[cfg.dim, w, cfg.eps.to_bits()]), 1); Ok(()) };
        rms(&hd.eb, &hd.cat_v[0], w2.f(&nm("nextn.enorm.weight"))?)?; rms(hsrc, &hd.cat_v[1], w2.f(&nm("nextn.hnorm.weight"))?)?; bar();
        self.lgemv_w(w2, cb, w2.wref(&nm("nextn.eh_proj.weight"))?, &hd.cat, &hd.cur, false)?; bar();
        rms(&hd.cur, &hd.xn, w2.f(&nm("attn_norm.weight"))?)?; bar();
        if !fill { self.lgemv_w(w2, cb, w2.wref(&nm("attn_q.weight"))?, &hd.xn, &hd.qv[0], false)?; }
        self.lgemv_w(w2, cb, w2.wref(&nm("attn_k.weight"))?, &hd.xn, &hd.qv[1], false)?;
        self.lgemv_w(w2, cb, w2.wref(&nm("attn_v.weight"))?, &hd.xn, &hd.qv[2], false)?; bar();
        let ds = self.set(&self.kv, &[&hd.qkv, &hd.kc, &hd.vc, &self.aux])?; c.dispatch(cb, &self.kv, ds, &push_u32(&[kvn, qn, qn + kvn, 0, 164]), kvn.div_ceil(256)); bar();
        if fill { return Ok(()) }
        let ds = self.set(&self.attn, &[&hd.qkv, &hd.kc, &hd.vc, &hd.part, &self.aux])?;
        c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, 0, self.nc, cfg.hd, 164, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0]), cfg.n_head, self.nc, 1); bar();
        let ds = self.set(&self.comb, &[&hd.part, &hd.ao, &hd.qkv])?; c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, 0, 0, 0, 0]), cfg.n_head); bar();
        self.lgemv_w(w2, cb, w2.wref(&nm("attn_output.weight"))?, &hd.ao, &hd.cur, true)?; bar();
        rms(&hd.cur, &hd.xn, w2.f(&nm("post_attention_norm.weight"))?)?; bar();
        let ds = self.set(&self.rlog, &[&hd.gate32, &hd.xn, &hd.prb])?; c.dispatch(cb, &self.rlog, ds, &push_u32(&[0, cfg.dim, cfg.n_exp]), cfg.n_exp.div_ceil(8)); bar();
        let ds = self.set(&self.router, &[&w2.f32b, &hd.xn, &self.aux, &hd.prb])?;
        c.dispatch(cb, &self.router, ds, &push_u32(&[0, w2.f(&nm("exp_probs_b.bias"))?, cfg.dim, cfg.n_exp, cfg.top_k, cfg.w_norm as u32, cfg.w_scale.to_bits()]), 1);
        let up = w2.wref(&nm("ffn_up_shexp.weight"))?;
        self.lgemv_w(w2, cb, up, &hd.xn, &hd.shz, false)?; self.lgemv_w(w2, cb, w2.wref(&nm("ffn_latent_down.weight"))?, &hd.xn, &hd.lat, false)?; bar();
        let ds = self.set(&self.relu2, &[&hd.shz])?; c.dispatch(cb, &self.relu2, ds, &push_u32(&[up.n]), up.n.div_ceil(256));
        let (ue, de) = (w2.wref(&nm("ffn_up_exps.weight"))?, w2.wref(&nm("ffn_down_exps.weight"))?);
        let ds = self.set(&self.moe_up, &[&w2.arenas[ue.arena], &hd.lat, &hd.hm, &w2.f32b, &self.aux])?;
        c.dispatch3(cb, &self.moe_up, ds, &push_u32(&[ue.off as u32, (ue.bytes / ue.e as u64) as u32, ue.n, ue.k, qtype(ue.ty)?, cfg.top_k]), ue.n.div_ceil(self.moe_rpw), cfg.top_k, 1); bar();
        self.lgemv_w(w2, cb, w2.wref(&nm("ffn_down_shexp.weight"))?, &hd.shz, &hd.cur, true)?;
        let ds = self.set(&self.moe_dn, &[&w2.arenas[de.arena], &hd.hm, &hd.lat2, &w2.f32b, &self.aux])?;
        c.dispatch(cb, &self.moe_dn, ds, &push_u32(&[de.off as u32, (de.bytes / de.e as u64) as u32, de.n, de.k, qtype(de.ty)?, cfg.top_k]), de.n.div_ceil(self.moe_rpw)); bar();
        self.lgemv_w(w2, cb, w2.wref(&nm("ffn_latent_up.weight"))?, &hd.lat2, &hd.cur, true)?; bar();
        let ds = self.set(&hd.ln, &[&hd.cur, &hd.hn, &w2.f32b])?; c.dispatch(cb, &hd.ln, ds, &push_u32(&[cfg.dim, w2.f(&nm("nextn.shared_head_norm.weight"))?, cfg.eps.to_bits()]), 1); bar();
        self.lgemv_w(w2, cb, w2.wref("output.weight")?, &hd.hn, &hd.logits, false)?; bar();
        let ds = self.set(&hd.am, &[&hd.logits, &self.aux])?; c.dispatch(cb, &hd.am, ds, &push_u32(&[cfg.vocab, cfg.vocab, 502, 0]), 1);
        Ok(())
    }
}

pub struct MtpStats { pub seed_s: f64, pub cycles: u32, pub accepted: u32, pub verify_s: f64, pub head_s: f64, pub wall_s: f64, pub head_calls: u32 }
impl Model {
    fn head_run(&self, hd: &mut Head, src: usize, q: u32) -> Result<(u32, f64)> {
        let au = self.au(); au[164] = hd.hpos; au[165] = q; hd.hpos += 1;
        let t = self.ctx.submit_repeated(hd.step[src], 1)?;
        let a = unsafe { std::slice::from_raw_parts(self.aux.ptr as *const u32, 512) }; Ok((a[502], t))
    }
    fn head_fill(&self, hd: &mut Head, src: usize, q: u32) -> Result<f64> {
        let au = self.au(); au[164] = hd.hpos; au[165] = q; au[166] = q; hd.hpos += 1;
        self.ctx.submit_repeated(hd.step[if src == 99 { 7 } else { 4 + src }], 1)
    }
    /// Greedy generation with MTP speculation (k = 1 draft per verify cycle). Output equals plain greedy when the numerics match (N35_DET=1 on both sides).
    pub fn mtp_generate(&self, s: &prefill::Scratch, v: &V2, hd: &mut Head, prompt: &[u32], n_new: usize) -> Result<(Vec<u32>, MtpStats)> {
        let l = prompt.len(); if l + n_new + 4 > self.cfg.max_ctx as usize { bail!("context too long") }
        let t0 = std::time::Instant::now();
        prefill::prefill(s, self, prompt)?;
        hd.hpos = 0; hd.kc.bytes_mut().fill(0); hd.vc.bytes_mut().fill(0);
        let mut seed_s = 0.0;
        if env("SEED", 1) > 0 && s.hseed.is_some() { for q in 0..(l as u32 - 1) { seed_s += self.head_fill(hd, 99, q)?; } }
        self.steps(1)?;
        let toks = self.tk(); let t1 = toks[l];
        let mut st = MtpStats { seed_s, cycles: 0, accepted: 0, verify_s: 0.0, head_s: 0.0, wall_s: 0.0, head_calls: 0 };
        let mut out = vec![t1];
        let (mut d, ht) = self.head_run(hd, 0, (l - 1) as u32)?; st.head_s += ht; st.head_calls += 1;
        let (mut a, mut p, mut flip) = (t1, l as u32, false);
        while out.len() < n_new {
            let (c0, c1, vt) = self.v2_step(v, a, d, p, flip)?; st.verify_s += vt; st.cycles += 1;
            if c0 == d {
                st.accepted += 1; out.push(c0); out.push(c1); toks[p as usize + 2] = c1;
                let h1 = if env("CHEAP", 1) > 0 { self.head_fill(hd, 1, p)? } else { self.head_run(hd, 1, p)?.1 }; let (nd, h2) = self.head_run(hd, 2, p + 1)?;
                st.head_s += h1 + h2; st.head_calls += 2; d = nd; a = c1; p += 2; flip = !flip;
            } else {
                out.push(c0); toks[p as usize + 1] = c0;
                let (nd, h1) = self.head_run(hd, 1, p)?; st.head_s += h1; st.head_calls += 1; d = nd; a = c0; p += 1;
            }
        }
        out.truncate(n_new); st.wall_s = t0.elapsed().as_secs_f64();
        Ok((out, st))
    }
}
