//! Batched prompt processing for Nemotron 3.5 Super (LatentMoE, Q4_0), fork of crates/nemotron/src/prefill.rs: quantized x f16 coop-matrix GEMMs from crates/prefill (Pf::gemm) plus
//! chunked Mamba-2 kernels (batched conv, register-resident chunk scan, grouped norm), batched causal attention and
//! squared-ReLU. Runs prompt[..L-1] in chunks of PF_M tokens and leaves the conv/SSM state and KV cache exactly where
//! the token-by-token path would; the decode step then consumes the last prompt token.
use crate::{Kind, Model};
use gguf as _;
use anyhow::{bail, Result};
use ash::vk;
use prefill::{GemmOp, Pf};
use rtcore::{qtype, DsCache, WRef};
use vkrt::{push_u32, shaders, Buffer, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }

thread_local! { static PQ: std::cell::Cell<Option<vk::QueryPool>> = std::cell::Cell::new(None); static PL: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(vec![]); static CUR: std::cell::Cell<&'static str> = std::cell::Cell::new("start"); }
fn lab(x: &'static str) { CUR.with(|c| c.set(x)) }
fn tick(ctx: &vkrt::Ctx, cb: vk::CommandBuffer) { if let Some(qp) = PQ.with(|q| q.get()) { PL.with(|l| { let mut l = l.borrow_mut(); let i = l.len() as u32; unsafe { ctx.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::COMPUTE_SHADER, qp, i) } l.push(CUR.with(|c| c.get()).to_string()); }) } }
/// Profile one chunk of n tokens at pos0 with a timestamp after every barrier; per-label (summed) seconds. State advances (garbage-free: same work as prefill).
pub fn profile_chunk(s: &Scratch, m: &Model, pos0: u32, n: u32) -> Result<Vec<(String, f64, u32)>> {
    let d = &m.ctx.device; let nq = 4096u32;
    let qp = unsafe { d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(nq), None)? };
    PL.with(|l| l.borrow_mut().clear()); lab("start"); let mut err = None;
    m.ctx.run(|cb| { unsafe { d.cmd_reset_query_pool(cb, qp, 0, nq) }; PQ.with(|q| q.set(Some(qp))); tick(&m.ctx, cb); if let Err(e) = record_chunk(s, m, cb, pos0, n) { err = Some(e) } PQ.with(|q| q.set(None)); })?;
    if let Some(e) = err { return Err(e) }
    let labs = PL.with(|l| l.borrow().clone()); let mut ts = vec![0u64; labs.len()];
    unsafe { d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; d.destroy_query_pool(qp, None); }
    let pr = m.ctx.info.ts_period_ns as f64 * 1e-9; let mut agg: Vec<(String, f64, u32)> = vec![];
    for i in 1..labs.len() { let t = ts[i].wrapping_sub(ts[i - 1]) as f64 * pr; if let Some(a) = agg.iter_mut().find(|a| a.0 == labs[i]) { a.1 += t; a.2 += 1 } else { agg.push((labs[i].clone(), t, 1)) } }
    Ok(agg)
}

pub struct Scratch { pub pf: Pf, pub mmax: u32, lda: u32, ldy: u32, x: Buffer, a: Buffer, y: Buffer, xbc: Buffer, yb: Buffer,
    r: Buffer, g: Buffer, gt: Buffer, ag: Buffer, yg: Buffer, hg: Buffer, yd: Buffer, yl: Buffer, al: Buffer, rl: Buffer, rw: Buffer, rwoff: Vec<u64>, psel2: Kernel, psel: Kernel, psort: Kernel, pcomb: Kernel,
    pub hseed: Option<Buffer>, hcap: Kernel, conv: Kernel, scan: Kernel, gnorm: Kernel, relu: Kernel, kv: Kernel, attn: Kernel, embed: Kernel, ds: DsCache,
    /// image embeddings [cnt][dim] f32 replacing the (placeholder) token rows ip..ip+cnt (set by the VL runner), and the row-copy kernel
    pub img: std::cell::RefCell<Option<(Buffer, u32, u32)>>, icopy: Kernel }

impl Scratch {
    pub fn new(m: &Model, mmax: u32) -> Result<Self> {
        let (c, ctx) = (&m.cfg, &m.ctx);
        let lda = c.dim.max(c.d_inner).max(c.ffn).max(c.n_head * c.hd).next_multiple_of(8);
        let ldy = c.in_proj().max(c.ffn).max((c.n_head + 2 * c.n_kv) * c.hd);
        let rows = (mmax + 256) as u64;
        let nk = (mmax * c.top_k.max(1)) as u64; let fe = c.ff_exp.max(64) as u64; let lat = c.latent.max(64) as u64;
        let maxt = nk / 16 + c.n_exp as u64 + 1;
        // router weights (BF16 in the GGUF, f32 in wt.f32b) as exact f16 for the GEMM router: 512 x dim per MoE layer
        let nmoe = c.kinds.iter().filter(|k| **k == Kind::Moe).count() as u64;
        let rw = ctx.buffer((nmoe * c.n_exp as u64 * c.dim as u64 * 2).max(256))?; let mut rwoff = vec![0u64; c.n_layer as usize];
        { let dst = unsafe { std::slice::from_raw_parts_mut(rw.ptr as *mut u16, rw.size as usize / 2) }; let src = m.wt.f32b.as_f32(); let mut o = 0usize;
          for l in 0..c.n_layer as usize { if c.kinds[l] != Kind::Moe { continue }
            let f = m.wt.f(&format!("blk.{l}.ffn_gate_inp.weight"))? as usize; let cnt = (c.n_exp * c.dim) as usize; rwoff[l] = (o * 2) as u64;
            for i in 0..cnt { dst[o + i] = half::f16::from_f32(src[f + i]).to_bits(); } o += cnt; } }
        let hseed = if std::env::var("MTP").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1) > 0 { Some(ctx.buffer((c.max_ctx * c.dim) as u64 * 4)?) } else { None };
        Ok(Scratch { pf: Pf::new(ctx)?, hseed, hcap: ctx.kernel(spv!("n35_hcap"), 3, 16, &[], None)?, rw, rwoff, mmax, lda, ldy,
            r: ctx.buffer(2 * nk * 4)?, g: ctx.buffer(2 * nk * 4)?, gt: ctx.buffer((1 + 3 * maxt) * 4)?,
            ag: ctx.buffer((nk + 256) * lat * 2)?, yg: ctx.buffer(nk * fe * 4)?, hg: ctx.buffer((nk + 256) * fe * 2)?, yd: ctx.buffer(nk * lat * 4)?, yl: ctx.buffer(rows * lat * 4)?, rl: ctx.buffer(rows * 512 * 4)?, psel2: ctx.kernel(spv!("n35_psel2"), 3, 28, &[], Some(32))?, al: ctx.buffer(rows * lat * 2)?,
            psel: ctx.kernel(spv!("n35_psel"), 3, 36, &[psel_tb(c.dim)], Some(32))?, psort: ctx.kernel(spv!("n35_psort"), 3, 16, &[], None)?, pcomb: ctx.kernel(spv!("n35_pcomb"), 4, 16, &[], None)?,
            x: ctx.buffer(rows * c.dim as u64 * 4)?, a: ctx.buffer(rows * lda as u64 * 2)?, y: ctx.buffer(rows * ldy as u64 * 4)?,
            xbc: ctx.buffer(mmax as u64 * c.conv_ch() as u64 * 4)?, yb: ctx.buffer(mmax as u64 * c.d_inner as u64 * 4)?,
            conv: ctx.kernel(spv!("nh_pconv"), 4, 28, &[], None)?, scan: ctx.kernel(spv!("nh_pscan"), 5, 44, &[c.ssm_hd(), scan_ps(c.ssm_hd())], None)?,
            gnorm: ctx.kernel(spv!("nh_pgnorm"), 3, 20, &[], None)?, relu: ctx.kernel(spv!("nh_prelu"), 2, 12, &[], None)?,
            kv: ctx.kernel(spv!("nh_pkv"), 3, 24, &[], None)?, attn: ctx.kernel(spv!("nh_pattn"), 4, 28, &[c.hd], None)?,
            embed: ctx.kernel(shaders::EMBED_ANY, 5, 24, &[], None)?, ds: DsCache::default(), img: std::cell::RefCell::new(None), icopy: ctx.kernel(spv!("n35_icopy"), 2, 12, &[], None)? })
    }
}

/// Head-dim split of the prefill scan (NH_PSPLIT, default 8 when it divides the head size).
fn scan_ps(hd: u32) -> u32 { let p: u32 = std::env::var("NH_PSPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(8); if p >= 1 && hd % p == 0 { p } else { 1 } }

/// Tokens per router workgroup (NH_PSEL_TB, default 4 when dim <= 3072, else 1; max 8 = subgroups).
fn psel_tb(dim: u32) -> u32 { let t: u32 = std::env::var("N35_PSEL_TB").ok().and_then(|v| v.parse().ok()).unwrap_or(2); if t >= 1 && t <= 8 && dim <= 4096 && (t == 1 || t.is_power_of_two()) { t } else { 1 } }

fn gemm(s: &Scratch, m: &Model, cb: vk::CommandBuffer, w: WRef, n: u32, y_col: u32, ldy: u32, y: &Buffer, acc: bool) -> Result<()> {
    s.pf.gemm(&m.ctx, cb, &m.wt.arenas[w.arena], &m.wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: s.lda, a_row0: 0, m: n, y, ldy, y_col, acc })
}

fn gemm_lat(s: &Scratch, m: &Model, cb: vk::CommandBuffer, w: WRef, n: u32) -> Result<()> {
    s.pf.gemm(&m.ctx, cb, &m.wt.arenas[w.arena], &m.wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: s.lda, a_row0: 0, m: n, y: &s.x, ldy: m.cfg.dim, y_col: 0, acc: true })
}

fn record_chunk(s: &Scratch, m: &Model, cb: vk::CommandBuffer, pos0: u32, n: u32) -> Result<()> {
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    let bar = || { ctx.barrier(cb); tick(ctx, cb); };
    // NH_SKIP bitmask (profiling only, output is garbage): 1 mamba gemms, 2 conv+scan+gnorm, 4 attn kv+attn, 8 attn gemms, 16 router/sort/gather/comb, 32 expert gemms, 64 shexp gemms+relu, 128 dense ffn, 256 rmsnorm
    let sk: u32 = std::env::var("NH_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let set = |k: &Kernel, b: &[&Buffer]| s.ds.get(ctx, k, b);
    let te = wt.wref("token_embd.weight")?;
    let ds = set(&s.embed, &[&wt.arenas[te.arena], &s.x, &m.aux, &m.toks, &wt.f32b])?;
    for t in 0..n { ctx.dispatch(cb, &s.embed, ds, &push_u32(&[te.off as u32, c.dim, qtype(te.ty)?, 160, t, t * c.dim]), 1); }
    bar();
    if let Some((eb, ip, cnt)) = &*s.img.borrow() {
        let (lo, hi) = (pos0.max(*ip), (pos0 + n).min(ip + cnt));
        if lo < hi { let ds = set(&s.icopy, &[eb, &s.x])?;
            ctx.dispatch(cb, &s.icopy, ds, &push_u32(&[(hi - lo) * c.dim, (lo - ip) * c.dim, (lo - pos0) * c.dim]), ((hi - lo) * c.dim).div_ceil(256)); bar(); }
    }
    let ch = c.conv_ch();
    for l in 0..c.n_layer {
        let nm = |x: &str| format!("blk.{l}.{x}");
        let i = m.idx[l as usize];
        lab("norm"); if sk & 256 == 0 { s.pf.run(ctx, "rmsnorm", &[&s.x, &s.a, &wt.f32b], &[c.dim, s.lda, wt.f(&nm("attn_norm.weight"))?, c.eps.to_bits(), 0], n, cb)?; bar(); }
        match c.kinds[l as usize] {
            Kind::Mamba => {
                lab("m.in"); if sk & 1 == 0 { gemm(s, m, cb, wt.wref(&nm("ssm_in.weight"))?, n, 0, s.ldy, &s.y, false)?; bar(); }
                if sk & 2 == 0 {
                lab("m.conv"); let ds = set(&s.conv, &[&s.y, &m.cst, &s.xbc, &wt.f32b])?;
                if sk & 512 == 0 { ctx.dispatch(cb, &s.conv, ds, &push_u32(&[ch, c.d_inner, i * ch * 3, wt.f(&nm("ssm_conv1d.weight"))?, wt.f(&nm("ssm_conv1d.bias"))?, n, s.ldy]), ch.div_ceil(64)); bar(); }
                lab("m.scan"); let ds = set(&s.scan, &[&s.y, &s.xbc, &m.sst, &s.yb, &wt.f32b])?;
                if sk & 1024 == 0 { ctx.dispatch(cb, &s.scan, ds, &push_u32(&[c.d_inner, c.n_group, c.ssm_heads, i * c.d_inner * c.d_state, wt.f(&nm("ssm_a"))?, wt.f(&nm("ssm_d"))?,
                    wt.f(&nm("ssm_dt.bias"))?, 2 * c.d_inner + 2 * c.n_group * c.d_state, n, s.ldy, ch]), c.ssm_heads * scan_ps(c.ssm_hd())); bar(); }
                lab("m.gnorm"); let ds = set(&s.gnorm, &[&s.yb, &s.a, &wt.f32b])?;
                if sk & 2048 == 0 { ctx.dispatch3(cb, &s.gnorm, ds, &push_u32(&[c.d_inner / c.n_group, wt.f(&nm("ssm_norm.weight"))?, c.eps.to_bits(), c.d_inner, s.lda]), c.n_group, n, 1); bar(); }
                }
                lab("m.out"); if sk & 1 == 0 { gemm(s, m, cb, wt.wref(&nm("ssm_out.weight"))?, n, 0, c.dim, &s.x, true)?; bar(); }
            }
            Kind::Attn => {
                let (qn, kvn) = (c.n_head * c.hd, c.n_kv * c.hd);
                let cache_off = i * c.max_ctx * kvn;
                lab("a.qkv"); if sk & 8 == 0 {
                gemm(s, m, cb, wt.wref(&nm("attn_q.weight"))?, n, 0, s.ldy, &s.y, false)?;
                gemm(s, m, cb, wt.wref(&nm("attn_k.weight"))?, n, qn, s.ldy, &s.y, false)?;
                gemm(s, m, cb, wt.wref(&nm("attn_v.weight"))?, n, qn + kvn, s.ldy, &s.y, false)?; bar(); }
                if sk & 4 == 0 {
                lab("a.kv"); let ds = set(&s.kv, &[&s.y, &m.kc, &m.vc])?;
                ctx.dispatch3(cb, &s.kv, ds, &push_u32(&[kvn, qn, qn + kvn, cache_off, pos0, s.ldy]), kvn.div_ceil(256), n, 1); bar();
                lab("a.attn"); let ds = set(&s.attn, &[&s.y, &m.kc, &m.vc, &s.a])?;
                ctx.dispatch3(cb, &s.attn, ds, &push_u32(&[c.n_head, c.n_kv, cache_off, pos0, s.ldy, (1.0f32 / (c.hd as f32).sqrt()).to_bits(), s.lda]), c.n_head, n, 1); bar();
                }
                lab("a.o"); if sk & 8 == 0 { gemm(s, m, cb, wt.wref(&nm("attn_output.weight"))?, n, 0, c.dim, &s.x, true)?; bar(); }
            }
            Kind::Moe => {
                let (d, tk, nk, fe, lat) = (c.dim, c.top_k, n * c.top_k, c.ff_exp, c.latent);
                if d > 4096 { bail!("router kernel stages <= 4096 inputs") }
                lab("e.psel"); if sk & 16 == 0 {
                if std::env::var("N35_PSEL").map(|v| v == "1").unwrap_or(false) {
                    let ds = set(&s.psel, &[&wt.f32b, &s.a, &s.r])?;
                    ctx.dispatch(cb, &s.psel, ds, &push_u32(&[wt.f(&nm("ffn_gate_inp.weight"))?, wt.f(&nm("exp_probs_b.bias"))?, d, c.n_exp, tk, c.w_norm as u32, c.w_scale.to_bits(), s.lda, n]), n.div_ceil(psel_tb(c.dim)));
                } else {
                    let gw = WRef { arena: 0, off: s.rwoff[l as usize], ty: gguf::GgmlType::F16, k: d, n: c.n_exp, e: 1, bytes: (c.n_exp * d * 2) as u64 };
                    s.pf.gemm(&m.ctx, cb, &s.rw, &m.wt.f32b, &GemmOp { w: gw, row0: 0, rows: c.n_exp, a: &s.a, lda: s.lda, a_row0: 0, m: n, y: &s.rl, ldy: 512, y_col: 0, acc: false })?; bar();
                    let ds = set(&s.psel2, &[&wt.f32b, &s.rl, &s.r])?;
                    ctx.dispatch(cb, &s.psel2, ds, &push_u32(&[wt.f(&nm("exp_probs_b.bias"))?, c.n_exp, tk, c.w_norm as u32, c.w_scale.to_bits(), 512, n]), n.div_ceil(8));
                }
                } bar(); lab("e.shup");
                let su = wt.wref(&nm("ffn_up_shexp.weight"))?;
                if sk & 64 == 0 { gemm(s, m, cb, su, n, 0, s.ldy, &s.y, false)?; bar(); }
                // latent down-projection of xn -> f32 yl [n][lat] -> f16 al
                lab("e.latdown"); if sk & 16 == 0 {
                s.pf.gemm(&m.ctx, cb, &m.wt.arenas[wt.wref(&nm("ffn_latent_down.weight"))?.arena], &m.wt.f32b, &GemmOp { w: wt.wref(&nm("ffn_latent_down.weight"))?, row0: 0, rows: lat, a: &s.a, lda: s.lda, a_row0: 0, m: n, y: &s.yl, ldy: lat, y_col: 0, acc: false })?; bar();
                s.pf.run(ctx, "f2h", &[&s.yl, &s.al], &[lat, lat, n], (lat * n).div_ceil(256), cb)?; }
                bar();
                let gm: u32 = std::env::var("PFG_M").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
                lab("e.sort"); if sk & 16 == 0 {
                let ds = set(&s.psort, &[&s.r, &s.g, &s.gt])?; ctx.dispatch(cb, &s.psort, ds, &push_u32(&[nk, tk, c.n_exp, gm & 0xff]), 1); bar();
                lab("e.gather"); s.pf.run(ctx, "gather", &[&s.al, &s.ag, &s.g], &[lat / 2, lat / 2, lat / 2, nk], (lat / 2 * nk).div_ceil(256), cb)?; bar();
                }
                lab("e.up_exps"); if sk & 64 == 0 { let ds = set(&s.relu, &[&s.y, &s.a])?; ctx.dispatch3(cb, &s.relu, ds, &push_u32(&[su.n, s.ldy, s.lda]), su.n.div_ceil(256), n, 1); }
                let (ue, de) = (wt.wref(&nm("ffn_up_exps.weight"))?, wt.wref(&nm("ffn_down_exps.weight"))?);
                let maxt = nk / (gm & 0xff) + c.n_exp + 1;
                if sk & 32 == 0 { s.pf.gemm_grouped(ctx, cb, &wt.arenas[ue.arena], &wt.f32b, ue, &s.gt, gm, maxt, &s.ag, lat, &s.yg, fe, 0)?; bar(); }
                lab("e.dsh+relu2"); if sk & 64 == 0 { gemm(s, m, cb, wt.wref(&nm("ffn_down_shexp.weight"))?, n, 0, d, &s.x, true)?; }
                let ds = set(&s.relu, &[&s.yg, &s.hg])?; ctx.dispatch3(cb, &s.relu, ds, &push_u32(&[fe, fe, fe]), fe.div_ceil(256), nk, 1); bar();
                lab("e.down_exps"); if sk & 32 == 0 { s.pf.gemm_grouped(ctx, cb, &wt.arenas[de.arena], &wt.f32b, de, &s.gt, gm, maxt, &s.hg, fe, &s.yd, lat, 0)?; bar(); }
                lab("e.comb"); if sk & 16 == 0 {
                let ds = set(&s.pcomb, &[&s.a, &s.yd, &s.r, &s.g])?; ctx.dispatch(cb, &s.pcomb, ds, &push_u32(&[lat, n, tk, s.lda]), (lat * n).div_ceil(256)); bar();
                lab("e.latup"); gemm_lat(s, m, cb, wt.wref(&nm("ffn_latent_up.weight"))?, n)?; bar(); }
            }
            Kind::Ffn => {
                let up = wt.wref(&nm("ffn_up.weight"))?; lab("f.up");
                gemm(s, m, cb, up, n, 0, s.ldy, &s.y, false)?; bar();
                lab("f.relu"); let ds = set(&s.relu, &[&s.y, &s.a])?;
                ctx.dispatch3(cb, &s.relu, ds, &push_u32(&[up.n, s.ldy, s.lda]), up.n.div_ceil(256), n, 1); bar();
                lab("f.down"); gemm(s, m, cb, wt.wref(&nm("ffn_down.weight"))?, n, 0, c.dim, &s.x, true)?; bar();
            }
        }
    }
    if let Some(hs) = &s.hseed { let ds = set(&s.hcap, &[&s.x, hs, &wt.f32b])?; ctx.dispatch(cb, &s.hcap, ds, &push_u32(&[c.dim, wt.f("output_norm.weight")?, c.eps.to_bits(), pos0]), n); ctx.barrier(cb); }
    Ok(())
}

/// Prefill prompt[..L-1]; afterwards Model::decode_from continues at position L-1. Returns GPU seconds.
pub fn prefill(s: &Scratch, m: &Model, prompt: &[u32]) -> Result<f64> {
    let l = prompt.len(); if l < 2 || l + 1 > m.cfg.max_ctx as usize { bail!("prompt length") }
    m.reset();
    let (toks, au) = (m.tk(), m.au());
    toks[..l].copy_from_slice(prompt);
    let (mut pos, mut gpu) = (0usize, 0.0);
    while pos < l - 1 {
        let n = (l - 1 - pos).min(s.mmax as usize);
        au[160] = pos as u32;
        let mut err = None;
        gpu += m.ctx.run(|cb| { if let Err(e) = record_chunk(s, m, cb, pos as u32, n as u32) { err = Some(e) } })?;
        if let Some(e) = err { return Err(e) }
        pos += n;
    }
    au[160] = (l - 1) as u32; au[161] = l as u32; au[162] = 0;
    Ok(gpu)
}
