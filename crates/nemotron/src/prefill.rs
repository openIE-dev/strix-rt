//! Batched prompt processing for Nemotron-H: quantized x f16 coop-matrix GEMMs from crates/prefill (Pf::gemm) plus
//! chunked Mamba-2 kernels (batched conv, register-resident chunk scan, grouped norm), batched causal attention and
//! squared-ReLU. Runs prompt[..L-1] in chunks of PF_M tokens and leaves the conv/SSM state and KV cache exactly where
//! the token-by-token path would; the decode step then consumes the last prompt token.
use crate::{Kind, Model};
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
    r: Buffer, g: Buffer, gt: Buffer, gt2: Buffer, ag: Buffer, yg: Buffer, hg: Buffer, yd: Buffer, psel: Kernel, psort: Kernel,
    conv: Kernel, scan: Kernel, gnorm: Kernel, relu: Kernel, kv: Kernel, attn: Kernel, embed: Kernel, ds: DsCache }

impl Scratch {
    pub fn new(m: &Model, mmax: u32) -> Result<Self> {
        let (c, ctx) = (&m.cfg, &m.ctx);
        let lda = c.dim.max(c.d_inner).max(c.ffn).max(c.n_head * c.hd).next_multiple_of(8);
        let ldy = c.in_proj().max(c.ffn).max((c.n_head + 2 * c.n_kv) * c.hd);
        let rows = (mmax + 256) as u64;
        let nk = (mmax * c.top_k.max(1)) as u64; let fe = c.ff_exp.max(64) as u64;
        let maxt = nk / 16 + c.n_exp as u64 + 1;
        Ok(Scratch { pf: Pf::new(ctx)?, mmax, lda, ldy,
            r: ctx.buffer(2 * nk * 4)?, g: ctx.buffer(2 * nk * 4)?, gt: ctx.buffer((1 + 3 * maxt) * 4)?, gt2: ctx.buffer((1 + 3 * maxt) * 4)?,
            ag: ctx.buffer((nk + 256) * c.dim as u64 * 2)?, yg: ctx.buffer(nk * fe * 4)?, hg: ctx.buffer((nk + 256) * fe * 2)?, yd: ctx.buffer(nk * c.dim as u64 * 4)?,
            psel: ctx.kernel(spv!("nh_psel"), 3, 36, &[psel_tb(c.dim)], Some(32))?, psort: ctx.kernel(spv!("nh_psort"), 4, 20, &[], None)?,
            x: ctx.buffer(rows * c.dim as u64 * 4)?, a: ctx.buffer(rows * lda as u64 * 2)?, y: ctx.buffer(rows * ldy as u64 * 4)?,
            xbc: ctx.buffer(mmax as u64 * c.conv_ch() as u64 * 4)?, yb: ctx.buffer(mmax as u64 * c.d_inner as u64 * 4)?,
            conv: ctx.kernel(spv!("nh_pconv"), 4, 28, &[], None)?, scan: ctx.kernel(spv!("nh_pscan"), 5, 44, &[c.ssm_hd(), scan_ps(c.ssm_hd())], None)?,
            gnorm: ctx.kernel(spv!("nh_pgnorm"), 3, 20, &[], None)?, relu: ctx.kernel(spv!("nh_prelu"), 2, 12, &[], None)?,
            kv: ctx.kernel(spv!("nh_pkv"), 3, 24, &[], None)?, attn: ctx.kernel(spv!("nh_pattn"), 4, 28, &[c.hd], None)?,
            embed: ctx.kernel(shaders::EMBED_ANY, 5, 24, &[], None)?, ds: DsCache::default() })
    }
}

/// Head-dim split of the prefill scan (NH_PSPLIT, default 8 when it divides the head size).
fn scan_ps(hd: u32) -> u32 { let p: u32 = std::env::var("NH_PSPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(8); if p >= 1 && hd % p == 0 { p } else { 1 } }

/// Tokens per router workgroup (NH_PSEL_TB, default 4 when dim <= 3072, else 1; max 8 = subgroups).
fn psel_tb(dim: u32) -> u32 { let t: u32 = std::env::var("NH_PSEL_TB").ok().and_then(|v| v.parse().ok()).unwrap_or(4); if t >= 1 && t <= 8 && dim <= 12288 / t && (t == 1 || t.is_power_of_two()) { t } else { 1 } }

fn gemm(s: &Scratch, m: &Model, cb: vk::CommandBuffer, w: WRef, n: u32, y_col: u32, ldy: u32, y: &Buffer, acc: bool) -> Result<()> {
    s.pf.gemm(&m.ctx, cb, &m.wt.arenas[w.arena], &m.wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: s.lda, a_row0: 0, m: n, y, ldy, y_col, acc })
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
                let (d, tk, nk, fe) = (c.dim, c.top_k, n * c.top_k, c.ff_exp);
                if d > 4096 { bail!("router kernel stages <= 4096 inputs") }
                lab("e.psel+shup"); if sk & 16 == 0 {
                let ds = set(&s.psel, &[&wt.f32b, &s.a, &s.r])?;
                ctx.dispatch(cb, &s.psel, ds, &push_u32(&[wt.f(&nm("ffn_gate_inp.weight"))?, wt.f(&nm("exp_probs_b.bias"))?, d, c.n_exp, tk, c.w_norm as u32, c.w_scale.to_bits(), s.lda, n]), n.div_ceil(psel_tb(c.dim)));
                }
                let su = wt.wref(&nm("ffn_up_shexp.weight"))?;
                if sk & 64 == 0 { gemm(s, m, cb, su, n, 0, s.ldy, &s.y, false)?; bar(); }
                let qm = std::env::var("PF_QMM").map(|v| v == "1").unwrap_or(false);
                let gm: u32 = if qm { prefill::Pf::qgemm_rows() } else { std::env::var("PFG_M").ok().and_then(|v| v.parse().ok()).unwrap_or(32) };
                let hyb: u32 = if qm { 0 } else { std::env::var("PFG_HYB").ok().and_then(|v| v.parse().ok()).unwrap_or(0) };
                lab("e.sort"); if sk & 16 == 0 {
                let ds = set(&s.psort, &[&s.r, &s.g, &s.gt, &s.gt2])?; ctx.dispatch(cb, &s.psort, ds, &push_u32(&[nk, tk, c.n_exp, gm & 0xff, hyb]), 1); bar();
                lab("e.gather"); s.pf.run(ctx, "gather", &[&s.a, &s.ag, &s.g], &[d / 2, s.lda / 2, d / 2, nk], (d / 2 * nk).div_ceil(256), cb)?; bar();
                }
                lab("e.up_exps"); if sk & 64 == 0 { let ds = set(&s.relu, &[&s.y, &s.a])?; ctx.dispatch3(cb, &s.relu, ds, &push_u32(&[su.n, s.ldy, s.lda]), su.n.div_ceil(256), n, 1); }
                let (ue, de) = (wt.wref(&nm("ffn_up_exps.weight"))?, wt.wref(&nm("ffn_down_exps.weight"))?);
                let maxt = nk / (gm & 0xff) + c.n_exp + 1;
                if sk & 32 == 0 { if qm { s.pf.qgemmh_grouped(ctx, cb, &wt.arenas[ue.arena], ue, &s.gt, maxt, &s.ag, d, &s.yg, fe, 0)?; } else { s.pf.gemm_grouped(ctx, cb, &wt.arenas[ue.arena], &wt.f32b, ue, &s.gt, gm, maxt, &s.ag, d, &s.yg, fe, 0)?;; if hyb > 0 { s.pf.gemm_grouped(ctx, cb, &wt.arenas[ue.arena], &wt.f32b, ue, &s.gt2, 16, c.n_exp + 1, &s.ag, d, &s.yg, fe, 0)?; } } bar(); }
                lab("e.dsh+relu2"); if sk & 64 == 0 { gemm(s, m, cb, wt.wref(&nm("ffn_down_shexp.weight"))?, n, 0, d, &s.x, true)?; }
                let ds = set(&s.relu, &[&s.yg, &s.hg])?; ctx.dispatch3(cb, &s.relu, ds, &push_u32(&[fe, fe, fe]), fe.div_ceil(256), nk, 1); bar();
                lab("e.down_exps"); if sk & 32 == 0 { if qm { s.pf.qgemmh_grouped(ctx, cb, &wt.arenas[de.arena], de, &s.gt, maxt, &s.hg, fe, &s.yd, d, 0)?; } else { s.pf.gemm_grouped(ctx, cb, &wt.arenas[de.arena], &wt.f32b, de, &s.gt, gm, maxt, &s.hg, fe, &s.yd, d, 0)?;; if hyb > 0 { s.pf.gemm_grouped(ctx, cb, &wt.arenas[de.arena], &wt.f32b, de, &s.gt2, 16, c.n_exp + 1, &s.hg, fe, &s.yd, d, 0)?; } } bar(); }
                lab("e.comb"); if sk & 16 == 0 { s.pf.run(ctx, "moecomb", &[&s.x, &s.yd, &s.r, &s.g], &[d, n, tk], (d * n).div_ceil(256), cb)?; bar(); }
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
