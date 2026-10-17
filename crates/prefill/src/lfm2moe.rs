//! Batched prefill for LFM2-MoE (lfm2moe): short-conv / GQA-attention mixers, dense lead FFN, and a grouped MoE FFN.
//! MoE: a router kernel picks top-k experts per token; the routing is read back (one sync per MoE layer), tokens are
//! grouped by expert on the CPU, gathered into contiguous rows, and each expert runs a row-range GEMM on its group
//! (gate, up -> SwiGLU -> down), followed by a weighted combine into the residual. Leaves kc/vc and the conv state
//! where the token-by-token path would and hands decode toks[L] / aux[160..162].
use crate::{GemmOp, Pf};
use anyhow::{bail, Result};
use ash::vk;
use gguf::dequant::dequant_row;
use lfm2::Model;
use rtcore::WRef;
use vkrt::Buffer;

pub struct Lf { pub mmax: u32, x: Buffer, a: Buffer, al: u32, y: Buffer, ldy: u32, ag: Buffer, yg: Buffer, hg: Buffer, yd: Buffer, r: Buffer, rw: Buffer, g: Buffer, gt: Buffer, gt2: Buffer,
    attn_idx: Vec<u32>, conv_idx: Vec<u32>, nkv: u32 }

impl Lf {
    pub fn new(m: &Model, mmax: u32) -> Result<Self> {
        let (c, ctx) = (&m.cfg, &m.ctx);
        if c.hd != 64 { bail!("hd {}", c.hd) }
        if crate::env("PF_PROF", 0) > 0 { eprintln!("lfm2moe: layers {} dim {} heads {} kv {:?} ffn {} ff_exp {} experts {} top_k {} dense_lead {} vocab {}",
            c.n_layer, c.dim, c.n_head, c.n_kv, c.ffn, c.ff_exp, c.n_exp, c.top_k, c.dense_lead, c.vocab); }
        let nkv = c.n_kv.iter().copied().max().unwrap_or(0);
        let al = c.dim.max(c.ffn).max(c.n_head * 64).next_multiple_of(8);
        let ldy = (3 * c.dim).max((c.n_head + 2 * nkv) * 64).max(2 * c.ffn);
        let (mm, mk, fe) = (mmax as u64, (mmax * c.top_k) as u64, c.ff_exp as u64);
        let (mut attn_idx, mut conv_idx, mut na, mut nc) = (vec![], vec![], 0, 0);
        for l in 0..c.n_layer as usize { if c.n_kv[l] > 0 { attn_idx.push(na); conv_idx.push(u32::MAX); na += 1 } else { attn_idx.push(u32::MAX); conv_idx.push(nc); nc += 1 } }
        // router weights (f32 in the model) as f16 [layer][n_exp][dim] so routing runs as a coop-matrix GEMM
        let rw = ctx.buffer((c.n_layer * c.n_exp * c.dim) as u64 * 2)?;
        { let f = m.wt.f32b.as_f32(); let h = unsafe { std::slice::from_raw_parts_mut(rw.ptr as *mut half::f16, (c.n_layer * c.n_exp * c.dim) as usize) };
          for l in c.dense_lead..c.n_layer { let o = m.wt.f(&format!("blk.{l}.ffn_gate_inp.weight"))? as usize; let ne = (c.n_exp * c.dim) as usize;
              for i in 0..ne { h[l as usize * ne + i] = half::f16::from_f32(f[o + i]); } } }
        Ok(Lf { mmax, rw, x: ctx.buffer(mm * c.dim as u64 * 4)?, a: ctx.buffer((mm + 256) * al as u64 * 2)?, al, y: ctx.buffer(mm * ldy as u64 * 4)?, ldy,
            ag: ctx.buffer((mk + 256) * c.dim as u64 * 2)?, yg: ctx.buffer(mk * 2 * fe * 4)?, hg: ctx.buffer((mk + 256) * fe * 2)?, yd: ctx.buffer(mk * c.dim as u64 * 4)?,
            r: ctx.buffer(mk * 8 + mm * 64 * 4)?, g: ctx.buffer(mk * 8)?, gt: ctx.buffer(((1 + 3 * (mk + c.n_exp as u64)) * 4).max(256))?, gt2: ctx.buffer(((1 + 3 * (mk + c.n_exp as u64)) * 4).max(256))?, attn_idx, conv_idx, nkv })
    }
}

fn expert(w: WRef, e: u32) -> WRef { let eb = w.bytes / w.e.max(1) as u64; WRef { off: w.off + e as u64 * eb, bytes: eb, e: 1, ..w } }
fn grid(n: u32) -> u32 { n.div_ceil(256) }

/// Mixer + FFN norm (+ dense FFN or router) of layer l on n tokens at pos0.
fn record_layer(pf: &Pf, m: &Model, s: &Lf, cb: vk::CommandBuffer, l: usize, n: u32, pos0: u32, normed: bool) -> Result<()> {
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    let nm = |x: &str| format!("blk.{l}.{x}");
    let skip = crate::env("PF_SKIP", 0);
    let gemm = |w: WRef, a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> {
        if skip & 1 != 0 { return Ok(()) }
        pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m: n, y, ldy, y_col, acc }) };
    // (normed: the previous MoE layer's fused combine already wrote this layer's normed input to s.a)
    if !normed { pf.run(ctx, "rmsnorm", &[&s.x, &s.a, &wt.f32b], &[c.dim, s.al, wt.f(&nm("attn_norm.weight"))?, c.eps.to_bits(), 0], n, cb)?; ctx.barrier(cb); }
    if c.n_kv[l] > 0 {
        let nk = c.n_kv[l]; let ldp = (c.n_head + 2 * nk) * 64;
        let cache_off = s.attn_idx[l] * c.max_ctx * s.nkv * 64;
        gemm(wt.wref(&nm("attn_q.weight"))?, &s.a, s.al, &s.y, ldp, 0, false)?;
        gemm(wt.wref(&nm("attn_k.weight"))?, &s.a, s.al, &s.y, ldp, c.n_head * 64, false)?;
        gemm(wt.wref(&nm("attn_v.weight"))?, &s.a, s.al, &s.y, ldp, (c.n_head + nk) * 64, false)?; ctx.barrier(cb);
        pf.run(ctx, "rope64", &[&s.y, &m.kc, &m.vc, &wt.f32b], &[c.n_head, nk, cache_off, c.theta.to_bits(), wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?,
            c.eps.to_bits(), pos0, ldp, n], n * (c.n_head + nk), cb)?; ctx.barrier(cb);
        if skip & 4 == 0 { pf.run(ctx, "attn64", &[&s.y, &m.kc, &m.vc, &s.a], &[c.n_head, nk, cache_off, ldp, pos0, (1.0f32 / 8.0).to_bits(), s.al, n], n.div_ceil(16) * c.n_head, cb)?; }
        ctx.barrier(cb);
        gemm(wt.wref(&nm("attn_output.weight"))?, &s.a, s.al, &s.x, c.dim, 0, true)?; ctx.barrier(cb);
    } else {
        gemm(wt.wref(&nm("shortconv.in_proj.weight"))?, &s.a, s.al, &s.y, 3 * c.dim, 0, false)?; ctx.barrier(cb);
        let so = s.conv_idx[l] * 2 * c.dim;
        pf.run(ctx, "sconv", &[&s.y, &m.cst, &s.a, &wt.f32b], &[c.dim, so, wt.f(&nm("shortconv.conv.weight"))?, n, 3 * c.dim, s.al], grid(c.dim * n), cb)?; ctx.barrier(cb);
        pf.run(ctx, "sconvst", &[&s.y, &m.cst], &[c.dim, so, n, 3 * c.dim], c.dim.div_ceil(256), cb)?; ctx.barrier(cb);
        gemm(wt.wref(&nm("shortconv.out_proj.weight"))?, &s.a, s.al, &s.x, c.dim, 0, true)?; ctx.barrier(cb);
    }
    pf.run(ctx, "rmsnorm", &[&s.x, &s.a, &wt.f32b], &[c.dim, s.al, wt.f(&nm("ffn_norm.weight"))?, c.eps.to_bits(), 0], n, cb)?; ctx.barrier(cb);
    if (l as u32) < c.dense_lead {
        let (gw, uw) = (wt.wref(&nm("ffn_gate.weight"))?, wt.wref(&nm("ffn_up.weight"))?);
        gemm(gw, &s.a, s.al, &s.y, 2 * c.ffn, 0, false)?; gemm(uw, &s.a, s.al, &s.y, 2 * c.ffn, c.ffn, false)?; ctx.barrier(cb);
        pf.run(ctx, "swiglu", &[&s.y, &s.a], &[c.ffn, n, s.al], grid(c.ffn * n), cb)?; ctx.barrier(cb);
        gemm(wt.wref(&nm("ffn_down.weight"))?, &s.a, s.al, &s.x, c.dim, 0, true)?; ctx.barrier(cb);
    } else {
        if crate::env("PF_SKIP", 0) & 8 == 0 { if crate::env("PF_RF32", 0) == 0 {
            let ne = c.n_exp * c.dim;
            let w = WRef { arena: 0, off: (l as u32 * ne * 2) as u64, ty: gguf::GgmlType::F16, k: c.dim, n: c.n_exp, e: 1, bytes: ne as u64 * 2 };
            pf.gemm(ctx, cb, &s.rw, &wt.f32b, &GemmOp { w, row0: 0, rows: c.n_exp, a: &s.a, lda: s.al, a_row0: 0, m: n, y: &s.r, ldy: 64, y_col: 2 * n * c.top_k, acc: false })?;
        } else { pf.run(ctx, "route", &[&s.a, &wt.f32b, &s.r], &[c.dim, s.al, wt.f(&nm("ffn_gate_inp.weight"))?, wt.f(&nm("exp_probs_b.bias"))?, c.n_exp, c.top_k, n], n.div_ceil(16) * c.n_exp.div_ceil(16), cb)?; }
            ctx.barrier(cb); if crate::env("PF_MSEL", 1) == 0 { pf.run(ctx, "rsel", &[&s.r, &wt.f32b], &[wt.f(&nm("exp_probs_b.bias"))?, c.n_exp, c.top_k, n], n, cb)?; } }
        ctx.barrier(cb);
    }
    Ok(())
}

/// Grouped expert FFN of MoE layer l + combine into the residual (routing in s.r from record_layer):
/// GPU sort by expert -> gather rows -> grouped GEMM gate, up -> SwiGLU -> grouped GEMM down -> weighted combine.
fn record_moe(pf: &Pf, m: &Model, s: &Lf, cb: vk::CommandBuffer, l: usize, n: u32, next_norm: Option<u32>) -> Result<bool> {
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    let nm = |x: &str| format!("blk.{l}.{x}");
    let (gw, uw, dw) = (wt.wref(&nm("ffn_gate_exps.weight"))?, wt.wref(&nm("ffn_up_exps.weight"))?, wt.wref(&nm("ffn_down_exps.weight"))?);
    let (nk, fe, d) = (n * c.top_k, c.ff_exp, c.dim);
    // PF_HYB=1: mixed 64/32-row expert tiles (gt: 64-row tiles, gt2: <=32-row tails), each GEMM = two dispatches without a barrier
    let hyb = crate::env("PF_HYB", 1) != 0 && crate::env("PF_MSEL", 1) != 0;
    // many rows per expert (pp2048: ~128): 32 x 256 gate/up tiles and 32-row down tiles win; few (pp512: ~32): 32 x 128 + 64-row down
    let wide = nk / c.n_exp >= crate::env("PF_WIDE_RPE", 64);
    // PF_QMM=1: packed-LDS int8 expert GEMMs (pf_qgemm, Q4_K gate/up; down too when Q4_K), q8 activations; plain tiles of 32*MI rows
    let qmm = crate::env("PF_QMM", 0) > 0 && rtcore::qtype(gw.ty)? == 0 && rtcore::qtype(uw.ty)? == 0;
    let hyb = hyb && !qmm;
    let (bm, bm2) = if qmm { (Pf::qgemm_rows(), Pf::qgemm_rows()) } else if hyb { (crate::env("PF_HYBM", 64), 32) } else { (crate::env("PF_GBM", 32), crate::env("PF_GBM2", if wide { 32 } else { 64 })) };
    let gmr = if wide && bm == 32 { 0x8020 } else { bm };   // tile-variant key for the gate/up grouped GEMMs
    let gmr2 = if wide && bm2 == 32 { 0x8020 } else { bm2 };
    let (mt, mt2) = (nk.div_ceil(if hyb { 32 } else { bm }) + c.n_exp, nk.div_ceil(bm2) + c.n_exp);
    let sk = crate::env("PF_SKIP", 0);
    // PF_MSEL=1 (default): top-k expert selection is fused into the grouping pass (pf_moesel, one dispatch instead of rsel + moesort)
    if crate::env("PF_MSEL", 1) != 0 && crate::env("PF_SKIP", 0) & 8 == 0 { pf.run(ctx, "moesel", &[&s.r, &s.g, &s.gt, &s.gt2, &wt.f32b], &[n, c.top_k, c.n_exp, bm, bm2, wt.f(&nm("exp_probs_b.bias"))?, hyb as u32], 1, cb)?; }
    else { pf.run(ctx, "moesort", &[&s.r, &s.g, &s.gt, &s.gt2], &[nk, c.top_k, c.n_exp, bm, bm2, 0], 1, cb)?; }
    ctx.barrier(cb);
    if sk & 48 == 0 { pf.run(ctx, "gather", &[&s.a, &s.ag, &s.g], &[d / 2, s.al / 2, d / 2, nk], grid(d / 2 * nk), cb)?; } ctx.barrier(cb);
    if qmm {
        let ldx = nk.div_ceil(64) * 64 + 64; let qb = pf.qbufs(ctx, ldx, d.max(fe))?; let (xq, xs, xu) = unsafe { (&*qb[0], &*qb[1], &*qb[2]) };
        let mt = nk.div_ceil(bm) + c.n_exp;
        if crate::env("PF_QMM", 0) == 2 {   // f16 activations, packed-LDS weights unpacked in registers (pf_qgemmh)
            pf.qgemmh_grouped(ctx, cb, &wt.arenas[gw.arena], gw, &s.gt, mt, &s.ag, d, &s.yg, 2 * fe, 0)?;
            pf.qgemmh_grouped(ctx, cb, &wt.arenas[uw.arena], uw, &s.gt, mt, &s.ag, d, &s.yg, 2 * fe, fe)?; ctx.barrier(cb);
            pf.run(ctx, "swiglu", &[&s.yg, &s.hg], &[fe, nk, fe], grid(fe * nk), cb)?; ctx.barrier(cb);
            if rtcore::qtype(dw.ty)? == 0 && fe % 256 == 0 { pf.qgemmh_grouped(ctx, cb, &wt.arenas[dw.arena], dw, &s.gt, mt, &s.hg, fe, &s.yd, d, 0)?; }
            else { pf.gemm_grouped(ctx, cb, &wt.arenas[dw.arena], &wt.f32b, dw, &s.gt, bm, mt, &s.hg, fe, &s.yd, d, 0)?; }
            ctx.barrier(cb);
        } else {
        pf.q8(ctx, cb, &s.ag, d, nk, d, ldx, xq, xs, xu)?; ctx.barrier(cb);
        pf.qgemm_grouped(ctx, cb, &wt.arenas[gw.arena], gw, &s.gt, mt, xq, xs, xu, ldx, &s.yg, 2 * fe, 0)?;
        pf.qgemm_grouped(ctx, cb, &wt.arenas[uw.arena], uw, &s.gt, mt, xq, xs, xu, ldx, &s.yg, 2 * fe, fe)?; ctx.barrier(cb);
        pf.run(ctx, "swiglu", &[&s.yg, &s.hg], &[fe, nk, fe], grid(fe * nk), cb)?; ctx.barrier(cb);
        if rtcore::qtype(dw.ty)? == 0 && fe % 256 == 0 { pf.q8(ctx, cb, &s.hg, fe, nk, fe, ldx, xq, xs, xu)?; ctx.barrier(cb);
            pf.qgemm_grouped(ctx, cb, &wt.arenas[dw.arena], dw, &s.gt, mt, xq, xs, xu, ldx, &s.yd, d, 0)?; }
        else { pf.gemm_grouped(ctx, cb, &wt.arenas[dw.arena], &wt.f32b, dw, &s.gt, bm, mt, &s.hg, fe, &s.yd, d, 0)?; }
        ctx.barrier(cb);
        }
    } else {
    let fused = crate::env("PF_MSWI", 1) > 0 && gw.arena == uw.arena && gw.ty == uw.ty && gw.bytes == uw.bytes;
    if fused { if sk & 1 == 0 { pf.gemm_grouped_swiglu(ctx, cb, &wt.arenas[gw.arena], &wt.f32b, gw, uw, &s.gt, gmr, mt, &s.ag, d, &s.hg, fe, 0)?;
        if hyb { pf.gemm_grouped_swiglu(ctx, cb, &wt.arenas[gw.arena], &wt.f32b, gw, uw, &s.gt2, bm2, mt2, &s.ag, d, &s.hg, fe, 0)?; } } ctx.barrier(cb); }
    else {
        if sk & 1 == 0 {
            pf.gemm_grouped(ctx, cb, &wt.arenas[gw.arena], &wt.f32b, gw, &s.gt, gmr, mt, &s.ag, d, &s.yg, 2 * fe, 0)?;
            pf.gemm_grouped(ctx, cb, &wt.arenas[uw.arena], &wt.f32b, uw, &s.gt, gmr, mt, &s.ag, d, &s.yg, 2 * fe, fe)?;
            if hyb { pf.gemm_grouped(ctx, cb, &wt.arenas[gw.arena], &wt.f32b, gw, &s.gt2, bm2, mt2, &s.ag, d, &s.yg, 2 * fe, 0)?;
                     pf.gemm_grouped(ctx, cb, &wt.arenas[uw.arena], &wt.f32b, uw, &s.gt2, bm2, mt2, &s.ag, d, &s.yg, 2 * fe, fe)?; } }
        ctx.barrier(cb);
        if sk & 80 == 0 { pf.run(ctx, "swiglu", &[&s.yg, &s.hg], &[fe, nk, fe], grid(fe * nk), cb)?; } ctx.barrier(cb);
    }
    if sk & 1 == 0 { if hyb { pf.gemm_grouped(ctx, cb, &wt.arenas[dw.arena], &wt.f32b, dw, &s.gt, gmr, mt, &s.hg, fe, &s.yd, d, 0)?; }
        pf.gemm_grouped(ctx, cb, &wt.arenas[dw.arena], &wt.f32b, dw, &s.gt2, gmr2, mt2, &s.hg, fe, &s.yd, d, 0)?; }
    ctx.barrier(cb);
    }
    if let (Some(nw), true) = (next_norm, crate::env("PF_COMBNORM", 1) > 0 && d <= 4096) {
        pf.run(ctx, "combnorm", &[&s.x, &s.yd, &s.r, &s.g, &s.a, &wt.f32b], &[d, n, c.top_k, s.al, nw, c.eps.to_bits()], n, cb)?; ctx.barrier(cb);
        return Ok(true);
    }
    if sk & 16 == 0 { pf.run(ctx, "moecomb", &[&s.x, &s.yd, &s.r, &s.g], &[d, n, c.top_k], grid(d * n), cb)?; } ctx.barrier(cb);
    Ok(false)
}

fn record_head(pf: &Pf, m: &Model, s: &Lf, cb: vk::CommandBuffer, row: u32) -> Result<()> {
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    pf.run(ctx, "rmsnorm", &[&s.x, &s.a, &wt.f32b], &[c.dim, s.al, wt.f("token_embd_norm.weight")?, c.eps.to_bits(), row], 1, cb)?; ctx.barrier(cb);
    let w = wt.wref("output.weight").or_else(|_| wt.wref("token_embd.weight"))?;
    pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: s.al, a_row0: 0, m: 1, y: &m.logits, ldy: w.n, y_col: 0, acc: false })?;
    ctx.barrier(cb); Ok(())
}

/// Batched prefill of `prompt`; leaves logits of the last token, toks[L] (greedy) and the decode position set. Returns wall s.
pub fn prefill(pf: &Pf, m: &Model, s: &Lf, prompt: &[u32]) -> Result<f64> {
    let (c, l) = (&m.cfg, prompt.len()); if l == 0 || l + 1 > c.max_ctx as usize { bail!("prompt length") }
    let t0 = std::time::Instant::now();
    m.cst.bytes_mut().fill(0);
    let te = m.wt.g.tensor("token_embd.weight").ok_or_else(|| anyhow::anyhow!("no token_embd"))?;
    let data = m.wt.g.data(te); let d = c.dim as usize; let rb = data.len() / c.vocab as usize;
    let mut pos = 0usize;
    while pos < l {
        let n = (l - pos).min(s.mmax as usize);
        let x = &mut s.x.as_f32_mut()[..n * d];
        for (i, &tk) in prompt[pos..pos + n].iter().enumerate() { dequant_row(te.ty, &data[tk as usize * rb..(tk as usize + 1) * rb], &mut x[i * d..(i + 1) * d]); }
        let last = pos + n == l;
        let (nl, lps) = (c.n_layer as usize, crate::env("PF_LPS", 24) as usize);
        let prof = crate::env("PF_PROF", 0) > 0; let mut tl = std::time::Instant::now();
        let normed = std::cell::Cell::new(false);
        for l0 in (0..nl).step_by(lps) {
            let mut err = None;
            m.ctx.run(|cb| {
                let r = (|| -> Result<()> {
                    for li in l0..(l0 + lps).min(nl) {
                        record_layer(pf, m, s, cb, li, n as u32, pos as u32, normed.get())?; normed.set(false);
                        if li as u32 >= c.dense_lead {
                            let nn = if li + 1 < nl { Some(m.wt.f(&format!("blk.{}.attn_norm.weight", li + 1))?) } else { None };
                            normed.set(record_moe(pf, m, s, cb, li, n as u32, nn)?);
                        }
                    }
                    if last && l0 + lps >= nl { record_head(pf, m, s, cb, n as u32 - 1)?; }
                    Ok(()) })();
                if let Err(e) = r { err = Some(e) } })?;
            if let Some(e) = err { return Err(e) }
            if prof { eprintln!("layers {l0}.. {:.2} ms", tl.elapsed().as_secs_f64() * 1e3); tl = std::time::Instant::now(); }
        }
        pos += n;
    }
    let toks = unsafe { std::slice::from_raw_parts_mut(m.toks.ptr as *mut u32, m.toks.size as usize / 4) };
    let au = unsafe { std::slice::from_raw_parts_mut(m.aux.ptr as *mut u32, 256) };
    toks[..l].copy_from_slice(prompt);
    let lg = &m.logits.as_f32()[..c.vocab as usize];
    toks[l] = lg.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    au[160] = l as u32; au[161] = (l + 1) as u32; au[162] = 0;
    Ok(t0.elapsed().as_secs_f64())
}

// ---------------------------------------------------------------------------------------------------------------
// Cross-prompt batched prefill (strix-serve1): one pass runs one chunk from each of several sequences. Row-wise work
// (norms, every projection GEMM, router GEMM, grouped MoE, combine) runs once over the packed rows, so expert GEMM
// groups get rows from several prompts. Sequence-dependent work (short conv + its state, RoPE + attention) runs per
// segment on the scratch `t` with the segment rows copied in / out, using that sequence's own kc / vc / cst.
pub struct Seg<'a> { pub toks: &'a [u32], pub pos0: u32, pub kc: &'a Buffer, pub vc: &'a Buffer, pub cst: &'a Buffer, pub last: bool }
fn xbar(m: &Model, cb: vk::CommandBuffer) {
    let a = vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ;
    let st = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
    let mb = [vk::MemoryBarrier::default().src_access_mask(a).dst_access_mask(a)];
    unsafe { m.ctx.device.cmd_pipeline_barrier(cb, st, st, vk::DependencyFlags::empty(), &mb, &[], &[]) };
}
fn cp(m: &Model, cb: vk::CommandBuffer, a: &Buffer, b: &Buffer, ao: u64, bo: u64, n: u64) {
    if n > 0 { unsafe { m.ctx.device.cmd_copy_buffer(cb, a.buf, b.buf, &[vk::BufferCopy { src_offset: ao, dst_offset: bo, size: n }]) } }
}
fn record_layer_m(pf: &Pf, m: &Model, s: &Lf, t: &Lf, cb: vk::CommandBuffer, l: usize, segs: &[Seg], r0: &[u32], n: u32, normed: bool) -> Result<()> {
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    let nm = |x: &str| format!("blk.{l}.{x}");
    let gemm = |w: WRef, a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> {
        pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m: n, y, ldy, y_col, acc }) };
    if !normed { pf.run(ctx, "rmsnorm", &[&s.x, &s.a, &wt.f32b], &[c.dim, s.al, wt.f(&nm("attn_norm.weight"))?, c.eps.to_bits(), 0], n, cb)?; ctx.barrier(cb); }
    if c.n_kv[l] > 0 {
        let nk = c.n_kv[l]; let ldp = (c.n_head + 2 * nk) * 64;
        let cache_off = s.attn_idx[l] * c.max_ctx * s.nkv * 64;
        gemm(wt.wref(&nm("attn_q.weight"))?, &s.a, s.al, &s.y, ldp, 0, false)?;
        gemm(wt.wref(&nm("attn_k.weight"))?, &s.a, s.al, &s.y, ldp, c.n_head * 64, false)?;
        gemm(wt.wref(&nm("attn_v.weight"))?, &s.a, s.al, &s.y, ldp, (c.n_head + nk) * 64, false)?; xbar(m, cb);
        for (g, &r) in segs.iter().zip(r0) { let k = g.toks.len() as u32;
            cp(m, cb, &s.y, &t.y, r as u64 * ldp as u64 * 4, 0, k as u64 * ldp as u64 * 4); xbar(m, cb);
            pf.run(ctx, "rope64", &[&t.y, g.kc, g.vc, &wt.f32b], &[c.n_head, nk, cache_off, c.theta.to_bits(), wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?,
                c.eps.to_bits(), g.pos0, ldp, k], k * (c.n_head + nk), cb)?; ctx.barrier(cb);
            pf.run(ctx, "attn64", &[&t.y, g.kc, g.vc, &t.a], &[c.n_head, nk, cache_off, ldp, g.pos0, (1.0f32 / 8.0).to_bits(), t.al, k], k.div_ceil(16) * c.n_head, cb)?; xbar(m, cb);
            cp(m, cb, &t.a, &s.a, 0, r as u64 * s.al as u64 * 2, k as u64 * s.al as u64 * 2); xbar(m, cb); }
        gemm(wt.wref(&nm("attn_output.weight"))?, &s.a, s.al, &s.x, c.dim, 0, true)?; ctx.barrier(cb);
    } else {
        gemm(wt.wref(&nm("shortconv.in_proj.weight"))?, &s.a, s.al, &s.y, 3 * c.dim, 0, false)?; xbar(m, cb);
        let so = s.conv_idx[l] * 2 * c.dim;
        for (g, &r) in segs.iter().zip(r0) { let k = g.toks.len() as u32;
            cp(m, cb, &s.y, &t.y, r as u64 * 3 * c.dim as u64 * 4, 0, k as u64 * 3 * c.dim as u64 * 4); xbar(m, cb);
            pf.run(ctx, "sconv", &[&t.y, g.cst, &t.a, &wt.f32b], &[c.dim, so, wt.f(&nm("shortconv.conv.weight"))?, k, 3 * c.dim, t.al], grid(c.dim * k), cb)?; ctx.barrier(cb);
            pf.run(ctx, "sconvst", &[&t.y, g.cst], &[c.dim, so, k, 3 * c.dim], c.dim.div_ceil(256), cb)?; xbar(m, cb);
            cp(m, cb, &t.a, &s.a, 0, r as u64 * s.al as u64 * 2, k as u64 * s.al as u64 * 2); xbar(m, cb); }
        gemm(wt.wref(&nm("shortconv.out_proj.weight"))?, &s.a, s.al, &s.x, c.dim, 0, true)?; ctx.barrier(cb);
    }
    pf.run(ctx, "rmsnorm", &[&s.x, &s.a, &wt.f32b], &[c.dim, s.al, wt.f(&nm("ffn_norm.weight"))?, c.eps.to_bits(), 0], n, cb)?; ctx.barrier(cb);
    if (l as u32) < c.dense_lead {
        let (gw, uw) = (wt.wref(&nm("ffn_gate.weight"))?, wt.wref(&nm("ffn_up.weight"))?);
        gemm(gw, &s.a, s.al, &s.y, 2 * c.ffn, 0, false)?; gemm(uw, &s.a, s.al, &s.y, 2 * c.ffn, c.ffn, false)?; ctx.barrier(cb);
        pf.run(ctx, "swiglu", &[&s.y, &s.a], &[c.ffn, n, s.al], grid(c.ffn * n), cb)?; ctx.barrier(cb);
        gemm(wt.wref(&nm("ffn_down.weight"))?, &s.a, s.al, &s.x, c.dim, 0, true)?; ctx.barrier(cb);
    } else {
        let ne = c.n_exp * c.dim;
        let w = WRef { arena: 0, off: (l as u32 * ne * 2) as u64, ty: gguf::GgmlType::F16, k: c.dim, n: c.n_exp, e: 1, bytes: ne as u64 * 2 };
        pf.gemm(ctx, cb, &s.rw, &wt.f32b, &GemmOp { w, row0: 0, rows: c.n_exp, a: &s.a, lda: s.al, a_row0: 0, m: n, y: &s.r, ldy: 64, y_col: 2 * n * c.top_k, acc: false })?;
        ctx.barrier(cb);
    }
    Ok(())
}
/// Run one packed pass over `segs` (sum of lengths <= s.mmax, each <= t.mmax). Returns (segment index, greedy next
/// token, logits) for every segment with `last`. Sequences start with zeroed conv state (caller) at pos0 = 0.
pub fn prefill_pass(pf: &Pf, m: &Model, s: &Lf, t: &Lf, segs: &[Seg]) -> Result<Vec<(usize, u32, Vec<f32>)>> {
    let c = &m.cfg;
    let nt: usize = segs.iter().map(|g| g.toks.len()).sum();
    if nt == 0 || nt > s.mmax as usize || segs.iter().any(|g| g.toks.len() > t.mmax as usize) { bail!("prefill_pass: {nt} rows") }
    if crate::env("PF_MSEL", 1) == 0 || crate::env("PF_RF32", 0) != 0 { bail!("prefill_pass needs PF_MSEL=1, PF_RF32=0") }
    let te = m.wt.g.tensor("token_embd.weight").ok_or_else(|| anyhow::anyhow!("no token_embd"))?;
    let data = m.wt.g.data(te); let d = c.dim as usize; let rb = data.len() / c.vocab as usize;
    { let x = &mut s.x.as_f32_mut()[..nt * d];
      for (i, &tk) in segs.iter().flat_map(|g| g.toks.iter()).enumerate() { dequant_row(te.ty, &data[tk as usize * rb..(tk as usize + 1) * rb], &mut x[i * d..(i + 1) * d]); } }
    let r0: Vec<u32> = segs.iter().scan(0u32, |a, g| { let r = *a; *a += g.toks.len() as u32; Some(r) }).collect();
    let n = nt as u32;
    let (nl, lps) = (c.n_layer as usize, crate::env("PF_LPS", 8) as usize);
    let normed = std::cell::Cell::new(false);
    for l0 in (0..nl).step_by(lps) {
        let mut err = None;
        m.ctx.run(|cb| {
            let r = (|| -> Result<()> {
                for li in l0..(l0 + lps).min(nl) {
                    record_layer_m(pf, m, s, t, cb, li, segs, &r0, n, normed.get())?; normed.set(false);
                    if li as u32 >= c.dense_lead {
                        let nn = if li + 1 < nl { Some(m.wt.f(&format!("blk.{}.attn_norm.weight", li + 1))?) } else { None };
                        normed.set(record_moe(pf, m, s, cb, li, n, nn)?);
                    }
                }
                Ok(()) })();
            if let Err(e) = r { err = Some(e) } })?;
        if let Some(e) = err { return Err(e) }
    }
    let mut out = vec![];
    for (i, (g, &r)) in segs.iter().zip(&r0).enumerate() { if !g.last { continue }
        let mut err = Ok(()); m.ctx.run(|cb| { err = record_head(pf, m, s, cb, r + g.toks.len() as u32 - 1); })?; err?;
        let lg = m.logits.as_f32()[..c.vocab as usize].to_vec();
        let am = lg.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        out.push((i, am as u32, lg)); }
    Ok(out)
}
