//! Batched prefill for the qwen35 trunk (Qwen3.8-27B): processes a chunk of prompt tokens per layer with
//! coop-matrix GEMMs and leaves kc/vc, conv state and DeltaNet state where the token-by-token path would.
//! `prefill` runs prompt[..L-1] batched and the last token through the model's own decode step, so the logits,
//! argmax and sampling state are produced by the decode code and generation continues with `submit_repeated(step, n)`.
use crate::{GemmOp, Pf};
use anyhow::{bail, Result};
use ash::vk;
use gguf::dequant::dequant_row;
use qwen38::Model;
use vkrt::Buffer;
use std::collections::HashSet;
#[path = "npu_split.rs"] pub mod npu_split;
#[path = "q35multi.rs"] pub mod multi;

/// Bonsai 2 (ternary PQ2_0/PTQ1_0, Prism Hadamard folding) state: rotated-weight names, sign vectors per width,
/// GDN v-grouped permutation for ssm_out inputs, and the f32 / rotated / int8 activation scratch.
pub struct Bon { pub rot: HashSet<String>, pub sg: Vec<(u32, Buffer)>, pub perm: (u32, u32, u32), pub inv_embed: bool,
    pub xf: Buffer, pub xr: Buffer, pub xq: Buffer, pub xs: Buffer, pub xh: Buffer }
impl Bon {
    pub fn rot(&self, name: &str) -> bool { self.rot.contains(name) || self.rot.contains(name.trim_end_matches(".weight")) }
    fn signs(&self, k: u32) -> Result<&Buffer> { self.sg.iter().find(|(w, _)| *w == k).map(|(_, b)| b).ok_or_else(|| anyhow::anyhow!("no Hadamard signs of width {k}")) }
}
fn is_tern(w: &rtcore::WRef) -> bool { matches!(w.ty, gguf::GgmlType::Other(142) | gguf::GgmlType::Other(143)) }

pub struct Q35 { pub auto: Option<Box<Q35>>, pub npu_on: std::cell::Cell<bool>, pub tb: Option<Box<Q35>>, pub tc: Option<Box<Q35>>, pub mmax: u32, pub al: u32, pub x: Buffer, pub a: Buffer, pub a2: Buffer, pub p: Buffer, pub cv: Buffer, pub o: Buffer, pub lidx: Vec<u32>, pub ldp_max: u32, pub bon: Option<Bon>, pub npu: Option<std::rc::Rc<npu_split::NpuSplit>> }

impl Q35 {
    pub fn new(m: &Model, mmax: u32) -> Result<Self> {
        // Default for ternary (Bonsai) models at PF_M >= 2048, unless NPU_TB / NPU_SPLIT is set (NPU_AUTO=0 disables):
        // pick the NPU mode per prompt length (see `prefill`): column split state + a 512+512+1024 three-block set
        let explicit = std::env::var("NPU_TB").is_ok() || std::env::var("NPU_SPLIT").is_ok();
        if !explicit && crate::env("NPU_AUTO", 1) > 0 && mmax >= 2048 && m.wt.w.values().any(is_tern) {
            let mut s = Self::new_inner(m, mmax, 1)?;
            // three-block pipeline (round 23): A, B 512 tokens each on the NPU, C 1024 on the GPU
            let mut a = Self::new_inner(m, 512, 3)?; let mut b = Self::new_inner(m, 512, 0)?; b.npu = a.npu.clone();
            a.tb = Some(Box::new(b)); a.tc = Some(Box::new(Self::new_inner(m, 1024, 0)?));
            s.auto = Some(Box::new(a)); return Ok(s)
        }
        // NPU_TB=1: token-block pipeline. Each chunk is split into halves A (this state, whole FFN on the NPU) and B
        // (s.tb, all on the GPU); the NPU runs A's FFN of layer l while the GPU runs B's mixer / FFN of layer l.
        // NPU_TB_A=tokens: size of block A (default half the chunk; NPU M must be a multiple of 512), B gets the rest
        // NPU_TB=3: blocks A, B (whole FFN on the NPU, NPU_TB_A tokens each, default 512) and C (GPU, the rest)
        if crate::env("NPU_TB", 0) == 3 { let h = (crate::env("NPU_TB_A", 512) as u32).min(mmax / 3).next_multiple_of(128);
            let mut a = Self::new_inner(m, h, 3)?; let mut b = Self::new_inner(m, h, 0)?; b.npu = a.npu.clone();
            a.tb = Some(Box::new(b)); a.tc = Some(Box::new(Self::new_inner(m, mmax - 2 * h, 0)?)); return Ok(a) }
        if crate::env("NPU_TB", 0) > 0 { let h = (crate::env("NPU_TB_A", mmax / 2) as u32).min(mmax - 1).next_multiple_of(128);
            let mut a = Self::new_inner(m, h, 2)?; a.tb = Some(Box::new(Self::new_inner(m, (mmax - h).max(h), 0)?)); return Ok(a) }
        Self::new_inner(m, mmax, crate::env("NPU_SPLIT", 0).min(1) as u8)
    }
    fn new_inner(m: &Model, mmax: u32, npu_mode: u8) -> Result<Self> {
        let c = &m.cfg; let ctx = &m.ctx;
        if c.hd != 256 || c.d_state != 128 { bail!("prefill: hd {} d_state {} unsupported", c.hd, c.d_state) }
        let conv_ch = 2 * c.nk * c.d_state + c.nv * c.d_state;
        let al = c.dim.max(c.nv * c.d_state).max(c.n_head * c.hd).max(c.ffn).next_multiple_of(8);
        let ldp_max = (conv_ch + c.nv * c.d_state + 2 * c.nv).max(c.n_head * c.hd * 2 + 2 * c.n_kv * c.hd).max(2 * c.ffn);
        let mm = mmax as u64;
        let (mut lidx, mut na, mut nr) = (vec![], 0u32, 0u32);
        for &r in &c.recr { if r { lidx.push(nr); nr += 1 } else { lidx.push(na); na += 1 } }
        let g = &m.wt.g;
        let strs = |k: &str| -> Vec<String> { match g.get(k) { Ok(gguf::Value::Arr(v)) => v.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(), _ => vec![] } };
        let ints = |k: &str| -> Vec<i64> { match g.get(k) { Ok(gguf::Value::Arr(v)) => v.iter().filter_map(|x| match x { gguf::Value::I32(i) => Some(*i as i64), o => o.as_u64().map(|u| u as i64) }).collect(), _ => vec![] } };
        let tern = m.wt.w.values().any(is_tern);
        let bon = if tern {
            let rot: HashSet<String> = strs("prism.hadamard.weight_names").into_iter().collect();
            let (widths, svals) = (ints("prism.hadamard.sign_widths"), ints("prism.hadamard.sign_values"));
            let mut sg = vec![]; let mut so = 0usize;
            for &wd in &widths { let wd = wd as usize; if so + wd > svals.len() { bail!("sign_values too short") }
                let b = ctx.buffer((wd * 4) as u64)?; for (d, v) in b.as_f32_mut().iter_mut().zip(&svals[so..so + wd]) { *d = *v as f32 } sg.push((wd as u32, b)); so += wd; }
            let grouped = matches!(g.get("prism.hadamard.gdn_v_grouped"), Ok(gguf::Value::Bool(true)));
            let perm = if grouped { (c.d_state, c.nk, c.nv / c.nk) } else { (1, 1, 1) };
            let inv_embed = strs("prism.hadamard.inverse_weight_names").iter().any(|n| n == "token_embd.weight" || n == "token_embd");
            let kmax = al as u64; let mp = (mm + 64).next_multiple_of(64);
            Some(Bon { rot, sg, perm, inv_embed, xf: ctx.buffer(mm * kmax * 4)?, xr: ctx.buffer(mm * kmax * 4)?, xq: ctx.buffer(mp * kmax)?, xs: ctx.buffer(mp * kmax / 128 * 4)?, xh: ctx.buffer((mm + 256) * kmax * 2)? })
        } else { None };
        let mut q = Q35 { auto: None, npu_on: std::cell::Cell::new(true), tb: None, tc: None, npu: None, bon, mmax, al, x: ctx.buffer(mm * c.dim as u64 * 4)?, a: ctx.buffer((mm + 256) * al as u64 * 2)?, a2: ctx.buffer((mm + 256) * al as u64 * 2)?, p: ctx.buffer(mm * ldp_max as u64 * 4)?,
            cv: ctx.buffer(mm * conv_ch as u64 * 4)?, o: ctx.buffer(mm * (c.nv * c.d_state) as u64 * 4)?, lidx, ldp_max };
        // NPU_SPLIT=1: Bonsai PQ2_0 projections split between NPU and GPU (npu_split.rs)
        if npu_mode > 0 { if let Some(b) = q.bon.as_ref() { let sp = npu_split::NpuSplit::new(m, mmax, b, npu_mode >= 2, if npu_mode == 3 { 2 } else { 1 })?; eprintln!("[npu split] weights folded in {:.1} s", sp.load_s); q.npu = Some(std::rc::Rc::new(sp)) } }
        Ok(q)
    }
}

/// CPU embedding of `toks` into s.x (token_embd rows, any GGUF type).
fn embed(m: &Model, s: &Q35, toks: &[u32]) -> Result<()> {
    let t = m.wt.g.tensor("token_embd.weight").ok_or_else(|| anyhow::anyhow!("no token_embd"))?;
    let data = m.wt.g.data(t); let d = m.cfg.dim as usize; let rb = data.len() / m.cfg.vocab as usize;
    let x = &mut s.x.as_f32_mut()[..toks.len() * d];
    let tf = match t.ty { gguf::GgmlType::Other(142) => Some(ternary::Fmt::Pq2_0), gguf::GgmlType::Other(143) => Some(ternary::Fmt::Ptq1_0), _ => None };
    for (i, &tk) in toks.iter().enumerate() { let row = &data[tk as usize * rb..(tk as usize + 1) * rb];
        match tf { Some(f) => ternary::dequant_row(f, row, &mut x[i * d..(i + 1) * d]), None => dequant_row(t.ty, row, &mut x[i * d..(i + 1) * d]) } }
    Ok(())
}

/// Record the batched forward of n = toks.len() tokens at positions pos0.. (s.x must hold their embeddings).
pub fn record(pf: &Pf, m: &Model, s: &Q35, cb0: vk::CommandBuffer, n: u32, pos0: u32, layers: std::ops::Range<usize>) -> Result<()> {
    record_part(pf, m, s, cb0, n, pos0, layers, 3)
}
/// `part`: bit 0 = token mixer (norm + attention / GDN + output projection), bit 1 = FFN.
pub fn record_part(pf: &Pf, m: &Model, s: &Q35, cb0: vk::CommandBuffer, n: u32, pos0: u32, layers: std::ops::Range<usize>, part: u32) -> Result<()> {
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    let sp = s.npu.as_ref().filter(|_| s.npu_on.get()); let cur = || sp.map_or(cb0, |x| x.cur.get()); // NPU_SPLIT cuts the command stream
    let ds_ = c.d_state; let conv_ch = 2 * c.nk * ds_ + c.nv * ds_; let al = s.al;
    let bar = || ctx.barrier(cur()); let sync = crate::env("PF_SYNC", 0) > 0; let skip = crate::env("PF_SKIP", 0); // profiling: 1 gemm, 2 scan, 4 attn, 8 conv/norm/rope/swiglu
    // PF_TF16=1 (default): ternary GEMMs via the f16 coop-matrix GEMM (pf_gemm, deq32 types 8/9); 0 = int8 pf_tgemm
    let tf16 = crate::env("PF_TF16", 1) != 0; // NPU_SPLIT: GPU half via f16 GEMM too (NPU A built from b.xr)
    let gemm = |name: &str, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> {
        let w = wt.wref(name)?; if skip & 1 != 0 { return Ok(()) }
        if is_tern(&w) { let b = s.bon.as_ref().unwrap();
            if let (Some(x), Some(nn)) = (sp, sp.and_then(|x| x.nn(name))) {
                return x.cut(ctx, name, &b.xr, n, y, ldy, y_col, acc, |hb| if w.n == nn { Ok(()) } else if tf16 { let (a, lda) = if b.rot(name) { (&b.xh, w.k) } else { (&s.a, al) };
                    pf.gemm(ctx, hb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: nn, rows: w.n - nn, a, lda, a_row0: 0, m: n, y, ldy, y_col: y_col + nn, acc }) }
                    else { pf.tgemm(ctx, hb, &wt.arenas[w.arena], w, nn, w.n - nn, &b.xq, &b.xs, n, y, ldy, y_col + nn, acc) }) }
            if tf16 { // ternary weights decoded to f16 in the coop-matrix GEMM (K-slab), A = f16 (rotated: b.xh, else s.a)
                let (a, lda) = if b.rot(name) { (&b.xh, w.k) } else { (&s.a, al) };
                pf.gemm(ctx, cur(), &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m: n, y, ldy, y_col, acc })?;
                if sync { ctx.barrier(cur()) } return Ok(()) }
            pf.tgemm(ctx, cur(), &wt.arenas[w.arena], w, 0, w.n, &b.xq, &b.xs, n, y, ldy, y_col, acc)?; if sync { ctx.barrier(cur()) } return Ok(()) }
        pf.gemm(ctx, cur(), &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: al, a_row0: 0, m: n, y, ldy, y_col, acc })?;
        if sync { ctx.barrier(cur()) } Ok(())
    };
    let run = |k: &str, b: &[&Buffer], p: &[u32], g: u32| -> Result<()> {
        let bit = match k { "gdn_scan" => 2, "attn" => 4, _ => 8 }; if skip & bit != 0 { return Ok(()) } pf.run(ctx, k, b, p, g, cur()) };
    let norm_to = |wn: &str, dst: &Buffer| -> Result<()> { if skip & 8 != 0 { return Ok(()) } pf.run(ctx, "rmsnorm", &[&s.x, dst, &wt.f32b], &[c.dim, al, wt.f(wn)?, c.eps.to_bits(), 0], n, cur())?; ctx.barrier(cur()); Ok(()) };
    let norm = |wn: &str| norm_to(wn, &s.a);
    // Bonsai: s.a (f16, K wide) -> f32 -> [x' = H_1024(s * perm(x)) if the weight is rotated] -> per-token int8 (xq, xs)
    let bprep = |wname: &str, k: u32, perm: bool| -> Result<()> {
        let (Some(b), Ok(w)) = (s.bon.as_ref(), wt.wref(wname)) else { return Ok(()) };
        if !is_tern(&w) || skip & 1 != 0 { return Ok(()) }
        if tf16 && !b.rot(wname) { return Ok(()) }   // f16 GEMM reads s.a directly
        pf.run(ctx, "h2f", &[&s.a, &b.xf], &[k, al, n], (k * n).div_ceil(256), cur())?; bar();
        let src = if b.rot(wname) {
            let (hd, nk, rep) = if perm { b.perm } else { (1, 1, 1) };
            pf.run(ctx, "fwht", &[&b.xf, &b.xr, b.signs(k)?], &[k, 0, 0, 1, hd, nk, rep], k / 1024 * n, cur())?; bar(); &b.xr
        } else { &b.xf };
        if tf16 { pf.run(ctx, "f2h", &[src, &b.xh], &[k, k, n], (k * n).div_ceil(256), cur())?; bar(); return Ok(()) }
        pf.run(ctx, "quantx", &[src, &b.xq, &b.xs], &[n, k], n, cur())?; bar();
        Ok(())
    };
    for l in layers {
        let nm = |x: &str| format!("blk.{l}.{x}");
        let li = s.lidx[l];
        if part & 1 != 0 {
        norm(&nm("attn_norm.weight"))?;
        if c.recr[l] {
            let ldp = conv_ch + c.nv * ds_ + 2 * c.nv; let (zo, ao, bo) = (conv_ch, conv_ch + c.nv * ds_, conv_ch + c.nv * ds_ + c.nv);
            bprep(&nm("attn_qkv.weight"), c.dim, false)?;
            gemm(&nm("attn_qkv.weight"), &s.p, ldp, 0, false)?; gemm(&nm("attn_gate.weight"), &s.p, ldp, zo, false)?;
            gemm(&nm("ssm_alpha.weight"), &s.p, ldp, ao, false)?; gemm(&nm("ssm_beta.weight"), &s.p, ldp, bo, false)?; bar();
            let st_off = li * conv_ch * (c.d_conv - 1);
            run("gdn_conv", &[&s.p, &m.cst, &s.cv, &wt.f32b], &[c.nk, st_off, wt.f(&nm("ssm_conv1d.weight"))?, c.d_conv, c.eps.to_bits(), ldp, conv_ch, conv_ch / 128], n * (conv_ch / 128))?; bar();
            run("gdn_convst", &[&s.p, &m.cst], &[st_off, c.d_conv, ldp, n], conv_ch / 128)?; if sync { bar() }
            if skip & 2 == 0 { pf.gdn(ctx, &[&s.cv, &s.p, &m.sst, &s.o, &wt.f32b], &[c.nk, c.nv, li * c.nv * ds_ * ds_ / 4, ao, bo, wt.f(&nm("ssm_dt.bias"))?, wt.f(&nm("ssm_a"))?, ldp, conv_ch, n], cur())?; } bar();
            run("gnorm", &[&s.o, &s.p, &s.a, &wt.f32b], &[c.nv, ldp, zo, wt.f(&nm("ssm_norm.weight"))?, c.eps.to_bits(), al], n * c.nv)?; bar();
            bprep(&nm("ssm_out.weight"), c.nv * ds_, true)?;
            gemm(&nm("ssm_out.weight"), &s.x, c.dim, 0, true)?; bar();
        } else {
            let (qn, kn) = (c.n_head * c.hd * 2, c.n_kv * c.hd); let ldp = qn + 2 * kn;
            let cache_off = li * c.max_ctx * c.n_kv * c.hd;
            bprep(&nm("attn_q.weight"), c.dim, false)?;
            gemm(&nm("attn_q.weight"), &s.p, ldp, 0, false)?; gemm(&nm("attn_k.weight"), &s.p, ldp, qn, false)?; gemm(&nm("attn_v.weight"), &s.p, ldp, qn + kn, false)?; bar();
            run("rope", &[&s.p, &m.kc, &m.vc, &wt.f32b], &[c.n_head, c.n_kv, cache_off, c.theta.to_bits(), wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?,
                c.eps.to_bits(), 2 * c.hd, qn, qn + kn, c.n_rot, pos0, ldp], n * (c.n_head + c.n_kv))?; bar();
            run("attn", &[&s.p, &m.kc, &m.vc, &s.a], &[c.n_head, c.n_kv, cache_off, 2 * c.hd, pos0, (1.0f32 / (c.hd as f32).sqrt()).to_bits(), ldp, al, n], n.div_ceil(16) * c.n_head)?; bar();
            bprep(&nm("attn_output.weight"), c.n_head * c.hd, false)?;
            gemm(&nm("attn_output.weight"), &s.x, c.dim, 0, true)?; bar();
        }
        }
        if part & 2 == 0 { continue }
        let (wg, wu) = (wt.wref(&nm("ffn_gate.weight"))?, wt.wref(&nm("ffn_up.weight"))?);
        let fused = s.bon.is_none() && wg.arena == wu.arena && crate::env("PF_SWI", 1) > 0 && skip & 1 == 0;
        norm_to(&nm("post_attention_norm.weight"), if fused { &s.a2 } else { &s.a })?;
        if fused {
            // fused gate|up GEMM + SwiGLU straight into the f16 activation buffer (s.a is not read by it: A input is s.a2)
            pf.gemm_swiglu(ctx, cur(), &wt.arenas[wg.arena], &wt.f32b, wg, wu, 0, c.ffn, &s.a2, al, n, &s.a, al, 0)?; bar();
        } else {
            bprep(&nm("ffn_gate.weight"), c.dim, false)?;
            gemm(&nm("ffn_gate.weight"), &s.p, 2 * c.ffn, 0, false)?; gemm(&nm("ffn_up.weight"), &s.p, 2 * c.ffn, c.ffn, false)?; bar();
            run("swiglu", &[&s.p, &s.a], &[c.ffn, n, al], (n * c.ffn).div_ceil(256))?; bar();
        }
        bprep(&nm("ffn_down.weight"), c.ffn, false)?;
        gemm(&nm("ffn_down.weight"), &s.x, c.dim, 0, true)?; bar();
    }
    Ok(())
}

/// Full prompt prefill. Resets the recurrent state, runs prompt[..L-1] in chunks of s.mmax and the last token through
/// the model's decode step (which writes m.logits, argmax, toks[L] and advances pos). Returns wall seconds.
/// NPU_AUTO (default at PF_M >= 2048 on ternary models), by prompt length L:
///   L <= 1024: GPU only (a fixed-M NPU GEMM on few tokens loses);
///   1024 < L <= 2048: three-block pipeline NPU_TB=3 (1.28x at pp2048, greedy 16/16);
///   L > 2048: NPU column split (1.12x at pp4096, greedy 16/16; NPU_TB is faster there but int8 activations flip a near-tie).
pub fn prefill(pf: &Pf, m: &Model, s: &Q35, prompt: &[u32]) -> Result<f64> {
    if let Some(t) = s.auto.as_deref() {
        let l = prompt.len();
        if l > 1024 && l <= 2048 { return prefill_body(pf, m, t, prompt) }
        if l <= 1024 { s.npu_on.set(false); let r = prefill_body(pf, m, s, prompt); s.npu_on.set(true); return r }
    }
    prefill_body(pf, m, s, prompt)
}
/// Prefill prompt[start..] on top of the state the scratch already holds for prompt[..start] (prefix-cache hit,
/// see qwen38::multi). GPU only (no NPU pipelines), PF_HEAD path: logits of the last token, greedy pick -> toks[L].
pub fn prefill_from(pf: &Pf, m: &Model, s: &Q35, prompt: &[u32], start: usize) -> Result<f64> {
    if start == 0 { return prefill(pf, m, s, prompt) }
    let st = s.auto.as_deref().map(|_| s).unwrap_or(s);
    if st.tb.is_some() { bail!("prefill_from: token-block pipeline state") }
    let on = st.npu_on.get(); st.npu_on.set(false);
    let r = prefill_body_at(pf, m, st, prompt, start, usize::MAX); st.npu_on.set(on); r
}
fn prefill_body(pf: &Pf, m: &Model, s: &Q35, prompt: &[u32]) -> Result<f64> { prefill_body_at(pf, m, s, prompt, 0, usize::MAX) }
/// Resumable prefill for strix-serve: process prompt[start..min(end, L)) (GPU only, PF_HEAD path); start == 0 resets the
/// recurrent state. Returns true once the whole prompt is done (logits + greedy toks[L] handed off, as `prefill`).
pub fn prefill_part(pf: &Pf, m: &Model, s: &Q35, prompt: &[u32], start: usize, end: usize) -> Result<bool> {
    if s.tb.is_some() { bail!("prefill_part: token-block pipeline state") }
    let on = s.npu_on.get(); s.npu_on.set(false);
    let r = prefill_body_at(pf, m, s, prompt, start, end); s.npu_on.set(on); r?;
    Ok(end >= prompt.len())
}
fn prefill_body_at(pf: &Pf, m: &Model, s: &Q35, prompt: &[u32], start: usize, end: usize) -> Result<f64> {
    let l = prompt.len(); if l == 0 || start >= l || l + 1 > m.cfg.max_ctx as usize { bail!("prompt length") }
    let t0 = std::time::Instant::now();
    if start == 0 { m.cst.bytes_mut().fill(0); m.sst.bytes_mut().fill(0); }
    // PF_HEAD=1 (default): all L tokens batched + M=1 output projection; 0: last token through the decode step
    let head = crate::env("PF_HEAD", 1) > 0;
    let lb = if head { l } else { l - 1 };
    let nl = (m.cfg.n_layer as usize).min(crate::env("PF_NL", 999) as usize);
    let lps = crate::env("PF_LPS", 8) as usize;
    if start > 0 && !head { bail!("prefill_from needs PF_HEAD=1") }
    let mut pos = start;
    let stop = lb.min(end);
    while pos < stop {
        if let Some(sb) = s.tb.as_deref() { pos = match s.tc.as_deref() { Some(sc) => tb3_chunk(pf, m, s, sb, sc, prompt, pos, lb, nl, head)?, None => tb_chunk(pf, m, s, sb, prompt, pos, lb, nl, head)? }; continue }
        let n = (stop - pos).min(s.mmax as usize);
        embed(m, s, &prompt[pos..pos + n])?;
        let last_chunk = pos + n == lb;
        // several submits per chunk keep each command buffer well under the GPU job timeout
        for l0 in (0..nl).step_by(lps) {
            let mut err = None;
            let tail = head && last_chunk && l0 + lps >= nl && nl == m.cfg.n_layer as usize;
            let body = |cb: vk::CommandBuffer| {
                let pre = if l0 == 0 { match s.bon.as_ref() { Some(b) if b.inv_embed => b.signs(m.cfg.dim).and_then(|sg| {
                    pf.run(&m.ctx, "fwht", &[&s.x, &s.x, sg], &[m.cfg.dim, 0, 0, 2, 1, 1, 1], m.cfg.dim / 1024 * n as u32, cb)?; m.ctx.barrier(cb); Ok(()) }), _ => Ok(()) } } else { Ok(()) };
                let r = pre.and_then(|_| record(pf, m, s, cb, n as u32, pos as u32, l0..(l0 + lps).min(nl))).and_then(|_| if tail { record_head(pf, m, s, s.npu.as_ref().filter(|_| s.npu_on.get()).map_or(cb, |x| x.cur.get()), n as u32 - 1) } else { Ok(()) });
                if let Err(e) = r { err = Some(e) } };
            match s.npu.as_ref().filter(|_| s.npu_on.get()) { Some(sp) => sp.run(&m.ctx, body)?, None => { m.ctx.run(body)?; } }
            if let Some(e) = err { return Err(e) }
        }
        pos += n;
    }
    if let Some(sp) = s.npu.as_ref().filter(|_| s.npu_on.get()) { eprintln!("[npu split] {} calls, npu {:.1} ms, host wait for prep {:.1} ms, total so far {:.1} ms", sp.calls.get(), sp.t_npu.get() * 1e3, sp.t_wait.get() * 1e3, t0.elapsed().as_secs_f64() * 1e3); if crate::env("NPU_STATS", 0) > 0 { eprintln!("[npu split] {}", sp.stats()) } }
    let toks = unsafe { std::slice::from_raw_parts_mut(m.toks.ptr as *mut u32, m.toks.size as usize / 4) };
    let au = unsafe { std::slice::from_raw_parts_mut(m.aux.ptr as *mut u32, 1024) };
    if nl < m.cfg.n_layer as usize || pos < lb { return Ok(t0.elapsed().as_secs_f64()) } // debug partial trunk / resumable part: no hand-off
    toks[..l].copy_from_slice(prompt);
    if head {
        // greedy pick of the first new token (what the decode argmax/sample kernels would do), then hand the
        // state to decode: the next step embeds toks[L] at position L and writes toks[L+1]
        let lg = &m.logits.as_f32()[..m.cfg.vocab as usize];
        let am = lg.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        toks[l] = am as u32;
        au[160] = l as u32; au[161] = (l + 1) as u32; au[162] = 0;
    } else {
        au[160] = (l - 1) as u32; au[161] = l as u32; au[162] = 0;
        m.ctx.submit_repeated(m.step, 1)?;
    }
    Ok(t0.elapsed().as_secs_f64())
}

/// Final norm of chunk row `row` + lm_head (M=1 GEMM) into m.logits.
/// NPU_TB: one chunk as two token blocks. Per layer: A's mixer (GPU) -> B's mixer (GPU) while the NPU runs A's
/// gate/up -> A's SwiGLU + down prep (GPU, tiny) -> B's FFN (GPU) while the NPU runs A's down -> A's residual add.
fn tb_chunk(pf: &Pf, m: &Model, s: &Q35, sb: &Q35, prompt: &[u32], pos: usize, lb: usize, nl: usize, head: bool) -> Result<usize> {
    let sp = s.npu.as_ref().unwrap();
    let n = (lb - pos).min((s.mmax + sb.mmax) as usize); let ha = (n / 2).min(s.mmax as usize); let hn = n - ha;
    let last = pos + n == lb;
    if ha > 0 { embed(m, s, &prompt[pos..pos + ha])? } embed(m, sb, &prompt[pos + ha..pos + n])?;
    let pre = |cb: vk::CommandBuffer, st: &Q35, k: usize, l: usize| -> Result<()> {
        if l == 0 { if let Some(b) = st.bon.as_ref() { if b.inv_embed { let sg = b.signs(m.cfg.dim)?;
            pf.run(&m.ctx, "fwht", &[&st.x, &st.x, sg], &[m.cfg.dim, 0, 0, 2, 1, 1, 1], m.cfg.dim / 1024 * k as u32, cb)?; m.ctx.barrier(cb); } } } Ok(()) };
    let every = crate::env("NPU_TB_SYNC", 16) as usize; // host sync every N layers (1 = round 18)
    for l in 0..nl {
        let tail = head && last && l + 1 == nl && nl == m.cfg.n_layer as usize;
        let (mut e0, mut e1, mut e2) = (None, None, None);
        let b0 = sp.rec(&m.ctx, |cb| if let Err(e) = pre(cb, sb, hn, l).and_then(|_| record_part(pf, m, sb, cb, hn as u32, (pos + ha) as u32, l..l + 1, 1)) { e0 = Some(e) })?;
        let b1 = sp.rec(&m.ctx, |cb| if let Err(e) = record_part(pf, m, sb, cb, hn as u32, (pos + ha) as u32, l..l + 1, 2)
            .and_then(|_| if tail { record_head(pf, m, sb, cb, hn as u32 - 1) } else { Ok(()) }) { e1 = Some(e) })?;
        sp.run_tb(&m.ctx, |cb| if ha > 0 { if let Err(e) = pre(cb, s, ha, l).and_then(|_| record_part(pf, m, s, cb, ha as u32, pos as u32, l..l + 1, 3)) { e2 = Some(e) } }, vec![b0, b1], l + 1 == nl || (l + 1) % every == 0)?;
        for e in [e0, e1, e2] { if let Some(e) = e { return Err(e) } }
    }
    Ok(pos + n)
}
/// NPU_TB=3: blocks A and B alternate on the NPU (whole FFN each, separate NPU buffers) while the GPU runs the
/// mixers of all three blocks and C's FFN. NPU order per layer: gate/up A, gate/up B, down A, down B. Layer l+1's A
/// mixer is queued right after A's residual add, so it runs on the GPU while the NPU runs B's down of layer l.
fn tb3_chunk(pf: &Pf, m: &Model, s: &Q35, sb: &Q35, sc: &Q35, prompt: &[u32], pos: usize, lb: usize, nl: usize, head: bool) -> Result<usize> {
    let sp = s.npu.as_ref().unwrap(); let ctx = &m.ctx;
    let n = (lb - pos).min((2 * s.mmax + sc.mmax) as usize);
    let ha = (n / 3).min(s.mmax as usize); let hc = n - 2 * ha;
    if ha == 0 || hc as u32 > sc.mmax { bail!("NPU_TB=3: chunk of {n} tokens does not fit") }
    let (pa, pb, pc) = (pos, pos + ha, pos + 2 * ha);
    embed(m, s, &prompt[pa..pb])?; embed(m, sb, &prompt[pb..pc])?; embed(m, sc, &prompt[pc..pos + n])?;
    let last = pos + n == lb;
    let pre = |cb: vk::CommandBuffer, st: &Q35, k: usize, l: usize| -> Result<()> {
        if l == 0 { if let Some(b) = st.bon.as_ref() { if b.inv_embed { let sg = b.signs(m.cfg.dim)?;
            pf.run(&m.ctx, "fwht", &[&st.x, &st.x, sg], &[m.cfg.dim, 0, 0, 2, 1, 1, 1], m.cfg.dim / 1024 * k as u32, cb)?; m.ctx.barrier(cb); } } } Ok(()) };
    let err: std::cell::RefCell<Option<anyhow::Error>> = std::cell::RefCell::new(None);
    let keep = |r: Result<()>| if let Err(e) = r { if err.borrow().is_none() { *err.borrow_mut() = Some(e) } };
    let rec_a = |l: usize| sp.rec_layer(ctx, 0, |cb| keep(pre(cb, s, ha, l).and_then(|_| record_part(pf, m, s, cb, ha as u32, pa as u32, l..l + 1, 3))));
    let rec_b = |l: usize| sp.rec_layer(ctx, 1, |cb| keep(pre(cb, sb, ha, l).and_then(|_| record_part(pf, m, sb, cb, ha as u32, pb as u32, l..l + 1, 3))));
    let tail = |l: usize| head && last && l + 1 == nl && nl == m.cfg.n_layer as usize;
    let rec_c = |l: usize, part: u32| sp.rec(ctx, |cb| keep((if part == 1 { pre(cb, sc, hc, l) } else { Ok(()) })
        .and_then(|_| record_part(pf, m, sc, cb, hc as u32, pc as u32, l..l + 1, part))
        .and_then(|_| if part == 2 && tail(l) { record_head(pf, m, sc, cb, hc as u32 - 1) } else { Ok(()) })));
    let (mut la, mut lbk) = (rec_a(0)?, rec_b(0)?);
    sp.sub_seg(ctx, &mut la, 0)?; sp.sub_seg(ctx, &mut lbk, 0)?; sp.sub(ctx, rec_c(0, 1)?)?;
    for l in 0..nl {
        if la.nseg() != 2 || lbk.nseg() != 2 { bail!("NPU_TB=3: expected 2 NPU segments per block-layer, got {} / {}", la.nseg(), lbk.nseg()) }
        sp.npu_seg(ctx, &la, 0)?;                                   // NPU: gate/up A  | GPU: mixer B, mixer C
        sp.sub_seg(ctx, &mut la, 1)?; sp.sub(ctx, rec_c(l, 2)?)?;
        sp.npu_seg(ctx, &lbk, 0)?;                                  // NPU: gate/up B  | GPU: SwiGLU A, FFN C
        sp.sub_seg(ctx, &mut lbk, 1)?;
        sp.npu_seg(ctx, &la, 1)?;                                   // NPU: down A     | GPU: FFN C, SwiGLU B
        sp.sub_last(ctx, &la)?;
        if l + 1 < nl { la = rec_a(l + 1)?; sp.sub_seg(ctx, &mut la, 0)?; }
        sp.npu_seg(ctx, &lbk, 1)?;                                  // NPU: down B     | GPU: add A, mixer A(l+1)
        sp.sub_last(ctx, &lbk)?;
        if l + 1 < nl { lbk = rec_b(l + 1)?; sp.sub_seg(ctx, &mut lbk, 0)?; sp.sub(ctx, rec_c(l + 1, 1)?)?; }
        // sync only at the end: a mid-chunk sync would destroy the fences of the already-submitted next-layer segments
        if l + 1 == nl { sp.sync(ctx)? }
        if let Some(e) = err.borrow_mut().take() { return Err(e) }
    }
    Ok(pos + n)
}
fn record_head(pf: &Pf, m: &Model, s: &Q35, cb: vk::CommandBuffer, row: u32) -> Result<()> {
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    pf.run(ctx, "rmsnorm", &[&s.x, &s.a, &wt.f32b], &[c.dim, s.al, wt.f("output_norm.weight")?, c.eps.to_bits(), row], 1, cb)?; ctx.barrier(cb);
    let w = wt.wref("output.weight").or_else(|_| wt.wref("token_embd.weight"))?;
    if is_tern(&w) {
        let b = s.bon.as_ref().unwrap(); let k = c.dim; let hn = if wt.wref("output.weight").is_ok() { "output.weight" } else { "token_embd.weight" };
        pf.run(ctx, "h2f", &[&s.a, &b.xf], &[k, s.al, 1], k.div_ceil(256), cb)?; ctx.barrier(cb);
        let src = if b.rot(hn) { pf.run(ctx, "fwht", &[&b.xf, &b.xr, b.signs(k)?], &[k, 0, 0, 1, 1, 1, 1], k / 1024, cb)?; ctx.barrier(cb); &b.xr } else { &b.xf };
        pf.run(ctx, "quantx", &[src, &b.xq, &b.xs], &[1, k], 1, cb)?; ctx.barrier(cb);
        pf.tgemm(ctx, cb, &wt.arenas[w.arena], w, 0, w.n, &b.xq, &b.xs, 1, &m.logits, w.n, 0, false)?; ctx.barrier(cb);
        return Ok(());
    }
    pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: s.al, a_row0: 0, m: 1, y: &m.logits, ldy: w.n, y_col: 0, acc: false })?;
    ctx.barrier(cb);
    Ok(())
}
