//! Cross-prompt batched prefill for strix-serve: one pass runs one chunk from each of several sequences. Row-wise work
//! (norms, every projection GEMM, SwiGLU) runs once over the packed rows, so the coop-matrix GEMM tile rows (256 tokens)
//! are filled by several prompts instead of each prompt paying for a partial tail tile. Sequence-dependent work
//! (GDN conv + scan + gated norm, RoPE + attention) runs per segment on the scratch `t`, with segment rows copied in/out,
//! and reads/writes that sequence's own state (`Seg::st` = kc, vc, cst, sst of its qwen38::multi::Slot).
//! A segment's numerics do not depend on its pass neighbours (GEMM rows are independent), so with fixed chunk
//! boundaries per sequence the result equals the single-sequence pass. Ternary (Bonsai) / NPU paths are not supported.
use super::*;

pub struct Seg<'a> { pub toks: &'a [u32], pub pos0: u32, pub st: [&'a Buffer; 4], pub last: bool }

fn xbar(m: &Model, cb: vk::CommandBuffer) {
    let a = vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ;
    let st = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
    let mb = [vk::MemoryBarrier::default().src_access_mask(a).dst_access_mask(a)];
    unsafe { m.ctx.device.cmd_pipeline_barrier(cb, st, st, vk::DependencyFlags::empty(), &mb, &[], &[]) };
}
fn cp(m: &Model, cb: vk::CommandBuffer, a: &Buffer, b: &Buffer, ao: u64, bo: u64, n: u64) {
    if n > 0 { unsafe { m.ctx.device.cmd_copy_buffer(cb, a.buf, b.buf, &[vk::BufferCopy { src_offset: ao, dst_offset: bo, size: n }]) } }
}

/// Run one packed pass over `segs` (sum of lengths <= s.mmax, each <= t.mmax). For each segment with `last`, returns
/// (segment index, greedy next token) and leaves its logits... (logits are overwritten per segment; greedy only).
pub fn prefill_pass(pf: &Pf, m: &Model, s: &Q35, t: &Q35, segs: &[Seg]) -> Result<Vec<(usize, u32, Vec<f32>)>> {
    if s.bon.is_some() || s.tb.is_some() { bail!("prefill_pass: ternary / token-block paths not supported") }
    let nt: usize = segs.iter().map(|g| g.toks.len()).sum();
    if nt == 0 || nt > s.mmax as usize || segs.iter().any(|g| g.toks.len() > t.mmax as usize) { bail!("prefill_pass: {nt} rows") }
    let all: Vec<u32> = segs.iter().flat_map(|g| g.toks.iter().copied()).collect();
    embed(m, s, &all)?;
    let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx);
    let n = nt as u32; let al = s.al;
    let ds_ = c.d_state; let conv_ch = 2 * c.nk * ds_ + c.nv * ds_;
    let r0: Vec<u32> = segs.iter().scan(0u32, |a, g| { let r = *a; *a += g.toks.len() as u32; Some(r) }).collect();
    let lps = crate::env("PF_LPS", 8) as usize; let nl = c.n_layer as usize;
    for l0 in (0..nl).step_by(lps) {
        let mut err = Ok(());
        m.ctx.run(|cb| { err = (|| -> Result<()> {
            let bar = || ctx.barrier(cb);
            let gemm = |name: &str, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> { let w = wt.wref(name)?;
                pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: al, a_row0: 0, m: n, y, ldy, y_col, acc }) };
            let norm_to = |wn: &str, dst: &Buffer| -> Result<()> { pf.run(ctx, "rmsnorm", &[&s.x, dst, &wt.f32b], &[c.dim, al, wt.f(wn)?, c.eps.to_bits(), 0], n, cb)?; bar(); Ok(()) };
            for l in l0..(l0 + lps).min(nl) {
                let nm = |x: &str| format!("blk.{l}.{x}");
                let li = s.lidx[l];
                norm_to(&nm("attn_norm.weight"), &s.a)?;
                if c.recr[l] {
                    let ldp = conv_ch + c.nv * ds_ + 2 * c.nv; let (zo, ao, bo) = (conv_ch, conv_ch + c.nv * ds_, conv_ch + c.nv * ds_ + c.nv);
                    gemm(&nm("attn_qkv.weight"), &s.p, ldp, 0, false)?; gemm(&nm("attn_gate.weight"), &s.p, ldp, zo, false)?;
                    gemm(&nm("ssm_alpha.weight"), &s.p, ldp, ao, false)?; gemm(&nm("ssm_beta.weight"), &s.p, ldp, bo, false)?; xbar(m, cb);
                    let st_off = li * conv_ch * (c.d_conv - 1);
                    for (g, &r) in segs.iter().zip(&r0) { let k = g.toks.len() as u32;
                        cp(m, cb, &s.p, &t.p, r as u64 * ldp as u64 * 4, 0, k as u64 * ldp as u64 * 4); xbar(m, cb);
                        pf.run(ctx, "gdn_conv", &[&t.p, g.st[2], &t.cv, &wt.f32b], &[c.nk, st_off, wt.f(&nm("ssm_conv1d.weight"))?, c.d_conv, c.eps.to_bits(), ldp, conv_ch, conv_ch / 128], k * (conv_ch / 128), cb)?; bar();
                        pf.run(ctx, "gdn_convst", &[&t.p, g.st[2]], &[st_off, c.d_conv, ldp, k], conv_ch / 128, cb)?;
                        pf.gdn(ctx, &[&t.cv, &t.p, g.st[3], &t.o, &wt.f32b], &[c.nk, c.nv, li * c.nv * ds_ * ds_ / 4, ao, bo, wt.f(&nm("ssm_dt.bias"))?, wt.f(&nm("ssm_a"))?, ldp, conv_ch, k], cb)?; bar();
                        pf.run(ctx, "gnorm", &[&t.o, &t.p, &t.a, &wt.f32b], &[c.nv, ldp, zo, wt.f(&nm("ssm_norm.weight"))?, c.eps.to_bits(), al], k * c.nv, cb)?; xbar(m, cb);
                        cp(m, cb, &t.a, &s.a, 0, r as u64 * al as u64 * 2, k as u64 * al as u64 * 2); xbar(m, cb); }
                    gemm(&nm("ssm_out.weight"), &s.x, c.dim, 0, true)?; bar();
                } else {
                    let (qn, kn) = (c.n_head * c.hd * 2, c.n_kv * c.hd); let ldp = qn + 2 * kn;
                    let cache_off = li * c.max_ctx * c.n_kv * c.hd;
                    gemm(&nm("attn_q.weight"), &s.p, ldp, 0, false)?; gemm(&nm("attn_k.weight"), &s.p, ldp, qn, false)?; gemm(&nm("attn_v.weight"), &s.p, ldp, qn + kn, false)?; xbar(m, cb);
                    for (g, &r) in segs.iter().zip(&r0) { let k = g.toks.len() as u32;
                        cp(m, cb, &s.p, &t.p, r as u64 * ldp as u64 * 4, 0, k as u64 * ldp as u64 * 4); xbar(m, cb);
                        pf.run(ctx, "rope", &[&t.p, g.st[0], g.st[1], &wt.f32b], &[c.n_head, c.n_kv, cache_off, c.theta.to_bits(), wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?,
                            c.eps.to_bits(), 2 * c.hd, qn, qn + kn, c.n_rot, g.pos0, ldp], k * (c.n_head + c.n_kv), cb)?; bar();
                        pf.run(ctx, "attn", &[&t.p, g.st[0], g.st[1], &t.a], &[c.n_head, c.n_kv, cache_off, 2 * c.hd, g.pos0, (1.0f32 / (c.hd as f32).sqrt()).to_bits(), ldp, al, k], k.div_ceil(16) * c.n_head, cb)?; xbar(m, cb);
                        cp(m, cb, &t.a, &s.a, 0, r as u64 * al as u64 * 2, k as u64 * al as u64 * 2); xbar(m, cb); }
                    gemm(&nm("attn_output.weight"), &s.x, c.dim, 0, true)?; bar();
                }
                let (wg, wu) = (wt.wref(&nm("ffn_gate.weight"))?, wt.wref(&nm("ffn_up.weight"))?);
                let fused = wg.arena == wu.arena && crate::env("PF_SWI", 1) > 0;
                norm_to(&nm("post_attention_norm.weight"), if fused { &s.a2 } else { &s.a })?;
                if fused { pf.gemm_swiglu(ctx, cb, &wt.arenas[wg.arena], &wt.f32b, wg, wu, 0, c.ffn, &s.a2, al, n, &s.a, al, 0)?; bar(); }
                else { gemm(&nm("ffn_gate.weight"), &s.p, 2 * c.ffn, 0, false)?; gemm(&nm("ffn_up.weight"), &s.p, 2 * c.ffn, c.ffn, false)?; bar();
                    pf.run(ctx, "swiglu", &[&s.p, &s.a], &[c.ffn, n, al], (n * c.ffn).div_ceil(256), cb)?; bar(); }
                gemm(&nm("ffn_down.weight"), &s.x, c.dim, 0, true)?; bar();
            }
            Ok(()) })(); })?;
        err?;
    }
    let mut out = vec![];
    for (i, (g, &r)) in segs.iter().zip(&r0).enumerate() { if !g.last { continue }
        let mut err = Ok(()); m.ctx.run(|cb| { err = record_head(pf, m, s, cb, r + g.toks.len() as u32 - 1); })?; err?;
        let lg = m.logits.as_f32()[..c.vocab as usize].to_vec();
        let am = lg.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        out.push((i, am as u32, lg)); }
    Ok(out)
}
