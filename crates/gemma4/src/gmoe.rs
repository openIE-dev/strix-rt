//! Grouped-GEMM MoE prefill for Gemma 4 (26B-A4B) on crates/prefill's grouped expert GEMMs:
//! route (g4_router) -> g4_msort (tile table, n_exp <= 128) -> gather f16 rows -> grouped gate_up GEMM (Q4_K, gate | up
//! rows of the fused tensor) -> GeGLU -> grouped down GEMM -> g4_mcomb (route weight * down_exps.scale).
//! prefill has no Q5_1 GEMM, so down_exps (Q5_1) is requantized once to Q8_0 for prefill only (decode keeps Q5_1).
//! The new error per weight is <= 1/254 of a block's range, against Q5_1's own 1/31 step. G4_PF_GMOE=0 falls back to
//! per-column GEMVs.
use crate::Model;
use anyhow::{bail, Result};
use gguf::GgmlType;
use rtcore::WRef;
use vkrt::{Buffer, Ctx};

fn h2f(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let v = if e == 0 { (m as f32) * 2f32.powi(-24) } else if e == 31 { f32::INFINITY } else { f32::from_bits(((e + 112) << 23) | (m << 13)) };
    if s == 1 { -v } else { v }
}
fn f2h(v: f32) -> u16 {
    let b = v.to_bits(); let s = ((b >> 16) & 0x8000) as u16; let a = v.abs();
    if a < 6.1e-5 { return s | (a / 2f32.powi(-24)).round() as u16 }
    let e = ((b >> 23) & 255) as i32 - 112; let m = b & 0x7f_ffff;
    let mut r = ((e as u32) << 10) | (m >> 13); if m & 0x1000 != 0 { r += 1 }
    if e >= 31 { return s | 0x7bff } s | r as u16
}

/// Q5_1 tensor bytes -> Q8_0 bytes (same block order).
fn q51_to_q80(src: &[u8], dst: &mut [u8]) {
    let nb = src.len() / 24;
    let th = 16; let per = nb.div_ceil(th);
    std::thread::scope(|sc| {
        for (si, (s, d)) in src.chunks(per * 24).zip(dst.chunks_mut(per * 34)).enumerate() { let _ = si;
            sc.spawn(move || for (b, o) in s.chunks_exact(24).zip(d.chunks_exact_mut(34)) {
                let (dd, mm) = (h2f(u16::from_le_bytes([b[0], b[1]])), h2f(u16::from_le_bytes([b[2], b[3]])));
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let mut v = [0f32; 32];
                for j in 0..16 {
                    v[j] = dd * ((b[8 + j] & 15) as u32 | ((qh >> j) & 1) << 4) as f32 + mm;
                    v[j + 16] = dd * ((b[8 + j] >> 4) as u32 | ((qh >> (j + 16)) & 1) << 4) as f32 + mm;
                }
                let amax = v.iter().fold(0f32, |a, x| a.max(x.abs()));
                let sc8 = amax / 127.0; let h = f2h(sc8); let inv = if sc8 > 0.0 { 1.0 / h2f(h) } else { 0.0 };
                o[0..2].copy_from_slice(&h.to_le_bytes());
                for j in 0..32 { o[2 + j] = (v[j] * inv).round().clamp(-127.0, 127.0) as i8 as u8; }
            });
        }
    });
}

/// Per-layer prefill copies of the expert down projections (Q8_0) when the model's down_exps is Q5_1.
pub fn down_q8(ctx: &Ctx, m: &Model) -> Result<Vec<Option<(Buffer, WRef)>>> {
    let mut out = Vec::new();
    for l in 0..m.cfg.n_layer {
        let nm = format!("blk.{l}.ffn_down_exps.weight");
        let Ok(w) = m.wt.wref(&nm) else { out.push(None); continue };
        if w.ty != GgmlType::Other(7) { out.push(None); continue }
        let t = m.wt.g.tensor(&nm).ok_or_else(|| anyhow::anyhow!("{nm}"))?;
        let src = &m.wt.g.mmap[m.wt.g.data_offset + t.offset as usize..][..(w.k / 32 * 24) as usize * w.n as usize * w.e as usize];
        let bytes = (w.k / 32 * 34) as u64 * w.n as u64 * w.e as u64;
        if bytes > u32::MAX as u64 { bail!("down_q8: {nm} too large") }
        let b = ctx.buffer(bytes)?;
        q51_to_q80(src, &mut b.bytes_mut()[..bytes as usize]);
        out.push(Some((b, WRef { arena: usize::MAX, off: 0, ty: GgmlType::Q8_0, k: w.k, n: w.n, e: w.e, bytes })));
    }
    Ok(out)
}
