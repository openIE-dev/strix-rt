//! Check the generic-format GEMV (gemvq) against the CPU dequant reference: one tensor per format found in a GGUF.
use anyhow::Result;
use gguf::{dequant::dequant_row, Gguf};
use rtcore::{gkq, qtype, WRef, GQ};
fn main() -> Result<()> {
    let p = std::env::args().nth(1).unwrap();
    let g = Gguf::open(&p)?; let ctx = vkrt::Ctx::new()?;
    let mut seen = std::collections::HashSet::new();
    let ks = [gkq(&ctx, "X", (8, 64), false, 32)?, gkq(&ctx, "X", (32, 128), false, 32)?];

    for t in &g.tensors {
        if qtype(t.ty).is_err() || !seen.insert(format!("{:?}", t.ty)) { continue }
        let k = t.dims[0] as usize; let rb = t.row_bytes(); let rows = (t.dims[1] as usize).min(64);
        let wb = ctx.buffer((rb * rows) as u64)?; wb.bytes_mut().copy_from_slice(&g.data(t)[..rb * rows]);
        let xb = ctx.buffer(k as u64 * 4)?; let yb = ctx.buffer(rows as u64 * 4)?;
        let f = ctx.buffer(rtcore::TBL_FLOATS as u64 * 4)?; { let fm = f.as_f32_mut(); for (i, &v) in gguf::iq3s_grid::IQ3S_GRID.iter().enumerate() { fm[i] = f32::from_bits(v); } for (i, &v) in gguf::dequant::KV_IQ4NL.iter().enumerate() { fm[512 + i] = v as f32; } } let aux = ctx.buffer(1024)?; let xw = ctx.buffer(rows as u64 * 4)?;
        let mut s = 7u64; for v in xb.as_f32_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *v = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5; }
        let x = xb.as_f32().to_vec();
        let mut row = vec![0f32; k]; let mut r = vec![0f32; rows];
        for i in 0..rows { let src = &g.data(t)[i * rb..(i + 1) * rb];
            match t.ty { gguf::GgmlType::Other(c) if ternary::Fmt::from_ggml(c).is_some() => ternary::dequant_row(ternary::Fmt::from_ggml(c).unwrap(), src, &mut row), _ => dequant_row(t.ty, src, &mut row) } r[i] = row.iter().zip(&x).map(|(a, b)| a * b).sum(); }
        let rmax = r.iter().fold(0f32, |m, v| m.max(v.abs()));
        let w = WRef { arena: 0, off: 0, ty: t.ty, k: k as u32, n: rows as u32, e: 1, bytes: (rb * rows) as u64 };
        for gk in &ks {
            let (gq, _) = GQ::segs(&[w], None, &[], 1e-6)?;
            let ds = ctx.bind(&gk.k, &[&wb, &xb, &yb, &f, &aux, &xw])?;
            yb.as_f32_mut().fill(f32::NAN);
            ctx.run(|cb| ctx.dispatch(cb, &gk.k, ds, &gq.bytes(), gq.n.div_ceil(gk.rows_per_wg)))?;
            let err = yb.as_f32().iter().zip(&r).fold(0f32, |m, (a, b)| if a.is_nan() { f32::INFINITY } else { m.max((a - b).abs()) });
            println!("{:<10} {:<28} K={k:<6} rows/wg={:<3} max_abs_err={:.3e} rel={:.3e} {}", format!("{:?}", t.ty), t.name, gk.rows_per_wg, err, err / rmax, if err / rmax < 1e-4 { "OK" } else { "FAIL" });
        }
    }
    Ok(())
}
