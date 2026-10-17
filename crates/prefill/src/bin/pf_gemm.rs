//! GEMM correctness + throughput on real weights: pf_gemm <model.gguf> [M=512] [filter]
use anyhow::Result;
use gguf::dequant::dequant_row;
use prefill::{GemmOp, Pf};
use vkrt::Ctx;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let filt = a.get(3).cloned().unwrap_or_default();
    let ctx = Ctx::new()?;
    let wt = rtcore::Weights::load(&ctx, &a[1])?;
    let pf = Pf::new(&ctx)?;
    // pick one tensor per (type, shape) among blk.0..blk.4 + output
    let mut seen = std::collections::HashSet::new();
    let mut names: Vec<String> = wt.w.keys().filter(|n| (n.starts_with("blk.0.") || n.starts_with("blk.3.") || n.starts_with("blk.1.") || n.as_str() == "output.weight") && n.contains(&filt)).cloned().collect();
    names.sort();
    let iters = prefill::env("ITERS", 5);
    println!("M={m}\n| tensor | type | N x K | us | TFLOPS | weight GB/s | max_abs/max|ref| |\n|---|---|---|---|---|---|---|");
    let mut tot_flop = 0f64; let mut tot_t = 0f64;
    for n in names {
        let mut w = wt.wref(&n)?; if w.e > 1 && prefill::env("QMMH", 0) > 0 { let eb = w.bytes / w.e as u64; w = rtcore::WRef { bytes: eb, e: 1, ..w }; } if w.e > 1 || w.k % 64 != 0 { continue }
        if !seen.insert((format!("{:?}", w.ty), w.k, w.n)) { continue }
        let (k, nn) = (w.k as usize, w.n as usize);
        let ab = ctx.buffer((m as usize * k * 2) as u64)?; let yb = ctx.buffer((m as usize * nn * 4) as u64)?;
        let mut s = 1u64; let av: Vec<f32> = (0..m as usize * k).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / 16777216.0) - 0.5 }).collect();
        for (i, v) in av.iter().enumerate() { let h = half::f16::from_f32(*v); ab.bytes_mut()[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes()); }
        let op = GemmOp { w, row0: 0, rows: w.n, a: &ab, lda: w.k, a_row0: 0, m, y: &yb, ldy: w.n, y_col: 0, acc: false };
        let ar = &wt.arenas[w.arena];
        ctx.run(|cb| { pf.gemm(&ctx, cb, ar, &wt.f32b, &op).unwrap(); })?;
        let y1 = yb.as_f32()[..m as usize * nn].to_vec(); yb.as_f32_mut().fill(1.0);
        ctx.run(|cb| { pf.gemm(&ctx, cb, ar, &wt.f32b, &op).unwrap(); })?;
        let nd = y1.iter().zip(&yb.as_f32()[..m as usize * nn]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        if nd > 0 { let fi = y1.iter().zip(yb.as_f32()).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap(); println!("NONDET {n}: {nd} differing, first at m={} n={}", fi / nn, fi % nn); }
        // reference: rows sampled, all m for a few tokens
        let t = wt.g.tensor(&n).unwrap(); let data = wt.g.data(t); let e0 = wt.wref(&n)?.e.max(1) as usize; let rb = data.len() / (nn * if prefill::env("QMMH", 0) > 0 { e0 } else { 1 }); // stacked expert tensors: slice 0 only (row bytes = total / (rows * experts))
        let (mut e, mut r) = (0f32, 0f32); let mut wrow = vec![0f32; k];
        let q50 = |src: &[u8], y: &mut [f32]| for (bi, yb) in y.chunks_mut(32).enumerate() { let b = &src[bi * 22..(bi + 1) * 22]; let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32(); let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            for j in 0..16 { yb[j] = d * ((((b[6 + j] & 15) as u32) | (((qh >> j) & 1) << 4)) as f32 - 16.0); yb[j + 16] = d * ((((b[6 + j] >> 4) as u32) | (((qh >> (j + 16)) & 1) << 4)) as f32 - 16.0); } };
        let mut qref: Vec<(usize, usize, f64)> = vec![];
        { for row in [0, 1, nn / 3, nn / 2, nn - 1] { if matches!(w.ty, gguf::GgmlType::Other(6)) { q50(&data[row * rb..(row + 1) * rb], &mut wrow) } else { dequant_row(w.ty, &data[row * rb..(row + 1) * rb], &mut wrow) }
            for tok in [0usize, 1, 77usize.min(m as usize - 1), (m as usize) / 2, m as usize - 1] { let ah: Vec<f32> = av[tok * k..(tok + 1) * k].iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
                qref.push((tok, row, wrow.iter().zip(&ah).map(|(a, b)| *a as f64 * *b as f64).sum())); } } }
        if matches!(w.ty, gguf::GgmlType::Other(6)) { let (mut e5, mut r5) = (0f32, 0f32); for &(tok, row, rv) in &qref { e5 = e5.max((yb.as_f32()[tok * nn + row] - rv as f32).abs()); r5 = r5.max(rv.abs() as f32); } println!("  Q5_0 f16-path vs CPU ref: max_abs {e5:.3e} / max|ref| {r5:.3e}"); }
        if !matches!(w.ty, gguf::GgmlType::Other(6)) { for row in [0, 1, nn / 3, nn / 2, nn - 1] { dequant_row(w.ty, &data[row * rb..(row + 1) * rb], &mut wrow);
            for tok in [0usize, 1, 77usize.min(m as usize - 1), (m as usize) / 2, m as usize - 1] { let ah: Vec<f32> = av[tok * k..(tok + 1) * k].iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
                let refv: f64 = wrow.iter().zip(&ah).map(|(a, b)| *a as f64 * *b as f64).sum(); let g = yb.as_f32()[tok * nn + row];
                e = e.max((g - refv as f32).abs()); r = r.max(refv.abs() as f32); } } }
        let mut tm = ctx.run(|cb| for _ in 0..iters { pf.gemm(&ctx, cb, ar, &wt.f32b, &op).unwrap(); ctx.barrier(cb); })? / iters as f64;
        if prefill::env("QMMH", 0) > 0 && Pf::qh_ok(w) {
            let y0 = yb.as_f32()[..m as usize * nn].to_vec(); yb.as_f32_mut().fill(0.0);
            ctx.run(|cb| { pf.qgemmh(&ctx, cb, ar, w, m, &ab, w.k, &yb, w.n, 0).unwrap(); })?;
            let yh = &yb.as_f32()[..m as usize * nn]; let nd = y0.iter().zip(yh).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let (mut eq, mut rq) = (0f32, 0f32); for (a, b) in y0.iter().zip(yh) { eq = eq.max((a - b).abs()); rq = rq.max(a.abs()); }
            let th = ctx.run(|cb| for _ in 0..iters { pf.qgemmh(&ctx, cb, ar, w, m, &ab, w.k, &yb, w.n, 0).unwrap(); ctx.barrier(cb); })? / iters as f64;
            let fl = 2.0 * m as f64 * k as f64 * nn as f64;
            if !qref.is_empty() { let (mut eh, mut rh) = (0f32, 0f32); for &(tok, row, rv) in &qref { eh = eh.max((yh[tok * nn + row] - rv as f32).abs()); rh = rh.max(rv.abs() as f32); } println!("  qgemmh vs CPU ref: max_abs {eh:.3e} / max|ref| {rh:.3e}"); }
            println!("  QMMH {n}: f16 {:.0} us {:.2} TF | qgemmh {:.0} us {:.2} TF | differing {nd}/{} max|h-f16|/max {:.2e}", tm * 1e6, fl / tm / 1e12, th * 1e6, fl / th / 1e12, y0.len(), eq / rq);
        }
        if prefill::env("QMM", 0) > 0 && w.ty == gguf::GgmlType::Q4K && k % 256 == 0 {
            // packed-LDS int8 GEMM (pf_qgemm): q8 activations + Q4_K; compare with the f16 path, time incl. the quant pass
            let ldx = m.div_ceil(64) * 64 + 64;
            let xq = ctx.buffer(ldx as u64 * k as u64)?; let xs = ctx.buffer(ldx as u64 * (k / 32) as u64 * 4)?; let xu = ctx.buffer(ldx as u64 * (k / 32) as u64 * 4)?;
            let y0 = yb.as_f32()[..m as usize * nn].to_vec(); yb.as_f32_mut().fill(0.0);
            ctx.run(|cb| { pf.q8(&ctx, cb, &ab, w.k, m, w.k, ldx, &xq, &xs, &xu).unwrap(); ctx.barrier(cb); pf.qgemm(&ctx, cb, ar, w, m, &xq, &xs, &xu, ldx, &yb, w.n, 0).unwrap(); })?;
            let (mut eq, mut rq) = (0f32, 0f32); for (a, b) in y0.iter().zip(&yb.as_f32()[..m as usize * nn]) { eq = eq.max((a - b).abs()); rq = rq.max(a.abs()); }
            let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>();
            let yq = &yb.as_f32()[..m as usize * nn]; let cos = dot(&y0, yq) / (dot(&y0, &y0).sqrt() * dot(yq, yq).sqrt());
            let tq = ctx.run(|cb| for _ in 0..iters { pf.q8(&ctx, cb, &ab, w.k, m, w.k, ldx, &xq, &xs, &xu).unwrap(); ctx.barrier(cb); pf.qgemm(&ctx, cb, ar, w, m, &xq, &xs, &xu, ldx, &yb, w.n, 0).unwrap(); ctx.barrier(cb); })? / iters as f64;
            let tg = ctx.run(|cb| for _ in 0..iters { pf.qgemm(&ctx, cb, ar, w, m, &xq, &xs, &xu, ldx, &yb, w.n, 0).unwrap(); ctx.barrier(cb); })? / iters as f64;
            let fl = 2.0 * m as f64 * k as f64 * nn as f64;
            println!("  QMM {n}: f16 {:.0} us {:.2} TF | q8+qgemm {:.0} us | qgemm {:.0} us {:.2} TF | max|q-f16|/max {:.2e} cos {:.7}", tm * 1e6, fl / tm / 1e12, tq * 1e6, tg * 1e6, fl / tg / 1e12, eq / rq, cos);
            ctx.free_buffer(xq); ctx.free_buffer(xs); ctx.free_buffer(xu);
        }
        if prefill::env("T8", 0) > 0 {
            // T8 repack: time, check against the original-format result, report memory
            let t0 = std::time::Instant::now(); let pk = prefill::repack_t8(data, w.ty, nn, k); let tc = t0.elapsed().as_secs_f64();
            let wb = ctx.buffer(pk.len() as u64)?; wb.bytes_mut()[..pk.len()].copy_from_slice(&pk);
            let y0 = yb.as_f32()[..m as usize * nn].to_vec();
            ctx.run(|cb| { pf.gemm_t8(&ctx, cb, &wb, &wt.f32b, w.n, w.k, &ab, w.k, m, &yb, w.n, 0, false).unwrap(); })?;
            let (mut e8, mut r8) = (0f32, 0f32); for (a, b) in y0.iter().zip(&yb.as_f32()[..m as usize * nn]) { e8 = e8.max((a - b).abs()); r8 = r8.max(a.abs()); }
            tm = ctx.run(|cb| for _ in 0..iters { pf.gemm_t8(&ctx, cb, &wb, &wt.f32b, w.n, w.k, &ab, w.k, m, &yb, w.n, 0, false).unwrap(); ctx.barrier(cb); })? / iters as f64;
            println!("  T8 {n}: repack {:.2} s, {:.1} MB (orig {:.1} MB), max|t8-orig|/max {:.2e}", tc, pk.len() as f64 / 1e6, w.bytes as f64 / 1e6, e8 / r8);
            ctx.free_buffer(wb);
        }
        let fl = 2.0 * m as f64 * k as f64 * nn as f64; tot_flop += fl; tot_t += tm;
        println!("| {n} | {:?} | {nn}x{k} | {:.0} | {:.2} | {:.1} | {:.2e} |", w.ty, tm * 1e6, fl / tm / 1e12, w.bytes as f64 / tm / 1e9, e / r);
        ctx.free_buffer(ab); ctx.free_buffer(yb);
    }
    println!("aggregate {:.2} TFLOPS", tot_flop / tot_t / 1e12);
    Ok(())
}
