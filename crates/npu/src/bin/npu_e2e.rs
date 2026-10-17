//! npu_e2e <i32-out build> M K N m k n [hadamard 0|1] [iters]
//! On-device prompt-processing linear: GPU (Vulkan) does blockwise-1024 Hadamard + per-token int8 quant straight
//! into the NPU's A buffer (shared dma-buf, tiled layout), the NPU runs the int8 GEMM with int32 output, the GPU
//! dequantizes the NPU's C (shared dma-buf) to f32 in GPU memory. No host f32 round trip.
use npu::*; use std::time::Instant;
use vkrt::push_u32;
fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }
fn fwht1024(row: &mut [f32]) {
    for blk in row.chunks_mut(1024) { let mut h = 1; while h < blk.len() { for i in (0..blk.len()).step_by(2 * h) { for j in i..i + h { let (u, v) = (blk[j], blk[j + h]); blk[j] = u + v; blk[j + h] = u - v; } } h *= 2; }
        for v in blk.iter_mut() { *v *= 0.03125; } }
}
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (dir, mm, kk, nn, m, k, n) = (&a[1], p(2), p(3), p(4), p(5), p(6), p(7));
    let had = a.len() > 8 && p(8) == 1;
    let iters = if a.len() > 9 { p(9) } else { 20 };
    if kk > 8192 || (had && kk % 1024 != 0) { anyhow::bail!("K must be <= 8192 (and a multiple of 1024 with hadamard)") }
    let mut g = Gemm::new(dir, (mm, kk, nn), (m, k, n), 4)?;
    let ops = g.ops();
    let mut s = 5u64;
    let fr = |s: &mut u64| (lcg(s) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
    let x: Vec<f32> = (0..mm * kk).map(|_| fr(&mut s) * (1.0 + (lcg(&mut s) % 4) as f32)).collect();
    let w: Vec<f32> = (0..kk * nn).map(|_| fr(&mut s) * 0.05).collect();
    let (wq, ws) = quantize_cols(&w, kk, nn);
    g.set_b(&wq)?;

    let ctx = vkrt::Ctx::new()?;
    let xb = ctx.buffer((mm * kk * 4) as u64)?; unsafe { std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, xb.ptr, mm * kk * 4) };
    let swb = ctx.buffer((nn * 4) as u64)?; unsafe { std::ptr::copy_nonoverlapping(ws.as_ptr() as *const u8, swb.ptr, nn * 4) };
    let sxb = ctx.buffer_readback((mm * 4) as u64)?;
    let yb = ctx.buffer_readback((mm * nn * 4) as u64)?;
    let av = unsafe { vkshare::import(&ctx, g.a.export_dmabuf()?, (mm * kk) as u64)? };
    let cv = unsafe { vkshare::import(&ctx, g.c.export_dmabuf()?, (mm * nn * 4) as u64)? };
    let kq = ctx.kernel(vkshare::QUANT_TILED, 3, 16, &[], None)?;
    let kd = ctx.kernel(vkshare::DEQUANT, 4, 8, &[], None)?;
    let dsq = ctx.bind(&kq, &[&xb, &av, &sxb])?;
    let dsd = ctx.bind(&kd, &[&cv, &sxb, &swb, &yb])?;
    let pq = push_u32(&[kk as u32, m as u32, k as u32, had as u32]);
    let total4 = (mm * nn / 4) as u32; let pd = push_u32(&[(nn / 4) as u32, total4]);
    let groups_d = (total4 + 255) / 256;
    let step = |tq: &mut f64, tn: &mut f64, td: &mut f64| -> anyhow::Result<()> {
        *tq += ctx.run(|cb| ctx.dispatch(cb, &kq, dsq, &pq, mm as u32))?;
        let t = Instant::now(); g.run()?; *tn += t.elapsed().as_secs_f64();
        *td += ctx.run(|cb| ctx.dispatch(cb, &kd, dsd, &pd, groups_d))?;
        Ok(())
    };
    let (mut tq, mut tn, mut td) = (0.0, 0.0, 0.0);
    step(&mut tq, &mut tn, &mut td)?;
    // verify against a host reference of the same pipeline
    let mut xr = x.clone(); if had { for r in 0..mm { fwht1024(&mut xr[r * kk..(r + 1) * kk]); } }
    let (xq, xs) = quantize_rows(&xr, mm, kk);
    let sxg = unsafe { std::slice::from_raw_parts(sxb.ptr as *const f32, mm) };
    let sx_bad = (0..mm).filter(|&r| ((sxg[r] - xs[r]) / xs[r]).abs() > 1e-6).count();
    g.a.from_device()?; let at = g.a.bytes();
    let (mut a_bad, mut a_big) = (0usize, 0usize);
    let mut xg = vec![0i8; mm * kk]; // the GPU-quantized A, back in row-major, used as the GEMM reference
    for mb in 0..mm / m { for kb in 0..kk / k { for i in 0..m { for j in 0..k {
        let o = ((mb * (kk / k) + kb) * m + i) * k + j; let ri = (mb * m + i) * kk + kb * k + j; xg[ri] = at[o] as i8;
        if at[o] as i8 != xq[ri] { a_bad += 1; if (at[o] as i8 as i32 - xq[ri] as i32).abs() > 1 { a_big += 1; } } } } } }
    let y = unsafe { std::slice::from_raw_parts(yb.ptr as *const f32, mm * nn) };
    let (mut y_bad, mut num, mut den) = (0, 0f64, 0f64);
    for _ in 0..2000 { let r = lcg(&mut s) as usize % mm; let c = lcg(&mut s) as usize % nn;
        let want = ref_dot(&xg, &wq, kk, nn, r, c) as f64 * xs[r] as f64 * ws[c] as f64;
        if ((y[r * nn + c] as f64 - want) / want.abs().max(1e-3)).abs() > 1e-5 { y_bad += 1; }
        let yf: f64 = (0..kk).map(|i| xr[r * kk + i] as f64 * w[i * nn + c] as f64).sum();
        num += (y[r * nn + c] as f64 - yf).powi(2); den += yf * yf; }
    println!("verify: sx mismatches {sx_bad}/{mm}; GPU vs host int8 A: {a_bad}/{} differ (rounding ties), {a_big} by more than 1; Y vs exact int ref on GPU-quantized A: {y_bad}/2000 -> {} ; rel RMS vs f32 {:.4}",
        mm * kk, if sx_bad + a_big + y_bad == 0 { "PASS" } else { "FAIL" }, (num / den).sqrt());
    for _ in 0..3 { step(&mut 0.0, &mut 0.0, &mut 0.0)?; }
    let (mut tq, mut tn, mut td) = (0.0, 0.0, 0.0);
    let t = Instant::now(); for _ in 0..iters { step(&mut tq, &mut tn, &mut td)?; } let wall = t.elapsed().as_secs_f64() / iters as f64;
    let it = iters as f64;
    println!("{mm}x{kk}x{nn} hadamard={had}: end-to-end wall {:.2} ms/iter ({:.2} TOPS-equiv) | GPU quant{} {:.3} ms | NPU GEMM {:.2} ms ({:.2} TOPS) | GPU dequant {:.3} ms",
        wall * 1e3, ops / wall / 1e12, if had { "+FWHT" } else { "" }, tq / it * 1e3, tn / it * 1e3, ops / (tn / it) / 1e12, td / it * 1e3);
    Ok(())
}
