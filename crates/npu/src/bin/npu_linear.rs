//! npu_linear <i32-out build_dir> M K N m k n [iters] : batched prompt-processing linear layer on the NPU.
//! X [M tokens, K] f32 quantized per token to int8, W [K, N] int8 per output channel (uploaded once),
//! int32 accumulate on the NPU, dequantize to f32 on the host. Checks exact int32 accumulators and f32 error.
use npu::*; use std::time::Instant;
fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }
fn f(s: &mut u64) -> f32 { (lcg(s) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 }
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (dir, mm, kk, nn, m, k, n) = (&a[1], p(2), p(3), p(4), p(5), p(6), p(7));
    let iters = if a.len() > 8 { p(8) } else { 50 };
    let mut g = Gemm::new(dir, (mm, kk, nn), (m, k, n), 4)?;
    let ops = g.ops();
    let mut s = 3u64;
    let x: Vec<f32> = (0..mm * kk).map(|_| f(&mut s) * (1.0 + (lcg(&mut s) % 4) as f32)).collect();
    let w: Vec<f32> = (0..kk * nn).map(|_| f(&mut s) * 0.05).collect();
    let (wq, ws) = quantize_cols(&w, kk, nn);
    g.set_b(&wq)?;
    // exact int32 check of the NPU accumulators
    let (xq, xs) = quantize_rows(&x, mm, kk);
    g.set_a(&xq)?; g.run()?;
    let acc = g.c_i32()?.to_vec();
    let ns = 3000; let mut bad = 0; let (mut num, mut den) = (0f64, 0f64);
    for _ in 0..ns {
        let r = lcg(&mut s) as usize % mm; let c = lcg(&mut s) as usize % nn;
        if acc[r * nn + c] != ref_dot(&xq, &wq, kk, nn, r, c) { bad += 1; }
        let yf: f64 = (0..kk).map(|i| x[r * kk + i] as f64 * w[i * nn + c] as f64).sum();
        let yq = acc[r * nn + c] as f64 * xs[r] as f64 * ws[c] as f64;
        num += (yq - yf).powi(2); den += yf.powi(2);
    }
    println!("int32 accumulators vs exact: {bad}/{ns} mismatches -> {}", if bad == 0 { "PASS" } else { "FAIL" });
    println!("quantized vs f32 reference: relative RMS error {:.4}", (num / den).sqrt());
    for _ in 0..5 { g.run()?; }
    let t = Instant::now(); for _ in 0..iters { g.run()?; } let tr = t.elapsed().as_secs_f64() / iters as f64;
    println!("NPU run only: {:.0} us = {:.2} TOPS", tr * 1e6, ops / tr / 1e12);
    let n2 = (iters / 5).max(3);
    let t = Instant::now(); for _ in 0..n2 { let y = linear_f32(&mut g, &x, &ws)?; std::hint::black_box(&y); } let te = t.elapsed().as_secs_f64() / n2 as f64;
    println!("linear_f32 end-to-end (quantize+pack+sync+run+sync+dequant, multithreaded host): {:.0} us = {:.2} TOPS-equivalent", te * 1e6, ops / te / 1e12);
    let y = linear_f32(&mut g, &x, &ws)?;
    let mut bad = 0;
    for _ in 0..ns { let r = lcg(&mut s) as usize % mm; let c = lcg(&mut s) as usize % nn;
        let want = acc[r * nn + c] as f32 * xs[r] * ws[c]; if y[r * nn + c] != want { bad += 1; } }
    println!("linear_f32 output vs acc*sx*sw: {bad}/{ns} mismatches -> {}", if bad == 0 { "PASS" } else { "FAIL" });
    Ok(())
}
