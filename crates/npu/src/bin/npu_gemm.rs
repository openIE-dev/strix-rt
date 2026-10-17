//! npu_gemm <build_dir> M K N m k n out_bytes(1|4) [iters] : run + verify the wa_dc GEMM build through the Rust path.
use npu::*; use std::time::Instant;
fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (dir, mm, kk, nn, m, k, n, ob) = (&a[1], p(2), p(3), p(4), p(5), p(6), p(7), p(8));
    let iters = if a.len() > 9 { p(9) } else { 100 };
    let t0 = Instant::now();
    let mut g = Gemm::new(dir, (mm, kk, nn), (m, k, n), ob)?;
    println!("open+alloc {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    let mut s = 42u64;
    let av: Vec<i8> = (0..mm * kk).map(|_| lcg(&mut s) as i8).collect();
    let bv: Vec<i8> = (0..kk * nn).map(|_| lcg(&mut s) as i8).collect();
    let t0 = Instant::now(); g.set_a(&av)?; g.set_b(&bv)?;
    println!("pack+sync A,B {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    for _ in 0..10 { g.run()?; }
    let mut ts = Vec::with_capacity(iters);
    for _ in 0..iters { let t = Instant::now(); g.run()?; ts.push(t.elapsed().as_secs_f64()); }
    let avg = ts.iter().sum::<f64>() / iters as f64;
    let mn = ts.iter().cloned().fold(f64::MAX, f64::min); let mx = ts.iter().cloned().fold(0.0, f64::max);
    let ops = g.ops();
    println!("run-only: avg {:.0} us  avg {:.2} TOPS  max {:.2}  min {:.2}  ({} iters)", avg * 1e6, ops / avg / 1e12, ops / mn / 1e12, ops / mx / 1e12, iters);
    let t = Instant::now(); g.run()?; let c = g.c_bytes()?.to_vec(); let e2e = t.elapsed().as_secs_f64();
    println!("run+C sync: {:.0} us  {:.2} TOPS", e2e * 1e6, ops / e2e / 1e12);
    // sampled verification
    let ns = 20000; let (mut bad_w, mut bad_s, mut bad_x) = (0, 0, 0);
    for _ in 0..ns {
        let r = lcg(&mut s) as usize % mm; let cc = lcg(&mut s) as usize % nn;
        let exact = ref_dot(&av, &bv, kk, nn, r, cc);
        if ob == 4 {
            let got = i32::from_le_bytes(c[(r * nn + cc) * 4..(r * nn + cc) * 4 + 4].try_into().unwrap());
            if got != exact { bad_x += 1; }
        } else {
            let got = c[r * nn + cc] as i8;
            if got != exact as i8 { bad_w += 1; }
            if got != exact.clamp(-128, 127) as i8 { bad_s += 1; }
        }
    }
    if ob == 4 { println!("verify {ns} samples vs exact int32: {} mismatches -> {}", bad_x, if bad_x == 0 { "PASS" } else { "FAIL" }); }
    else { println!("verify {ns} samples: wrap-mismatch {bad_w}, saturate-mismatch {bad_s} -> {}", if bad_w == 0 || bad_s == 0 { "PASS" } else { "FAIL" }); }
    Ok(())
}
