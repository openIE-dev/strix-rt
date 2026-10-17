//! npu_q4 <build_dir> M K N m k n [iters] [ternary=1] : int8 x int4/ternary g128 GEMM bench + verification.
use npu::*; use std::time::Instant;
fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (dir, mm, kk, nn, m, k, n) = (&a[1], p(2), p(3), p(4), p(5), p(6), p(7));
    let iters = if a.len() > 8 { p(8) } else { 50 };
    let tern = a.len() <= 9 || p(9) == 1;
    let qint = std::env::var("Q4INT").is_ok();
    let mut g = GemmQ4::new(dir, (mm, kk, nn), (m, k, n), qint)?;
    let mut s = 11u64;
    let pa = std::env::var("PAT_A").unwrap_or_default(); let pw = std::env::var("PAT_W").unwrap_or_default(); let ps = std::env::var("PAT_S").unwrap_or_default();
    let av: Vec<i8> = (0..mm * kk).map(|i| match pa.as_str() { "one" => 1, "row" => ((i / kk) % 7) as i8 - 3, "col" => ((i % kk) % 5) as i8 - 2, "onehot" => (i % kk == (i / kk) % 16 + std::env::var("OH").map(|v| v.parse::<usize>().unwrap()).unwrap_or(0)) as i8, _ => lcg(&mut s) as i8 }).collect();
    let wv: Vec<i8> = (0..kk * nn).map(|i| match pw.as_str() { "one" => 1, "col" => ((i % nn) % 3) as i8 - 1, "row" => ((i / nn) % 3) as i8 - 1, "c16" => ((i % nn) % 16) as i8 - 8, "k16" => ((i / nn) % 16) as i8 - 8, "g" => if (i / nn) / 128 == std::env::var("G").map(|v| v.parse::<usize>().unwrap()).unwrap_or(0) { ((i / nn) % 16) as i8 - 8 } else { 0 }, _ => if tern { (lcg(&mut s) % 3) as i8 - 1 } else { (lcg(&mut s) % 16) as i8 - 8 } }).collect();
    // scales are per 128-group (g128); with k < 128 each tile repeats its group's scale row
    let g128: Vec<f32> = (0..(kk / 128) * nn).map(|gi| if ps == "grp" { (gi / nn + 1) as f32 } else if ps == "one" { 1.0 } else { 0.5 + (lcg(&mut s) % 1000) as f32 / 1000.0 }).collect();
    let sc: Vec<f32> = (0..(kk / k) * nn).map(|i| g128[((i / nn) * k / 128) * nn + i % nn]).collect();
    let (qv, cs) = GemmQ4::int_scales(&sc, kk / k, nn);
    let t = Instant::now(); g.set_a(&av)?; g.set_b(&wv, if qint { &qv } else { &sc })?; println!("pack+sync {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    g.run()?;
    let c: Vec<f32> = if qint {
        let ci = g.c_i32()?.to_vec();
        // exact integer reference of the epilogue on samples
        let mut bad = 0; let mut s2 = 99u64;
        for _ in 0..2000 { let r = lcg(&mut s2) as usize % mm; let cc = lcg(&mut s2) as usize % nn; let mut acc_c: i64 = 0;
            for gi in 0..kk / k { let mut acc = 0i64; for kx in gi * k..(gi + 1) * k { acc += av[r * kk + kx] as i64 * wv[kx * nn + cc] as i64; }
                let x = (acc_c << 15) + acc * qv[gi * nn + cc] as i64; let fl = x >> 15; let rem = x - (fl << 15); acc_c = if rem > 16384 || (rem == 16384 && fl & 1 == 1) { fl + 1 } else { fl }; }
            if acc_c != ci[r * nn + cc] as i64 { bad += 1; if bad <= 3 { println!("  int mismatch r{r} c{cc}: got {} want {acc_c}", ci[r * nn + cc]); } } }
        println!("integer epilogue exact check: {bad}/2000 mismatches -> {}", if bad == 0 { "PASS" } else { "FAIL" });
        ci.iter().enumerate().map(|(i, &v)| v as f32 * cs[i % nn]).collect()
    } else { g.c_f32()?.to_vec() };
    if std::env::var("DUMP").is_ok() { for r in (0..16usize).chain([16, 17]) { let row: Vec<String> = (0..3).map(|cc| { let mut want = 0f64; for gi in 0..kk / k { let mut acc = 0i64; for kx in gi * k..(gi + 1) * k { acc += av[r * kk + kx] as i64 * wv[kx * nn + cc] as i64; } want += acc as f64 * sc[gi * nn + cc] as f64; } format!("{}/{}", c[r * nn + cc], want) }).collect(); println!("row {r}: {}", row.join(" ")); } }
    let ns = 3000; let mut bad = 0; let mut maxrel = 0f64;
    for _ in 0..ns {
        let r = lcg(&mut s) as usize % mm; let cc = lcg(&mut s) as usize % nn;
        let mut want = 0f64;
        for gi in 0..kk / k { let mut acc = 0i64; for kx in gi * k..(gi + 1) * k { acc += av[r * kk + kx] as i64 * wv[kx * nn + cc] as i64; } want += acc as f64 * sc[gi * nn + cc] as f64; }
        let got = c[r * nn + cc] as f64;
        let rel = (got - want).abs() / (128.0 * (kk as f64).sqrt()); if rel > maxrel { maxrel = rel; }
        if rel > 1e-4 { bad += 1; if bad <= 5 { println!("  mismatch r{r} c{cc}: got {got} want {want}"); } }
    }
    println!("verify {ns} samples ({}): {bad} over 1e-4 (|err| / (128*sqrt(K)), the typical |y| scale), max {maxrel:.2e} -> {}", if tern { "ternary" } else { "int4" }, if bad == 0 { "PASS" } else { "FAIL" });
    for _ in 0..5 { g.run()?; }
    let mut ts = vec![]; for _ in 0..iters { let t = Instant::now(); g.run()?; ts.push(t.elapsed().as_secs_f64()); }
    let avg = ts.iter().sum::<f64>() / iters as f64; let mn = ts.iter().cloned().fold(9.0, f64::min);
    println!("run-only avg {:.0} us = {:.2} TOPS (best {:.2}), {iters} iters", avg * 1e6, g.ops() / avg / 1e12, g.ops() / mn / 1e12);
    Ok(())
}
