//! laya_parity <laya-f16.gguf> <golden.bin>: Rust/Vulkan Laya vs the PyTorch reference (ref/golden_rows.py, fp32 CPU).
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let b = std::fs::read(&a[2])?; let mut p = 0usize;
    let rdu = |p: &mut usize, n: usize| -> Vec<u32> { let v = b[*p..*p + 4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(); *p += 4 * n; v };
    let rdf = |p: &mut usize, n: usize| -> Vec<f32> { let v = b[*p..*p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(); *p += 4 * n; v };
    let m = laya::Laya::load(&a[1], 512)?; let d = m.d as usize;
    let n = rdu(&mut p, 1)[0];
    let cos = |x: &[f32], y: &[f32]| { let dt = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| *p as f64 * *q as f64).sum::<f64>(); dt(x, y) / (dt(x, x) * dt(y, y)).sqrt() };
    let rowcos = |x: &[f32], y: &[f32], l: usize| (0..l).map(|i| cos(&x[i * d..(i + 1) * d], &y[i * d..(i + 1) * d])).fold(1f64, f64::min);
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    let (mut agree, mut wp, mut wc) = (0, 0f32, 1f64); let mut lat = vec![];
    for r in 0..n {
        let l = rdu(&mut p, 1)[0] as usize; let ids = rdu(&mut p, l); let h = rdu(&mut p, 2); let (qt, k) = (h[0], h[1] as usize);
        let mk: Vec<usize> = rdu(&mut p, k).into_iter().map(|x| x as usize).collect();
        let (genc, ghead, glog, gact) = (rdf(&mut p, l * d), rdf(&mut p, l * d), rdf(&mut p, k), rdf(&mut p, 2));
        m.forward(&ids, qt, &mk, 1)?; let xr = m.x_rows(l); let ce = rowcos(&xr, &genc, l);
        if std::env::var("LAYA_DIAG").is_ok() { let nrm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let mut w: Vec<(f64, usize)> = (0..l).map(|i| (cos(&xr[i * d..(i + 1) * d], &genc[i * d..(i + 1) * d]), i)).collect(); w.sort_by(|a, b| a.0.total_cmp(&b.0));
            for &(c, i) in w.iter().take(4) { let (a, g) = (&xr[i * d..(i + 1) * d], &genc[i * d..(i + 1) * d]);
                let top = |v: &[f32]| { let mut ix: Vec<usize> = (0..d).collect(); ix.sort_by(|x, y| v[*y].abs().total_cmp(&v[*x].abs())); ix[..3].iter().map(|&j| (j, v[j])).collect::<Vec<_>>() };
                println!("  diag row {r} tok {i}/{l} id {} cos {c:.6} |rust| {:.2} |ref| {:.2} top rust {:?} ref {:?}", ids[i], nrm(a), nrm(g), top(a), top(g)); } }
        m.forward(&ids, qt, &mk, 2)?; let ch = rowcos(&m.x_rows(l), &ghead, l);
        let t = std::time::Instant::now(); let o = m.forward(&ids, qt, &mk, 0)?; let ms = t.elapsed().as_secs_f64() * 1e3; lat.push((ms, o.gpu_ms));
        let (pr, gp) = (laya::softmax(&o.logits), laya::softmax(&glog));
        let dp = pr.iter().zip(&gp).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        if am(&pr) == am(&gp) { agree += 1 } wp = wp.max(dp); wc = wc.min(ce.min(ch));
        println!("row {r}: L={l} K={k} qtype {qt}: encoder min row cos {ce:.6}, head min row cos {ch:.6}, max|dprob| {dp:.4}, act {:.3?} vs {:.3?}, argmax {} vs {} | {ms:.2} ms wall, {:.2} ms GPU\n   rust {pr:.4?}\n   ref  {gp:.4?}", o.act, gact, am(&pr), am(&gp), o.gpu_ms);
    }
    assert_eq!(p, b.len());
    println!("SUMMARY {n} rows: argmax agree {agree}/{n}, worst max|dprob| {wp:.4} (raw logits, before temperature), worst row cos {wc:.6}");
    if prefill::env("BENCH", 0) > 0 { let ids: Vec<u32> = (0..512).map(|i| 1000 + (i * 7919 % 40000) as u32).collect(); let mk = vec![10usize, 20, 30, 40];
        for _ in 0..3 { let t = std::time::Instant::now(); let o = m.forward(&ids, 0, &mk, 0)?; println!("BENCH L=512: {:.2} ms wall, {:.2} ms GPU", t.elapsed().as_secs_f64() * 1e3, o.gpu_ms); } }
    Ok(())
}
