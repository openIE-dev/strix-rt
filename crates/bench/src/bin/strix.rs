//! strix gen <ids comma-sep> [n]   |  strix kbench
use anyhow::Result;
use gguf::{dequant::dequant_row, GgmlType};
use qwen2::{gpt2_bytes, Gemvs, Model};
use std::time::Instant;

fn env(k: &str, d: u32) -> u32 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = std::env::var("MODEL").unwrap_or_else(|_| format!("{}/models/q7.gguf", std::env::var("HOME").unwrap()));
    let (ls, sg) = (env("GEMV_LS", 64), env("GEMV_SG", 32));
    let t0 = Instant::now();
    let m = Model::load(&path, 4096, ls, sg)?;
    eprintln!("load {:.2}s, gemv ls={ls} sg={sg}", t0.elapsed().as_secs_f64());
    match args.get(1).map(|s| s.as_str()) {
        Some("gen") => gen(&m, &args[2], args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64)),
        Some("kbench") => kbench(&m),
        Some("fast") => fast(&m, &args[2], args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64)),
        _ => { eprintln!("usage: strix gen <ids> [n] | strix kbench"); Ok(()) }
    }
}

fn gen(m: &Model, ids: &str, n: usize) -> Result<()> {
    let prompt: Vec<u32> = ids.trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect();
    let vocab = m.vocab_strings();
    let rec = m.record_token()?;
    let mut out = vec![]; let mut pos = 0u32; let mut gpu = 0.0; let mut tok = prompt[0];
    let tp = Instant::now();
    for (i, &t) in prompt.iter().enumerate() { m.embed(t); m.set_pos(pos); m.ctx.submit(&rec)?; pos += 1; if i + 1 == prompt.len() { tok = m.argmax(); } }
    let prompt_s = tp.elapsed().as_secs_f64();
    out.push(tok);
    let td = Instant::now();
    for _ in 1..n { m.embed(tok); m.set_pos(pos); gpu += m.ctx.submit(&rec)?; pos += 1; tok = m.argmax();
        if std::env::var("TOP").map(|v| v.parse::<usize>().ok() == Some(out.len())).unwrap_or(false) {
            let l = m.b.logits.as_f32(); let mut ix: Vec<usize> = (0..l.len()).collect(); ix.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap());
            for &i in &ix[..3] { eprintln!("step {} top: {} {:?} logit {:.4}", out.len(), i, String::from_utf8_lossy(&gpt2_bytes(&vocab[i])), l[i]); } }
        out.push(tok); }
    let dec = td.elapsed().as_secs_f64();
    let text: Vec<u8> = out.iter().flat_map(|&t| gpt2_bytes(&vocab[t as usize])).collect();
    println!("ids: {:?}", out);
    println!("text: {}", String::from_utf8_lossy(&text));
    println!("prompt: {} tok in {:.3}s ({:.1} tok/s, token-by-token)", prompt.len(), prompt_s, prompt.len() as f64 / prompt_s);
    println!("decode: {} tok in {:.3}s wall = {:.2} tok/s; GPU-only {:.2} ms/tok = {:.2} tok/s", n - 1, dec, (n - 1) as f64 / dec, gpu / (n - 1) as f64 * 1e3, (n - 1) as f64 / gpu);
    Ok(())
}

/// Per-kernel GB/s on real weights, cycling over all 28 layers (defeats the 32MB MALL), plus correctness vs CPU on layer 0/first match.
fn kbench(m: &Model) -> Result<()> {
    let iters = env("ITERS", 112);
    let mut seed = 7u64;
    let mut rnd = |b: &vkrt::Buffer| for v in b.as_f32_mut() { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *v = ((seed >> 40) as f32 / 16777216.0) * 2.0 - 1.0; };
    rnd(&m.b.xn); rnd(&m.b.hb);
    let cases: &[(&str, Option<&str>, GgmlType)] = &[("attn_q.weight", None, GgmlType::Q4K), ("attn_output.weight", None, GgmlType::Q4K), ("attn_k.weight", None, GgmlType::Q4K),
        ("attn_v.weight", None, GgmlType::Q6K), ("ffn_gate.weight", None, GgmlType::Q4K), ("ffn_gate.weight", Some("ffn_up.weight"), GgmlType::Q4K),
        ("ffn_down.weight", None, GgmlType::Q4K), ("ffn_down.weight", None, GgmlType::Q6K), ("output.weight", None, GgmlType::Q6K)];
    let configs: Vec<(u32, u32)> = match std::env::var("SWEEP") { Ok(_) => vec![(32, 32), (64, 32), (128, 32), (256, 32), (64, 64), (128, 64), (256, 64)], Err(_) => vec![(m.gv.ls, m.gv.sg)] };
    println!("| tensor | type | N x K | ls/sg | us/call | GB/s | % of 256 | max_abs | max_abs/max|ref| |");
    for &(ls, sg) in &configs {
        let gv = Gemvs::new(&m.ctx, ls, sg, m.cfg.dim)?;
        for &(name, up, ty) in cases {
            let refs: Vec<_> = if name == "output.weight" { vec![(m.wref(name)?, None)] } else {
                (0..m.cfg.n_layer).filter_map(|l| { let w = m.wref(&format!("blk.{l}.{name}")).ok()?; if w.ty != ty { return None } Some((w, up.map(|u| m.wref(&format!("blk.{l}.{u}")).unwrap()))) }).collect() };
            if refs.is_empty() { continue }
            let (w0, u0) = refs[0];
            let x = if w0.k == m.cfg.dim { &m.b.xn } else { &m.b.hb };
            let y = if w0.n as usize * 4 <= m.b.hb.size as usize && x.buf != m.b.hb.buf { &m.b.hb } else { &m.b.logits };
            let rec = m.ctx.record(|cb| { m.gemv(&gv, cb, w0, u0, x, y, None, false).unwrap(); })?;
            m.ctx.submit(&rec)?;
            // CPU reference
            let tname = m.w.iter().find(|(_, r)| r.arena == w0.arena && r.off == w0.off).map(|(n, _)| n.clone()).unwrap();
            let cpu = |tn: &str| -> Vec<f32> { let t = m.g.tensor(tn).unwrap(); let d = m.g.data(t); let rb = t.row_bytes(); let xs = x.as_f32();
                let mut out = vec![0f32; w0.n as usize];
                std::thread::scope(|s| { for (ci, c) in out.chunks_mut(w0.n.div_ceil(16) as usize).enumerate() { let ch = w0.n.div_ceil(16) as usize; s.spawn(move || { let mut buf = vec![0f32; w0.k as usize];
                    for (i, o) in c.iter_mut().enumerate() { let r = ci * ch + i; dequant_row(t.ty, &d[r * rb..(r + 1) * rb], &mut buf); *o = buf.iter().zip(&xs[..w0.k as usize]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32; } }); } });
                out };
            let mut r = cpu(&tname);
            if let Some(u) = u0 { let un = m.w.iter().find(|(_, r)| r.arena == u.arena && r.off == u.off).map(|(n, _)| n.clone()).unwrap(); let ru = cpu(&un);
                for (g, u) in r.iter_mut().zip(ru) { *g = *g / (1.0 + (-*g).exp()) * u; } }
            let gpu = &y.as_f32()[..w0.n as usize];
            let rmax = r.iter().fold(0f32, |a, v| a.max(v.abs()));
            let mabs = gpu.iter().zip(&r).fold(0f32, |a, (g, c)| if g.is_finite() { a.max((g - c).abs()) } else { f32::INFINITY });
            let rec = m.ctx.record(|cb| for i in 0..iters { let (w, u) = refs[i as usize % refs.len()]; m.gemv(&gv, cb, w, u, x, y, None, false).unwrap(); m.ctx.barrier(cb); })?;
            m.ctx.submit(&rec)?;
            let s = m.ctx.submit(&rec)? / iters as f64;
            let bytes = w0.bytes * if u0.is_some() { 2 } else { 1 } + w0.k as u64 * 4 + w0.n as u64 * 4;
            let gbs = bytes as f64 / s / 1e9;
            println!("| {}{} | {:?} | {} x {} | {ls}/{sg} | {:.1} | {:.1} | {:.1}% | {:.2e} | {:.2e} |", name, if u0.is_some() { "+up (swiglu)" } else { "" }, ty, w0.n, w0.k, s * 1e6, gbs, gbs / 2.56, mabs, mabs / rmax);
        }
    }
    Ok(())
}

fn fast(m: &Model, ids: &str, n: usize) -> Result<()> {
    let prompt: Vec<u32> = ids.trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect();
    let f = qwen2::fast::Fast::new(m, env("GEMV_LS", 64), env("GEMV_SG", 32))?;
    let vocab = m.vocab_strings();
    for run in 0..env("RUNS", 1) {
        let (out, pg, pw, dg, dw, steps) = f.generate(m, &prompt, n)?;
        if run == 0 {
            println!("ids: {:?}", out);
            let text: Vec<u8> = out.iter().flat_map(|&t| gpt2_bytes(&vocab[t as usize])).collect();
            println!("text: {}", String::from_utf8_lossy(&text));
        }
        println!("prompt {} steps: GPU {:.3}s wall {:.3}s | decode {} tok: wall {:.3}s = {:.2} tok/s; GPU {:.2} ms/tok = {:.2} tok/s",
            prompt.len(), pg, pw, steps, dw, steps as f64 / dw, dg / steps as f64 * 1e3, steps as f64 / dg);
    }
    if std::env::var("PROFILE").is_ok() { let p = f.profile(m)?; let tot: f64 = p.iter().map(|x| x.1).sum();
        for (l, us) in &p { println!("  {l:<8} {us:8.1} us ({:.1}%)", us / tot * 100.0); } println!("  total    {tot:8.1} us"); }
    Ok(())
}
