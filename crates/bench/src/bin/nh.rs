//! strix-rt Nemotron-H runner: `nh <model.gguf> <ids> [n]`  (env: RUNS, MAX_CTX, FORCE=<ref ids file>, TOPK)
use anyhow::Result;
fn env(n: &str, d: usize) -> usize { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn ids(s: &str) -> Vec<u32> { s.trim().trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect() }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m = nemotron::Model::load(&a[1], env("MAX_CTX", 1024) as u32)?;
    let prompt = ids(&a[2]);
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    if let Ok(f) = std::env::var("FORCE") {
        let seq = ids(&std::fs::read_to_string(&f)?);
        let pred = m.predict_all(&seq)?;
        let (mut ok, mut tot, mut first) = (0, 0, None);
        for i in prompt.len() - 1..seq.len() - 1 { tot += 1; if pred[i] == seq[i + 1] { ok += 1 } else if first.is_none() { first = Some(i + 1 - prompt.len()) } }
        println!("teacher-forced top-1 agreement: {ok}/{tot}; first mismatch at generated index {first:?}");
        return Ok(());
    }
    if let Ok(k) = std::env::var("TOPK") {
        m.generate(&prompt, 1)?;
        let l = m.logits.as_f32(); let mut ix: Vec<usize> = (0..l.len()).collect(); ix.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap());
        println!("top: {:?}", ix[..k.parse().unwrap_or(5)].iter().map(|&i| (i, l[i])).collect::<Vec<_>>());
    }
    if env("PROF", 0) > 0 {
        m.generate(&prompt, 1)?;
        let mut last = vec![]; for _ in 0..env("PROF", 0) { last = m.profile()?; }
        let tot: f64 = last.iter().map(|x| x.1).sum(); let mut agg: Vec<(String, f64, usize)> = vec![];
        for (l, t) in &last { if let Some(e) = agg.iter_mut().find(|e| &e.0 == l) { e.1 += t; e.2 += 1 } else { agg.push((l.clone(), *t, 1)) } }
        agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("step {:.3} ms, {} segments (barriers)", tot * 1e3, last.len());
        for (l, t, n) in agg { println!("{:>14} {:8.3} ms {:5.1}%  x{}", l, t * 1e3, 100.0 * t / tot, n); }
        return Ok(());
    }
    if env("PFPROF", 0) > 0 {
        let s = nemotron::prefill::Scratch::new(&m, env("PF_M", 512) as u32)?;
        let n = (prompt.len() - 1).min(env("PF_M", 512)) as u32;
        let mut agg: Vec<(String, f64, u32)> = vec![];
        for rep in 0..env("PFPROF", 0) { let r = nemotron::prefill::profile_chunk(&s, &m, 0, n)?; if rep + 1 == env("PFPROF", 0) { agg = r } }
        let tot: f64 = agg.iter().map(|a| a.1).sum(); agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("prefill chunk {} tok: {:.3} ms ({:.1} tok/s timestamps)", n, tot * 1e3, n as f64 / tot);
        for (l, t, c) in agg { println!("{:>14} {:8.3} ms {:5.1}%  x{}", l, t * 1e3, 100.0 * t / tot, c) }
        return Ok(());
    }
    if env("PF", 0) > 0 {
        let s = nemotron::prefill::Scratch::new(&m, env("PF_M", 512) as u32)?;
        for run in 0..env("RUNS", 1) {
            let (out, pg, pw, dg, dw, st) = m.generate_pf(&s, &prompt, n)?;
            if run == 0 { println!("ids: {:?}", out); }
            println!("prefill {} tok: {:.1} tok/s gpu ({:.3}s wall) | decode {} tok: {:.2} tok/s gpu, {:.2} tok/s wall", prompt.len() - 1, (prompt.len() - 1) as f64 / pg, pw, st, st as f64 / dg, st as f64 / dw);
        }
        return Ok(());
    }
    for run in 0..env("RUNS", 1) {
        let (out, pg, pw, dg, dw, st) = m.generate(&prompt, n)?;
        if run == 0 { println!("ids: {:?}", out); }
        println!("prompt {} tok: {:.1} tok/s gpu ({:.3}s wall) | decode {} tok: {:.2} tok/s gpu, {:.2} tok/s wall", prompt.len(), prompt.len() as f64 / pg, pw, st, st as f64 / dg, st as f64 / dw);
    }
    Ok(())
}
