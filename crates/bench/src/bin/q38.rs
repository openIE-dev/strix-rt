//! strix-rt Qwen3.5/3.8 runner: `q38 <model.gguf> <ids> [n]`  (env: RUNS, PROFILE, ZEROCOPY=dmabuf, MAX_CTX)
use anyhow::Result;
fn env(n: &str, d: usize) -> usize { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m = qwen38::Model::load(&a[1], env("MAX_CTX", 1024) as u32)?;
    let prompt: Vec<u32> = a[2].trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect();
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let vocab = m.wt.vocab_strings();
    if let Ok(f) = std::env::var("FORCE") {
        // teacher-forced agreement: FORCE=file with the reference id sequence (prompt + reference continuation)
        let seq: Vec<u32> = std::fs::read_to_string(&f)?.trim().trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect();
        let pred = m.predict_all(&seq)?;
        let (mut ok, mut tot, mut first_miss) = (0, 0, None);
        for i in prompt.len() - 1..seq.len() - 1 { tot += 1; if pred[i] == seq[i + 1] { ok += 1 } else if first_miss.is_none() { first_miss = Some(i + 1 - prompt.len()) } }
        println!("teacher-forced top-1 agreement: {ok}/{tot} generated positions; first mismatch at generated index {:?}", first_miss);
        let misses: Vec<String> = (prompt.len() - 1..seq.len() - 1).filter(|&i| pred[i] != seq[i + 1]).map(|i| format!("{}:{}->{}", i + 1 - prompt.len(), seq[i + 1], pred[i])).collect();
        println!("mismatches (gen idx: ref->ours): {}", misses.join(" "));
        return Ok(());
    }
    if let Ok(k) = std::env::var("TOPK") {
        // logits after each of the first TOPK_STEPS generated positions
        for st in 0..env("TOPK_STEPS", 1) { m.generate(&prompt, st + 1)?;
            let l = m.logits.as_f32(); let mut ix: Vec<usize> = (0..l.len()).collect(); ix.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap());
            let top: Vec<String> = ix[..k.parse().unwrap_or(5)].iter().map(|&i| format!("{i}:{:?}={:.4}", String::from_utf8_lossy(&rtcore::gpt2_bytes(&vocab[i])), l[i])).collect();
            println!("step {st}: {}", top.join("  ")); }
    }
    if std::env::var("VPROF").is_ok() {
        for mm in 1..=4 { println!("verify m={mm}: {:.2} ms GPU", m.time_verify(mm, 10)?); }
        return Ok(());
    }
    if let Ok(ks) = std::env::var("MTP") {
        let ks: Vec<usize> = ks.split(',').map(|v| v.parse().unwrap_or(1)).collect();
        let (base, _, _, dg, dw, st) = m.generate(&prompt, n)?;
        let (_, _, _, dg, dw, st) = { let _ = (dg, dw, st); m.generate(&prompt, n)? };
        println!("plain greedy baseline: {st} tok wall {dw:.3}s = {:.2} tok/s; GPU {:.2} ms/tok", st as f64 / dw, dg / st as f64 * 1e3);
        for &k in &ks { for run in 0..env("RUNS", 1) {
            let (out, dw, cyc, acc) = m.generate_spec(&prompt, n, k)?;
            let agree = out.iter().zip(&base).take_while(|(a, b)| a == b).count();
            if run == 0 { let text: Vec<u8> = out.iter().flat_map(|&t| rtcore::gpt2_bytes(&vocab[t as usize])).collect(); println!("ids: {:?}\ntext: {}", out, String::from_utf8_lossy(&text)); }
            let gen = out.len() - 1;
            println!("MTP k={k}: {cyc} verify cycles, {acc}/{} drafts accepted = {:.1}% acceptance, {:.3} tok/cycle | {gen} tok in {dw:.3}s wall = {:.2} tok/s | matches plain greedy for {agree}/{} tokens",
                cyc * k, acc as f64 / (cyc * k) as f64 * 100.0, gen as f64 / cyc as f64, gen as f64 / dw, out.len());
        } }
        return Ok(());
    }
    for run in 0..env("RUNS", 1) {
        let (out, pg, pw, dg, dw, steps) = m.generate(&prompt, n)?;
        if run == 0 {
            println!("ids: {:?}", out);
            let text: Vec<u8> = out.iter().flat_map(|&t| rtcore::gpt2_bytes(&vocab[t as usize])).collect();
            println!("text: {}", String::from_utf8_lossy(&text));
        }
        println!("prompt {} steps: GPU {:.3}s wall {:.3}s | decode {} tok: wall {:.3}s = {:.2} tok/s; GPU {:.3} ms/tok = {:.2} tok/s",
            prompt.len(), pg, pw, steps, dw, steps as f64 / dw, dg / steps as f64 * 1e3, steps as f64 / dg);
    }
    if std::env::var("PROFILE").is_ok() { let p = m.profile()?; let tot: f64 = p.iter().map(|x| x.1).sum();
        for (l, us) in &p { println!("  {l:<9} {us:8.1} us ({:.1}%)", us / tot * 100.0); } println!("  total     {tot:8.1} us"); }
    Ok(())
}
