//! go_run <model.gguf> <ids csv | @file> [n_new] [ref ids csv | @file]: greedy decode, tok/s, parity vs reference ids. Env RUNS.
use anyhow::Result;
fn ids(s: &str) -> Result<Vec<u32>> { let t = if let Some(p) = s.strip_prefix('@') { std::fs::read_to_string(p)? } else { s.to_string() };
    Ok(t.trim().trim_matches(|c| c == '[' || c == ']').split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()) }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m = gptoss::Model::load(&a[1], std::env::var("MAX_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048))?;
    let p = ids(&a[2])?; let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let r = a.get(4).map(|s| ids(s)).transpose()?;
    if let (Ok(_), Some(r)) = (std::env::var("PFORCED"), &r) {
        // teacher-forced after prefill (GO_PF selects batched vs stepwise prompt); margins at every step
        let o = m.pf_forced(&p, r)?;
        let bad: Vec<String> = o.iter().enumerate().filter(|(j, x)| x.0 != r[*j]).map(|(j, x)| format!("{j}(ours {} ref {} gap {:.4})", x.0, r[j], x.2)).collect();
        println!("PFORCED agree {}/{} mismatch [{}]", o.len() - bad.len(), o.len(), bad.join(", "));
        if let Ok(k) = std::env::var("MARGIN_AT") { let k: usize = k.parse()?; println!("MARGIN_AT {k}: pred {} top2 margin {:.4}", o[k].0, o[k].1); }
        return Ok(());
    }
    if let (Ok(_), Some(r)) = (std::env::var("FORCED"), &r) {
        let seq: Vec<u32> = p.iter().chain(r.iter()).copied().collect();
        let pred = m.forced(&seq)?;
        let (mut ok, mut bad) = (0, vec![]);
        for i in p.len() - 1..seq.len() - 1 { if pred[i] == seq[i + 1] { ok += 1 } else { bad.push(i + 1 - p.len()) } }
        println!("FORCED agree {ok}/{} (mismatch at gen idx {:?})", seq.len() - p.len(), bad);
        return Ok(());
    }
    for run in 0..std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1) {
        let (out, tp, dg, dw) = m.generate(&p, n)?;
        if run == 0 { println!("ids: {:?}", out); }
        println!("prompt {} tok in {:.3}s ({:.1} tok/s stepwise) | decode {} tok GPU {:.3}s = {:.2} tok/s | wall {:.3}s = {:.2} tok/s", p.len(), tp, p.len() as f64 / tp, n - 1, dg, (n - 1) as f64 / dg, dw, (n - 1) as f64 / dw);
        if let Some(r) = &r { let k = out.iter().zip(r).take_while(|(x, y)| x == y).count(); println!("parity: first {k}/{} match reference", out.len().min(r.len())); }
    }
    Ok(())
}
