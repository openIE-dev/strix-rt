//! Nemotron 3.5 Super text-tower runner: `n35 <model.gguf> <ids> [n]`
//! `n35 <model.gguf> PARITY <ref.txt>`: per line `prompt ids|llama greedy ids|llama top1-top2 logprob margins`: greedy and teacher-forced agreement.
//! env: RUNS, MAX_CTX, FORCE=<ref ids file> (teacher-forced agreement), TOPK=k (top-k logits after the prompt), LOGITS=<out.f32>
use anyhow::Result;
fn env(n: &str, d: usize) -> usize { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn ids(s: &str) -> Vec<u32> { s.trim().trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect() }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m = nemotron35::Model::load(&a[1], env("MAX_CTX", 1024) as u32)?;
    let pfs = if env("PF", 1) > 0 { Some(nemotron35::prefill::Scratch::new(&m, env("PF_M", 512) as u32)?) } else { None };
    if a[2] == "PARITY" {
        let (mut tg, mut tt, mut fg, mut ft) = (0, 0, 0, 0);
        for (li, line) in std::fs::read_to_string(&a[3])?.lines().enumerate() {
            let f: Vec<&str> = line.split('|').collect(); let (p, r) = (ids(f[0]), ids(f[1]));
            let mg: Vec<f32> = f[2].split(',').map(|x| x.trim().parse().unwrap()).collect();
            let (out, ..) = if let Some(sc) = &pfs { m.generate_pf(sc, &p, r.len())? } else { m.generate(&p, r.len())? };
            let same = out.iter().zip(&r).take_while(|(a, b)| a == b).count();
            let mut seq = p.clone(); seq.extend(&r); let pred = m.predict_all(&seq)?;
            let mut ok = 0; let mut miss = vec![];
            for i in 0..r.len() { if pred[p.len() - 1 + i] == r[i] { ok += 1 } else { miss.push((i, mg[i])) } }
            println!("prompt {li}: greedy identical prefix {same}/{} | teacher-forced {ok}/{} | misses (gen idx, llama margin nats): {:?}", r.len(), r.len(), miss);
            if same < r.len() { println!("   ours: {:?}\n   ref:  {:?}", &out[same..(same + 6).min(out.len())], &r[same..(same + 6).min(r.len())]); }
            tg += same; tt += r.len(); fg += ok; ft += r.len();
        }
        println!("TOTAL greedy identical {tg}/{tt}, teacher-forced {fg}/{ft}");
        return Ok(());
    }
    let prompt = ids(&a[2]);
    if env("MTP_CMP", 0) > 0 {
        let sc = pfs.as_ref().expect("PF=1"); let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
        let mp = std::env::var("MTP_GGUF").unwrap_or_else(|_| format!("{}/models/new/nemotron35vl/mtp-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf", std::env::var("HOME").unwrap()));
        let v = m.v2_new()?; let mut hd = m.head_new(&v, sc, &mp)?;
        if env("ONLY", 0) == 1 { let (_, _, _, dg, _, st) = m.generate_pf(sc, &prompt, n)?; println!("ONLYPLAIN decode GPU {:.3}s for {} tokens = {:.2} tok/s", dg, st, st as f64 / dg); return Ok(()) }
        let (plain, pg, _, dg, dw, st) = if env("ONLY", 0) == 2 { (Vec::new(), 0.0, 0.0, 1.0, 1.0, 0) } else { m.generate_pf(sc, &prompt, n)? };
        println!("plain (N35_DET={}): decode {:.2} tok/s gpu, {:.2} wall", env("N35_DET", 0), st as f64 / dg, st as f64 / dw); let _ = pg;
        for r in 0..env("MTP_CMP", 1) {
            let (out, s) = m.mtp_generate(sc, &v, &mut hd, &prompt, n)?;
            let same = out.iter().zip(&plain).take_while(|(x, y)| x == y).count();
            let dec_s = s.wall_s - 0.0;
            println!("MTP run {r}: {} cycles, {}/{} drafts accepted ({:.1}%), {:.3} tok/cycle | verify {:.2} ms/cycle, head {:.2} ms/call x{} | greedy match {same}/{}", s.cycles, s.accepted, s.cycles,
                100.0 * s.accepted as f64 / s.cycles.max(1) as f64, (out.len() as f64 - 1.0) / s.cycles.max(1) as f64, s.verify_s / s.cycles.max(1) as f64 * 1e3, s.head_s / s.head_calls.max(1) as f64 * 1e3, s.head_calls, out.len());
            let gpu = s.verify_s + s.head_s; println!("   head seed {:.1} ms", s.seed_s * 1e3); println!("   decode GPU {:.3}s for {} tokens = {:.2} tok/s (GPU), total wall incl prefill {:.3}s", gpu, out.len() - 1, (out.len() - 1) as f64 / gpu, dec_s);
            if same < out.len() { println!("   plain: {:?}\n   mtp:   {:?}", &plain[same.saturating_sub(2)..(same + 4).min(plain.len())], &out[same.saturating_sub(2)..(same + 4).min(out.len())]); }
        }
        return Ok(());
    }
    if env("V2TEST", 0) > 0 {
        let sc = pfs.as_ref().expect("PF=1"); let l = prompt.len();
        let (sq, ..) = m.generate(&prompt, 6)?;
        println!("plain greedy: {:?}", sq);
        let v = m.v2_new()?;
        nemotron35::prefill::prefill(sc, &m, &prompt)?;
        let a = prompt[l - 1];
        let (c0, c1, t) = m.v2_step(&v, a, sq[0], (l - 1) as u32, false)?;
        println!("verify(A={a}, B=true {}): col0 {c0} (expect {}), col1 {c1} (expect {}) | {:.2} ms", sq[0], sq[0], sq[1], t * 1e3);
        nemotron35::prefill::prefill(sc, &m, &prompt)?;
        let (c0, c1, t) = m.v2_step(&v, a, sq[0] ^ 1, (l - 1) as u32, false)?;
        println!("verify(A={a}, B=wrong): col0 {c0} (expect {}), col1 {c1} | {:.2} ms", sq[0], t * 1e3);
        for _ in 0..2 { m.v2_time(&v, 3)?; }
        let t = m.v2_time(&v, env("V2TEST", 5))?; println!("2-token verify: {:.2} ms/cycle (GPU, repeated)", t * 1e3);
        return Ok(());
    }
    if env("PFPROF", 0) > 0 {
        let sc = pfs.as_ref().expect("PF=1"); let n = env("PF_N", 511) as u32; m.generate_pf(sc, &prompt, 1)?;
        let mut agg = vec![]; for rep in 0..env("PFPROF", 0) { let r = nemotron35::prefill::profile_chunk(sc, &m, 0, n)?; if rep + 1 == env("PFPROF", 0) { agg = r } }
        let tot: f64 = agg.iter().map(|x| x.1).sum(); agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("prefill chunk {} tok: {:.3} ms ({:.1} tok/s timestamps)", n, tot * 1e3, n as f64 / tot);
        for (l, t, c) in agg { println!("{:>16} {:9.3} ms {:5.1}%  x{}", l, t * 1e3, 100.0 * t / tot, c); }
        return Ok(());
    }
    if env("PROF", 0) > 0 {
        m.generate(&prompt, 1)?;
        let mut last = vec![]; for _ in 0..env("PROF", 0) { last = m.profile()?; }
        let tot: f64 = last.iter().map(|x| x.1).sum(); let mut agg: Vec<(String, f64, usize)> = vec![];
        for (l, t) in &last { if let Some(e) = agg.iter_mut().find(|e| &e.0 == l) { e.1 += t; e.2 += 1 } else { agg.push((l.clone(), *t, 1)) } }
        agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("step {:.3} ms, {} segments", tot * 1e3, last.len());
        for (l, t, n) in agg { println!("{:>28} {:8.3} ms {:5.1}%  x{}", l, t * 1e3, 100.0 * t / tot, n); }
        return Ok(());
    }
    if env("DEBUG1", 0) > 0 {   // one decode step, then hidden-state / logits health
        let (out, pg, ..) = m.generate(&prompt[..1], 1)?;
        let st = |v: &[f32]| { let nan = v.iter().filter(|x| !x.is_finite()).count(); let rms = (v.iter().filter(|x| x.is_finite()).map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt(); (nan, rms) };
        println!("1 step {:.1} ms: next id {:?} | x nonfinite/rms {:?} | logits nonfinite/rms {:?}", pg * 1e3, out, st(m.x.as_f32()), st(m.logits.as_f32()));
        return Ok(());
    }
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
        if let Some(sc) = &pfs { m.generate_pf(sc, &prompt, 1)?; } else { m.generate(&prompt, 1)?; }
        let l = m.logits.as_f32(); let mut ix: Vec<usize> = (0..l.len()).collect(); ix.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap());
        println!("top: {:?}", ix[..k.parse().unwrap_or(5)].iter().map(|&i| (i, l[i])).collect::<Vec<_>>());
        if let Ok(p) = std::env::var("LOGITS") { std::fs::write(p, l.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?; }
        return Ok(());
    }
    if env("MTP", 1) > 0 && pfs.is_some() {
        let sc = pfs.as_ref().unwrap();
        let mp = std::env::var("MTP_GGUF").unwrap_or_else(|_| format!("{}/models/new/nemotron35vl/mtp-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf", std::env::var("HOME").unwrap()));
        if std::path::Path::new(&mp).exists() {
            let v = m.v2_new()?; let mut hd = m.head_new(&v, sc, &mp)?;
            for run in 0..env("RUNS", 1) {
                let (out, s) = m.mtp_generate(sc, &v, &mut hd, &prompt, n)?;
                if run == 0 { println!("ids: {:?}", out); }
                let gpu = s.verify_s + s.head_s;
                println!("MTP (exact greedy): prompt {} tok | {} cycles, {:.1}% drafts accepted, {:.3} tok/cycle | verify {:.2} ms/cycle, head {:.2} ms/call, head seed {:.1} ms ({} prompt positions) | decode {} tok: {:.2} tok/s gpu, wall incl prefill {:.3}s",
                    prompt.len(), s.cycles, 100.0 * s.accepted as f64 / s.cycles.max(1) as f64, (out.len() as f64 - 1.0) / s.cycles.max(1) as f64, s.verify_s / s.cycles.max(1) as f64 * 1e3, s.head_s / s.head_calls.max(1) as f64 * 1e3, s.seed_s * 1e3, prompt.len() - 1, out.len() - 1, (out.len() - 1) as f64 / gpu, s.wall_s);
            }
            return Ok(());
        }
        eprintln!("MTP gguf not found ({mp}); falling back to plain decode");
    }
    for run in 0..env("RUNS", 1) {
        let (out, pg, pw, dg, dw, st) = if let Some(sc) = &pfs { m.generate_pf(sc, &prompt, n)? } else { m.generate(&prompt, n)? };
        if run == 0 { println!("ids: {:?}", out); }
        println!("prompt {} tok: {:.1} tok/s gpu ({:.3}s wall) | decode {} tok: {:.2} tok/s gpu, {:.2} tok/s wall", prompt.len(), (prompt.len() - if pfs.is_some() { 1 } else { 0 }) as f64 / pg, pw, st, st as f64 / dg, st as f64 / dw);
    }
    Ok(())
}
