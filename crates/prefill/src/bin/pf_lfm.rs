//! lfm2moe prefill check + pp bench: pf_q38 <model.gguf> [L=512] [n_gen=16]
//! Compares logits / greedy continuation of the batched prefill against the token-by-token decode path.
use anyhow::Result;
use prefill::{lfm2moe, Pf};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let l: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let ng: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let m = lfm2::Model::load(&a[1], (l + ng + 8) as u32)?;
    let pf = Pf::new(&m.ctx)?;
    let s = lfm2moe::Lf::new(&m, prefill::env("PF_M", 512))?;
    // PF_TOKS=<file>: real-text prompt as token ids (e.g. from llama-tokenize --ids), truncated to L; else random ids
    let prompt: Vec<u32> = match std::env::var("PF_TOKS") { Ok(f) => std::fs::read_to_string(f)?.split(|c: char| !c.is_ascii_digit()).filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).take(l).collect(),
        Err(_) => { let mut x = 12345u64; (0..l).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); 1000 + ((x >> 33) % 60000) as u32 }).collect() } };
    let l = prompt.len();
    let toks = |n: usize| unsafe { std::slice::from_raw_parts(m.toks.ptr as *const u32, n).to_vec() };
    if prefill::env("NOREF", 0) > 0 {
        let mut ts = vec![]; for _ in 0..prefill::env("REPS", 3) { ts.push(lfm2moe::prefill(&pf, &m, &s, &prompt)?); }
        let best = ts.iter().cloned().fold(f64::MAX, f64::min);
        println!("prefill {l}: best {:.1} ms (all {:?}) -> pp{l} {:.2} tok/s", best * 1e3, ts.iter().map(|t| (t * 1e3) as u32).collect::<Vec<_>>(), l as f64 / best);
        // DEPTH_TG=1: time n_gen-1 greedy decode steps right after the prompt (decode at context depth l)
        if prefill::env("DEPTH_TG", 0) > 0 && ng > 1 { let t = std::time::Instant::now(); let g = m.ctx.submit_repeated(m.step, ng - 1)?;
            let w = t.elapsed().as_secs_f64(); println!("decode at depth {l}: {} tok wall {:.3}s = {:.2} tok/s (GPU {:.3}s)", ng - 1, w, (ng - 1) as f64 / w, g); }
        return Ok(());
    }
    // reference: token-by-token
    let t = std::time::Instant::now();
    let (ref_gen, ..) = m.generate(&prompt, ng)?;
    let ref_t = t.elapsed().as_secs_f64();
    // logits of the last prompt token: rerun generate(prompt,1)
    m.generate(&prompt, 1)?; let ref_logits = m.logits.as_f32()[..m.cfg.vocab as usize].to_vec();
    // prefill
    let mut best = f64::MAX;
    let mut prev: Option<Vec<f32>> = None;
    for _ in 0..prefill::env("REPS", 3) { best = best.min(lfm2moe::prefill(&pf, &m, &s, &prompt)?);
        let cur = m.logits.as_f32()[..m.cfg.vocab as usize].to_vec();
        if let Some(p) = &prev { println!("rep-to-rep logits max|diff| {:.5}", p.iter().zip(&cur).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)); }
        prev = Some(cur); }
    let lg = m.logits.as_f32()[..m.cfg.vocab as usize].to_vec();
    let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let (mut e, mut mx) = (0f32, 0f32); for (p, r) in lg.iter().zip(&ref_logits) { e = e.max((p - r).abs()); mx = mx.max(r.abs()); }
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>();
    let cos = dot(&lg, &ref_logits) / (dot(&lg, &lg).sqrt() * dot(&ref_logits, &ref_logits).sqrt());
    let top = |v: &[f32]| { let mut i: Vec<usize> = (0..v.len()).collect(); i.sort_by(|a, b| v[*b].total_cmp(&v[*a])); i.truncate(10); i };
    let (tp, tr) = (top(&lg), top(&ref_logits));
    println!("logits cosine {cos:.6}, top10 overlap {}/10, top5 pf {:?} ref {:?}", tp.iter().filter(|i| tr.contains(i)).count(), &tp[..5], &tr[..5]);
    println!("top5 ref logits pf {:?} ref {:?}", tr[..5].iter().map(|&i| lg[i]).collect::<Vec<_>>(), tr[..5].iter().map(|&i| ref_logits[i]).collect::<Vec<_>>());
    let first = toks(l + 1)[l];
    if ng > 1 { m.ctx.submit_repeated(m.step, ng - 1)?; }
    let pf_gen = toks(l + ng)[l..].to_vec();
    let agree = pf_gen.iter().zip(&ref_gen).take_while(|(a, b)| a == b).count();
    println!("L={l}: logits max|diff| {e:.4} (max|ref| {mx:.2}), argmax pf {} ref {}, first tok {first}", am(&lg), am(&ref_logits));
    if prefill::env("PF_FORCED", 0) > 0 {
        // teacher-forced along the token-by-token reference: re-prefill, then feed ref_gen[j] and compare each argmax with ref_gen[j+1]
        lfm2moe::prefill(&pf, &m, &s, &prompt)?;
        let tk = unsafe { std::slice::from_raw_parts_mut(m.toks.ptr as *mut u32, l + ng + 2) };
        let mut ok = (tk[l] == ref_gen[0]) as usize; let mut bad = vec![]; if ok == 0 { bad.push(0); }
        for j in 0..ng - 1 { tk[l + j] = ref_gen[j]; m.ctx.submit_repeated(m.step, 1)?; if tk[l + j + 1] == ref_gen[j + 1] { ok += 1 } else { bad.push(j + 1); let v = &m.logits.as_f32()[..m.cfg.vocab as usize]; let pr = tk[l + j + 1] as usize; println!("  mismatch {}: pred {pr} ref {} logit gap {:.4}", j + 1, ref_gen[j + 1], v[pr] - v[ref_gen[j + 1] as usize]); } }
        println!("PF_FORCED agree {ok}/{ng} mismatch {bad:?}");
    }
    println!("greedy continuation agreement {agree}/{ng}\n  pf  {pf_gen:?}\n  ref {ref_gen:?}");
    println!("prefill {l} tokens: {:.1} ms -> pp{l} {:.2} tok/s   (token-by-token generate incl. {ng} new: {:.1} ms)", best * 1e3, l as f64 / best, ref_t * 1e3);
    if prefill::env("CHECK", 0) > 0 { for k in 1..4.min(ng) {
        let mut p2 = prompt.clone(); p2.extend_from_slice(&ref_gen[..k]); let (g, ..) = m.generate(&p2, 1)?;
        let mut p3 = prompt.clone(); p3.extend_from_slice(&pf_gen[..k]); let (g3, ..) = m.generate(&p3, 1)?;
        println!("teacher-forced generate(prompt+ref[..{k}]) -> {:?} (ref {}), generate(prompt+pf[..{k}]) -> {:?} (pf {})", g, ref_gen[k], g3, pf_gen[k]); } }
    Ok(())
}
