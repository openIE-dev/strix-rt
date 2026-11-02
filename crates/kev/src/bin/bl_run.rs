//! bl_run <model.gguf> <ref.jsonl> [n_slots_override]: Base Labs (joey-1 / elephant-1) parity + timing on strix-rt.
//! ref.jsonl from tools/baselabs/ref.py (OneJev qev @81ce62f + serve.patch, CPU fp32): line 0 {meta, slot_ids}, then per item
//! {prefix, suffix, labels, logits}. Runs the prefix block once (block-bidirectional full attention), the suffix block from
//! its snapshot, reads the slot letters' LM logits at the last row, and prints per-item max |dlogit|, max |dprob|, argmax match.
//! BL_REPS=r repeats the whole pass r times (energy windows); BL_OUT=f writes {id, logits} JSONL.
use anyhow::Result;
use kev::{json::{self, J}, Branch, Kev};
fn nums(j: Option<&J>) -> Vec<f64> { match j { Some(J::Arr(v)) => v.iter().map(|x| match x { J::Num(t) => t.parse().unwrap_or(f64::NAN), _ => f64::NAN }).collect(), _ => vec![] } }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let lines: Vec<J> = std::fs::read_to_string(&a[2])?.lines().map(|l| json::parse(l)).collect::<Result<_>>()?;
    let slots: Vec<u32> = nums(lines[0].get("slot_ids")).iter().map(|x| *x as u32).collect();
    let mut k = Kev::load(&a[1], "", prefill::env("KEV_MAX_CTX", 4096), prefill::env("KEV_CHUNK", 2048))?;
    let t0 = std::time::Instant::now(); let reps = prefill::env("BL_REPS", 1);
    let (mut agree, mut n, mut wl, mut wp) = (0, 0, 0f64, 0f64); let mut outs = vec![];
    for rep in 0..reps { for it in &lines[1..] {
        let pre: Vec<u32> = nums(it.get("prefix")).iter().map(|x| *x as u32).collect(); let suf: Vec<u32> = nums(it.get("suffix")).iter().map(|x| *x as u32).collect();
        let want = nums(it.get("logits")); let ns = want.len();
        k.slots = Some(slots[..ns].to_vec()); if !k.bidir && std::env::var("BL_CAUSAL").is_err() { k.base_labs(slots[..ns].to_vec())? }
        k.set_state(&pre)?;
        let got = k.branches(pre.len(), &[Branch { ids: suf.clone(), decide: suf.len() - 1, opts: vec![] }])?.remove(0);
        if rep > 0 { continue }
        let sm = |z: &[f64]| { let m = z.iter().cloned().fold(f64::MIN, f64::max); let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect(); let s: f64 = e.iter().sum(); e.into_iter().map(|v| v / s).collect::<Vec<_>>() };
        let g: Vec<f64> = got.iter().map(|x| *x as f64).collect(); let (pg, pw) = (sm(&g), sm(&want));
        let dl = g.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max); let dp = pg.iter().zip(&pw).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let am = |v: &[f64]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
        n += 1; if am(&g) == am(&want) { agree += 1 } wl = wl.max(dl); wp = wp.max(dp);
        let id = it.get("id").map(|x| x.dump()).unwrap_or_default();
        println!("{id} P={} S={} dlogit={dl:.4} dprob={dp:.4} argmax {}", pre.len(), suf.len(), if am(&g) == am(&want) { "ok" } else { "DIFF" });
        outs.push(format!("{{\"id\":{id},\"logits\":{}}}", J::Arr(g.iter().map(|x| J::num(*x)).collect()).dump()));
    } }
    let dt = t0.elapsed().as_secs_f64();
    println!("SUMMARY items={n} argmax_agree={agree}/{n} max_dlogit={wl:.4} max_dprob={wp:.4} wall={dt:.2}s per_decision_ms={:.1}", dt * 1e3 / (n * reps as usize) as f64);
    if let Ok(f) = std::env::var("BL_OUT") { std::fs::write(f, outs.join("\n") + "\n")? }
    Ok(())
}
