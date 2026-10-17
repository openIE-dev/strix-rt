//! S3 decoder (CPU interpreter): `ktdec <rt/decoder prefix> <voices.json> <voice> <audio_ids.json> <out.wav> [seed]`
use anyhow::Result;
use kitten::{dec, json};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t = std::time::Instant::now();
    kitten::arena::init(1_000_000_000, std::env::var("DEC_GPU").map(|v| v != "0").unwrap_or(true))?;
    let d = dec::Decoder::load(&a[1])?;
    let vj = json::parse(&std::fs::read_to_string(&a[2])?)?; let v = vj.at(&a[3])?;
    let pt = v.at("prompt_token")?.arr()[0].i64s();
    let pf: Vec<f32> = v.at("prompt_feat")?.arr()[0].arr().iter().flat_map(|r| r.f32s()).collect();
    let emb = v.at("embedding")?.arr()[0].f32s();
    let mut toks = json::parse(&std::fs::read_to_string(&a[4])?)?.i64s(); toks.retain(|&x| (0..6561).contains(&x)); toks.extend([4299; 3]);
    let seed: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(1);
    eprintln!("load {:.2}s", t.elapsed().as_secs_f64());
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut wav = vec![];
    for _ in 0..reps { let t = std::time::Instant::now(); wav = d.run(&toks, &pt, &pf, &emb, seed)?; eprintln!("decode {:.3}s -> {} samples ({:.2}s audio)", t.elapsed().as_secs_f64(), wav.len(), wav.len() as f64 / 24000.0); }
    if std::env::var("DEC_PROF").is_ok() { let mut p: Vec<_> = d.prof.borrow().iter().map(|(k, v)| (k.clone(), *v)).collect(); p.sort_by(|x, y| y.1 .0.partial_cmp(&x.1 .0).unwrap()); for (k, (s, n)) in p.iter().take(40) { eprintln!("  {k:40} {:.3}s x{n}", s / reps as f64) } }
    if let Some(g) = kitten::arena::gpu() { eprintln!("gpu: {} dispatches, {} flushes, submit+wait {:.3}s total; arena peak {:.0} MB", g.n_disp.get(), g.n_flush.get(), g.t_wait.get(), kitten::arena::peak_mb()); }
    kitten::arena::GPROF.with(|p| { let p = p.borrow(); let tot: f64 = p.values().map(|v| v.0).sum(); if tot > 0.0 { eprintln!("GPU kernel time {:.3}s (all reps)", tot); let mut v: Vec<_> = p.iter().collect(); v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap()); for (k, (t, n)) in v.iter().take(40) { eprintln!("  {k:50} {:.4}s x{n}", t) } } });
    kitten::write_wav(&a[5], &wav, 24000)?;
    Ok(())
}
