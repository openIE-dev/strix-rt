//! KittenTTS 2 speech LM: `kt <model.gguf> <voice_dir> <prompt_ids.json> [max_new]` (voice_dir from tools/kitten_voice.py:
//! speaker.f32). Greedy with the fork's penalties; prints the generated ids as JSON (audio ids = id - 151675).
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m = kitten::Model::load(&a[1], 4096)?;
    let spk: Vec<f32> = std::fs::read(format!("{}/speaker.f32", a[2]))?.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let prompt: Vec<u32> = std::fs::read_to_string(&a[3])?.trim().trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect();
    let n: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let p = kitten::Penalties { base: 151675, count: 6561, speech_end: 151670, stop: 151674, repetition: 1.1, window: 50, run_penalty: 1.3, grace: 10 };
    let reps: usize = std::env::var("KT_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut r = m.generate(&prompt, &spk, n, &p)?;
    for i in 1..reps { let r2 = m.generate(&prompt, &spk, n, &p)?; eprintln!("rep {i}: prefill {:.3}s decode {:.3}s identical {}", r2.1, r2.2, r2.0 == r.0); r = r2; }
    if std::env::var("KT_LMPROF").is_ok() { let pos = (prompt.len() + 60) as u32; let seg = m.lm_profile(pos, 50)?;
        let names = ["xq8+qkv", "rope", "attn", "comb", "out+norm", "swiglu", "down+norm"]; let mut tot = [0f64; 7]; let (mut pre, mut post) = (0.0, 0.0);
        // layout: [first barrier], embed, rmsprep, 28 x (xq8? qkv, rope, attn, comb, xq8? out, xq8? swiglu, xq8? down), head
        eprintln!("LM profile at pos {pos}: {} segments, total {:.3} ms", seg.len(), seg.iter().sum::<f64>() * 1e3);
        let per = (seg.len() - 4) / 28; eprintln!("segments per layer {per}");
        for l in 0..28 { for k in 0..per { tot[k.min(6)] += seg[3 + l * per + k] } } pre += seg[..3].iter().sum::<f64>(); post += seg[3 + 28 * per..].iter().sum::<f64>();
        for (n, t) in names.iter().zip(tot) { eprintln!("  {n:10} {:.3} ms", t * 1e3) } eprintln!("  pre {:.3} ms, head {:.3} ms", pre * 1e3, post * 1e3); }
    let (out, pw, dw, dg) = r;
    let gen = out.len();
    eprintln!("prompt {} tok: {:.3}s | {} generated: wall {:.3}s = {:.1} tok/s, GPU {:.2} ms/tok | audio {:.2}s (25 tok/s)", prompt.len(), pw, gen, dw, gen as f64 / dw, dg / gen.max(1) as f64 * 1e3, gen as f64 / 25.0);
    println!("{:?}", out);
    Ok(())
}
