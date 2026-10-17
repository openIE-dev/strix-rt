//! End-to-end KittenTTS 2 in Rust/Vulkan, no C++: `ktts <model.gguf> <rt/decoder prefix> <voices.json> <voice> <text> <out.wav> [reps]`
//! text -> Rust BPE -> fork prompt layout -> speech LM (GPU, greedy + fork penalties) -> S3 flow decoder + vocoder (GPU) -> wav.
//! Prints per-rep LM / decoder / total seconds and RTF; rep 1 is cold (no cached reference prefix), later reps warm.
use anyhow::Result;
use kitten::{dec, json, tok};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t = std::time::Instant::now();
    let m = kitten::Model::load(&a[1], 4096)?;
    let tk = tok::Tokenizer::from_gguf(&m.wt.g)?;
    kitten::arena::init(std::env::var("DEC_ARENA_MF").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1000) * 1_000_000, true)?;
    let d = dec::Decoder::load(&a[2])?;
    let vj = json::parse(&std::fs::read_to_string(&a[3])?)?; let v = vj.at(&a[4])?;
    let pt = v.at("prompt_token")?.arr()[0].i64s();
    let pf: Vec<f32> = v.at("prompt_feat")?.arr()[0].arr().iter().flat_map(|r| r.f32s()).collect();
    let emb = v.at("embedding")?.arr()[0].f32s();
    let spk = v.at("speaker")?.f32s();
    let ref_text = tk.encode(&json_str(v.at("transcript")?)?);
    let ref_audio = v.at("reference_tokens")?.i64s();
    // fork text front end: normalize() rules (FST backend not ported), split into <= 380-char chunks, emotion prefix when tags present
    let text = kitten::text::normalize(&a[5]);
    if text.chars().any(|c| c.is_ascii_digit()) { eprintln!("warning: digits present; the fork's FST normalizer (number expansion) is not ported"); }
    let chunks = kitten::text::split(&text, 380, 130);
    let emotion = if kitten::text::has_expression(&a[5]) { tk.encode("{emo: 1}") } else { vec![] };
    let (base, sp_start, sp_end, txt_start, start, stop, final_seg, rts, rte, rss, rse) = (151675u32, 151669u32, 151670u32, 151671u32, 151673u32, 151674u32, 158236u32, 158239u32, 158240u32, 158237u32, 158238u32);
    let prompts: Vec<Vec<u32>> = chunks.iter().map(|c| {
        let mut p = vec![start, rts]; p.extend(&ref_text); p.push(rte); p.push(rss);
        p.extend(ref_audio.iter().map(|&x| base + x as u32)); p.push(rse);
        p.extend(&emotion); p.push(txt_start); p.extend(tk.encode(c)); p.push(final_seg); p.push(sp_start); p }).collect();
    if let Ok(p) = std::env::var("KT_DUMP_PROMPT") { std::fs::write(p, format!("{:?}", prompts[0]))?; }
    let p = kitten::Penalties { base, count: 6561, speech_end: sp_end, stop, repetition: 1.1, window: 50, run_penalty: 1.3, grace: 10 };
    eprintln!("load {:.2}s, {} chunk(s), prompt {} tokens, normalized: {text}", t.elapsed().as_secs_f64(), chunks.len(), prompts.iter().map(|p| p.len()).sum::<usize>());
    let reps: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut wav = vec![];
    for r in 0..reps {
        let t0 = std::time::Instant::now(); let (mut lm, mut dt, mut pf_s, mut ntok) = (0.0, 0.0, 0.0, 0); let mut waves = vec![]; let mut all = vec![];
        for (ci, prompt) in prompts.iter().enumerate() {
            let budget = if chunks.len() == 1 { 1000 } else { 200.max(1000.min((kitten::text::characters(&chunks[ci]) as f64 / 20.0 * 25.0 * 1.8) as usize)) };
            let tl = std::time::Instant::now(); let (out, pw, _, _) = m.generate(prompt, &spk, budget, &p)?; lm += tl.elapsed().as_secs_f64(); pf_s += pw; ntok += out.len();
            let mut toks: Vec<i64> = out.iter().filter(|&&x| x >= base && x < base + 6561).map(|&x| (x - base) as i64).collect(); toks.extend([4299; 3]);
            let td = std::time::Instant::now(); waves.push(d.run(&toks, &pt, &pf, &emb, 1)?); dt += td.elapsed().as_secs_f64(); all.push(out);
        }
        wav = kitten::text::join(&waves, 0.16);
        let au = wav.len() as f64 / 24000.0; let tot = t0.elapsed().as_secs_f64();
        println!("rep {}: {} tokens | LM {:.3}s (prefill {:.3}) | decoder {:.3}s | total {:.3}s | audio {:.2}s | RTF {:.3}", r + 1, ntok, lm, pf_s, dt, tot, au, tot / au);
        if r == 0 { std::fs::write(format!("{}.tokens.json", a[6]), format!("{:?}", if all.len() == 1 { all[0].clone() } else { all.concat() }))?; }
    }
    kitten::write_wav(&a[6], &wav, 24000)?;
    Ok(())
}
fn json_str(v: &json::J) -> Result<String> { Ok(v.str().to_string()) }
