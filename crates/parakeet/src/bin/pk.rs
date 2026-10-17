//! pk [--golden DIR] [--reps N] clip.wav ...  (16 kHz WAVs)
//! Transcribes each clip; prints one JSON line per clip (text, timings, RTF). With --golden, also compares the mel
//! features and encoder hidden state with the HF reference dumps (DIR/{name}_mel.npy, DIR/{name}_enc.npy).
use anyhow::{bail, Result};
use parakeet::{audio, Model};
fn stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let m = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let (n, d) = a.iter().zip(b).fold((0f64, 0f64), |(n, d), (x, y)| (n + ((x - y) as f64).powi(2), d + (*y as f64).powi(2)));
    (m, (n / d.max(1e-30)).sqrt() as f32)
}
fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut golden = None; let mut reps = 1usize;
    while args.first().map(|a| a.starts_with("--")).unwrap_or(false) { let a = args.remove(0); let v = args.remove(0);
        match a.as_str() { "--golden" => golden = Some(v), "--reps" => reps = v.parse()?, _ => bail!("unknown {a}") } }
    let home = std::env::var("HOME")?;
    let dir = std::env::var("PK_MODEL").unwrap_or(format!("{home}/models/parakeet-redux"));
    let vocab = std::env::var("PK_VOCAB").unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/ref/vocab.json").to_string());
    let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?;
    let mut m = Model::load(&ctx, &dir, &vocab)?; eprintln!("loaded in {:.2}s", m.load_s);
    let fe = audio::MelFrontend::new();
    for path in &args {
        let (a, sr) = audio::read_wav(path)?; if sr != 16000 { bail!("{path}: {sr} Hz (need 16 kHz)") }
        let name = std::path::Path::new(path).file_stem().unwrap().to_str().unwrap().to_string();
        let sec = a.len() as f64 / 16000.0;
        if let Some(g) = &golden {
            let mel = fe.features(&a); let (gm, gs) = audio::read_npy(&format!("{g}/{name}_mel.npy"))?;
            let n = mel.len().min(gm.len()); let (mx, rel) = stats(&mel[..n], &gm[..n]);
            let (ep, hid, t, _) = m.encode(&ctx, &pf, &mel, true)?; let _ = ep;
            let (ge, es) = audio::read_npy(&format!("{g}/{name}_enc.npy"))?; let hid = hid.unwrap(); let n2 = hid.len().min(ge.len());
            let (emx, erel) = stats(&hid[..n2], &ge[..n2]);
            eprintln!("{name}: mel {:?} vs ours {} frames: max|d| {mx:.4} rel {rel:.2e} | enc {:?} vs T={t}: max|d| {emx:.4} rel {erel:.2e}", gs, mel.len() / 128, es);
        }
        let mut runs = vec![];
        for _ in 0..reps.max(1) { let t0 = std::time::Instant::now(); let r = m.transcribe(&ctx, &pf, &fe, &a)?; runs.push((t0.elapsed().as_secs_f64(), r)); }
        runs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); let (tot, (text, _ids, tm)) = &runs[runs.len() / 2];
        println!("{}", serde_json::json!({"name": name, "text": text, "sec": sec, "total_s": tot, "rtf": tot / sec,
            "mel_ms": tm.mel_ms, "enc_ms": tm.enc_ms, "enc_gpu_ms": tm.enc_gpu_ms, "dec_ms": tm.dec_ms}));
    }
    Ok(())
}
