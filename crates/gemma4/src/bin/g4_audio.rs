//! g4_audio <mmproj.gguf> <16 kHz wav>: CPU gemma4a encoder only (no GPU); prints mel/embedding stats and timing.
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t0 = std::time::Instant::now();
    let au = gemma4::audio::Audio::load(&a[1])?;
    let w = gemma4::audio::read_wav(&a[2])?;
    let t1 = std::time::Instant::now();
    let (mel, nf) = au.mel(&w)?;
    let ms: f64 = mel.iter().map(|v| *v as f64).sum::<f64>() / mel.len() as f64;
    let (e, n) = au.encode_mel(&mel, nf)?;
    let ss: f64 = e.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    println!("load {:.2}s; {} samples -> {nf} mel frames (mean {ms:.4}) -> {n} tokens x {}; emb rms {:.4} first {:?}; encode {:.2}s",
        t1.duration_since(t0).as_secs_f32(), w.len(), au.out_dim, (ss / e.len() as f64).sqrt(), &e[..4], t1.elapsed().as_secs_f32());
    Ok(())
}
