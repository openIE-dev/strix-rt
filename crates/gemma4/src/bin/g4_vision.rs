//! g4_vision <mmproj.gguf> <image.ppm> [max_tok=1120] [out.f32]: run the Gemma 4 vision encoder, print stats,
//! optionally dump embeddings (raw f32 [T][out_dim]).
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ctx = vkrt::Ctx::new()?;
    let v = gemma4::vision::Vision::load(&ctx, &a[1])?;
    let (rgb, w, h) = gemma4::vision::read_ppm(&a[2])?;
    let mt = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1120);
    let t0 = std::time::Instant::now();
    let (emb, pc, pr) = v.encode(&rgb, w, h, mt)?;
    let dt = t0.elapsed().as_secs_f64();
    let rms = (emb.iter().map(|x| (x * x) as f64).sum::<f64>() / emb.len() as f64).sqrt();
    println!("image {w}x{h} -> tokens {pc}x{pr}={} dim {} in {:.3}s rms {rms:.4} sum {:.3} first {:?}", pc * pr, v.out_dim, dt, emb.iter().map(|&x| x as f64).sum::<f64>(), &emb[..4]);
    if let Some(o) = a.get(4) { std::fs::write(o, emb.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?; }
    Ok(())
}
