//! glm_vis <mmproj.gguf> <img.rgb (u32 w, u32 h LE + RGB u8)> <out.f32> [ref.f32]: encode with the Glimmer ViT; cosine vs a llama.cpp mtmd dump.
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let v = glimmer::vision::Vision::load(&a[1])?;
    let raw = std::fs::read(&a[2])?; let w = u32::from_le_bytes(raw[0..4].try_into()?) as usize; let h = u32::from_le_bytes(raw[4..8].try_into()?) as usize;
    let runs: usize = std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut o = None;
    for r in 0..runs { let t = std::time::Instant::now(); let x = v.encode(&raw[8..], w, h)?;
        println!("run {r}: {w}x{h} -> {} tokens, wall {:.3}s (CPU prep {:.3}s, GPU {:.3}s)", x.n_out, t.elapsed().as_secs_f64(), x.cpu_s, x.gpu_s); o = Some(x); }
    let o = o.unwrap();
    std::fs::write(&a[3], unsafe { std::slice::from_raw_parts(o.emb.as_ptr() as *const u8, o.emb.len() * 4) })?;
    if let Some(rp) = a.get(4) {
        let rb = std::fs::read(rp)?; let rf: Vec<f32> = rb.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let d = glimmer::vision::OUT as usize; let n = o.n_out.min(rf.len() / d);
        let (mut mn, mut sum) = (2.0f64, 0.0f64); let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64); let mut mx = 0f32;
        for t in 0..n { let (mut dd, mut x2, mut y2) = (0f64, 0f64, 0f64);
            for i in 0..d { let (p, q) = (o.emb[t * d + i] as f64, rf[t * d + i] as f64); dd += p * q; x2 += p * p; y2 += q * q; mx = mx.max((p - q).abs() as f32); }
            let c = dd / (x2.sqrt() * y2.sqrt()); mn = mn.min(c); sum += c; dot += dd; na += x2; nb += y2; }
        println!("COSINE tokens {n} (ours {} ref {}): mean per-token {:.6} min {:.6} overall {:.6} max|diff| {:.4} rms ours {:.4} ref {:.4}", o.n_out, rf.len() / d, sum / n as f64, mn, dot / (na.sqrt() * nb.sqrt()), mx,
            (na / (n * d) as f64).sqrt(), (nb / (n * d) as f64).sqrt());
    }
    Ok(())
}
