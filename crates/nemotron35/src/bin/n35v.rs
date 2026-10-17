//! Nemotron 3.5 VL vision tower: `n35v <mmproj_st dir> [size=512] [ref.f32]` encodes the mtmd-debug rainbow image
//! and compares with llama.cpp's dump (MTMD_DUMP from llama-mtmd-debug -p encode --image rainbow -n size).
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ctx = vkrt::Ctx::new()?;
    let v = nemotron35::vision::Vision::load(&ctx, &a[1])?;
    let s: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(512);
    let px = nemotron35::vision::rainbow(s);
    let (_, _) = v.encode(&ctx, &px, s, s)?; // warm-up
    let (out, dt) = v.encode(&ctx, &px, s, s)?;
    println!("encoded {s}x{s}: {} tokens x 4096 in {:.1} ms", out.len() / 4096, dt * 1e3);
    println!("out[0][..6] = {:?}", &out[..6]);
    if let Some(r) = a.get(3) {
        let rb = std::fs::read(r)?; let rf: Vec<f32> = rb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if rf.len() != out.len() { anyhow::bail!("ref has {} floats, ours {}", rf.len(), out.len()) }
        println!("ref[0][..6] = {:?}", &rf[..6]);
        let (mut dot, mut na, mut nb, mut mx, mut mn) = (0f64, 0f64, 0f64, 0f32, f64::MAX);
        for t in 0..out.len() / 4096 { let (x, y) = (&out[t * 4096..(t + 1) * 4096], &rf[t * 4096..(t + 1) * 4096]);
            let (mut d, mut a2, mut b2) = (0f64, 0f64, 0f64);
            for i in 0..4096 { d += x[i] as f64 * y[i] as f64; a2 += (x[i] as f64).powi(2); b2 += (y[i] as f64).powi(2); mx = mx.max((x[i] - y[i]).abs()); }
            mn = mn.min(d / (a2.sqrt() * b2.sqrt())); dot += d; na += a2; nb += b2; }
        println!("parity: cosine {:.6} (min per-token {:.6}), max |diff| {:.4}, ref rms {:.4}", dot / (na.sqrt() * nb.sqrt()), mn, mx, (nb / rf.len() as f64).sqrt());
    }
    Ok(())
}
