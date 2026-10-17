//! Vision parity: mcv_vis <vision_st dir> <image.ppm> <ref img_embd.f32> [reps]
use anyhow::Result;
use mcv::vision::{preprocess, read_ppm, Vision};
use vkrt::Ctx;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let ctx = Ctx::new()?;
    let t0 = std::time::Instant::now();
    let v = Vision::load(&ctx, &a[1])?;
    eprintln!("vision load {:.2}s", t0.elapsed().as_secs_f64());
    let (rgb, w, h) = read_ppm(&a[2])?;
    let tp = std::time::Instant::now();
    let (tiles, grid) = preprocess(&rgb, w, h);
    let tpre = tp.elapsed().as_secs_f64();
    println!("image {w}x{h} -> {} tiles (grid {:?}): {:?}  preprocess {:.1} ms", tiles.len(), grid, tiles.iter().map(|t| (t.w, t.h)).collect::<Vec<_>>(), tpre * 1e3);
    let mut out = vec![]; let mut times = vec![];
    for r in 0..reps { let mut o = vec![]; let mut tt = 0.0; for t in &tiles { let (e, dt) = v.encode(&ctx, t)?; o.extend(e); tt += dt; } times.push(tt); if r == 0 { out = o; } }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("encode times {:?} -> median {:.1} ms", times.iter().map(|t| (t * 1e4).round() / 10.0).collect::<Vec<_>>(), times[times.len() / 2] * 1e3);
    let raw = std::fs::read(&a[3])?;
    let rf: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let d = 2048; let n = out.len() / d; let nr = rf.len() / d;
    println!("tokens ours {n} ref {nr}");
    let mut worst = 1.0f64; let mut sum = 0.0;
    for i in 0..n.min(nr) {
        let (x, y) = (&out[i * d..(i + 1) * d], &rf[i * d..(i + 1) * d]);
        let (mut xy, mut xx, mut yy) = (0f64, 0f64, 0f64);
        for j in 0..d { xy += x[j] as f64 * y[j] as f64; xx += (x[j] as f64).powi(2); yy += (y[j] as f64).powi(2); }
        let c = xy / (xx.sqrt() * yy.sqrt() + 1e-30); worst = worst.min(c); sum += c;
        if i < 3 || i + 2 >= n { println!("  tok {i}: cos {c:.6}  |ours| {:.3} |ref| {:.3}", xx.sqrt(), yy.sqrt()); }
    }
    println!("COS mean {:.6} min {:.6}", sum / n.min(nr) as f64, worst);
    std::fs::write("/tmp/mcv_vis_out.f32", out.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(())
}
