//! `n35img <mmproj_st dir> <image.png|jpg> [ref_px.bin] [ref_emb.f32]`: real-image preprocessing + ViT encode, compared with
//! llama.cpp mtmd dumps (MTMD_DUMP_PX: i32 w,h + interleaved f32 [h][w][3]; MTMD_DUMP_EMB: f32 [tokens][4096]).
//! `n35img mk <in> <out.png|jpg> <w> <h>`: make a test image (Triangle-resize, for odd-size / jpeg cases). env: RUNS=n timed repeats.
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "mk" { let im = image::open(&a[2])?.resize_exact(a[4].parse()?, a[5].parse()?, image::imageops::FilterType::Triangle); im.to_rgb8().save(&a[3])?; return Ok(()) }
    let t0 = std::time::Instant::now();
    let (px, w, h) = nemotron35::imgprep::preprocess_file(&a[2])?; let cpu = t0.elapsed().as_secs_f64();
    println!("preprocess {} -> {w}x{h} ({} tokens) CPU {:.2} ms", a[2], (w / 32) * (h / 32), cpu * 1e3);
    if let Some(r) = a.get(3).filter(|r| r.as_str() != "-") {
        let b = std::fs::read(r)?; let (rw, rh) = (i32::from_le_bytes(b[0..4].try_into()?) as usize, i32::from_le_bytes(b[4..8].try_into()?) as usize);
        let rf: Vec<f32> = b[8..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        println!("ref pixels {rw}x{rh}; size match {}", rw == w && rh == h);
        if rw == w && rh == h { let n = w * h; let (mut mx, mut ne) = (0f32, 0usize);
            for i in 0..n { for c in 0..3 { let d = (px[c * n + i] - rf[i * 3 + c]).abs(); mx = mx.max(d); if d != 0.0 { ne += 1 } } }
            println!("pixel tensor: max |diff| {mx:e}, {ne}/{} elements differ", 3 * n); }
    }
    let ctx = vkrt::Ctx::new()?; let v = nemotron35::vision::Vision::load(&ctx, &a[1])?;
    let _ = v.encode(&ctx, &px, h, w)?;
    let runs: usize = std::env::var("RUNS").ok().and_then(|x| x.parse().ok()).unwrap_or(1);
    let mut out = vec![]; let mut ts = vec![]; let (rgb, sw, sh) = nemotron35::imgprep::decode(&std::fs::read(&a[2])?)?; let mut tc = vec![];
    // each run = CPU preprocess (resize+normalise from decoded RGB) + ViT encode, as in a real request (file decode excluded)
    for _ in 0..runs { let t = std::time::Instant::now(); let (px2, ..) = nemotron35::imgprep::preprocess_rgb(&rgb, sw, sh); tc.push(t.elapsed().as_secs_f64()); let (o, dt) = v.encode(&ctx, &px2, h, w)?; out = o; ts.push(dt) }
    tc.sort_by(|x, y| x.partial_cmp(y).unwrap()); println!("preprocess CPU median {:.2} ms over {runs} runs", tc[tc.len() / 2] * 1e3);
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!("encoded {} tokens x 4096: median {:.1} ms (GPU wall, {runs} runs)", out.len() / 4096, ts[ts.len() / 2] * 1e3);
    if let Some(p) = std::env::var("OUT_EMB").ok() { std::fs::write(p, out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?; }
    if let Some(r) = a.get(4) {
        let rb = std::fs::read(r)?; let rf: Vec<f32> = rb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if rf.len() != out.len() { anyhow::bail!("ref has {} floats, ours {}", rf.len(), out.len()) }
        let (mut dot, mut na, mut nb, mut mx, mut mn) = (0f64, 0f64, 0f64, 0f32, f64::MAX);
        for t in 0..out.len() / 4096 { let (x, y) = (&out[t * 4096..(t + 1) * 4096], &rf[t * 4096..(t + 1) * 4096]);
            let (mut d, mut a2, mut b2) = (0f64, 0f64, 0f64);
            for i in 0..4096 { d += x[i] as f64 * y[i] as f64; a2 += (x[i] as f64).powi(2); b2 += (y[i] as f64).powi(2); mx = mx.max((x[i] - y[i]).abs()); }
            mn = mn.min(d / (a2.sqrt() * b2.sqrt())); dot += d; na += a2; nb += b2; }
        println!("embedding parity: cosine {:.6} (min per-token {:.6}), max |diff| {:.4}, ref rms {:.4}", dot / (na.sqrt() * nb.sqrt()), mn, mx, (nb / rf.len() as f64).sqrt());
    }
    Ok(())
}
