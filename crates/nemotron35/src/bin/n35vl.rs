//! Nemotron 3.5 Super VL end to end on one real image: `n35vl <model.gguf> <mmproj_st dir> <image> <pre ids> <post ids> [n_new]`
//! prompt ids = pre ++ [placeholder x n_img_tokens] ++ post (pre ends with <img>, post starts with </img>; make them with llama-tokenize).
//! Pipeline: nemotron35::imgprep (decode/resize/normalise) -> Vision ViT (GPU) -> embeddings replace the placeholder rows in prefill -> greedy decode.
//! env: MAX_CTX (default 4096), PF_M (512), RUNS (repeat the whole generate), IMG_ONLY=1 (stop after the image encode, loop RUNS times).
use anyhow::Result;
fn env(n: &str, d: usize) -> usize { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn ids(s: &str) -> Vec<u32> { s.trim().trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().parse().unwrap()).collect() }
fn rd(p: &str) -> Result<Vec<u32>> { Ok(ids(&std::fs::read_to_string(p)?)) }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n_new: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(80);
    let (pre, post) = (rd(&a[4])?, rd(&a[5])?);
    let m = nemotron35::Model::load(&a[1], env("MAX_CTX", 4096) as u32)?;
    let sc = nemotron35::prefill::Scratch::new(&m, env("PF_M", 512) as u32)?;
    let v = nemotron35::vision::Vision::load(&m.ctx, &a[2])?;
    let (rgb, w, h) = nemotron35::imgprep::decode(&std::fs::read(&a[3])?)?;
    let enc = |v: &nemotron35::vision::Vision| -> Result<(Vec<f32>, usize, usize, f64, f64)> {
        let t = std::time::Instant::now(); let (px, tw, th) = nemotron35::imgprep::preprocess_rgb(&rgb, w, h); let cpu = t.elapsed().as_secs_f64();
        let (o, gpu) = v.encode(&m.ctx, &px, th, tw)?; Ok((o, tw, th, cpu, gpu)) };
    let _ = enc(&v)?; // warm-up
    let (emb, tw, th, cpu, gpu) = enc(&v)?; let cnt = emb.len() / 4096;
    println!("image {w}x{h} -> {tw}x{th}, {cnt} tokens: preprocess CPU {:.1} ms, ViT GPU {:.1} ms", cpu * 1e3, gpu * 1e3);
    if env("IMG_ONLY", 0) > 0 { for _ in 0..env("RUNS", 1) { enc(&v)?; } return Ok(()) }
    let eb = m.ctx.buffer((emb.len() * 4) as u64)?; eb.as_f32_mut()[..emb.len()].copy_from_slice(&emb);
    let mut prompt = pre.clone(); let ip = prompt.len() as u32; prompt.extend(std::iter::repeat(0).take(cnt)); prompt.extend(&post);
    *sc.img.borrow_mut() = Some((eb, ip, cnt as u32));
    println!("prompt {} tokens (image rows {ip}..{})", prompt.len(), ip as usize + cnt);
    for run in 0..env("RUNS", 1) {
        let (out, pg, pw, dg, _dw, st) = m.generate_pf(&sc, &prompt, n_new)?;
        if run == 0 { println!("ids: {:?}", out); }
        println!("prefill {:.3}s gpu ({:.3}s wall) = {:.1} tok/s | decode {} tok {:.2} tok/s", pg, pw, (prompt.len() - 1) as f64 / pg, st, st as f64 / dg);
    }
    Ok(())
}
