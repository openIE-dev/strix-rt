//! qi [--golden DIR] [--size N] [--steps N] [--out PNG] [--prompt S] [--seed-latents NPY] [--stage te|dit|all]
//! Qwen-Image-2.1 text-to-image. With --golden, checks ids, prompt embeds, sigmas, per-step velocity (teacher-forced) vs the diffusers dumps.
use anyhow::{bail, Result};
use qimage::*;
pub fn read_npy(path: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let b = std::fs::read(path)?; if &b[..6] != b"\x93NUMPY" { bail!("{path}: not npy") }
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = std::str::from_utf8(&b[off..off + hl])?; if !h.contains("'<f4'") { bail!("{path}: need <f4: {h}") }
    let sh = h.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> = sh.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    Ok((b[off + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), shape))
}
fn stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let m = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let (n, d) = a.iter().zip(b).fold((0f64, 0f64), |(n, d), (x, y)| (n + ((x - y) as f64).powi(2), d + (*y as f64).powi(2)));
    (m, (n / d.max(1e-30)).sqrt() as f32)
}
/// image [4][H][W] in [-1, 1] -> PNG (RGB, or RGBA if the 4th channel is not all opaque) + raw f32 [0,1]; PSNR vs golden image.npy.
fn save(out: &str, img: &[f32], size: u32, golden: Option<&str>) -> Result<()> {
    let n = (size * size) as usize; let u: Vec<f32> = img.iter().map(|x| (x * 0.5 + 0.5).clamp(0.0, 1.0)).collect();
    let rgba = u[3 * n..].iter().any(|a| *a < 0.5); let ch = if rgba { 4 } else { 3 };
    let mut px = vec![0u8; n * ch]; for p in 0..n { for c in 0..ch { px[p * ch + c] = (u[c * n + p] * 255.0 + 0.5) as u8 } }
    png::write(out, size, size, ch as u32, &px)?; std::fs::write(format!("{out}.img.f32"), u.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    eprintln!("wrote {out} ({}x{} {})", size, size, if rgba { "RGBA" } else { "RGB" });
    if let Some(g) = golden { if let Ok((gi, gs)) = read_npy(&format!("{g}/image.npy")) {
        let m = gi.len().min(u.len()); let rgb = 3 * n; let k = m.min(rgb);
        let mse = u[..k].iter().zip(&gi[..k]).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / k as f64;
        eprintln!("vs golden {gs:?}: RGB PSNR {:.2} dB", 10.0 * (1.0 / mse.max(1e-12)).log10()); } }
    Ok(())
}
fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let (mut golden, mut size, mut steps, mut out, mut stage) = (None::<String>, 1024u32, 40usize, "qi.png".to_string(), "all".to_string());
    let mut prompt = "A red fox sitting in a snowy pine forest at sunrise, soft golden light, a wooden sign that reads \"STRIX\", highly detailed photograph".to_string();
    while !args.is_empty() { let a = args.remove(0); let v = args.remove(0);
        match a.as_str() { "--golden" => golden = Some(v), "--size" => size = v.parse()?, "--steps" => steps = v.parse()?, "--out" => out = v, "--prompt" => prompt = v, "--stage" => stage = v, _ => bail!("unknown {a}") } }
    let home = std::env::var("HOME")?; let dir = std::env::var("QI_MODEL").unwrap_or(format!("{home}/models/qwen-image-2.1"));
    let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?; let kn = Kern::new(&ctx)?;
    let tk = tok::Tok::load(&format!("{dir}/processor/tokenizer.json"))?;
    let (ids, drop_idx) = tok::prompt_ids(&tk, &prompt);
    let gj: Option<serde_json::Value> = golden.as_ref().and_then(|g| std::fs::read_to_string(format!("{g}/ref.json")).ok()).and_then(|t| serde_json::from_str(&t).ok());
    if let Some(j) = &gj { let gi: Vec<u32> = j["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        eprintln!("ids match: {} ({} vs {}), drop {} vs {}", gi == ids, ids.len(), gi.len(), drop_idx, j["drop_idx"]); }
    let t0 = std::time::Instant::now();
    let te = TextEncoder::load(&ctx, &format!("{dir}/text_encoder"))?; eprintln!("te loaded {:.2}s zc={}", te.st.load_s, te.st.zero_copy);
    let (mut emb, teg) = te.encode(&ctx, &pf, &kn, &ids, drop_idx)?;
    if std::env::var("QI_GOLDEN_EMB").is_ok() { emb = read_npy(&format!("{}/prompt_embeds.npy", golden.as_ref().unwrap()))?.0; eprintln!("using golden prompt embeds"); } let lt = (emb.len() / 4096) as u32;
    eprintln!("te: Lt={lt} gpu {:.1} ms", teg * 1e3);
    if let Some(g) = &golden { let (ge, gs) = read_npy(&format!("{g}/prompt_embeds.npy"))?; let (mx, rel) = stats(&emb, &ge); eprintln!("prompt_embeds {gs:?}: max|d| {mx:.4} rel {rel:.2e}"); }
    if stage == "te" { eprintln!("ids {:?} drop {drop_idx}", ids); let (full, _) = te.encode(&ctx, &pf, &kn, &ids, 0)?;
        std::fs::write(format!("{out}.te.f32"), full.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?; return Ok(()) }
    drop(te);
    if stage == "vae" { // decode the golden final latents (last golden step input + its Euler update)
        let g = golden.as_ref().unwrap(); let (gl, _) = read_npy(&format!("{g}/step_lat_in.npy"))?; let (gv, _) = read_npy(&format!("{g}/step_v.npy"))?;
        let (h, w) = (size / 16, size / 16); let nn = (h * w * 64) as usize; let sig = sigmas(steps, (h * w) as usize);
        let lat: Vec<f32> = (0..nn).map(|i| gl[(steps - 1) * nn + i] + (sig[steps] - sig[steps - 1]) * gv[(steps - 1) * nn + i]).collect();
        let vae = vae::Vae::load(&ctx, &format!("{dir}/vae"))?; eprintln!("vae loaded {:.2}s", vae.load_s);
        let t1 = std::time::Instant::now(); let (img, vg) = vae.decode(&ctx, &pf, &kn, &lat, h, w)?;
        eprintln!("vae decode gpu {:.2}s wall {:.2}s", vg, t1.elapsed().as_secs_f64());
        return save(&out, &img, size, Some(g)); }
    let dit = Dit::load(&ctx, &format!("{dir}/transformer"))?; eprintln!("dit loaded {:.2}s zc={} fused={}", dit.st.load_s, dit.st.zero_copy, dit.fused);
    let (h, w) = (size / 16, size / 16); let n = h * w;
    let s = dit.state(&ctx, lt, n)?; Dit::rope_table(&s, h, w);
    let pg = dit.prefix(&ctx, &pf, &kn, &s, &emb)?; eprintln!("prefix gpu {:.1} ms", pg * 1e3);
    if stage == "prof" { let lat0 = vec![0.1f32; (n * 64) as usize]; dit.step(&ctx, &pf, &kn, &s, &lat0, 0.5)?;
        let pr = dit.prof_block(&ctx, &pf, &kn, &s, 5)?; let tot: f64 = pr.iter().map(|x| x.1).sum();
        let d = 4096f64; let m = n as f64; let fl = |name: &str| match name { "qkv" => 2.0 * m * d * 3.0 * d, "o" => 2.0 * m * d * d, "swiglu" => 2.0 * m * d * 2.0 * 12288.0, "down" => 2.0 * m * 12288.0 * d,
            "flash" => 4.0 * m * (m + lt as f64) * d, _ => 0.0 };
        for (k, t) in &pr { eprintln!("{k:8} {:8.2} ms {:5.1}%  {:6.1} TFLOPS", t * 1e3, 100.0 * t / tot, fl(k) / t / 1e12); }
        eprintln!("block total {:.2} ms -> x32 = {:.2} s/step", tot * 1e3, tot * 32.0);
        let st = dit.step(&ctx, &pf, &kn, &s, &lat0, 0.5)?; eprintln!("full step gpu {:.2} s", st); return Ok(()) }
    let sig = sigmas(steps, n as usize);
    if let Some(j) = &gj { let gsg: Vec<f32> = j["sigmas"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        eprintln!("sigmas max|d| {:.2e}", sig.iter().zip(&gsg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)); }
    let mut lat = match &golden { Some(g) => read_npy(&format!("{g}/latents0.npy"))?.0, None => { let mut x = 42u64; (0..n * 64).map(|_| { let mut s = 0f32; for _ in 0..12 { x ^= x << 13; x ^= x >> 7; x ^= x << 17; s += (x >> 40) as f32 / (1u64 << 24) as f32 } s - 6.0 }).collect() } };
    let gsteps = golden.as_ref().filter(|_| std::env::var("QI_NO_TF").is_err()).and_then(|g| Some((read_npy(&format!("{g}/step_lat_in.npy")).ok()?.0, read_npy(&format!("{g}/step_v.npy")).ok()?.0)));
    let nn = (n * 64) as usize; let mut gpu = 0.0;
    for i in 0..steps {
        if let Some((gl, gv)) = &gsteps { // teacher-forced check on the golden input
            dit.step(&ctx, &pf, &kn, &s, &gl[i * nn..(i + 1) * nn], sig[i])?; let v = &s.v.as_f32()[..nn];
            let (mx, rel) = stats(v, &gv[i * nn..(i + 1) * nn]); let (_, drift) = stats(&lat, &gl[i * nn..(i + 1) * nn]);
            eprintln!("step {i:2} sigma {:.4}: v max|d| {mx:.4} rel {rel:.2e} | free-run latent rel {drift:.2e}", sig[i]);
        }
        gpu += dit.step(&ctx, &pf, &kn, &s, &lat, sig[i])?; let v = &s.v.as_f32()[..nn]; let ds = sig[i + 1] - sig[i];
        for (l, x) in lat.iter_mut().zip(v) { *l += ds * x }
    }
    eprintln!("denoise: {steps} steps gpu {:.2}s total {:.2}s", gpu, t0.elapsed().as_secs_f64());
    let raw: Vec<u8> = lat.iter().flat_map(|x| x.to_le_bytes()).collect(); std::fs::write(format!("{out}.lat.f32"), raw)?;
    drop(s); drop(dit);
    let vae = vae::Vae::load(&ctx, &format!("{dir}/vae"))?; eprintln!("vae loaded {:.2}s", vae.load_s);
    let t1 = std::time::Instant::now(); let (img, vg) = vae.decode(&ctx, &pf, &kn, &lat, h, w)?;
    eprintln!("vae decode gpu {:.2}s wall {:.2}s | total {:.2}s", vg, t1.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
    save(&out, &img, size, golden.as_deref())
}
