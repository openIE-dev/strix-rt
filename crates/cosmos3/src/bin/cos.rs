//! cos --golden DIR [--stage tower|tf|run|text] [--out PREFIX]
//! Cosmos3-Nano text-to-video on strix-rt; with --golden checks positions, per-layer hidden states, per-call velocity
//! (teacher forced on the diffusers inputs), sigmas and the free-running final latent against the diffusers dumps (ref.py).
use anyhow::{bail, Result};
use cosmos3::*;
pub fn read_npy(path: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let b = std::fs::read(path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?; if &b[..6] != b"\x93NUMPY" { bail!("{path}: not npy") }
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = std::str::from_utf8(&b[off..off + hl])?;
    let sh = h.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    if h.contains("'<i8'") { return Ok((b[off + hl..].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(), sh.split(',').filter_map(|x| x.trim().parse().ok()).collect())) }
    if !h.contains("'<f4'") { bail!("{path}: need <f4: {h}") }
    Ok((b[off + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), sh.split(',').filter_map(|x| x.trim().parse().ok()).collect()))
}
pub fn cmp(a: &[f32], b: &[f32]) -> String {
    let (mut d, mut n, mut aa, mut bb, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f32);
    for (x, y) in a.iter().zip(b) { let (x, y) = (*x as f64, *y as f64); d += (x - y) * (x - y); n += x * y; aa += x * x; bb += y * y; mx = mx.max((x - y).abs() as f32); }
    format!("cos {:.6} rel {:.3e} max|d| {:.4} (n {} vs {})", n / (aa * bb).sqrt().max(1e-30), (d / bb.max(1e-30)).sqrt(), mx, a.len(), b.len())
}
pub fn now() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64() }
fn main() -> Result<()> {
    if std::env::var_os("QI_WALIGN").is_none() { std::env::set_var("QI_WALIGN", "1") } // 4 KB aligned weight copies: gen eval 450 -> 279 ms, fixes 720p zero-copy submission ENOMEM; QI_WALIGN=0 = old zero-copy path
    let mut args: Vec<String> = std::env::args().skip(1).collect(); let args0 = args.clone();
    if args.first().map_or(false, |a| a == "ask") { let home = std::env::var("HOME")?; let dir = std::env::var("COS_MODEL").unwrap_or(format!("{home}/models/new/cosmos3/Cosmos3-Nano")); return ask(&dir, &args[1..].to_vec()); }
    if args.first().map_or(false, |a| a == "vae") { // cos vae --lat F --size HxW --frames N --out P : decode a saved gen latent in its own process
        let home = std::env::var("HOME")?; let dir = std::env::var("COS_MODEL").unwrap_or(format!("{home}/models/new/cosmos3/Cosmos3-Nano"));
        let mut o = std::collections::HashMap::new(); let a = &args[1..]; let mut i = 0; while i + 1 < a.len() { o.insert(a[i].trim_start_matches('-').to_string(), a[i + 1].clone()); i += 2; }
        let (hh, ww) = { let s = o["size"].clone(); let (x, y) = s.split_once('x').unwrap(); (x.parse::<usize>()?, y.parse::<usize>()?) };
        let frames: usize = o["frames"].parse()?; let (t, h, w) = ((frames - 1) / 4 + 1, hh / 16, ww / 16);
        let lat: Vec<f32> = std::fs::read(&o["lat"])?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?; let mut kn = qimage::Kern::new(&ctx)?; vae::add_kernels(&ctx, &mut kn)?;
        return decode_save(&ctx, &pf, &kn, &dir, &lat, t, h, w, o.get("out").map(|s| s.as_str()).unwrap_or("cos_vae"), "", ""); }
    if args.first().map_or(false, |a| a == "gen") { let home = std::env::var("HOME")?; let dir = std::env::var("COS_MODEL").unwrap_or(format!("{home}/models/new/cosmos3/Cosmos3-Nano")); return gen(&dir, &args[1..].to_vec()); }
    let (mut golden, mut stage, mut out, mut latf, mut vref) = (String::new(), "tf".to_string(), "cos".to_string(), String::new(), String::new());
    while !args.is_empty() { let a = args.remove(0); let v = args.remove(0);
        match a.as_str() { "--golden" => golden = v, "--stage" => stage = v, "--out" => out = v, "--lat" => latf = v, "--vref" => vref = v, _ => bail!("unknown {a}") } }
    let home = std::env::var("HOME")?; let dir = std::env::var("COS_MODEL").unwrap_or(format!("{home}/models/new/cosmos3/Cosmos3-Nano"));
    if golden.is_empty() { return gen(&dir, &args0); }
    let g = |n: &str| read_npy(&format!("{golden}/{n}.npy"));
    let run: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{golden}/run.json"))?)?;
    let steps = run["steps"].as_u64().unwrap() as usize; let ncalls = run["calls"].as_u64().unwrap() as usize;
    let (_, vs) = g("c0_vis_in")?; let (t, h, w) = (vs[2], vs[3], vs[4]); let n = (t * h * w / 4) as u32;
    let ids = |i: usize| -> Result<Vec<u32>> { Ok(g(&format!("c{i}_input_ids"))?.0.iter().map(|x| *x as u32).collect()) };
    let (mut ic, mut iu) = (ids(0)?, ids(1)?);
    { let tk = prompt::Tok::load(&format!("{dir}/text_tokenizer/tokenizer.json"))?; let pr = run["prompt"].as_str().unwrap_or("").to_string();
      let (fr, hh, ww) = (run["F"].as_u64().unwrap() as u32, run["H"].as_u64().unwrap() as u32, run["W"].as_u64().unwrap() as u32);
      let (oc, ou) = (prompt::ids(&tk, &pr, false, fr, hh, ww, 24.0)?, prompt::ids(&tk, "", true, fr, hh, ww, 24.0)?);
      eprintln!("tokenizer: cond ids match {} ({} vs {}), uncond match {} ({} vs {})", oc == ic, oc.len(), ic.len(), ou == iu, ou.len(), iu.len());
      if oc != ic { eprintln!("ours {:?}\nref  {:?}", oc, ic); }
      if std::env::var("COS_OWN_IDS").is_ok() { ic = oc; iu = ou; } }
    eprintln!("latent t{t} h{h} w{w} -> {n} tokens; cond {} uncond {} text tokens; {steps} steps, {ncalls} calls", ic.len(), iu.len());
    if stage == "sched" { // CPU: scheduler alone on the golden CFG velocities
        let (sg, _) = sched::karras_flow(steps, 0.147, 200.0); let mut sch = sched::UniPc::new(sg); let mut lat = g("c0_vis_in")?.0;
        for i in 0..steps { let (vc, _) = g(&format!("c{}_vel", 2 * i))?; let (vu, _) = g(&format!("c{}_vel", 2 * i + 1))?;
            let v: Vec<f32> = vu.iter().zip(&vc).map(|(u, c)| u + 6.0 * (c - u)).collect(); lat = sch.step(&v, &lat);
            if let Ok((r, _)) = g(&format!("c{}_vis_in", 2 * i + 2)) { eprintln!("sched step {i}: {}", cmp(&lat, &r)); } }
        return Ok(()) }
    let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?; let mut kn = qimage::Kern::new(&ctx)?; vae::add_kernels(&ctx, &mut kn)?;
    if stage == "vae" { let lat: Vec<f32> = std::fs::read(&latf)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        return decode_save(&ctx, &pf, &kn, &dir, &lat, t, h, w, &out, &vref, &golden); }
    let t0 = std::time::Instant::now();
    let m = Model::load(&ctx, &format!("{dir}/transformer"), stage == "text")?; eprintln!("loaded {:.2}s zc={} layers {}", m.st.load_s, m.st.zero_copy, m.nl);
    let mut sc = m.state(&ctx, ic.len() as u32, n)?; let mut su = m.state(&ctx, iu.len() as u32, n)?;
    Model::set_grid(&sc, t as u32, h as u32 / 2, w as u32 / 2); Model::set_grid(&su, t as u32, h as u32 / 2, w as u32 / 2);
    { let (pi, _) = g("c0_position_ids")?; let seq = pi.len() / 3; let lt = ic.len();
      let mut bad = 0; for j in 0..seq { let (et, eh, ew) = if j < lt { (j as f64, j as f64, j as f64) } else { let k = j - lt; let hw = h * w / 4;
          ((k / hw) as f64 + lt as f64 + 15000.0, ((k % hw) / (w / 2)) as f64, (k % (w / 2)) as f64) };
          if (pi[j] as f64 - et).abs() > 1e-3 || (pi[seq + j] as f64 - eh).abs() > 1e-3 || (pi[2 * seq + j] as f64 - ew).abs() > 1e-3 { bad += 1 } }
      eprintln!("position ids: {} / {seq} mismatches", bad); }
    let full = stage == "text"; let e0 = now();
    let (gp, logits) = m.prefix(&ctx, &pf, &kn, &mut sc, &ic, full)?; let (gq, _) = m.prefix(&ctx, &pf, &kn, &mut su, &iu, false)?;
    eprintln!("prefix gpu {:.1} + {:.1} ms", gp * 1e3, gq * 1e3); eprintln!("PHASE prefix {e0:.3} {:.3}", now());
    if let Some(lg) = logits { let am = lg.iter().enumerate().fold((0, f32::MIN), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }); eprintln!("und last-token argmax {} ({:.3})", am.0, am.1); }
    let tsr = g("timesteps")?.0; let (sg, ts) = sched::karras_flow(steps, 0.147, 200.0); let mut sch = sched::UniPc::new(sg);
    let gs = g("sigmas")?.0; eprintln!("sigmas max|d| {:.2e}  timesteps max|d| {:.2e}", sch.sigmas.iter().zip(&gs).map(|(a, b)| (*a as f32 - b).abs()).fold(0f32, f32::max),
        ts.iter().zip(&tsr).map(|(a, b)| (*a as f32 - b).abs()).fold(0f32, f32::max));
    if stage == "tower" || stage == "text" {
        let (vi, _) = g("c0_vis_in")?; let (v, gs0) = m.step(&ctx, &pf, &kn, &mut sc, &patchify(&vi, t, h, w), ts[0])?;
        for r in 0..3 { let mut s2 = m.state(&ctx, ic.len() as u32, n)?; let _ = r; std::mem::swap(&mut s2.dumps, &mut sc.dumps); let (_, gg) = m.step(&ctx, &pf, &kn, &mut sc, &patchify(&vi, t, h, w), ts[0])?; eprintln!("step gpu {:.1} ms (first {:.1} ms)", gg * 1e3, gs0 * 1e3); std::mem::swap(&mut s2.dumps, &mut sc.dumps); }
        if std::env::var("COS_PROF").is_ok() { let pr = m.prof(&ctx, &pf, &kn, &sc, 7)?; let tot: f64 = pr.iter().map(|x| x.1).sum(); let nn = n as f64; let d = 4096f64;
            let fl = |k: &str| match k { "qkv" => 2.0 * nn * d * 6144.0, "o" => 2.0 * nn * d * d, "swiglu" => 2.0 * nn * d * 24576.0, "down" => 2.0 * nn * 12288.0 * d, "attn" => 4.0 * nn * (nn + ic.len() as f64) * d, _ => 0.0 };
            for (k, t) in &pr { eprintln!("{k:7} {:8.3} ms {:5.1}% {:6.2} TFLOPS", t * 1e3, 100.0 * t / tot, fl(k) / t / 1e12); } eprintln!("layer {:.2} ms x36 = {:.1} ms", tot * 1e3, tot * 36e3); }
        for (name, x) in sc.dumps.iter() { let key = if name == "in_gen" { "c0_L0_in_gen".to_string() } else { format!("c0_{name}") };
            if let Ok((r, _)) = g(&key) { eprintln!("{name:10} {}", cmp(x, &r)); } }
        eprintln!("vel call0  {}", cmp(&unpatchify(&v, t, h, w), &g("c0_vel")?.0)); return Ok(()) }
    let e1 = now(); let nn = 48 * t * h * w; let mut lat = g("c0_vis_in")?.0; let gsc = 6.0f32; let mut gpu = 0.0;
    let tf = stage == "tf";
    for i in 0..steps {
        if tf { for ci in [2 * i, 2 * i + 1] { let (vi, _) = g(&format!("c{ci}_vis_in"))?; let st = if ci % 2 == 0 { &mut sc } else { &mut su };
            let (v, _) = m.step(&ctx, &pf, &kn, st, &patchify(&vi, t, h, w), ts[i])?; eprintln!("step {i} {} vel {}", if ci % 2 == 0 { "cond  " } else { "uncond" }, cmp(&unpatchify(&v, t, h, w), &g(&format!("c{ci}_vel"))?.0)); } }
        let (vc, g1) = m.step(&ctx, &pf, &kn, &mut sc, &patchify(&lat, t, h, w), ts[i])?; let (vu, g2) = m.step(&ctx, &pf, &kn, &mut su, &patchify(&lat, t, h, w), ts[i])?; gpu += g1 + g2;
        let v: Vec<f32> = unpatchify(&vu.iter().zip(&vc).map(|(u, c)| u + gsc * (c - u)).collect::<Vec<_>>(), t, h, w);
        lat = sch.step(&v, &lat);
        if i + 1 < steps { if let Ok((r, _)) = g(&format!("c{}_vis_in", 2 * i + 2)) { eprintln!("step {i} free-run latent vs ref {}", cmp(&lat, &r)); } }
    }
    eprintln!("denoise {steps} steps: gpu {:.2}s wall {:.2}s", gpu, t0.elapsed().as_secs_f64()); eprintln!("PHASE denoise {e1:.3} {:.3}", now());
    std::fs::write(format!("{out}.lat.f32"), lat[..nn].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    eprintln!("wrote {out}.lat.f32 [48][{t}][{h}][{w}]");
    drop(sc); drop(su); drop(m);
    pf.ds.clear(); // freed model/state buffers: no stale descriptor sets for recycled VkBuffer handles in the VAE
    decode_save(&ctx, &pf, &kn, &dir, &lat, t, h, w, &out, &vref, &golden)
}
fn decode_save(ctx: &vkrt::Ctx, pf: &prefill::Pf, kn: &qimage::Kern, dir: &str, lat: &[f32], t: usize, h: usize, w: usize, out: &str, vref: &str, golden: &str) -> Result<()> {
    if std::env::var_os("PF_MINWG").is_none() { std::env::set_var("PF_MINWG", "100000000") } // VAE convs: 128x128 GEMM tiles (-5% VAE time; +18% on the gen tower, so only here)
    let e2 = now(); let t0 = std::time::Instant::now(); let v = vae::Vae::load(ctx, &format!("{dir}/vae"))?; eprintln!("vae loaded {:.2}s", v.load_s);
    let mut dumps = vec![]; let t1 = std::time::Instant::now();
    let (vid, g) = v.decode(ctx, pf, kn, lat, t as u32, h as u32, w as u32, &mut dumps)?;
    eprintln!("vae decode gpu {:.2}s wall {:.2}s (load+decode {:.2}s)", g, t1.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64()); eprintln!("PHASE vae {e2:.3} {:.3}", now());
    for (name, x) in &dumps { if let Ok((r, s)) = read_npy(&format!("{vref}/{name}.npy")) { eprintln!("vae {name:4} {s:?} {}", cmp(x, &r)); } }
    let psnr = |a: &[f32], b: &[f32]| { let mse = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64; 10.0 * (1.0 / mse.max(1e-20)).log10() };
    let u: Vec<f32> = vid.iter().map(|x| (x * 0.5 + 0.5).clamp(0.0, 1.0)).collect();
    if let Ok((r, s)) = read_npy(&format!("{vref}/video.npy")) { let ru: Vec<f32> = r.iter().map(|x| (x * 0.5 + 0.5).clamp(0.0, 1.0)).collect(); eprintln!("video vs vae_ref {s:?}: PSNR {:.2} dB {}", psnr(&u, &ru), cmp(&u, &ru)); }
    let (nf, ho, wo) = (vid.len() / (3 * 256 * h * w), 16 * h, 16 * w);
    // golden pipeline video (output_type np: [F][H][W][3] in [0, 1])
    if let Ok((r, s)) = read_npy(&format!("{golden}/video.npy")) { if r.len() == u.len() { let mut cf = vec![0f32; r.len()];
        for f in 0..nf { for p in 0..ho * wo { for c in 0..3 { cf[(c * nf + f) * ho * wo + p] = r[(f * ho * wo + p) * 3 + c] } } }
        eprintln!("video vs golden pipeline {s:?}: PSNR {:.2} dB {}", psnr(&u, &cf), cmp(&u, &cf)); } }
    std::fs::write(format!("{out}.video.f32"), u.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    let mut rgb = vec![0u8; nf * ho * wo * 3]; for f in 0..nf { for p in 0..ho * wo { for c in 0..3 { rgb[(f * ho * wo + p) * 3 + c] = (u[(c * nf + f) * ho * wo + p] * 255.0 + 0.5) as u8 } } }
    std::fs::write(format!("{out}.rgb"), &rgb)?;
    let st = std::process::Command::new(std::env::var("COS_FFMPEG").unwrap_or(format!("{}/cosmos_ref/.venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2", std::env::var("HOME").unwrap_or_default()))).args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{wo}x{ho}"), "-r", "24", "-i", &format!("{out}.rgb"), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "16", &format!("{out}.mp4")]).status();
    eprintln!("wrote {out}.video.f32 [3][{nf}][{ho}][{wo}], {out}.rgb, mp4: {:?}", st.map(|s| s.success()));
    Ok(())
}

/// cos gen [--prompt S] [--negative S] [--size HxW] [--frames F] [--steps N] [--cfg G] [--seed N] [--out P]: text in, video out.
fn gen(dir: &str, a: &[String]) -> Result<()> {
    let mut o = std::collections::HashMap::new(); let mut i = 0; while i + 1 < a.len() { o.insert(a[i].trim_start_matches('-').to_string(), a[i + 1].clone()); i += 2; }
    let gs = |k: &str, d: &str| o.get(k).cloned().unwrap_or(d.to_string());
    let prompt = gs("prompt", "A robot arm is cleaning a plate in the kitchen"); let neg = gs("negative", "");
    let (hh, ww) = { let s = gs("size", "256x256"); let (x, y) = s.split_once('x').unwrap(); (x.parse::<u32>()?, y.parse::<u32>()?) };
    let frames: u32 = gs("frames", "9").parse()?; let steps: usize = gs("steps", "35").parse()?; let cfg: f32 = gs("cfg", "6").parse()?; let seed: u64 = gs("seed", "42").parse()?; let out = gs("out", "cos_gen");
    anyhow::ensure!(hh % 32 == 0 && ww % 32 == 0 && (frames - 1) % 4 == 0, "size must be multiples of 32, frames 4k+1");
    let (t, h, w) = ((frames as usize - 1) / 4 + 1, hh as usize / 16, ww as usize / 16); let n = (t * h * w / 4) as u32;
    let t00 = std::time::Instant::now();
    let tk = prompt::Tok::load(&format!("{dir}/text_tokenizer/tokenizer.json"))?;
    let ic = prompt::ids(&tk, &prompt, false, frames, hh, ww, 24.0)?; let iu = prompt::ids(&tk, &neg, true, frames, hh, ww, 24.0)?;
    eprintln!("{hh}x{ww} {frames} frames -> latent t{t} h{h} w{w}, {n} tokens; text {} / {} tokens; {steps} steps cfg {cfg}", ic.len(), iu.len());
    let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?; let mut kn = qimage::Kern::new(&ctx)?; vae::add_kernels(&ctx, &mut kn)?;
    let m = Model::load(&ctx, &format!("{dir}/transformer"), false)?; eprintln!("loaded {:.2}s", m.st.load_s);
    let mut sc = m.state(&ctx, ic.len() as u32, n)?; let mut su = m.state(&ctx, iu.len() as u32, n)?;
    Model::set_grid(&sc, t as u32, h as u32 / 2, w as u32 / 2); Model::set_grid(&su, t as u32, h as u32 / 2, w as u32 / 2);
    let e0 = now(); m.prefix(&ctx, &pf, &kn, &mut sc, &ic, false)?; m.prefix(&ctx, &pf, &kn, &mut su, &iu, false)?; eprintln!("PHASE prefix {e0:.3} {:.3}", now());
    let (sg, ts) = sched::karras_flow(steps, 0.147, 200.0); let mut sch = sched::UniPc::new(sg);
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1; let mut u = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
    let nn = 48 * t * h * w; let mut lat: Vec<f32> = (0..nn).map(|_| { let (a, b) = (u(), u()); ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32 }).collect();
    let e1 = now(); let mut gpu = 0.0;
    for i in 0..steps { let tok = patchify(&lat, t, h, w);
        let (vc, g1) = m.step(&ctx, &pf, &kn, &mut sc, &tok, ts[i])?; let (vu, g2) = m.step(&ctx, &pf, &kn, &mut su, &tok, ts[i])?; gpu += g1 + g2;
        let v = unpatchify(&vu.iter().zip(&vc).map(|(u, c)| u + cfg * (c - u)).collect::<Vec<_>>(), t, h, w); lat = sch.step(&v, &lat); }
    eprintln!("denoise {steps} steps: gpu {gpu:.2}s wall {:.2}s", now() - e1); eprintln!("PHASE denoise {e1:.3} {:.3}", now());
    std::fs::write(format!("{out}.lat.f32"), lat.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    drop(sc); drop(su); drop(m);
    if std::env::var("COS_NO_VAE").is_ok() { eprintln!("latent only; decode with: cos vae --lat {out}.lat.f32 --size {hh}x{ww} --frames {frames}"); return Ok(()) }
    if std::env::var("COS_VAE_SAME").is_err() { // the VAE in a fresh process: after the 12-step denoise in the same process the 720p VAE lost the device twice
        drop(kn); drop(pf); drop(ctx);
        use std::os::unix::process::CommandExt; // exec: the denoise process image (and its Vulkan device) is replaced, the VAE starts clean
        let e = std::process::Command::new(std::env::current_exe()?).args(["vae", "--lat", &format!("{out}.lat.f32"), "--size", &format!("{hh}x{ww}"), "--frames", &frames.to_string(), "--out", &out]).exec();
        anyhow::bail!("exec vae: {e}") }
    decode_save(&ctx, &pf, &kn, dir, &lat, t, h, w, &out, "", "")?; eprintln!("total {:.2}s", t00.elapsed().as_secs_f64()); Ok(())
}

fn ffmpeg() -> String { std::env::var("COS_FFMPEG").unwrap_or(format!("{}/cosmos_ref/.venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2", std::env::var("HOME").unwrap_or_default())) }
/// Decode + resize an image with ffmpeg (bicubic) to the Qwen2-VL smart_resize size; returns RGB8 and (h, w).
fn load_image(path: &str, max_px: usize) -> Result<(Vec<u8>, usize, usize)> {
    let probe = std::process::Command::new(ffmpeg()).args(["-hide_banner", "-i", path]).output()?; let s = String::from_utf8_lossy(&probe.stderr);
    let dims = s.split(|c: char| c == ',' || c == ' ').filter_map(|t| { let (a, b) = t.split_once('x')?; Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)) }).next().ok_or(anyhow::anyhow!("no image size in ffmpeg output"))?;
    let (h, w) = cosmos3::vit::smart_resize(dims.1, dims.0, 65536, max_px);
    let o = std::process::Command::new(ffmpeg()).args(["-loglevel", "error", "-i", path, "-vf", &format!("scale={w}:{h}:flags=bicubic"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).output()?;
    anyhow::ensure!(o.stdout.len() == h * w * 3, "ffmpeg gave {} bytes for {w}x{h}", o.stdout.len()); Ok((o.stdout, h, w))
}
/// cos ask (--image FILE | --ref DIR) [--question Q] [--max-px N] [--new N]: Qwen3-VL understanding path
/// (ViT + DeepStack into the und tower, greedy decode; each token re-runs the und prefix). --ref = vit_ref.py dump: parity.
fn ask(dir: &str, a: &[String]) -> Result<()> {
    let mut o = std::collections::HashMap::new(); let mut i = 0; while i + 1 < a.len() { o.insert(a[i].trim_start_matches('-').to_string(), a[i + 1].clone()); i += 2; }
    let gs = |k: &str, d: &str| o.get(k).cloned().unwrap_or(d.to_string());
    let rf = gs("ref", ""); let question = gs("question", "Describe this image in one sentence."); let new: usize = gs("new", "32").parse()?;
    let tk = prompt::Tok::load(&format!("{dir}/text_tokenizer/tokenizer.json"))?; let dt = prompt::Detok::load(&format!("{dir}/text_tokenizer/tokenizer.json"))?;
    let img_tok = tk.id("<|image_pad|>").ok_or(anyhow::anyhow!("no image_pad"))?; let eos = tk.id("<|im_end|>").unwrap(); let eot = tk.id("<|endoftext|>").unwrap_or(eos);
    let (patches, gh, gw) = if !rf.is_empty() { let (p, _) = read_npy(&format!("{rf}/pixel_values.npy"))?; let (g, _) = read_npy(&format!("{rf}/grid_thw.npy"))?; (p, g[1] as usize, g[2] as usize) }
        else { let (rgb, h, w) = load_image(&gs("image", ""), gs("max-px", "200704").parse()?)?; cosmos3::vit::patchify_image(&rgb, h, w) };
    let nv = gh * gw / 4;
    let (mut ids, r0) = prompt::und_ids(&tk, &question, nv)?;
    if !rf.is_empty() { let (ri, _) = read_npy(&format!("{rf}/input_ids.npy"))?; let ri: Vec<u32> = ri.iter().map(|x| *x as u32).collect();
        eprintln!("input ids: ours {} ref {} match {}", ids.len(), ri.len(), ids == ri); if ids != ri { ids = ri.clone(); } }
    let r0 = ids.iter().position(|&t| t == img_tok).unwrap_or(r0);
    let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?; let mut kn = qimage::Kern::new(&ctx)?; vae::add_kernels(&ctx, &mut kn)?;
    let t0 = std::time::Instant::now(); let vit = cosmos3::vit::Vit::load(&ctx, &format!("{dir}/vision_encoder"))?; eprintln!("vit loaded {:.2}s, grid {gh}x{gw} -> {nv} tokens", t0.elapsed().as_secs_f64());
    let e0 = now(); let mut vo = vit.encode(&ctx, &patches, gh, gw)?; let mut best = vo.gpu_s; for _ in 0..2 { let v2 = vit.encode(&ctx, &patches, gh, gw)?; best = best.min(v2.gpu_s); vo = v2; }
    eprintln!("PHASE vit {e0:.3} {:.3}", now()); eprintln!("vit gpu {:.1} ms (best of 3)", best * 1e3);
    if !rf.is_empty() { let (r, _) = read_npy(&format!("{rf}/vit_out.npy"))?; eprintln!("vit merger   {}", cmp(&vo.emb, &r));
        for k in 0..3 { let (r, _) = read_npy(&format!("{rf}/vit_ds{k}.npy"))?; eprintln!("vit deepstack{k} {}", cmp(&vo.ds[k], &r)); } }
    let m = Model::load(&ctx, &format!("{dir}/transformer"), true)?; eprintln!("transformer loaded {:.2}s", m.st.load_s);
    let mut pos = prompt::und_positions(&ids, img_tok, gh, gw);
    if !rf.is_empty() { let (rp, _) = read_npy(&format!("{rf}/position_ids.npy"))?; let l = ids.len(); let mut bad = 0;
        for j in 0..l { for ax in 0..3 { if rp[ax * l + j] as f64 != pos[j][ax] { bad += 1 } } } eprintln!("positions: {bad} mismatches of {}", 3 * l); }
    let mut s = m.state(&ctx, (ids.len() + new + 1) as u32, 1)?; s.ds = Some((r0, vo.ds.to_vec()));
    let mut out: Vec<u32> = vec![]; let e1 = now(); let mut gpu = 0.0; let mut first = None;
    let kv = std::env::var("COS_ASK_REPREFILL").is_err(); let mut ttft = 0.0;
    for step in 0..new {
        let lg = if step == 0 || !kv {
            let mut e = m.embed(&ids)?; e[r0 * 4096..(r0 + nv) * 4096].copy_from_slice(&vo.emb);
            s.lt = ids.len() as u32; s.dec = 0; s.ds = Some((r0, vo.ds.to_vec())); Model::set_rope_u(&s, &pos);
            let (g, lg) = m.prefix_emb(&ctx, &pf, &kn, &mut s, &e, true)?; gpu += g; if step == 0 { ttft = now() - e1; } lg.unwrap()
        } else {   // KV cache: only the newest token goes through the und tower
            s.ds = None; s.dec = (ids.len() - 1) as u32; let e = m.embed(&ids[ids.len() - 1..])?;
            let (g, lg) = m.decode_step(&ctx, &pf, &kn, &mut s, &e, *pos.last().unwrap())?; gpu += g; lg
        };
        if step == 0 { first = Some(lg.clone()); }
        let nt = lg.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
        if nt == eos || nt == eot { break } out.push(nt); ids.push(nt); let p = pos.last().unwrap()[0] + 1.0; pos.push([p; 3]);
    }
    eprintln!("PHASE decode {e1:.3} {:.3}", now()); eprintln!("decode: {} tokens, gpu {:.2}s, wall {:.2}s (prefill + first token {:.2}s, then {:.1} ms/token, kv cache {kv})", out.len(), gpu, now() - e1, ttft, (now() - e1 - ttft) * 1e3 / (out.len().max(2) - 1) as f64);
    if !rf.is_empty() { let (rl, _) = read_npy(&format!("{rf}/logits0.npy"))?; let f = first.unwrap(); let mut ti: Vec<usize> = (0..rl.len()).collect(); ti.sort_by(|a, b| rl[*b].partial_cmp(&rl[*a]).unwrap());
        let mut oi: Vec<usize> = (0..f.len().min(rl.len())).collect(); oi.sort_by(|a, b| f[*b].partial_cmp(&f[*a]).unwrap());
        eprintln!("logits0 {} top5 ours {:?} ref {:?}", cmp(&f[..rl.len()], &rl), &oi[..5], &ti[..5]);
        let (ra, _) = read_npy(&format!("{rf}/answer_ids.npy"))?; let ra: Vec<u32> = ra.iter().map(|x| *x as u32).collect();
        let same = out.iter().zip(&ra).take_while(|(a, b)| a == b).count(); eprintln!("greedy: first {same} of {} ref tokens identical", ra.len());
        eprintln!("ref answer:  {}", dt.decode(&ra)); }
    println!("answer: {}", dt.decode(&out));
    Ok(())
}
