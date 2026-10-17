//! yue2 parity --ref DIR            per-stage parity vs the fixed-seed torch reference dumps (yue_ref/ref.py)
//! yue2 gen --ref DIR [--seed N] [--maxsem N] [--out WAV] [--teacher 1]
//!     full generation from the reference prompt ids: ABC plan + semantic tokens (GPU logits, host sampling), NAR
//!     32-step midpoint flow matching, VAE decode. --teacher 1 reuses the reference ABC/semantic tokens and noise.
use anyhow::{bail, Result};
use std::time::Instant;
use yue2::*;

fn cmp(name: &str, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len()); let (mut dot, mut na, mut nb, mut d2, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f32);
    for i in 0..n { let (x, y) = (a[i] as f64, b[i] as f64); dot += x * y; na += x * x; nb += y * y; d2 += (x - y).powi(2); mx = mx.max((a[i] - b[i]).abs()) }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    eprintln!("  {name:<28} n={n:<9} cos {cos:.6}  rel {:.3e}  max|d| {mx:.4}", (d2 / nb.max(1e-30)).sqrt()); cos
}
fn read_wav(path: &str) -> Result<(Vec<f32>, u16)> {
    let b = std::fs::read(path)?; let mut i = 12; let (mut ch, mut fmt, mut bits) = (2u16, 3u16, 32u16);
    while i + 8 <= b.len() { let id = &b[i..i + 4]; let n = u32::from_le_bytes(b[i + 4..i + 8].try_into()?) as usize;
        if id == b"fmt " { fmt = u16::from_le_bytes([b[i + 8], b[i + 9]]); ch = u16::from_le_bytes([b[i + 10], b[i + 11]]); bits = u16::from_le_bytes([b[i + 22], b[i + 23]]); }
        if id == b"data" { let d = &b[i + 8..(i + 8 + n).min(b.len())];
            let v = if bits == 32 && (fmt == 3 || fmt == 0xfffe) { d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
                else if bits == 16 { d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect() } else { bail!("wav fmt {fmt}/{bits}") };
            return Ok((v, ch)) }
        i += 8 + n + (n & 1); }
    bail!("{path}: no data")
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect(); if args.is_empty() { bail!("usage: yue2 parity|gen --ref DIR ...") }
    let mode = args.remove(0);
    let (mut rdir, mut seed, mut maxsem, mut out, mut teacher, mut steps) = ("".to_string(), 1234u64, 750usize, "yue2_out.wav".to_string(), false, 32usize);
    while args.len() >= 2 { let a = args.remove(0); let v = args.remove(0);
        match a.as_str() { "--ref" => rdir = v, "--seed" => seed = v.parse()?, "--maxsem" => maxsem = v.parse()?, "--out" => out = v, "--teacher" => teacher = v == "1", "--steps" => steps = v.parse()?, _ => bail!("unknown {a}") } }
    let home = std::env::var("HOME")?;
    let snap = |repo: &str| -> Result<String> { let d = format!("{home}/.cache/huggingface/hub/models--m-a-p--{repo}/snapshots"); Ok(std::fs::read_dir(&d)?.next().unwrap()?.path().display().to_string()) };
    let mdir = std::env::var("YE_MODEL").or_else(|_| snap("YuE2-3B"))?; let vdir = std::env::var("YE_VAE").or_else(|_| snap("YuE2-Vae"))?;
    let t0 = Instant::now();
    let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?; let kn = qimage::Kern::new(&ctx)?; let yk = YK::new(&ctx)?;
    let e = Env { ctx: &ctx, pf: &pf, kn: &kn, yk: &yk };
    let m = Model::load(&ctx, &mdir)?; let vae = vae::Vae::load(&ctx, &vdir)?;
    eprintln!("loaded in {:.2}s (zero copy {})", t0.elapsed().as_secs_f64(), m.st.zero_copy);
    let prefix = npy::read_u32(&format!("{rdir}/prefix.npy"))?; let abc_ref = npy::read_u32(&format!("{rdir}/abc.npy")).unwrap_or_default();
    let sem_ref = npy::read_u32(&format!("{rdir}/semantic.npy"))?;
    let rmax = std::env::var("YE_RMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048u32);
    let s = State::new(&ctx, rmax, (prefix.len() + abc_ref.len() + 4096 + maxsem.max(sem_ref.len()) + 64) as u32)?;
    // the reference prefix ends with ABC_START for cot=full; the semantic prefix is prefix + abc + [ABC_END, MUSIC_START]
    // prefix.npy is the reference *semantic* prefix (instruction + ABC_START + abc + ABC_END + MUSIC_START); recover the planning prefix
    let semref = prefix.clone(); let prefix: Vec<u32> = if abc_ref.is_empty() { semref[..semref.len() - 1].to_vec() } else { semref[..semref.len() - abc_ref.len() - 2].to_vec() };
    assert!(abc_ref.is_empty() || prefix.last() == Some(&ABC_START));
    let sem_prefix = |abc: &[u32]| -> Vec<u32> { let mut p = prefix.clone(); p.extend_from_slice(abc); p.push(ABC_END); p.push(MUSIC_START); p };
    if mode == "dbg" {
        let nd: usize = std::env::var("YE_DN").ok().and_then(|v| v.parse().ok()).unwrap_or(64); let rep: Vec<u32> = prefix.iter().cycle().take(nd).copied().collect(); m.ar_forward(&e, &s, &rep, 0, None)?;
        let x = s.x.as_f32()[..nd * 2048].to_vec();
        if std::env::var("YE_STOP").as_deref() == Ok("4") { let a = unsafe { std::slice::from_raw_parts(s.a.ptr as *const half::f16, 64 * 2048) }; let a: Vec<f32> = a.iter().map(|v| v.to_f32()).collect(); cmp("n0", &a, &npy::read(&format!("{rdir}/dbg_n0.npy"))?.0); return Ok(()) }
        if std::env::var("YE_STOP").as_deref() == Ok("3") { let y = s.y.as_f32(); let q: Vec<f32> = (0..64).flat_map(|t| y[t * 4096..t * 4096 + 2048].to_vec()).collect(); cmp("q", &q, &npy::read(&format!("{rdir}/dbg_q.npy"))?.0);
            let a = unsafe { std::slice::from_raw_parts(s.a.ptr as *const half::f16, 64 * 2048) }; let wb = m.st.bytes("model.layers.0.self_attn.q_proj.weight")?;
            let cpu: Vec<f32> = (0..2048).map(|n| (0..2048).map(|k| a[k].to_f32() * half::bf16::from_le_bytes([wb[2 * (n * 2048 + k)], wb[2 * (n * 2048 + k) + 1]]).to_f32()).sum()).collect();
            cmp("q row0 gpu vs cpu", &q[..2048], &cpu); cmp("q row0 ref vs cpu", &npy::read(&format!("{rdir}/dbg_q.npy"))?.0[..2048], &cpu);
            let w = m.st.wref("model.layers.0.self_attn.q_proj.weight")?; eprintln!("wref arena {} off {} n {} k {} bytes {} nbufs {}", w.arena, w.off, w.n, w.k, w.bytes, m.st.bufs.len());
            return Ok(()) }
        if let Ok(nl) = std::env::var("YE_NL") { let i: usize = nl.parse::<usize>()? - 1; if let Ok((r, _)) = npy::read(&format!("{rdir}/dbg_L{i}.npy")) { cmp(&format!("L{i}"), &x, &r); return Ok(()) } }
        for nm in ["dbg_emb", "dbg_a0", "dbg_l0f", "dbg_l0", "dbg_l1"] { if let Ok((r, _)) = npy::read(&format!("{rdir}/{nm}.npy")) { cmp(nm, &x, &r); } }
        return Ok(())
    }
    if mode == "parity" {
        eprintln!("== stage 1: AR logits (teacher forced, {} prefix + {} codec)", prefix.len(), sem_ref.len());
        let sp = sem_prefix(&abc_ref);
        let mut full = sp.clone(); full.extend(sem_ref.iter().map(|t| t + CODEC_OFFSET));
        let pos = npy::read_u32(&format!("{rdir}/ar_logit_pos.npy"))?; let (rl, rs) = npy::read(&format!("{rdir}/ar_logits.npy"))?;
        let ours = m.ar_logits_rows(&e, &s, &full, &pos)?; let v = rs[1];
        cmp("ar logits (all rows)", &ours, &rl);
        let (mut agree, mut cmin) = (0, 1f64);
        for i in 0..pos.len() { let (a, b) = (&ours[i * v..(i + 1) * v], &rl[i * v..(i + 1) * v]);
            let am = |x: &[f32]| x[MUSIC_END as usize..(MUSIC_END + CODEC_SIZE + 1) as usize].iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
            if am(a) == am(b) { agree += 1 }
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64); for j in 0..v { d += a[j] as f64 * b[j] as f64; na += (a[j] as f64).powi(2); nb += (b[j] as f64).powi(2) } cmin = cmin.min(d / (na * nb).sqrt()); }
        eprintln!("  codec argmax agreement {agree}/{} rows, min row cos {cmin:.6}", pos.len());
        eprintln!("== stage 2: NAR velocity at t=1 (reference noise)");
        let mut chunk = sp.clone(); chunk.extend(sem_ref.iter().map(|t| t + CODEC_OFFSET)); chunk.push(MUSIC_END);
        let (noise, _) = npy::read(&format!("{rdir}/noise.npy"))?;
        let (nar, gp) = m.nar_prefill(&e, &s, &chunk, sem_ref.len() as u32)?;
        let (v0, gv) = m.velocity(&e, &s, &nar, &noise, 1.0)?; let (rv0, _) = npy::read(&format!("{rdir}/v0.npy"))?;
        cmp("v0", &v0, &rv0); eprintln!("  prefill gpu {:.1} ms, one velocity gpu {:.1} ms", gp * 1e3, gv * 1e3);
        eprintln!("== stage 3: {steps}-step midpoint solve (reference noise)");
        let t = Instant::now(); let (lat, gs) = m.solve(&e, &s, &nar, &noise, steps)?; let ws = t.elapsed().as_secs_f64();
        let (rlat, _) = npy::read(&format!("{rdir}/latent.npy"))?; cmp("latents", &lat, &rlat); eprintln!("  solve wall {ws:.2}s gpu {gs:.2}s");
        npy::write(&format!("{rdir}/ours_latent.npy"), &lat, &[lat.len() / 64, 64])?;
        eprintln!("== stage 4: VAE decode");
        let (wref, _) = read_wav(&format!("{rdir}/ref.wav"))?;
        let t = Instant::now(); let (au, gvae) = vae.decode(&e, &rlat)?; let wv = t.elapsed().as_secs_f64();
        cmp("audio (ref latents)", &au.iter().map(|x| x.clamp(-1.0, 1.0)).collect::<Vec<_>>(), &wref); eprintln!("  vae wall {wv:.2}s gpu {gvae:.2}s  ({} samples ref {})", au.len() / 2, wref.len() / 2);
        npy::write_wav(&format!("{rdir}/ours_vae.wav"), &au.iter().map(|x| x.clamp(-1.0, 1.0)).collect::<Vec<_>>(), 2, 48000)?;
        let (au2, _) = vae.decode(&e, &lat)?; let au2: Vec<f32> = au2.iter().map(|x| x.clamp(-1.0, 1.0)).collect();
        cmp("audio (our latents)", &au2, &wref); npy::write_wav(&format!("{rdir}/ours_tf.wav"), &au2, 2, 48000)?;
        return Ok(())
    }
    if mode == "gen" {
        let mut rng = Rng(seed ^ 0x5eed_0000);
        let t = Instant::now();
        let (abc, abc_eos, g_abc, w_abc) = if teacher { (abc_ref.clone(), true, 0.0, 0.0) } else if prefix.last() == Some(&ABC_START) {
            m.generate(&e, &s, &prefix, false, &Sampling::abc(), &mut rng, |_| {})? } else { (vec![], true, 0.0, 0.0) };
        let sp = sem_prefix(&abc);
        let mut ss = Sampling::semantic(); ss.max_tokens = maxsem; ss.min_tokens = ss.min_tokens.min(maxsem);
        let (sem, sem_eos, g_sem, w_sem) = if teacher { (sem_ref.iter().map(|t| t + CODEC_OFFSET).collect(), true, 0.0, 0.0) } else { m.generate(&e, &s, &sp, true, &ss, &mut rng, |_| {})? };
        let codec: Vec<u32> = sem.iter().map(|t| t - CODEC_OFFSET).collect();
        let tn = Instant::now(); let mut lat = vec![]; let mut g_nar = 0.0;
        let noise_all: Vec<f32> = if teacher { npy::read(&format!("{rdir}/noise.npy"))?.0 } else { (0..codec.len() * 64).map(|_| rng.normal()).collect() };
        for (a, b) in chunk_ranges(codec.len(), sp.len()) {
            let mut ch = sp.clone(); ch.extend(codec[a..b].iter().map(|t| t + CODEC_OFFSET)); ch.push(MUSIC_END);
            let (nar, gp) = m.nar_prefill(&e, &s, &ch, (b - a) as u32)?; let (l, g) = m.solve(&e, &s, &nar, &noise_all[a * 64..b * 64], steps)?; lat.extend(l); g_nar += gp + g; }
        let w_nar = tn.elapsed().as_secs_f64();
        let tv = Instant::now(); let (au, g_vae) = vae.decode(&e, &lat)?; let w_vae = tv.elapsed().as_secs_f64();
        let au: Vec<f32> = au.iter().map(|x| x.clamp(-1.0, 1.0)).collect(); let secs = au.len() as f64 / 2.0 / 48000.0;
        let wall = t.elapsed().as_secs_f64();
        npy::write_wav(&out, &au, 2, 48000)?; npy::write(&format!("{out}.latent.npy"), &lat, &[lat.len() / 64, 64])?;
        std::fs::write(format!("{out}.tokens.json"), serde_json::json!({"abc": abc, "semantic": codec, "abc_eos": abc_eos, "sem_eos": sem_eos}).to_string())?;
        eprintln!("abc {} tok ({:.2}s, gpu {:.2}s, {:.1} tok/s) | semantic {} tok ({:.2}s, gpu {:.2}s, {:.1} tok/s) | nar {:.2}s (gpu {:.2}) | vae {:.2}s (gpu {:.2})",
            abc.len(), w_abc, g_abc, abc.len() as f64 / w_abc.max(1e-9), codec.len(), w_sem, g_sem, codec.len() as f64 / w_sem.max(1e-9), w_nar, g_nar, w_vae, g_vae);
        println!("RESULT audio_s {secs:.2} wall_s {wall:.2} audio_per_s {:.3} abc_s {w_abc:.2} sem_s {w_sem:.2} nar_s {w_nar:.2} vae_s {w_vae:.2} out {out}", secs / wall);
        return Ok(())
    }
    bail!("mode {mode}")
}
