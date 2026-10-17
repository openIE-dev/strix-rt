//! g4_run <model.gguf> <ids csv | @file> [n_new] [ref ids csv | @file]: greedy decode, tok/s, parity vs reference ids.
//! FORCED=1 with ref: teacher-forced agreement + margins. Env RUNS, MAX_CTX.
//! IMG=<mmproj.gguf>,<image.ppm>[,max_tok]: the prompt id 999999 is replaced by the image's embeddings (needs G4_PF=1).
use anyhow::Result;
fn ids(s: &str) -> Result<Vec<u32>> {
    if let Some(n) = s.strip_prefix("rand:") { let n: usize = n.parse()?; let mut x = 0x9e3779b9u64; return Ok(std::iter::once(2).chain((1..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; 1000 + (x % 200000) as u32 })).collect()) } let t = if let Some(p) = s.strip_prefix('@') { std::fs::read_to_string(p)? } else { s.to_string() };
    Ok(t.trim().trim_matches(|c| c == '[' || c == ']').split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()) }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m = gemma4::Model::load(&a[1], std::env::var("MAX_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048))?;
    let mut pf = if std::env::var("G4_PF").map(|v| v != "0").unwrap_or(false) { Some(gemma4::Prefill::new(&m, std::env::var("G4_PF_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512))?) } else { None };
    let mut p = ids(&a[2])?; let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let r = a.get(4).map(|s| ids(s)).transpose()?;
    if let Ok(spec) = std::env::var("IMG") {
        let f: Vec<&str> = spec.split(',').collect();
        let v = gemma4::vision::Vision::load(&m.ctx, f[0])?;
        let (rgb, w, h) = gemma4::vision::read_ppm(f[1])?;
        let (emb, pc, pr) = v.encode(&rgb, w, h, f.get(2).and_then(|x| x.parse().ok()).unwrap_or(1120))?;
        let at = p.iter().position(|&t| t == 999999).ok_or_else(|| anyhow::anyhow!("no 999999 image marker in prompt"))?;
        let cnt = pc * pr;
        p.splice(at..at + 1, std::iter::repeat(0).take(cnt));
        let pfm = pf.as_mut().ok_or_else(|| anyhow::anyhow!("IMG needs G4_PF=1"))?;
        pfm.img = Some((gemma4::pf::image_buf(&m, &emb)?, at as u32, cnt as u32));
        println!("image {w}x{h} -> {pc}x{pr} = {cnt} tokens at {at}, prompt {} tok", p.len());
    }
    if let Ok(spec) = std::env::var("AUD") {
        // AUD=<mmproj.gguf>,<16 kHz wav>: the prompt id 999998 is replaced by the audio embeddings (needs G4_PF=1)
        let f: Vec<&str> = spec.split(',').collect();
        let t0 = std::time::Instant::now();
        let au = gemma4::audio::Audio::load(f[0])?;
        let wav = gemma4::audio::read_wav(f[1])?;
        let (emb, cnt) = au.encode(&wav)?;
        let at = p.iter().position(|&t| t == 999998).ok_or_else(|| anyhow::anyhow!("no 999998 audio marker in prompt"))?;
        p.splice(at..at + 1, std::iter::repeat(0).take(cnt));
        let s: f64 = emb.iter().map(|v| *v as f64).sum(); let ss: f64 = emb.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        println!("audio {} samples -> {cnt} tokens at {at}, prompt {} tok, emb sum {s:.4} rms {:.4}, CPU encode {:.2}s", wav.len(), p.len(), (ss / emb.len() as f64).sqrt(), t0.elapsed().as_secs_f32());
        let pfm = pf.as_mut().ok_or_else(|| anyhow::anyhow!("AUD needs G4_PF=1"))?;
        pfm.img = Some((gemma4::pf::image_buf(&m, &emb)?, at as u32, cnt as u32));
    }
    if std::env::var("DUMPX").is_ok() {
        // residual after DBG_LAYERS layers at the last prompt position (compare with llama-eval-callback l_out-<L-1>)
        m.generate(pf.as_ref(), &p, 1)?;
        let x = m.x.as_f32(); let d = m.cfg.dim as usize;
        let ss: f64 = x[..d].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        println!("DUMPX first {:?} last {:?} sum {:.4} rms {:.4}", &x[..4], &x[d - 4..d], x[..d].iter().map(|v| *v as f64).sum::<f64>(), (ss / d as f64).sqrt());
        if std::env::var("DBG_PART").is_ok() {
            let m1 = m.m1.as_f32(); let au = unsafe { std::slice::from_raw_parts(m.aux.ptr as *const u32, 1024) }; let af = m.aux.as_f32();
            println!("M1 first {:?} last {:?} sum {:.4}", &m1[..3], &m1[d - 3..d], m1[..d].iter().map(|v| *v as f64).sum::<f64>());
            println!("ROUTE ids {:?} w {:?}", &au[440..448], &af[448..456]);
        }
        return Ok(());
    }
    if let (Ok(_), Some(r)) = (std::env::var("FORCED"), &r) {
        let o = m.forced(pf.as_ref(), &p, r)?;
        let bad: Vec<String> = o.iter().enumerate().filter(|(j, x)| x.0 != r[*j]).map(|(j, x)| format!("{j}(ours {} ref {} gap {:.4} margin {:.4})", x.0, r[j], x.2, x.1)).collect();
        let minm = o.iter().map(|x| x.1).fold(f32::MAX, f32::min);
        println!("FORCED agree {}/{} min top2 margin {:.4} mismatch [{}]", o.len() - bad.len(), o.len(), minm, bad.join(", "));
        return Ok(());
    }
    for run in 0..std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1) {
        let (out, tp, dg, dw) = m.generate(pf.as_ref(), &p, n)?;
        if run == 0 { println!("ids: {:?}", out); }
        println!("prompt {} tok in {:.3}s ({:.1} tok/s) | decode {} tok GPU {:.3}s = {:.2} tok/s | wall {:.3}s = {:.2} tok/s", p.len(), tp, p.len() as f64 / tp, n - 1, dg, (n - 1) as f64 / dg, dw, (n - 1) as f64 / dw);
        if let Some(r) = &r { let k = out.iter().zip(r).take_while(|(x, y)| x == y).count(); println!("parity: first {k}/{} match reference", out.len().min(r.len())); }
    }
    Ok(())
}
