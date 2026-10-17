//! glm_run <model.gguf> <ids csv | @file> [n_new] [ref ids csv | @file]: greedy decode, tok/s, parity vs reference ids.
//! <ids> may also be `text:<prompt>` or `textfile:<path>` (tokenized by glimmer::tok, BOS (NO_BOS=1 drops it; the text refs in ref_llama.json were made without BOS) + parse_special, raw completion like the llama.cpp refs; no chat template).
//! A `<__media__>` marker in the text is replaced by the image (env IMAGE=<png|jpg>, MMPROJ=<gguf>, default: mmproj-*.gguf beside the model): ViT embeddings are injected into prefill.
//! ref arg may be `json:<file>#<idx>` ("gen" array of a ref_llama*.json). FORCED=1 with ref: teacher-forced agreement + margins. Env RUNS, MAX_CTX, DBG_LAYERS, DUMPX.
use anyhow::Result;
fn ids(s: &str) -> Result<Vec<u32>> {
    let t = if let Some(f) = s.strip_prefix('@') { std::fs::read_to_string(f)? } else { s.to_string() };
    Ok(t.trim().trim_matches(|c| c == '[' || c == ']').split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()) }
fn refs(s: &str) -> Result<Vec<u32>> {
    if let Some(r) = s.strip_prefix("json:") { let (f, i) = r.rsplit_once("#").unwrap_or((r, "0")); let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(f)?)?;
        return Ok(v[i.parse::<usize>()?]["gen"].as_array().ok_or_else(|| anyhow::anyhow!("no gen"))?.iter().map(|x| x.as_u64().unwrap() as u32).collect()); }
    ids(s)
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mut text_ids: Option<Vec<u32>> = None; let mut img: Option<(Vec<f32>, usize)> = None; let mut tk = None;
    if a[2].starts_with("text:") || a[2].starts_with("textfile:") {
        let text = if let Some(f) = a[2].strip_prefix("textfile:") { std::fs::read_to_string(f)? } else { a[2][5..].to_string() };
        let g = gguf::Gguf::open(&a[1])?; let t = glimmer::tok::Tokenizer::from_gguf(&g)?;
        let mut n_img = 0;
        if text.contains(glimmer::tok::MEDIA) {
            let ip = std::env::var("IMAGE").map_err(|_| anyhow::anyhow!("prompt has <__media__> but IMAGE is not set"))?;
            let mp = std::env::var("MMPROJ").unwrap_or_else(|_| { let d = std::path::Path::new(&a[1]).parent().unwrap(); std::fs::read_dir(d).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.file_name().unwrap().to_string_lossy().starts_with("mmproj") && p.extension().map_or(false, |e| e == "gguf")).unwrap().to_string_lossy().into_owned() });
            let (rgb, w, h) = glimmer::vision::load_image(&ip)?;
            let v = glimmer::vision::Vision::load(&mp)?; let o = v.encode(&rgb, w, h)?; n_img = o.n_out;
            println!("image {ip}: {w}x{h} -> {} tokens (ViT GPU {:.3}s, CPU prep {:.3}s)", o.n_out, o.gpu_s, o.cpu_s);
            img = Some((o.emb, 0)); drop(v);
        }
        let (mut ids, mut pos) = t.build_prompt(&text, n_img); if std::env::var("NO_BOS").is_ok() && t.bos == ids.first().copied() { ids.remove(0); pos = pos.map(|p| p - 1); } if let Some(i) = img.as_mut() { i.1 = pos.unwrap_or(0); }
        println!("prompt: {} tokens", ids.len()); text_ids = Some(ids); tk = Some(t);
    }
    let chunk: u32 = std::env::var("PF_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let m = glimmer::Model::load(&a[1], std::env::var("MAX_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(4096), chunk)?;
    let mut pf = if chunk > 0 { Some(glimmer::Prefill::new(&m, chunk)?) } else { None };
    if let (Some((e, pos)), Some(pf)) = (&img, pf.as_mut()) {
        let bytes = e.len() * 4; let buf = m.ctx.buffer(bytes as u64)?; unsafe { std::ptr::copy_nonoverlapping(e.as_ptr() as *const u8, buf.ptr as *mut u8, bytes); }
        pf.img = Some((buf, *pos as u32, (e.len() / m.cfg.dim as usize) as u32));
    }
    // IMG_EMB=file.f32 IMG_POS=p: [cnt][dim] f32 image embeddings replace the (placeholder) token rows p..p+cnt
    if let (Ok(f), Some(pf)) = (std::env::var("IMG_EMB"), pf.as_mut()) {
        let raw = std::fs::read(&f)?; let cnt = (raw.len() / 4 / m.cfg.dim as usize) as u32;
        let buf = m.ctx.buffer(raw.len() as u64)?; unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), buf.ptr as *mut u8, raw.len()); }
        pf.img = Some((buf, std::env::var("IMG_POS").ok().and_then(|v| v.parse().ok()).unwrap_or(0), cnt));
        println!("image: {cnt} embedding rows at position {}", pf.img.as_ref().unwrap().1);
    }
    let c = &m.cfg;
    println!("glimmer: {} layers dim {} heads {}/{} hd {} ff {} vocab {} win {} swa layers {} softcap {} logit_scale {} eps {}", c.n_layer, c.dim, c.n_head, c.n_kv, c.hd, c.ff, c.vocab, c.win,
        c.layers.iter().filter(|l| l.swa).count(), c.softcap, c.logit_scale, c.eps);
    let p = match text_ids { Some(t) => t, None => ids(&a[2])? }; let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let r = a.get(4).map(|s| refs(s)).transpose()?;
    if std::env::var("DUMPX").is_ok() {
        m.generate(pf.as_ref(), &p, 1)?;
        let x = m.x.as_f32(); let d = m.cfg.dim as usize;
        let ss: f64 = x[..d].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        println!("DUMPX first {:?} last {:?} sum {:.4} rms {:.4}", &x[..4], &x[d - 4..d], x[..d].iter().map(|v| *v as f64).sum::<f64>(), (ss / d as f64).sqrt());
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
        if run == 0 { println!("ids: {:?}", out); if let Some(t) = &tk { println!("text: {:?}", t.decode(&out)); } }
        println!("prompt {} tok in {:.3}s ({:.1} tok/s) | decode {} tok GPU {:.3}s = {:.2} tok/s | wall {:.3}s = {:.2} tok/s", p.len(), tp, p.len() as f64 / tp, n - 1, dg, (n - 1) as f64 / dg, dw, (n - 1) as f64 / dw);
        if let Some(r) = &r { let k = out.iter().zip(r).take_while(|(x, y)| x == y).count(); println!("parity: first {k}/{} match reference", out.len().min(r.len())); }
    }
    Ok(())
}
