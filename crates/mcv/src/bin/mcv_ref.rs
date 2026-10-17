//! Parity vs the llama.cpp reference dump (crates/mcv/ref/mcvdump.cpp):
//!   mcv_ref <model.gguf> <refdir> [n_gen]
//! Feeds the reference prompt (text tokens from chunks.txt; image chunks from img_embd_*.f32 + img_pos_*.txt),
//! decodes greedily and compares token ids with refdir/gen.txt.
use anyhow::Result;
use mcv::llm::{Item, Llm};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (model, dir) = (&a[1], &a[2]);
    let n_gen: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let chunks = std::fs::read_to_string(format!("{dir}/chunks.txt"))?;
    let gen = std::fs::read_to_string(format!("{dir}/gen.txt")).unwrap_or_default();
    let ref_ids: Vec<u32> = gen.lines().find(|l| l.starts_with("IDS")).map(|l| l.split_whitespace().skip(1).filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
    let t0 = std::time::Instant::now();
    let m = Llm::load(model, std::env::var("MAX_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192))?;
    eprintln!("load {:.1}s", t0.elapsed().as_secs_f64());
    // build items + mrope positions
    let mut embs: Vec<Vec<f32>> = vec![];
    let mut plan: Vec<(bool, u32, (u32, u32, u32))> = vec![]; // (is_emb, tok or emb idx, pos)
    let mut pos = 0u32;
    let dim = m.cfg.dim as usize;
    // OWN_VIS=<image.ppm> (VIS_DIR = vision safetensors dir): use our vision tower instead of the dumped embeddings
    let own: Option<Vec<Vec<f32>>> = match std::env::var("OWN_VIS") { Ok(ppm) => {
        let vd = std::env::var("VIS_DIR").unwrap_or_else(|_| format!("{}/models/minicpmv47/vision_st", std::env::var("HOME").unwrap()));
        let v = mcv::vision::Vision::load(&m.ctx, &vd)?;
        let (rgb, w, h) = mcv::vision::read_ppm(&ppm)?;
        let (tiles, _) = mcv::vision::preprocess(&rgb, w, h);
        let mut o = vec![]; let mut tt = 0.0; for t in &tiles { let (e, dt) = v.encode(&m.ctx, t)?; o.push(e); tt += dt; }
        let mut tt2 = 0.0; for t in &tiles { tt2 += v.encode(&m.ctx, t)?.1; }
        eprintln!("own vision: {} tiles, {} rows, encode {:.1} ms (warm {:.1} ms)", tiles.len(), o.iter().map(|e| e.len()).sum::<usize>() / dim, tt * 1e3, tt2 * 1e3);
        Some(o) } Err(_) => None };
    let mut ti = 0usize;
    for (ci, line) in chunks.lines().enumerate() {
        if let Some(r) = line.strip_prefix("TEXT ") {
            for t in r.split(':').nth(1).unwrap_or("").split_whitespace() { plan.push((false, t.parse()?, (pos, pos, pos))); pos += 1; }
        } else if let Some(r) = line.strip_prefix("IMAGE ") {
            let nt: usize = r.split_whitespace().next().unwrap().trim_start_matches("n_tokens=").parse()?;
            let npos: u32 = r.split_whitespace().nth(1).unwrap().trim_start_matches("n_pos=").parse()?;
            let raw = std::fs::read(format!("{dir}/img_embd_{ci}.f32"))?;
            let rf: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let f: Vec<f32> = match &own { Some(o) => { let all: Vec<f32> = if o.get(ti).map(|e| e.len()) != Some(nt * dim) { let a: Vec<f32> = o[ti.min(o.len())..].concat(); ti = o.len(); a } else { ti += 1; o[ti - 1].clone() }; let o = &all;
                    if o.len() != nt * dim { anyhow::bail!("own vision rows {} != ref {}", o.len() / dim, nt) }
                    let cs: Vec<f64> = (0..nt).map(|j| { let (a, b) = (&o[j * dim..(j + 1) * dim], &rf[j * dim..(j + 1) * dim]);
                        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64); for k in 0..dim { ab += (a[k] * b[k]) as f64; aa += (a[k] * a[k]) as f64; bb += (b[k] * b[k]) as f64; } ab / (aa * bb).sqrt() }).collect();
                    eprintln!("  chunk {ci}: {nt} rows, emb cos mean {:.5} min {:.5}", cs.iter().sum::<f64>() / nt as f64, cs.iter().cloned().fold(1.0, f64::min));
                    { let mut w: Vec<(usize, f64)> = cs.iter().cloned().enumerate().collect(); w.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap()); eprintln!("  worst rows {:?}", &w[..w.len().min(8)]);
                      let nr = |v: &[f32], j: usize| v[j * dim..(j + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                      let med = { let mut q: Vec<f32> = (0..nt).map(|j| nr(&rf, j)).collect(); q.sort_by(|a, b| a.partial_cmp(b).unwrap()); q[nt / 2] };
                      for &(j, _) in &w[..w.len().min(3)] { let dd: f32 = (0..dim).map(|k| (o[j * dim + k] - rf[j * dim + k]).powi(2)).sum::<f32>().sqrt(); eprintln!("    row {j}: |ours| {:.3} |ref| {:.3} |diff| {:.3} (median |ref| {:.3})", nr(o, j), nr(&rf, j), dd, med); } }
                    o.clone() }
                None => rf };
            let ps = std::fs::read_to_string(format!("{dir}/img_pos_{ci}.txt"))?;
            // dumped with pos_0 = 0 (base = pos_0 - 1 wraps): rebase onto the running position
            for (j, l) in ps.lines().enumerate().take(nt) {
                let v: Vec<u32> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
                let p = (pos.wrapping_add(v[0]), pos.wrapping_add(v[1]), pos.wrapping_add(v[2]));
                embs.push(f[j * dim..(j + 1) * dim].to_vec());
                plan.push((true, (embs.len() - 1) as u32, p));
            }
            pos += npos;
        }
    }
    let items: Vec<Item> = plan.iter().map(|(e, v, _)| if *e { Item::Emb(&embs[*v as usize]) } else { Item::Tok(*v) }).collect();
    let mpos: Vec<(u32, u32, u32)> = plan.iter().map(|p| p.2).collect();
    eprintln!("prompt: {} positions ({} embedded), rope n_pos {}", items.len(), embs.len(), pos);
    if std::env::var("FORCED").is_ok() {
        // teacher-forced: prompt + reference continuation; per step argmax / top-2 margin vs the reference token
        let np = items.len();
        let mut it2 = items; let mut mp2 = mpos.clone();
        for (j, &t) in ref_ids.iter().enumerate().take(ref_ids.len().saturating_sub(1)) { it2.push(Item::Tok(t)); let p = pos + j as u32; mp2.push((p, p, p)); }
        let mut agree = 0; let mut tot = 0; let mut worst = vec![];
        m.forced(&it2, &mp2, |i, lg| {
            if i + 1 < np { return }
            let j = i + 1 - np; if j >= ref_ids.len() { return }
            let mut b = (0usize, f32::MIN); let mut s2 = f32::MIN;
            for (v, &x) in lg.iter().enumerate() { if x > b.1 { s2 = b.1; b = (v, x) } else if x > s2 { s2 = x } }
            let r = ref_ids[j] as usize; tot += 1;
            if b.0 == r { agree += 1 } else { worst.push((j, b.0, r, b.1 - lg[r], b.1 - s2)); }
        })?;
        println!("FORCED agree {agree}/{tot}");
        for w in &worst { println!("  step {} ours {} ref {} logit gap {:.3} (top2 margin {:.3})", w.0, w.1, w.2, w.3, w.4); }
        return Ok(());
    }
    // PFCHK=1: top-5 logits after the prompt and after one decode step, batched vs stepwise
    if std::env::var("PFCHK").is_ok() {
        let top = |m: &Llm| { let l = &m.logits.as_f32_mut()[..m.cfg.vocab as usize]; let mut v: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); v.truncate(5); v };
        let pre = mcv::prefill::Prefill::new(&m, 512)?;
        pre.run(&m, &items, &mpos, pos)?; eprintln!("batched  prompt {:?}", top(&m)); m.decode(items.len(), 2, &[])?; eprintln!("batched  step1  {:?}", top(&m));
        m.prompt_stepwise(&items, &mpos, pos)?; eprintln!("stepwise prompt {:?}", top(&m)); m.decode(items.len(), 2, &[])?; eprintln!("stepwise step1  {:?}", top(&m));
        return Ok(());
    }
    // PREFILL=1: batched prefill (chunk PF_M rows; REPS timed repeats, median reported)
    let (_first, pt) = if std::env::var("PREFILL").map(|v| v == "1").unwrap_or(false) {
        let pre = mcv::prefill::Prefill::new(&m, std::env::var("PF_M").ok().and_then(|v| v.parse().ok()).unwrap_or(512))?;
        let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let mut ts = vec![]; let mut r = (0, 0.0);
        for _ in 0..reps { r = pre.run(&m, &items, &mpos, pos)?; ts.push(r.1); }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ts[ts.len() / 2];
        { let lg = unsafe { std::slice::from_raw_parts(m.logits.ptr as *const f32, m.cfg.vocab as usize) };
          let mut ix: Vec<usize> = (0..lg.len()).collect(); ix.select_nth_unstable_by(1, |a, b| lg[*b].partial_cmp(&lg[*a]).unwrap());
          let (i0, i1) = if lg[ix[0]] >= lg[ix[1]] { (ix[0], ix[1]) } else { (ix[1], ix[0]) };
          eprintln!("TOP2 {} {:.4} | {} {:.4} | margin {:.4}", i0, lg[i0], i1, lg[i1], lg[i0] - lg[i1]); }
        eprintln!("prefill (batched): {} positions, median {:.4} s over {} -> {:.1} tok/s (first pred {})", items.len(), med, reps, items.len() as f64 / med, r.0);
        (r.0, med)
    } else { m.prompt_stepwise(&items, &mpos, pos)? };
    // PFORCED=<ids file> (with PREFILL=1): teacher-forced continuation over the given ids after the batched prefill;
    // agreement counts the first prefill prediction plus each forced step's argmax vs the next id.
    if let Ok(f) = std::env::var("PFORCED") {
        let ids: Vec<u32> = std::fs::read_to_string(&f)?.split(|c: char| c == ',' || c.is_whitespace()).filter_map(|t| t.parse().ok()).collect();
        let np = items.len();
        let mut agree = (_first == ids[0]) as usize; let mut tot = 1; let mut bad = vec![];
        if _first != ids[0] { bad.push(0); }
        let mp: Vec<(u32, u32, u32)> = (0..ids.len()).map(|j| { let p = pos + j as u32; (p, p, p) }).collect();
        m.forced_from(np, &ids[..ids.len() - 1], &mp, |i, lg| {
            let j = i - np + 1; let mut b = (0usize, f32::MIN);
            for (v, &x) in lg.iter().enumerate() { if x > b.1 { b = (v, x) } }
            tot += 1; if b.0 == ids[j] as usize { agree += 1 } else { bad.push(j) }
        })?;
        println!("PFORCED agree {agree}/{tot} mismatch {:?}", bad);
        return Ok(());
    }
    let eos = [248046u32, 248044];
    let (out, dt) = m.decode(items.len(), n_gen, &eos)?;
    let nd = n_gen.saturating_sub(1).max(1); // decode steps actually executed (EOS truncation happens afterwards)
    println!("prompt(stepwise) {:.2}s  decode {:.1} tok/s ({} tok in {:.3}s)", pt, nd as f64 / dt, out.len(), dt);
    println!("OURS {:?}", out);
    let k = out.iter().zip(&ref_ids).take_while(|(a, b)| a == b).count();
    println!("REF  {:?}", &ref_ids[..ref_ids.len().min(out.len())]);
    println!("MATCH first {} / {} tokens", k, out.len().min(ref_ids.len()));
    Ok(())
}
