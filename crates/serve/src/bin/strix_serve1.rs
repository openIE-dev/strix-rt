//! strix-serve1: single-stream OpenAI-compatible server for models without a batched-decode trunk:
//!   LFM2.5-24B-A2B (lfm2moe; batched prefill via prefill::lfm2moe) and MiniCPM-V 4.7 (mcv; image input via data: URLs,
//!   own vision tower, batched prefill via mcv::prefill). Up to --slots sequences decode concurrently (round robin, each in its own state slot).
//! usage: strix-serve1 <model.gguf> [--host H] [--port P] [--alias NAME] [--ctx N] [--vision DIR] [--slots N]
use anyhow::{anyhow, bail, Result};
use serve::{listen, recv_all, sample, Ev, Front, Img, Job, Template, IMG_TOK};
use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;

fn arg(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()) }

enum Eng { Lfm(lfm2::Model, prefill::Pf, prefill::lfm2moe::Lf, prefill::lfm2moe::Lf), Mcv(mcv::llm::Llm, mcv::prefill::Prefill, mcv::vision::Vision, Option<mcv::prefill::Prefill>) }

/// MiniCPM-V prompt plan: items + M-RoPE positions + rope n_pos. Image rows follow llama.cpp mtmd (crates/mcv/ref/out/img):
/// per tile, the nx*ny merged rows (nx = w/56, ny = h/56) sit at (p, p-1+y, p-1+x), the separator rows at p+1, and the
/// tile advances the position by 2. Verified against the reference for single-tile images.
fn mcv_plan(m: &mcv::llm::Llm, v: &mcv::vision::Vision, ids: &[u32], imgs: &[Img]) -> Result<(Vec<u32>, Vec<Vec<f32>>, Vec<(bool, usize)>, Vec<(u32, u32, u32)>, u32)> {
    let dim = m.cfg.dim as usize; let (mut embs, mut plan, mut mpos) = (vec![], vec![], vec![]); let mut p = 0u32; let mut ki = 0usize;
    for &t in ids {
        if t != IMG_TOK { plan.push((false, t as usize)); mpos.push((p, p, p)); p += 1; continue }
        let im = imgs.get(ki).ok_or_else(|| anyhow!("image index"))?; ki += 1;
        let (tiles, _) = mcv::vision::preprocess(&im.rgb, im.w, im.h);
        for tl in &tiles {
            let (e, _) = v.encode(&m.ctx, tl)?; let rows = e.len() / dim; let (nx, ny) = (tl.w / 56, tl.h / 56); let g = (nx * ny).min(rows);
            for r in 0..rows { embs.push(e[r * dim..(r + 1) * dim].to_vec()); plan.push((true, embs.len() - 1));
                mpos.push(if r < g { (p, (p + (r / nx) as u32).wrapping_sub(1), (p + (r % nx) as u32).wrapping_sub(1)) } else { (p + 1, p + 1, p + 1) }); }
            p += 2;
        }
    }
    Ok((vec![], embs, plan, mpos, p))
}

#[allow(dead_code)]
fn main() -> Result<()> { run(std::env::args().collect(), None) }

/// load the model and run its engine loop; with `ready`, hand the Front to the caller (multi-model server) instead of listening
pub fn run(a: Vec<String>, ready: Option<std::sync::mpsc::Sender<std::sync::Arc<serve::Front>>>) -> Result<()> {
    if a.len() < 2 { bail!("usage: strix-serve1 <model.gguf> [--host H] [--port P] [--alias NAME] [--ctx N] [--vision DIR] [--slots N]") }
    let path = &a[1];
    let ctx: u32 = arg(&a, "--ctx").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let addr = format!("{}:{}", arg(&a, "--host").unwrap_or("127.0.0.1".into()), arg(&a, "--port").unwrap_or("8080".into()));
    let name = arg(&a, "--alias").unwrap_or_else(|| std::path::Path::new(path).file_stem().unwrap().to_string_lossy().into_owned());
    let t = std::time::Instant::now();
    let arch = { let g = gguf::Gguf::open(path)?; g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default() };
    if std::env::var("LFM_DET").is_err() { std::env::set_var("LFM_DET", "1"); } // reproducible greedy (gemv3 flag 128)
    // MiniCPM-V: batched decode (mcv record_multi) over up to --slots columns; activation buffers sized at load
    let mslots: usize = arg(&a, "--slots").and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let mmulti = std::env::var("SERVE_MULTI").map_or(true, |v| v != "0");
    if mmulti && std::env::var("NCOL").is_err() { std::env::set_var("NCOL", mslots.to_string()); }
    let pfm = prefill::env("PF_M", 512);
    // LFM2.5 cross-prompt batched prefill: every loop runs one packed pass with the next fixed-boundary chunk (<= chunk
    // rows) of up to `pack` rows' worth of prefilling sequences (prefill::lfm2moe::prefill_pass)
    let pfb = std::env::var("SERVE_PFB").map_or(true, |v| v != "0");
    let (chunk, pack) = (prefill::env("SERVE_CHUNK", 512) as usize, prefill::env("SERVE_PACK", 2048) as usize);
    if pfb && std::env::var("PF_WIDE_RPE").is_err() { std::env::set_var("PF_WIDE_RPE", "1000000000"); }
    if pfb && std::env::var("PF_SPLITK").is_err() { std::env::set_var("PF_SPLITK", "1"); } // tile choice independent of packing
    let eng = if arch.starts_with("lfm2") {
        let m = lfm2::Model::load(path, ctx)?; let pf = prefill::Pf::new(&m.ctx)?; let s = prefill::lfm2moe::Lf::new(&m, if pfb { pack as u32 } else { pfm })?;
        let t = prefill::lfm2moe::Lf::new(&m, if pfb { chunk as u32 } else { 16 })?; Eng::Lfm(m, pf, s, t)
    } else {
        let m = mcv::llm::Llm::load(path, ctx)?; let p = mcv::prefill::Prefill::new(&m, if pfb { pack as u32 } else { pfm })?; let tq = if pfb { Some(mcv::prefill::Prefill::new(&m, chunk as u32)?) } else { None };
        let vd = arg(&a, "--vision").unwrap_or_else(|| format!("{}/vision_st", std::path::Path::new(path).parent().unwrap().display()));
        let v = mcv::vision::Vision::load(&m.ctx, &vd)?; Eng::Mcv(m, p, v, tq)
    };
    let (wt, vocab_n, mctx) = match &eng { Eng::Lfm(m, ..) => (&m.wt, m.cfg.vocab as usize, m.cfg.max_ctx), Eng::Mcv(m, ..) => (&m.wt, m.cfg.vocab as usize, m.cfg.max_ctx) };
    let g = &wt.g;
    let mut tok = kev::tok::Tokenizer::from_gguf(g)?; if !arch.starts_with("lfm2") { tok.marks_in_letters = true; }
    let b2u = kev::tok::byte_map(); let u2b: HashMap<char, u8> = b2u.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect();
    let vocab = wt.vocab_strings();
    let types: Vec<u64> = match g.get("tokenizer.ggml.token_type") { Ok(gguf::Value::Arr(v)) => v.iter().map(|x| x.as_u64().unwrap_or(1)).collect(), _ => vec![1; vocab.len()] };
    let pieces: Vec<Vec<u8>> = vocab.iter().map(|s| s.chars().map(|c| u2b.get(&c).copied().unwrap_or(b'?')).collect()).collect();
    let special: Vec<bool> = (0..vocab.len()).map(|i| types.get(i) == Some(&3)).collect();
    let mut eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.id(s)).collect();
    if let Ok(e) = g.u64("tokenizer.ggml.eos_token_id") { eos.push(e as u32) }
    let vision = matches!(eng, Eng::Mcv(..));
    let tmpl = if arch.starts_with("lfm2") { Template::Lfm2 } else { Template::ChatMl };
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    let front = std::sync::Arc::new(Front { name: name.clone(), tok, pieces, special, tmpl, vision, jobs: std::sync::Mutex::new(tx), max_ctx: mctx as usize });
    match &ready { Some(r) => { let _ = r.send(front.clone()); } None => listen(&addr, front.clone())? }
    eprintln!("strix-serve1: {name} ({arch}), ctx {mctx}, vision {vision}, loaded in {:.1}s; http://{addr}/v1", t.elapsed().as_secs_f64());
    // multi-sequence serving: every sequence owns a state slot (KV, conv/DeltaNet state, token ring, aux) with its own
    // recorded step; prefill and steps run with the slot swapped into the model, so each sequence computes exactly as
    // single-stream. One prefill per loop iteration, then one step for every active sequence (round robin).
    enum S { L(lfm2::St), M(mcv::llm::St) }
    let nslot: usize = arg(&a, "--slots").and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let mut eng = eng;
    let mut slots: Vec<S> = (0..nslot).map(|_| match &mut eng { Eng::Lfm(m, ..) => m.new_st().map(S::L), Eng::Mcv(m, ..) => m.new_st().map(S::M) }).collect::<Result<_>>()?;
    struct Q { job: Job, sl: usize, n: usize, k: usize, rng: u64, ttft: f64 }
    let mut free: Vec<usize> = (0..nslot).rev().collect(); let mut act: Vec<Q> = vec![]; let mut queue: std::collections::VecDeque<Job> = Default::default();
    let tk = |s: &S| -> &mut [u32] { let b = match s { S::L(x) => &x.toks, S::M(x) => &x.toks }; unsafe { std::slice::from_raw_parts_mut(b.ptr as *mut u32, b.size as usize / 4) } };
    let batch_sub = std::env::var("SERVE_BATCHSUB").map_or(true, |v| v != "0");
    let (barcb, fence) = { let c: &vkrt::Ctx = match &eng { Eng::Lfm(m, ..) => &m.ctx, Eng::Mcv(m, ..) => &m.ctx };
        (c.record_raw(|cb| c.barrier(cb))?, unsafe { c.device.create_fence(&ash::vk::FenceCreateInfo::default(), None)? }) };
    let mut lmu: Option<lfm2::Multi> = None; let mut lcbs: std::collections::HashMap<Vec<usize>, ash::vk::CommandBuffer> = Default::default();
    let mut mmu: Option<mcv::llm::Multi> = None; let mut mcbs: std::collections::HashMap<Vec<usize>, ash::vk::CommandBuffer> = Default::default();
    let mut pfq: Vec<(Job, usize, usize)> = vec![];
    let mut mpfq: Vec<(Job, usize, usize, Vec<Vec<f32>>, Vec<(bool, usize)>, Vec<(u32, u32, u32)>, u32)> = vec![];
    let verbose = std::env::var("SERVE_LOG").is_ok(); let (mut nst, mut tst) = (0u64, 0f64);
    loop {
        for j in recv_all(&rx, act.is_empty() && queue.is_empty() && pfq.is_empty() && mpfq.is_empty()) { queue.push_back(j) }
        queue.retain(|j| !j.cancel.load(Relaxed));
        // LFM2.5: admit every waiting prompt that has a slot, then one packed prefill pass
        if let (Eng::Lfm(m, pf, s, t), true) = (&eng, pfb) {
            while let (Some(&sl), true) = (free.last(), !queue.is_empty()) {
                let job = queue.pop_front().unwrap();
                if job.ids.contains(&IMG_TOK) || job.ids.is_empty() || job.ids.len() + job.max_tokens + 2 > mctx as usize {
                    let _ = job.tx.send(Ev::Err(if job.ids.contains(&IMG_TOK) { "no image input".into() } else { "prompt + max_tokens exceeds the context".into() })); continue }
                free.pop(); if let S::L(st) = &slots[sl] { st.cst.bytes_mut().fill(0); }
                pfq.push((job, sl, 0usize));
            }
            if !pfq.is_empty() {
                let mut used = vec![]; let mut rows = 0;
                for (i, (job, _, done)) in pfq.iter().enumerate() { let k = (job.ids.len() - done).min(chunk); if rows + k > pack { break } rows += k; used.push((i, k)); }
                let segs: Vec<prefill::lfm2moe::Seg> = used.iter().map(|&(i, k)| { let (job, sl, done) = &pfq[i]; let S::L(st) = &slots[*sl] else { unreachable!() };
                    prefill::lfm2moe::Seg { toks: &job.ids[*done..*done + k], pos0: *done as u32, kc: &st.kc, vc: &st.vc, cst: &st.cst, last: *done + k == job.ids.len() } }).collect();
                let t0 = std::time::Instant::now();
                match prefill::lfm2moe::prefill_pass(pf, m, s, t, &segs) {
                    Err(e) => { for &(i, _) in used.iter().rev() { let (job, sl, _) = pfq.remove(i); let _ = job.tx.send(Ev::Err(e.to_string())); free.push(sl); } }
                    Ok(res) => {
                        if verbose { eprintln!("prefill pass: {} segs {rows} rows {:.0} ms", segs.len(), t0.elapsed().as_secs_f64() * 1e3); }
                        drop(segs);
                        for &(i, k) in &used { pfq[i].2 += k; }
                        let mut fin: Vec<(usize, u32)> = vec![];
                        for (si, tok, lg) in res { let i = used[si].0; let (job, _, _) = &mut pfq[i];
                            let tok = if job.temperature > 0.0 { let mut rng = job.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1; let t = sample(&lg[..vocab_n], job.temperature, job.top_p, &mut rng); t } else { tok };
                            fin.push((i, tok)); }
                        fin.sort_by(|a, b| b.0.cmp(&a.0));
                        for (i, tok) in fin { let (job, sl, _) = pfq.remove(i); let n = job.ids.len(); let tk_ = tk(&slots[sl]); tk_[..n].copy_from_slice(&job.ids); tk_[n] = tok;
                            act.push(Q { ttft: job.t0.elapsed().as_secs_f64() * 1e3, rng: job.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1, job, sl, n, k: 0 }); }
                    }
                }
            }
        }
        // MiniCPM-V: same scheme (mcv::prefill::Prefill::run_multi), image embeddings planned at admission
        if let (Eng::Mcv(m, p, v, Some(t)), true) = (&eng, pfb) {
            while let (Some(&sl), true) = (free.last(), !queue.is_empty()) {
                let job = queue.pop_front().unwrap();
                match mcv_plan(m, v, &job.ids, &job.images) {
                    Err(e) => { let _ = job.tx.send(Ev::Err(e.to_string())); }
                    Ok((_, embs, plan, mpos, npos)) => {
                        if plan.is_empty() || plan.len() + job.max_tokens + 2 > mctx as usize { let _ = job.tx.send(Ev::Err(format!("prompt {} positions + max_tokens exceeds the context {mctx}", plan.len()))); continue }
                        free.pop(); mpfq.push((job, sl, 0usize, embs, plan, mpos, npos)); }
                }
            }
            if !mpfq.is_empty() {
                let mut used = vec![]; let mut rows = 0;
                for (i, q) in mpfq.iter().enumerate() { let k = (q.4.len() - q.2).min(chunk); if rows + k > pack { break } rows += k; used.push((i, k)); }
                let items: Vec<Vec<mcv::llm::Item>> = used.iter().map(|&(i, k)| { let q = &mpfq[i];
                    q.4[q.2..q.2 + k].iter().map(|&(e, j)| if e { mcv::llm::Item::Emb(&q.3[j]) } else { mcv::llm::Item::Tok(j as u32) }).collect() }).collect();
                let segs: Vec<mcv::prefill::PSeg> = used.iter().zip(&items).map(|(&(i, k), it)| { let q = &mpfq[i]; let S::M(st) = &slots[q.1] else { unreachable!() };
                    mcv::prefill::PSeg { items: it, mpos: &q.5[q.2..q.2 + k], pos0: q.2 as u32, kc: &st.kc, vc: &st.vc, cst: &st.cst, sst: &st.sst, last: q.2 + k == q.4.len() } }).collect();
                let t0 = std::time::Instant::now();
                let r = p.run_multi(t, m, &segs);
                drop(segs); drop(items);
                match r {
                    Err(e) => { for &(i, _) in used.iter().rev() { let q = mpfq.remove(i); let _ = q.0.tx.send(Ev::Err(e.to_string())); free.push(q.1); } }
                    Ok(res) => {
                        if verbose { eprintln!("prefill pass: {} segs {rows} rows {:.0} ms", used.len(), t0.elapsed().as_secs_f64() * 1e3); }
                        for &(i, k) in &used { mpfq[i].2 += k; }
                        let mut fin: Vec<(usize, u32)> = vec![];
                        for (si, tok, lg) in res { let i = used[si].0; let job = &mpfq[i].0;
                            let tok = if job.temperature > 0.0 { let mut rng = job.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1; sample(&lg[..vocab_n], job.temperature, job.top_p, &mut rng) } else { tok };
                            fin.push((i, tok)); }
                        fin.sort_by(|a, b| b.0.cmp(&a.0));
                        for (i, tok) in fin { let q = mpfq.remove(i); let n = q.4.len(); let S::M(st) = &slots[q.1] else { unreachable!() };
                            let sa = unsafe { std::slice::from_raw_parts_mut(st.aux.ptr as *mut u32, 256) }; sa[164] = n as u32 - q.6; sa[167] = 2;
                            tk(&slots[q.1])[n] = tok;
                            act.push(Q { ttft: q.0.t0.elapsed().as_secs_f64() * 1e3, rng: q.0.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1, job: q.0, sl: q.1, n, k: 0 }); }
                    }
                }
            }
        }
        // admit one prompt
        if let (Some(&sl), true, false) = (free.last(), !queue.is_empty(), pfb) {
            let job = queue.pop_front().unwrap(); free.pop();
            let r: Result<usize> = match (&mut eng, &mut slots[sl]) {
                (Eng::Lfm(m, pf, s, _), S::L(st)) => m.with_st(st, |m| { if job.ids.contains(&IMG_TOK) { bail!("no image input") } prefill::lfm2moe::prefill(pf, m, s, &job.ids)?; Ok(job.ids.len()) }),
                (Eng::Mcv(m, p, v, _), S::M(st)) => m.with_st(st, |m| { let (_, embs, plan, mpos, npos) = mcv_plan(m, v, &job.ids, &job.images)?;
                    if plan.len() + job.max_tokens + 2 > mctx as usize { bail!("prompt {} positions + max_tokens exceeds the context {mctx}", plan.len()) }
                    let items: Vec<mcv::llm::Item> = plan.iter().map(|&(e, i)| if e { mcv::llm::Item::Emb(&embs[i]) } else { mcv::llm::Item::Tok(i as u32) }).collect();
                    p.run(m, &items, &mpos, npos)?; Ok(items.len()) }),
                _ => unreachable!() };
            match r { Err(e) => { let _ = job.tx.send(Ev::Err(e.to_string())); free.push(sl); }
                Ok(n) => act.push(Q { ttft: job.t0.elapsed().as_secs_f64() * 1e3, rng: job.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1, job, sl, n, k: 0 }) }
            // the first token is emitted below as k = 0 (no step needed: prefill left it in toks[n])
        }
        // one step per active sequence; k = number of tokens already emitted. Greedy sequences' steps go to the GPU in ONE
        // submission (separated by barrier command buffers, the steps share activation scratch) so the host never waits
        // between them; sampled sequences need their logits right after their own step and are submitted one by one.
        let ctxr: &vkrt::Ctx = match &eng { Eng::Lfm(m, ..) => &m.ctx, Eng::Mcv(m, ..) => &m.ctx };
        let mut pre = vec![false; act.len()];
        // LFM2.5 batched step (lfm2 record_multi): Multi buffers per column count, step cbs cached per slot set
        let mut sampled = vec![false; act.len()];
        if let (Eng::Lfm(m, ..), true) = (&eng, mmulti) {
            let idx: Vec<usize> = (0..act.len()).filter(|&j| act[j].k > 0).collect();
            if !idx.is_empty() {
                if lmu.is_none() { lmu = Some(m.new_multi(nslot as u32)?); }
                let mu = lmu.as_ref().unwrap();
                for (c, &j) in idx.iter().enumerate() { let q = &act[j]; let w = unsafe { std::slice::from_raw_parts_mut(mu.a[c].ptr as *mut u32, 256) }; w[160] = (q.n + q.k - 1) as u32; pre[j] = true; }
                let key: Vec<usize> = idx.iter().map(|&j| act[j].sl).collect();
                if !lcbs.contains_key(&key) {
                    let sts: Vec<&lfm2::St> = key.iter().map(|&s| match &slots[s] { S::L(st) => st, _ => unreachable!() }).collect();
                    let mut err = Ok(()); let cb = m.ctx.record_raw(|cb| { err = m.record_multi(cb, &sts, mu); })?; err?;
                    if lcbs.len() > 64 { lcbs.clear(); } lcbs.insert(key.clone(), cb);
                }
                let t0 = std::time::Instant::now(); m.ctx.submit_repeated(lcbs[&key], 1)?; tst += t0.elapsed().as_secs_f64(); nst += 1;
                let ba = unsafe { std::slice::from_raw_parts(mu.ba.ptr as *const u32, 1024) }; let lg = mu.logits.as_f32();
                for (c, &j) in idx.iter().enumerate() { let q = &mut act[j]; let pos = q.n + q.k;
                    let t = if q.job.temperature > 0.0 { sampled[j] = true; sample(&lg[c * vocab_n..(c + 1) * vocab_n], q.job.temperature, q.job.top_p, &mut q.rng) } else { ba[600 + c] };
                    tk(&slots[q.sl])[pos] = t; }
            }
        }
        // MiniCPM-V batched step: all active sequences past their first token decode as one record_multi step
        let mut col = vec![usize::MAX; act.len()];
        if let (Eng::Mcv(m, ..), true) = (&eng, mmulti) {
            let idx: Vec<usize> = (0..act.len()).filter(|&j| act[j].k > 0).collect();
            if !idx.is_empty() {
                let mu = mmu.get_or_insert_with(|| m.new_multi().expect("new_multi"));
                let au = m.au_all();
                for (c, &j) in idx.iter().enumerate() { let q = &act[j]; let pos = q.n + q.k; let S::M(st) = &slots[q.sl] else { unreachable!() };
                    let sa = unsafe { std::slice::from_raw_parts(st.aux.ptr as *const u32, 256) };
                    au[(mcv::llm::MTOKW as usize) + c] = tk(&slots[q.sl])[pos - 1]; au[(mcv::llm::MPOSW as usize) + c] = (pos - 1) as u32;
                    let r = mcv::llm::MRW as usize + 4 * c; au[r] = sa[164]; au[r + 1] = 0; au[r + 2] = 0; au[r + 3] = 2;
                    col[j] = c; pre[j] = true; }
                let key: Vec<usize> = idx.iter().map(|&j| act[j].sl).collect();
                if !mcbs.contains_key(&key) {
                    let sts: Vec<&mcv::llm::St> = key.iter().map(|&s| match &slots[s] { S::M(st) => st, _ => unreachable!() }).collect();
                    let mut err = Ok(()); let cb = m.ctx.record_raw(|cb| { err = m.record_multi(cb, &sts, mu); })?; err?;
                    if mcbs.len() > 64 { mcbs.clear(); } mcbs.insert(key.clone(), cb);
                }
                let t0 = std::time::Instant::now(); m.ctx.submit_repeated(mcbs[&key], 1)?; tst += t0.elapsed().as_secs_f64(); nst += 1;
                for (c, &j) in idx.iter().enumerate() { let q = &act[j]; tk(&slots[q.sl])[q.n + q.k] = au[600 + c]; }
            }
        }
        if batch_sub && !mmulti {
            let mut cbs = vec![];
            for (j, q) in act.iter().enumerate() { if q.k > 0 && q.job.temperature <= 0.0 { pre[j] = true; cbs.push(match &slots[q.sl] { S::L(st) => st.step, S::M(st) => st.step }); cbs.push(barcb); } }
            if !cbs.is_empty() { let t0 = std::time::Instant::now(); let d = &ctxr.device;
                unsafe { d.reset_fences(&[fence])?; d.queue_submit(ctxr.queue, &[ash::vk::SubmitInfo::default().command_buffers(&cbs)], fence)?; d.wait_for_fences(&[fence], true, u64::MAX)?; }
                tst += t0.elapsed().as_secs_f64(); nst += cbs.len() as u64 / 2; }
        }
        let mut i = 0;
        while i < act.len() {
            let q = &mut act[i]; let pos = q.n + q.k;
            if q.k > 0 && !pre[i] { let t0 = std::time::Instant::now();
                let r = match (&eng, &slots[q.sl]) { (Eng::Lfm(m, ..), S::L(st)) => m.ctx.submit_repeated(st.step, 1), (Eng::Mcv(m, ..), S::M(st)) => m.ctx.submit_repeated(st.step, 1), _ => unreachable!() };
                tst += t0.elapsed().as_secs_f64(); nst += 1;
                if let Err(e) = r { let _ = q.job.tx.send(Ev::Err(e.to_string())); free.push(q.sl); act.remove(i); pre.remove(i); col.remove(i); sampled.remove(i); continue } }
            let toks = tk(&slots[q.sl]);
            if q.job.temperature > 0.0 && !sampled[i] { let lo = if col[i] == usize::MAX { 0 } else { col[i] * vocab_n }; toks[pos] = sample(&logits_of(&eng)[lo..lo + vocab_n], q.job.temperature, q.job.top_p, &mut q.rng); }
            let t = toks[pos]; q.k += 1;
            let _ = q.job.tx.send(Ev::Tok(t));
            let fin = if q.job.cancel.load(Relaxed) || (!q.job.ignore_eos && eos.contains(&t)) { Some("stop") }
                else if q.k >= q.job.max_tokens || pos + 2 >= mctx as usize { Some("length") } else { None };
            if let Some(reason) = fin { let _ = q.job.tx.send(Ev::Done { reason, prompt: q.job.ids.len(), completion: q.k, cached: 0, ttft_ms: q.ttft }); free.push(q.sl); act.remove(i); pre.remove(i); col.remove(i); sampled.remove(i); continue }
            i += 1;
        }
        if verbose && nst > 0 && nst % 256 == 0 { eprintln!("steps {nst}: {:.2} ms/step, active {}", tst / nst as f64 * 1e3, act.len()); }
    }
}
fn logits_of(e: &Eng) -> &[f32] { match e { Eng::Lfm(m, ..) => m.logits.as_f32(), Eng::Mcv(m, ..) => m.logits.as_f32() } }
