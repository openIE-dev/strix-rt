//! strix-serve: OpenAI-compatible text server with continuous batching on strix-rt.
//!   strix-serve <model.gguf> [--host 127.0.0.1] [--port 8080] [--alias NAME] [--slots 8] [--ctx 4096] [--cache 2]
//! Endpoints: POST /v1/chat/completions, POST /v1/completions (SSE with "stream": true, stream_options.include_usage),
//! GET /v1/models, GET /health. Greedy (temperature 0) runs the argmax on the GPU; temperature > 0 samples on the host.
//! Engine (qwen35 trunk: Qwen3.8-27B, Bonsai 2): every loop iteration admits at most one waiting request (batched
//! prefill into the prefill scratch, with prefix-cache reuse), then runs ONE decode step for all active sequences
//! (qwen38::multi::record_multi: every projection is a multi-column gemvq over the batch). Each sequence owns a KV
//! block + GDN conv/recurrent state slot. Prefix cache: up to --cache snapshots of finished prompt prefills; a new prompt
//! that extends a cached one restores that state and prefills only the suffix.
use anyhow::{bail, Result};
use ash::vk;
use qwen38::multi::{Slot, POSW, TOKW};
use serve::*;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering::Relaxed;

struct Seq { job: Job, slot: usize, pos: u32, last: u32, n_out: usize, rng: u64, ttft: f64, cached: usize }
struct Pre { job: Job, pos: usize, cached: usize, t: f64 }
struct PreS { job: Job, slot: usize, pos: usize, cached: usize, t: f64 }
struct Ent { ids: Vec<u32>, slot: Slot, used: u64 }

fn arg(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()) }

#[path = "strix_serve1.rs"] #[allow(dead_code)] mod s1;

/// Unified server: `strix-serve a.gguf [b.gguf ...] [--aliases n1,n2] [--alias N] [--slots N] [--ctx N] [--host H] [--port P]`.
/// Each model gets its own engine thread (qwen35/qwen38 dense -> the strix-serve engine; lfm2moe / MiniCPM-V qwen3 MoE
/// -> the strix-serve1 engine); one HTTP listener routes requests by the JSON "model" field. A single model behaves as before.
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let flags_v = ["--host", "--port", "--alias", "--aliases", "--slots", "--ctx", "--cache", "--mmproj", "--vision"];
    let mut models = vec![]; let mut rest = vec![]; let mut i = 1;
    while i < a.len() {
        if flags_v.contains(&a[i].as_str()) && i + 1 < a.len() { if a[i] != "--aliases" { rest.push(a[i].clone()); rest.push(a[i + 1].clone()); } i += 2; }
        else if a[i].starts_with("--") { rest.push(a[i].clone()); i += 1 } else { models.push(a[i].clone()); i += 1 }
    }
    if models.len() == 1 { let ar = gguf_arch(&models[0]).unwrap_or_default(); return if is_dense(&ar) { run(a, None) } else { s1::run(a, None) } }
    if models.is_empty() { return run(a, None) }
    let aliases: Vec<String> = arg(&a, "--aliases").map(|x| x.split(',').map(|s| s.to_string()).collect()).unwrap_or_default();
    let (rtx, rrx) = std::sync::mpsc::channel(); let mut hs = vec![];
    for (k, m) in models.iter().enumerate() {
        let mut v = vec![a[0].clone(), m.clone()];
        let mut it = rest.iter(); while let Some(x) = it.next() { if x == "--alias" { it.next(); continue } v.push(x.clone()) }
        if let Some(n) = aliases.get(k) { v.push("--alias".into()); v.push(n.clone()) }
        let arch = gguf_arch(m).unwrap_or_default();
        let dense = is_dense(&arch);
        eprintln!("[strix-serve] model {k}: {m} arch={arch} engine={}", if dense { "serve" } else { "serve1" });
        let tx = rtx.clone();
        hs.push(std::thread::spawn(move || { let r = if dense { run(v, Some(tx)) } else { s1::run(v, Some(tx)) }; if let Err(e) = &r { eprintln!("[strix-serve] engine for {k} failed: {e:#}"); std::process::exit(1) } }));
        // load one model at a time (VRAM / startup memory peak)
        let f = rrx.recv()?; eprintln!("[strix-serve] ready: {}", f.name); FRONTS.lock().unwrap().push(f);
    }
    let fs = FRONTS.lock().unwrap().clone();
    let names: Vec<_> = fs.iter().map(|f| f.name.clone()).collect();
    for i in 0..names.len() { for j in 0..i { if names[i] == names[j] { bail!("duplicate model name {} (use --aliases)", names[i]) } } }
    let addr = format!("{}:{}", arg(&a, "--host").unwrap_or("127.0.0.1".into()), arg(&a, "--port").unwrap_or("8080".into()));
    serve::listen_many(&addr, fs)?; eprintln!("[strix-serve] listening on {addr}: {}", names.join(", "));
    for h in hs { let _ = h.join(); }
    Ok(())
}
static FRONTS: std::sync::Mutex<Vec<std::sync::Arc<serve::Front>>> = std::sync::Mutex::new(vec![]);

fn is_dense(ar: &str) -> bool { matches!(ar, "qwen35" | "qwen38" | "qwen3") }
/// general.architecture from a GGUF header
fn gguf_arch(p: &str) -> Result<String> {
    use std::io::Read; let mut f = std::fs::File::open(p)?; let mut b = vec![0u8; 1 << 16]; let n = f.read(&mut b)?; let b = &b[..n];
    let k = b"general.architecture"; let i = b.windows(k.len()).position(|w| w == k).ok_or_else(|| anyhow::anyhow!("no arch"))? + k.len();
    let len = u64::from_le_bytes(b[i + 4..i + 12].try_into()?) as usize; Ok(String::from_utf8_lossy(&b[i + 12..i + 12 + len]).into())
}

/// load the model and run its engine loop; with `ready`, hand the Front to the caller (multi-model server) instead of listening
pub fn run(a: Vec<String>, ready: Option<std::sync::mpsc::Sender<std::sync::Arc<serve::Front>>>) -> Result<()> {
    if a.len() < 2 { bail!("usage: strix-serve <model.gguf> [--host H] [--port P] [--alias NAME] [--slots N] [--ctx N] [--cache N]") }
    let path = &a[1];
    let nslot: usize = arg(&a, "--slots").and_then(|v| v.parse().ok()).unwrap_or(8);
    let ctx: u32 = arg(&a, "--ctx").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let ncache: usize = arg(&a, "--cache").and_then(|v| v.parse().ok()).unwrap_or(2);
    let addr = format!("{}:{}", arg(&a, "--host").unwrap_or("127.0.0.1".into()), arg(&a, "--port").unwrap_or("8080".into()));
    let name = arg(&a, "--alias").unwrap_or_else(|| std::path::Path::new(path).file_stem().unwrap().to_string_lossy().into_owned());
    std::env::set_var("NCOL", nslot.max(5).to_string());
    let t = std::time::Instant::now();
    let m = qwen38::Model::load(path, ctx)?;
    let pf = prefill::Pf::new(&m.ctx)?;
    let q = prefill::qwen35::Q35::new(&m, prefill::env("PF_M", 512))?;
    let slots: Vec<Slot> = (0..nslot).map(|_| m.new_slot()).collect::<Result<_>>()?;
    let parts = m.new_parts()?;
    m.multi_init();
    let g = &m.wt.g;
    let mut tok = kev::tok::Tokenizer::from_gguf(g)?; tok.marks_in_letters = true; // qwen35 pre-tokenizer: [\p{L}\p{M}]+
    let b2u = kev::tok::byte_map(); let u2b: HashMap<char, u8> = b2u.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect();
    let vocab = m.wt.vocab_strings();
    let types: Vec<u64> = match g.get("tokenizer.ggml.token_type") { Ok(gguf::Value::Arr(v)) => v.iter().map(|x| x.as_u64().unwrap_or(1)).collect(), _ => vec![1; vocab.len()] };
    let pieces: Vec<Vec<u8>> = vocab.iter().map(|s| s.chars().map(|c| u2b.get(&c).copied().unwrap_or(b'?')).collect()).collect();
    let special: Vec<bool> = (0..vocab.len()).map(|i| types.get(i) == Some(&3)).collect();
    let mut eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.id(s)).collect();
    if let Ok(e) = g.u64("tokenizer.ggml.eos_token_id") { eos.push(e as u32) }
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    let front = std::sync::Arc::new(Front { name: name.clone(), tok, pieces, special, tmpl: Template::ChatMl, vision: false, jobs: std::sync::Mutex::new(tx), max_ctx: ctx as usize });
    match &ready { Some(r) => { let _ = r.send(front.clone()); } None => listen(&addr, front.clone())? }
    eprintln!("strix-serve: {name} ({} layers, vocab {}), {nslot} slots x ctx {ctx}, prefix cache {ncache}, loaded in {:.1}s; http://{addr}/v1", m.cfg.n_layer, m.cfg.vocab, t.elapsed().as_secs_f64());
    let vocab_n = m.cfg.vocab as usize;
    let toks = unsafe { std::slice::from_raw_parts(m.toks.ptr as *const u32, m.toks.size as usize / 4) };
    let au = m.au();
    let mut free: Vec<usize> = (0..nslot).rev().collect();
    let mut queue: VecDeque<Job> = VecDeque::new(); let mut act: Vec<Seq> = vec![];
    let mut cache: Vec<Ent> = vec![]; let mut tick = 0u64; let mut pre: Option<Pre> = None; let mut pres: Vec<PreS> = vec![]; let (mut npass, mut prows) = (0usize, 0usize);
    let packed = q.bon.is_none() && std::env::var("SERVE_PACKED").map_or(true, |v| v != "0");
    let pack: usize = std::env::var("SERVE_PACK").ok().and_then(|v| v.parse().ok()).unwrap_or(512); // 1024-row passes cost 2.4x a 512-row pass (logs/serve/pf2_q38.log)
    let qt = prefill::qwen35::Q35::new(&m, 256)?; // per-segment scratch (chunk <= 256)
    let pf_passes: usize = std::env::var("SERVE_PF_PASSES").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let chunk: usize = std::env::var("SERVE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256).min(256);
    let mut cbs: HashMap<Vec<usize>, vk::CommandBuffer> = HashMap::new();
    let verbose = std::env::var("SERVE_LOG").is_ok();
    let (mut nsteps, mut ntok, mut tstep) = (0u64, 0u64, 0f64);
    // finish check after emitting token t; returns Some(reason)
    let done = |s: &Seq, t: u32| -> Option<&'static str> {
        if s.job.cancel.load(Relaxed) { return Some("stop") }
        if !s.job.ignore_eos && eos.contains(&t) { return Some("stop") }
        if s.n_out >= s.job.max_tokens || s.pos as usize + 2 >= ctx as usize { return Some("length") }
        None };
    loop {
        for j in recv_all(&rx, act.is_empty() && queue.is_empty() && pre.is_none() && pres.is_empty()) { queue.push_back(j) }
        queue.retain(|j| !j.cancel.load(Relaxed));
        if packed {
            // packed prefill: every waiting prompt gets a slot (up to free slots) and advances by one fixed CH-token chunk per
            // pass; one pass packs chunks of several prompts (<= PACK rows), then one decode step runs for the active sequences
            while !free.is_empty() { let Some(job) = queue.pop_front() else { break };
                tick += 1; let l = job.ids.len(); let sl = free.pop().unwrap();
                let hit = cache.iter_mut().filter(|e| e.ids.len() < l && job.ids.starts_with(&e.ids)).max_by_key(|e| e.ids.len());
                let n = match hit { Some(e) => { e.used = tick; let n = e.ids.len(); m.slot_copy(&e.slot, &slots[sl], n as u32)?; n } None => { m.slot_reset(&slots[sl]); 0 } };
                pres.push(PreS { job, slot: sl, pos: n, cached: n, t: 0.0 }); }
            // prefill priority: up to SERVE_PF_PASSES packed passes per decode step (default 1)
            for _pp in 0..pf_passes { if pres.is_empty() { break }
                let mut segs = vec![]; let mut used = 0usize; let mut who = vec![];
                for (i, p) in pres.iter().enumerate() { let l = p.job.ids.len(); let k = (l - p.pos).min(chunk);
                    if used + k > pack { continue } used += k; who.push((i, k));
                    let sl = &slots[p.slot]; segs.push(prefill::qwen35::multi::Seg { toks: &p.job.ids[p.pos..p.pos + k], pos0: p.pos as u32, st: [&sl.kc, &sl.vc, &sl.cst, &sl.sst], last: p.pos + k == l }); }
                let tp = std::time::Instant::now();
                let r = prefill::qwen35::multi::prefill_pass(&pf, &m, &q, &qt, &segs);
                drop(segs); let dt = tp.elapsed().as_secs_f64(); npass += 1; prows += used;
                match r { Err(e) => { for p in pres.drain(..) { let _ = p.job.tx.send(Ev::Err(format!("prefill: {e}"))); free.push(p.slot); } }
                    Ok(fin) => {
                        for &(i, k) in &who { pres[i].pos += k; pres[i].t += dt; }
                        let mut firsts: HashMap<usize, (u32, Vec<f32>)> = fin.into_iter().map(|(si, t, lg)| (who[si].0, (t, lg))).collect();
                        let mut keep = vec![];
                        for (i, p) in pres.drain(..).enumerate() {
                            let Some((tg, lg)) = firsts.remove(&i) else { keep.push(p); continue };
                            let (job, cached, sl, l) = (p.job, p.cached, p.slot, p.pos);
                            if ncache > 0 && !cache.iter().any(|e| e.ids == job.ids) {
                                if cache.len() < ncache { cache.push(Ent { ids: vec![], slot: m.new_slot()?, used: 0 }) }
                                let e = cache.iter_mut().min_by_key(|e| e.used).unwrap(); m.slot_copy(&slots[sl], &e.slot, l as u32)?; e.ids = job.ids.clone(); e.used = tick; }
                            let mut rng = job.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
                            let t0 = if job.temperature <= 0.0 { tg } else { sample(&lg, job.temperature, job.top_p, &mut rng) };
                            let s = Seq { ttft: job.t0.elapsed().as_secs_f64() * 1e3, job, slot: sl, pos: l as u32, last: t0, n_out: 1, rng, cached };
                            if verbose { eprintln!("admit slot {sl}: prompt {l} (cached {cached}), packed prefill {:.0} ms of passes, ttft {:.0} ms; passes {npass}, mean rows {:.0}", p.t * 1e3, s.ttft, prows as f64 / npass as f64) }
                            let _ = s.job.tx.send(Ev::Tok(t0));
                            match done(&s, t0) { Some(r) => { let _ = s.job.tx.send(Ev::Done { reason: r, prompt: l, completion: 1, cached, ttft_ms: s.ttft }); free.push(sl); }
                                None => act.push(s) }
                        }
                        pres = keep;
                    } }
            }
        } else {
        // chunked prefill interleaved with decode: start the next waiting prompt (prefix-cache aware), then run one chunk of
            // at most SERVE_CHUNK tokens per loop iteration (whole prompt when nothing is decoding), then one decode step
            if pre.is_none() && !free.is_empty() { if let Some(job) = queue.pop_front() {
                tick += 1; let l = job.ids.len();
                let hit = cache.iter_mut().filter(|e| e.ids.len() < l && job.ids.starts_with(&e.ids)).max_by_key(|e| e.ids.len());
                let n = match hit { Some(e) => { e.used = tick; let n = e.ids.len(); m.slot_restore(&e.slot, n as u32)?; n } None => 0 };
                pre = Some(Pre { job, pos: n, cached: n, t: 0.0 });
            } }
            if let Some(mut p) = pre.take() {
                let l = p.job.ids.len(); let end = p.pos + chunk; // fixed chunk boundaries (also when idle): same prefill numerics at any concurrency
                let tp = std::time::Instant::now();
                let r = prefill::qwen35::prefill_part(&pf, &m, &q, &p.job.ids, p.pos, end);
                p.t += tp.elapsed().as_secs_f64(); p.pos = end.min(l);
                match r { Err(e) => { let _ = p.job.tx.send(Ev::Err(format!("prefill: {e}"))); }
                    Ok(false) => pre = Some(p),
                    Ok(true) => {
                        let (job, cached) = (p.job, p.cached); let tc = std::time::Instant::now();
                        let sl = free.pop().unwrap();
                        m.slot_load(&slots[sl], l as u32)?;
                        if ncache > 0 && !cache.iter().any(|e| e.ids == job.ids) {
                            if cache.len() < ncache { cache.push(Ent { ids: vec![], slot: m.new_slot()?, used: 0 }) }
                            let e = cache.iter_mut().min_by_key(|e| e.used).unwrap(); m.slot_load(&e.slot, l as u32)?; e.ids = job.ids.clone(); e.used = tick; }
                        let mut rng = job.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
                        let t0 = if job.temperature <= 0.0 { toks[l] } else { sample(&m.logits.as_f32()[..vocab_n], job.temperature, job.top_p, &mut rng) };
                        let s = Seq { ttft: job.t0.elapsed().as_secs_f64() * 1e3, job, slot: sl, pos: l as u32, last: t0, n_out: 1, rng, cached };
                        if verbose { eprintln!("admit slot {sl}: prompt {l} (cached {cached}), prefill GPU-path {:.0} ms, copies {:.0} ms, ttft {:.0} ms", p.t * 1e3, tc.elapsed().as_secs_f64() * 1e3, s.ttft) }
                        let _ = s.job.tx.send(Ev::Tok(t0));
                        match done(&s, t0) { Some(r) => { let _ = s.job.tx.send(Ev::Done { reason: r, prompt: l, completion: 1, cached, ttft_ms: s.ttft }); free.push(sl); }
                            None => act.push(s) }
                    } }
            }
        }
        if act.is_empty() { continue }
        act.sort_by_key(|s| s.slot);
        let key: Vec<usize> = act.iter().map(|s| s.slot).collect();
        if !cbs.contains_key(&key) {
            if cbs.len() >= 300 { cbs.clear(); } // (command buffers stay allocated; bounded by the subsets seen)
            let sl: Vec<&Slot> = key.iter().map(|&i| &slots[i]).collect();
            let mut err = Ok(()); let cb = m.ctx.record_raw(|cb| { err = m.record_multi(cb, &sl, &parts); })?; err?;
            cbs.insert(key.clone(), cb);
        }
        for (j, s) in act.iter().enumerate() { au[POSW as usize + j] = s.pos; au[TOKW as usize + j] = s.last; }
        let ts = std::time::Instant::now();
        m.ctx.submit_repeated(cbs[&key], 1)?;
        tstep += ts.elapsed().as_secs_f64(); nsteps += 1; ntok += act.len() as u64;
        let lg = m.logits.as_f32();
        let mut keep = vec![];
        for (j, mut s) in act.drain(..).enumerate() {
            let t = if s.job.temperature <= 0.0 { au[600 + j] } else { sample(&lg[j * vocab_n..(j + 1) * vocab_n], s.job.temperature, s.job.top_p, &mut s.rng) };
            s.pos += 1; s.last = t; s.n_out += 1;
            let _ = s.job.tx.send(Ev::Tok(t));
            match done(&s, t) { Some(r) => { let _ = s.job.tx.send(Ev::Done { reason: r, prompt: s.job.ids.len(), completion: s.n_out, cached: s.cached, ttft_ms: s.ttft }); free.push(s.slot); }
                None => keep.push(s) }
        }
        act = keep;
        if verbose && nsteps % 64 == 0 { eprintln!("decode: {nsteps} steps, {ntok} tokens, {:.1} ms/step, batch {:.2}", tstep / nsteps as f64 * 1e3, ntok as f64 / nsteps as f64) }
    }
}
