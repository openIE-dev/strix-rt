//! kev_serve <merged.gguf> <head.bin> [addr=127.0.0.1:8009] [model=kev-0.8b]: TypeSafe System One server on strix-rt.
//! POST /v1/systemone {state, model?, questions{id: {type noul|choice|score, instructions, criteria}}} -> {model, answers,
//! usage{input_tokens, output_tokens}, latency_ms}; GET /v1/models. KEV_CHECK=golden.json: check the tokenizer/encoder
//! against kev's PyTorch encodings and exit. LAYA=laya-f16.gguf (LAYA_TOK=tokenizer.json, LAYA_CFG=rl_agent_config.json,
//! LAYA_NAME=laya) also serves Laya on the same endpoint, picked by the request's "model"; pass "-" as <merged.gguf> for Laya only.
//! /v1/decisions also returns a signed JouleContract-style slice receipt (crates/receipt; docs/infinity-suspend-resume.md):
//! request may carry thread_id and input_id (stripped before content hashing). compute_j = PPT power (amdgpu hwmon, or
//! KEV_POWER=<uW file>) integrated over the request, gross, ModelBased; wait_j = 0 (stateless, nothing resident), Estimator.
//! KEV_RECEIPT_KEY=<32-byte or hex seed file> (else an ephemeral key per run); KEV_RECEIPT_LEDGER=<tsv> persists the
//! per-thread seq/prev-hash chain (default ~/.local/state/strix-rt/receipts.tsv; "-" = in memory only).
use anyhow::{anyhow, Result};
use kev::{decisions, json::{self, J}, systemone, Kev};
use std::io::{BufRead, BufReader, Read, Write};
use receipt::{ledger::{self, Ledger}, meter::{self, Meter}, Energy, Provenance, Tier};

struct Rc { ledger: Ledger, power: Option<String> }

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let addr = a.get(3).cloned().unwrap_or("127.0.0.1:8009".into()); let model = a.get(4).cloned().unwrap_or("kev-0.8b".into());
    let max_ctx = prefill::env("KEV_MAX_CTX", 16384);
    let ly = match std::env::var("LAYA") { Ok(g) => { let d = std::env::var("HOME").unwrap_or_default() + "/models/decision/laya";
        Some((kev::laya_s1::LayaS1::load(&g, &std::env::var("LAYA_TOK").unwrap_or(d.clone() + "/tokenizer/tokenizer.json"), &std::env::var("LAYA_CFG").unwrap_or(d + "/rl_agent_config.json"))?,
            std::env::var("LAYA_NAME").unwrap_or("laya".into()))) } Err(_) => None };
    let mut kk = if a[1] == "-" { None } else { let kv = Kev::load(&a[1], &a[2], max_ctx, prefill::env("KEV_CHUNK", 512))?; let tk = kev::tok::Tokenizer::from_gguf(&kv.m.wt.g)?; Some((kv, tk)) };
    let model = if kk.is_none() { ly.as_ref().map(|l| l.1.clone()).unwrap_or(model) } else { model };
    if let (Ok(p), Some((kv, tk))) = (std::env::var("KEV_CHECK"), kk.as_mut()) { let _ = &kv;
        let g = json::parse(&std::fs::read_to_string(p)?)?; let (mut ok, mut n) = (0, 0);
        if let Some(J::Arr(recs)) = g.get("records") { for r in recs {
            let rec = r.get("rec").unwrap(); let state = rec.get("state").and_then(|s| s.as_str()).unwrap();
            let qs: Vec<(String, Vec<String>)> = match rec.get("questions") { Some(J::Arr(v)) => v.iter().map(|q| (q.get("instr").and_then(|s| s.as_str()).unwrap().to_string(),
                match q.get("options") { Some(J::Arr(o)) => o.iter().map(|x| x.as_str().unwrap().to_string()).collect(), _ => vec![] })).collect(), _ => vec![] };
            let (st, brs) = kev::encode(&tk, state, &qs)?;
            if let Some(J::Arr(gq)) = r.get("questions") { for (b, q) in brs.iter().zip(gq) {
                let want: Vec<u32> = match q.get("ids") { Some(J::Arr(v)) => v.iter().map(|x| match x { J::Num(t) => t.parse().unwrap(), _ => 0 }).collect(), _ => vec![] };
                let mut got = st.clone(); got.extend(&b.ids); n += 1; if got == want { ok += 1 } else { println!("MISMATCH\n got  {got:?}\n want {want:?}") } } }
        } }
        println!("encoder check: {ok}/{n} question rows identical to kev.model.encode"); return Ok(());
    }
    let key = match std::env::var("KEV_RECEIPT_KEY") { Ok(p) => ledger::load_key(&p)?, Err(_) => ledger::ephemeral_key()? };
    let lp = match std::env::var("KEV_RECEIPT_LEDGER") { Ok(p) if p == "-" => None, Ok(p) => Some(p.into()),
        Err(_) => Some(std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/state/strix-rt/receipts.tsv")) };
    let mut rc = Rc { ledger: Ledger::open(lp, key)?, power: meter::detect() };
    eprintln!("kev_serve: receipts signed by {} (power: {})", rc.ledger.pubkey_hex(), rc.power.as_deref().unwrap_or("none, compute_j = null"));
    let ln = std::net::TcpListener::bind(&addr)?; eprintln!("kev_serve: {}{} on http://{addr}/v1/systemone", if kk.is_some() { model.as_str() } else { "" }, ly.as_ref().map(|l| format!(" {}", l.1)).unwrap_or_default());
    for s in ln.incoming() {
        let mut s = match s { Ok(s) => s, Err(_) => continue };
        let (code, body) = match handle(&mut s, kk.as_mut(), ly.as_ref(), &model, &mut rc) { Ok(x) => x, Err(e) => (422, J::obj(vec![("detail", J::Str(e.to_string()))]).dump()) };
        let reason = match code { 200 => "OK", 404 => "Not Found", _ => "Unprocessable Entity" };
        let _ = write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    }
    Ok(())
}

fn handle(s: &mut std::net::TcpStream, kk: Option<&mut (Kev, kev::tok::Tokenizer)>, ly: Option<&(kev::laya_s1::LayaS1, String)>, model: &str, rc: &mut Rc) -> Result<(u16, String)> {
    let mut r = BufReader::new(s.try_clone()?); let mut line = String::new(); r.read_line(&mut line)?;
    let mut parts = line.split_whitespace(); let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let mut len = 0usize;
    loop { let mut h = String::new(); r.read_line(&mut h)?; let h = h.trim_end(); if h.is_empty() { break }
        if let Some((k, v)) = h.split_once(':') { if k.eq_ignore_ascii_case("content-length") { len = v.trim().parse()? } } }
    let mut body = vec![0u8; len]; r.read_exact(&mut body)?;
    let path = path.split('?').next().unwrap_or("");
    match (method, path) {
        ("GET", "/v1/models") => { let mut v = vec![]; if kk.is_some() { v.push(J::obj(vec![("name", J::Str(model.into()))])) } if let Some(l) = ly { v.push(J::obj(vec![("name", J::Str(l.1.clone()))])) }
            Ok((200, J::obj(vec![("models", J::Arr(v))]).dump())) }
        ("GET", "/healthz") => Ok((200, "{\"ok\":true}".into())),
        ("POST", "/v1/systemone") => { let req = json::parse(std::str::from_utf8(&body)?)?; systemone_resp(&req, kk, ly, model) }
        ("POST", "/v1/decisions") => {
            // OpenAI Decisions-compatible: {model, input, questions:[{type predicate|choice|score, name?, instructions, choices|levels}]}
            // -> System One request (predicate = noul) -> {id, object:"decision", model, answers:[...in question order], usage}
            let t0 = std::time::Instant::now(); let req = json::parse(std::str::from_utf8(&body)?)?;
            let ids = |k: &str| req.get(k).and_then(|x| x.as_str()).map(|s| s.to_string());
            let (thread, input_id) = (ids("thread_id").unwrap_or("default".into()), ids("input_id"));
            let canon = match &req { J::Obj(v) => J::Obj(v.iter().filter(|(k, _)| k != "thread_id" && k != "input_id").cloned().collect()), o => o.clone() }.dump();
            let (s1, meta) = decisions::to_systemone(&req)?;
            let m = rc.power.as_deref().and_then(|p| Meter::start(p, std::time::Duration::from_millis(5)));
            let res = systemone_resp(&s1, kk, ly, model);
            let (j, _, _) = m.map(|m| m.stop()).unwrap_or((f64::NAN, 0.0, 0));
            let (code, b) = res?; if code != 200 { return Ok((code, b)) }
            let r = json::parse(&b)?; let mut out = decisions::from_systemone(&r, &meta, &req, t0)?;
            let name = r.get("model").and_then(|x| x.as_str()).unwrap_or(model).to_string();
            let nq = meta.kinds.len().min(u16::MAX as usize) as u16;
            let tier = if name.to_ascii_lowercase().contains("9b") { Tier::Kev9b { batch: nq } } else { Tier::Small { batch: nq } };
            let tok = |k: &str| match r.get("usage").and_then(|u| u.get(k)) { Some(J::Num(n)) => n.parse::<f64>().unwrap_or(0.0) as u32, _ => 0 };
            let rid = match out.get("id") { Some(J::Str(s)) => s.clone(), _ => String::new() };
            let sr = rc.ledger.record(&thread, input_id.as_deref().unwrap_or(&rid), receipt::decision_hash(canon.as_bytes(), &name, tier), tier,
                tok("input_tokens"), tok("output_tokens"), Energy { joules: j, provenance: Provenance::ModelBased },
                Energy { joules: 0.0, provenance: Provenance::Estimator }, None)?;
            if let J::Obj(v) = &mut out { v.push(("receipt".into(), json::parse(&ledger::to_json(&sr))?)) }
            Ok((200, out.dump()))
        }
        _ => Ok((404, J::obj(vec![("detail", J::Str("not found".into()))]).dump())),
    }
}

fn systemone_resp(req: &J, kk: Option<&mut (Kev, kev::tok::Tokenizer)>, ly: Option<&(kev::laya_s1::LayaS1, String)>, model: &str) -> Result<(u16, String)> {
        {
            let t0 = std::time::Instant::now();
            let want = req.get("model").and_then(|m| m.as_str());
            if let Some((l, name)) = ly { if want == Some(name.as_str()) || (kk.is_none() && want.map_or(true, |w| w == model)) {
                let (answers, usage, _gpu) = l.system_one(req)?;
                return Ok((200, J::obj(vec![("model", J::Str(want.unwrap_or(name).to_string())), ("answers", answers), ("usage", usage), ("latency_ms", J::num((t0.elapsed().as_secs_f64() * 1e4).round() / 10.0))]).dump())) } }
            let Some((kv, tk)) = kk else { anyhow::bail!("unknown model {want:?}") };
            let (state, qs) = systemone::to_record(req)?;
            let (st, brs) = kev::encode(tk, &state, &qs.iter().map(|q| (q.instr.clone(), q.options.clone())).collect::<Vec<_>>())?;
            if st.len() + brs.iter().map(|b| b.ids.len()).max().unwrap_or(0) + 1 > kv.m.cfg.max_ctx as usize { return Err(anyhow!("state is {} tokens, over this server's context", st.len())) }
            let want_h = matches!(req.get("hidden"), Some(J::Bool(true)));
            if want_h { *kv.hsink.borrow_mut() = Some(vec![]) }
            let probs = kv.decide_all(&st, &brs); let hs = kv.hsink.borrow_mut().take(); let probs = probs?;
            let answers = systemone::to_answers(&probs, &qs);
            let out_toks = tk.encode_plain(&answers.dump()).len();
            let input = st.len() + brs.iter().map(|b| b.ids.len()).sum::<usize>();
            let name = req.get("model").and_then(|m| m.as_str()).unwrap_or(model).to_string();
            Ok((200, J::obj(vec![("model", J::Str(name)), ("answers", answers), ("usage", J::obj(vec![("input_tokens", J::num(input as f64)), ("output_tokens", J::num(out_toks as f64))])),
                ("latency_ms", J::num((t0.elapsed().as_secs_f64() * 1e4).round() / 10.0))]).dump())).map(|(c, b)| match hs { Some(hs) if want_h => (c, format!("{},\"hidden\":{}}}", &b[..b.len() - 1], J::Obj(qs.iter().zip(hs).map(|(q, h)| (q.id.clone(), J::Arr(h.iter().map(|v| J::num(*v as f64)).collect()))).collect()).dump())), _ => (c, b) })
        }
}
