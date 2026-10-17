//! kev_serve <merged.gguf> <head.bin> [addr=127.0.0.1:8009] [model=kev-0.8b]: TypeSafe System One server on strix-rt.
//! POST /v1/systemone {state, model?, questions{id: {type noul|choice|score, instructions, criteria}}} -> {model, answers,
//! usage{input_tokens, output_tokens}, latency_ms}; GET /v1/models. KEV_CHECK=golden.json: check the tokenizer/encoder
//! against kev's PyTorch encodings and exit. LAYA=laya-f16.gguf (LAYA_TOK=tokenizer.json, LAYA_CFG=rl_agent_config.json,
//! LAYA_NAME=laya) also serves Laya on the same endpoint, picked by the request's "model"; pass "-" as <merged.gguf> for Laya only.
use anyhow::{anyhow, Result};
use kev::{json::{self, J}, systemone, Kev};
use std::io::{BufRead, BufReader, Read, Write};

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
    let ln = std::net::TcpListener::bind(&addr)?; eprintln!("kev_serve: {}{} on http://{addr}/v1/systemone", if kk.is_some() { model.as_str() } else { "" }, ly.as_ref().map(|l| format!(" {}", l.1)).unwrap_or_default());
    for s in ln.incoming() {
        let mut s = match s { Ok(s) => s, Err(_) => continue };
        let (code, body) = match handle(&mut s, kk.as_mut(), ly.as_ref(), &model) { Ok(x) => x, Err(e) => (422, J::obj(vec![("detail", J::Str(e.to_string()))]).dump()) };
        let reason = match code { 200 => "OK", 404 => "Not Found", _ => "Unprocessable Entity" };
        let _ = write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    }
    Ok(())
}

fn handle(s: &mut std::net::TcpStream, kk: Option<&mut (Kev, kev::tok::Tokenizer)>, ly: Option<&(kev::laya_s1::LayaS1, String)>, model: &str) -> Result<(u16, String)> {
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
        ("POST", "/v1/systemone") => {
            let t0 = std::time::Instant::now();
            let req = json::parse(std::str::from_utf8(&body)?)?;
            let want = req.get("model").and_then(|m| m.as_str());
            if let Some((l, name)) = ly { if want == Some(name.as_str()) || (kk.is_none() && want.map_or(true, |w| w == model)) {
                let (answers, usage, _gpu) = l.system_one(&req)?;
                return Ok((200, J::obj(vec![("model", J::Str(want.unwrap_or(name).to_string())), ("answers", answers), ("usage", usage), ("latency_ms", J::num((t0.elapsed().as_secs_f64() * 1e4).round() / 10.0))]).dump())) } }
            let Some((kv, tk)) = kk else { anyhow::bail!("unknown model {want:?}") };
            let (state, qs) = systemone::to_record(&req)?;
            let (st, brs) = kev::encode(tk, &state, &qs.iter().map(|q| (q.instr.clone(), q.options.clone())).collect::<Vec<_>>())?;
            if st.len() + brs.iter().map(|b| b.ids.len()).max().unwrap_or(0) + 1 > kv.m.cfg.max_ctx as usize { return Err(anyhow!("state is {} tokens, over this server's context", st.len())) }
            let probs = kv.decide_all(&st, &brs)?;
            let answers = systemone::to_answers(&probs, &qs);
            let out_toks = tk.encode_plain(&answers.dump()).len();
            let input = st.len() + brs.iter().map(|b| b.ids.len()).sum::<usize>();
            let name = req.get("model").and_then(|m| m.as_str()).unwrap_or(model).to_string();
            Ok((200, J::obj(vec![("model", J::Str(name)), ("answers", answers), ("usage", J::obj(vec![("input_tokens", J::num(input as f64)), ("output_tokens", J::num(out_toks as f64))])),
                ("latency_ms", J::num((t0.elapsed().as_secs_f64() * 1e4).round() / 10.0))]).dump()))
        }
        _ => Ok((404, J::obj(vec![("detail", J::Str("not found".into()))]).dump())),
    }
}
