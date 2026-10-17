//! strix-serve front end: minimal HTTP/1.1 + OpenAI-compatible request/response shapes (model independent).
//! The engine (one GPU thread) receives `Job`s over a channel and streams `Ev`s back to the connection thread.
use anyhow::{anyhow, bail, Result};
use kev::json::{self, J};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, Sender};

pub enum Ev { Tok(u32), Done { reason: &'static str, prompt: usize, completion: usize, cached: usize, ttft_ms: f64 }, Err(String) }
/// decoded RGB8 image from an image_url content part
pub struct Img { pub rgb: Vec<u8>, pub w: usize, pub h: usize }
/// image sentinel in Job::ids: IMG_TOK marks where the i-th image of Job::images goes
pub const IMG_TOK: u32 = u32::MAX;
pub struct Job { pub ids: Vec<u32>, pub images: Vec<Img>, pub max_tokens: usize, pub ignore_eos: bool, pub temperature: f32, pub top_p: f32, pub seed: u64, pub tx: Sender<Ev>,
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>, pub t0: std::time::Instant }

/// What the connection threads need: tokenizer, detokenizer table, template, model name, job queue.
pub struct Front { pub name: String, pub tok: kev::tok::Tokenizer, pub pieces: Vec<Vec<u8>>, pub special: Vec<bool>, pub tmpl: Template, pub vision: bool, pub jobs: std::sync::Mutex<Sender<Job>>, pub max_ctx: usize }

#[derive(Clone, Copy, PartialEq)] pub enum Template { ChatMl, Lfm2 }
impl Template {
    /// messages (role, text) + generation prompt. ChatML follows the Qwen3 template (enable_thinking=false adds the
    /// empty <think> block); LFM2 = <|startoftext|> + ChatML turns.
    pub fn render(&self, msgs: &[(String, String)], think: bool) -> String {
        let mut s = String::new();
        if *self == Template::Lfm2 { s.push_str("<|startoftext|>"); }
        for (r, c) in msgs { s.push_str(&format!("<|im_start|>{r}\n{c}<|im_end|>\n")); }
        s.push_str("<|im_start|>assistant\n");
        if *self == Template::ChatMl && !think { s.push_str("<think>\n\n</think>\n\n"); }
        s
    }
}

pub struct Req { pub method: String, pub path: String, pub body: Vec<u8> }
pub fn read_req(s: &TcpStream) -> Result<Req> {
    let mut r = BufReader::new(s.try_clone()?); let mut line = String::new(); r.read_line(&mut line)?;
    let mut p = line.split_whitespace(); let (method, path) = (p.next().unwrap_or("").to_string(), p.next().unwrap_or("").split('?').next().unwrap_or("").to_string());
    let mut len = 0usize;
    loop { let mut h = String::new(); if r.read_line(&mut h)? == 0 { break } let h = h.trim_end(); if h.is_empty() { break }
        if let Some((k, v)) = h.split_once(':') { if k.eq_ignore_ascii_case("content-length") { len = v.trim().parse()? } } }
    if len > 64 << 20 { bail!("body too large") }
    let mut body = vec![0u8; len]; r.read_exact(&mut body)?;
    Ok(Req { method, path, body })
}
pub fn respond(s: &mut TcpStream, code: u16, body: &str) {
    let reason = match code { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", _ => "Error" };
    let _ = write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}", body.len());
}
fn err_json(msg: &str) -> String { J::obj(vec![("error", J::obj(vec![("message", J::Str(msg.into())), ("type", J::Str("invalid_request_error".into()))]))]).dump() }
fn now() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as f64).unwrap_or(0.0) }
fn num(j: Option<&J>) -> Option<f64> { match j { Some(J::Num(n)) => n.parse().ok(), _ => None } }
fn boolv(j: Option<&J>) -> Option<bool> { match j { Some(J::Bool(b)) => Some(*b), _ => None } }

/// incremental UTF-8: emit the longest valid prefix, keep an incomplete tail
pub struct Utf8 { buf: Vec<u8> }
impl Utf8 { pub fn new() -> Self { Utf8 { buf: vec![] } }
    pub fn push(&mut self, b: &[u8]) -> String { self.buf.extend_from_slice(b);
        let n = match std::str::from_utf8(&self.buf) { Ok(_) => self.buf.len(), Err(e) => if e.error_len().is_some() { self.buf.len() } else { e.valid_up_to() } };
        let s = String::from_utf8_lossy(&self.buf[..n]).into_owned(); self.buf.drain(..n); s }
    pub fn flush(&mut self) -> String { let s = String::from_utf8_lossy(&self.buf).into_owned(); self.buf.clear(); s } }

/// One connection for a server with several models: completions go to the Front whose name equals the request's
/// "model" field (the first model when absent or unknown is rejected with 404 if several are served); /v1/models lists all.
pub fn handle_many(fs: &[std::sync::Arc<Front>], mut s: TcpStream, id: u64) {
    let _ = s.set_nodelay(true);
    let r = match read_req(&s) { Ok(r) => r, Err(e) => return respond(&mut s, 400, &err_json(&e.to_string())) };
    let pick = || -> Option<&Front> {
        let want = json::parse(&String::from_utf8_lossy(r.body.as_ref())).ok().and_then(|j| j.get("model").and_then(|m| m.as_str().map(|x| x.to_string())));
        match want { Some(w) => fs.iter().find(|f| f.name == w).map(|f| &**f).or(if fs.len() == 1 { Some(&*fs[0]) } else { None }), None => Some(&*fs[0]) } };
    match (r.method.as_str(), r.path.as_str()) {
        ("GET", "/v1/models") | ("GET", "/models") => respond(&mut s, 200, &J::obj(vec![("object", J::Str("list".into())), ("data", J::Arr(fs.iter().map(|f| J::obj(vec![("id", J::Str(f.name.clone())),
            ("object", J::Str("model".into())), ("created", J::num(0.0)), ("owned_by", J::Str("strix-rt".into())), ("meta", J::obj(vec![("n_ctx", J::num(f.max_ctx as f64))]))])).collect()))]).dump()),
        ("GET", "/health") | ("GET", "/healthz") => respond(&mut s, 200, "{\"status\":\"ok\"}"),
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
            let chat = r.path.contains("chat");
            let Some(f) = pick() else { return respond(&mut s, 404, &err_json(&format!("unknown model; served: {}", fs.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", ")))) };
            if let Err(e) = complete(f, &mut s, &r.body, chat, id) { respond(&mut s, 400, &err_json(&e.to_string())) }
        }
        ("OPTIONS", _) => { let _ = write!(s, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n"); }
        _ => respond(&mut s, 404, &err_json("not found")),
    }
}

const MARK: char = '\u{E000}';
fn decode_image(url: &str) -> Result<Img> {
    use base64::Engine;
    let b64 = url.strip_prefix("data:").and_then(|r| r.split_once(";base64,").map(|x| x.1)).ok_or_else(|| anyhow!("image_url must be a data:<mime>;base64, URL"))?;
    let raw = base64::engine::general_purpose::STANDARD.decode(b64.trim())?;
    let im = image::load_from_memory(&raw)?.to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize);
    Ok(Img { rgb: im.into_raw(), w, h })
}
fn text_of(c: &J, imgs: &mut Option<&mut Vec<Img>>) -> Result<String> {
    match c { J::Str(t) => Ok(t.replace(MARK, "")), J::Null => Ok(String::new()),
        J::Arr(parts) => { let mut o = String::new(); for p in parts { match p.get("type").and_then(|t| t.as_str()) {
            Some("text") => o.push_str(&p.get("text").and_then(|t| t.as_str()).unwrap_or("").replace(MARK, "")),
            Some("image_url") => { let Some(v) = imgs.as_mut() else { bail!("this model does not take image input") };
                let u = p.get("image_url").and_then(|u| u.get("url").or(Some(u))).and_then(|u| u.as_str()).ok_or_else(|| anyhow!("image_url.url required"))?;
                let im = decode_image(u)?; // MiniCPM-V layout: <image_id>k</image_id><image> [tile rows incl. </image>] \n
                o.push_str(&format!("<image_id>{}</image_id><image>{MARK}\n", v.len())); v.push(im); }
            Some(t) => bail!("content part type {t:?} is not supported"), None => bail!("content part without type") } } Ok(o) }
        _ => bail!("bad message content") }
}

fn complete(f: &Front, s: &mut TcpStream, body: &[u8], chat: bool, id: u64) -> Result<()> {
    let t0 = std::time::Instant::now();
    let q = json::parse(std::str::from_utf8(body)?)?;
    let mut images: Vec<Img> = vec![];
    let prompt = if chat {
        let Some(J::Arr(ms)) = q.get("messages") else { bail!("messages required") };
        let mut v = vec![]; for m in ms { let mut im = if f.vision { Some(&mut images) } else { None }; v.push((m.get("role").and_then(|r| r.as_str()).ok_or_else(|| anyhow!("message role"))?.to_string(), text_of(m.get("content").unwrap_or(&J::Null), &mut im)?)); }
        let think = q.get("chat_template_kwargs").and_then(|k| boolv(k.get("enable_thinking"))).unwrap_or(true);
        f.tmpl.render(&v, think)
    } else { match q.get("prompt") { Some(J::Str(p)) => p.clone(), Some(J::Arr(a)) if a.len() == 1 => a[0].as_str().unwrap_or("").to_string(), _ => bail!("prompt (string) required") } };
    let mut ids = vec![]; for (k, piece) in prompt.split(MARK).enumerate() { if k > 0 { ids.push(IMG_TOK) } ids.extend(f.tok.encode(piece)); }
    if f.tmpl == Template::Lfm2 && !chat { if let Some(b) = f.tok.id("<|startoftext|>") { if ids.first() != Some(&b) { ids.insert(0, b) } } }
    if ids.is_empty() { bail!("empty prompt") }
    let max_tokens = num(q.get("max_completion_tokens")).or(num(q.get("max_tokens"))).unwrap_or(256.0).max(1.0) as usize;
    if ids.len() + max_tokens + 1 > f.max_ctx { bail!("prompt {} + max_tokens {max_tokens} exceeds the context {}", ids.len(), f.max_ctx) }
    let stream = boolv(q.get("stream")).unwrap_or(false);
    let usage_in_stream = q.get("stream_options").and_then(|o| boolv(o.get("include_usage"))).unwrap_or(false);
    let stops: Vec<String> = match q.get("stop") { Some(J::Str(x)) => vec![x.clone()], Some(J::Arr(a)) => a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(), _ => vec![] };
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let job = Job { ids, images, max_tokens, ignore_eos: boolv(q.get("ignore_eos")).unwrap_or(false), temperature: num(q.get("temperature")).unwrap_or(1.0) as f32,
        top_p: num(q.get("top_p")).unwrap_or(1.0) as f32, seed: num(q.get("seed")).unwrap_or(id as f64) as u64, tx, cancel: cancel.clone(), t0 };
    f.jobs.lock().unwrap().send(job).map_err(|_| anyhow!("engine stopped"))?;
    let rid = format!("{}-{id}", if chat { "chatcmpl" } else { "cmpl" });
    let obj = if chat { if stream { "chat.completion.chunk" } else { "chat.completion" } } else { "text_completion" };
    let model = f.name.clone(); let created = now();
    let choice = |text: Option<String>, fin: Option<&str>, first: bool| -> J {
        let fr = fin.map(|x| J::Str(x.into())).unwrap_or(J::Null);
        if chat { if stream { let mut d = vec![]; if first { d.push(("role", J::Str("assistant".into()))) } if let Some(t) = text { d.push(("content", J::Str(t))) }
                J::obj(vec![("index", J::num(0.0)), ("delta", J::obj(d)), ("finish_reason", fr)]) }
            else { J::obj(vec![("index", J::num(0.0)), ("message", J::obj(vec![("role", J::Str("assistant".into())), ("content", J::Str(text.unwrap_or_default()))])), ("finish_reason", fr)]) } }
        else { J::obj(vec![("index", J::num(0.0)), ("text", J::Str(text.unwrap_or_default())), ("logprobs", J::Null), ("finish_reason", fr)]) } };
    let envl = |choices: Vec<J>, usage: Option<J>| -> String { let mut v = vec![("id", J::Str(rid.clone())), ("object", J::Str(obj.into())), ("created", J::num(created)), ("model", J::Str(model.clone())), ("choices", J::Arr(choices))];
        if let Some(u) = usage { v.push(("usage", u)) } J::obj(v).dump() };
    let usage = |p: usize, c: usize, cached: usize, ttft: f64| J::obj(vec![("prompt_tokens", J::num(p as f64)), ("completion_tokens", J::num(c as f64)), ("total_tokens", J::num((p + c) as f64)),
        ("prompt_tokens_details", J::obj(vec![("cached_tokens", J::num(cached as f64))])), ("ttft_ms", J::num((ttft * 10.0).round() / 10.0))]);
    let mut u8s = Utf8::new(); let mut text = String::new(); let mut sent = 0usize; let mut first = true; let mut stopped = false;
    if stream { s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n")?; }
    let send = |s: &mut TcpStream, d: String| -> bool { let ok = s.write_all(format!("data: {d}\n\n").as_bytes()).and_then(|_| s.flush()).is_ok(); if !ok { cancel.store(true, Relaxed) } ok };
    let hold = stops.iter().map(|x| x.len().saturating_sub(1)).max().unwrap_or(0);
    let mut fin = None;
    for ev in rx.iter() {
        match ev {
            Ev::Tok(t) => { if stopped { continue }
                if !f.special.get(t as usize).copied().unwrap_or(true) { text.push_str(&u8s.push(&f.pieces[t as usize])); }
                // stop strings: cut at the first match and cancel the sequence in the engine
                let from = { let mut b = sent.saturating_sub(hold); while !text.is_char_boundary(b) { b -= 1 } b };
                if let Some(p) = stops.iter().filter(|x| !x.is_empty()).filter_map(|x| text[from..].find(x.as_str()).map(|i| i + from)).min() { text.truncate(p); stopped = true; cancel.store(true, Relaxed); }
                let mut upto = if stopped { text.len() } else { text.len().saturating_sub(hold) }; while !text.is_char_boundary(upto) { upto -= 1 }
                if stream && upto > sent { let d = envl(vec![choice(Some(text[sent..upto].to_string()), None, first)], None); first = false; sent = upto; if !send(s, d) { return Ok(()) } } }
            Ev::Done { reason, prompt, completion, cached, ttft_ms } => { fin = Some((if stopped { "stop" } else { reason }, prompt, completion, cached, ttft_ms)); break }
            Ev::Err(e) => bail!("{e}"),
        }
    }
    let (reason, p, c, cached, ttft) = fin.ok_or_else(|| anyhow!("engine dropped the request"))?;
    if !stopped { text.push_str(&u8s.flush()); }
    if stream {
        if text.len() > sent { let d = envl(vec![choice(Some(text[sent..].to_string()), None, first)], None); send(s, d); }
        send(s, envl(vec![choice(None, Some(reason), false)], None));
        if usage_in_stream { send(s, envl(vec![], Some(usage(p, c, cached, ttft)))); }
        let _ = s.write_all(b"data: [DONE]\n\n");
    } else { respond(s, 200, &envl(vec![choice(Some(text), Some(reason), true)], Some(usage(p, c, cached, ttft)))); }
    Ok(())
}

/// spawn the acceptor; each connection gets its own thread
pub fn listen(addr: &str, f: std::sync::Arc<Front>) -> Result<()> { listen_many(addr, vec![f]) }
/// acceptor for several models (requests routed by "model", see handle_many)
pub fn listen_many(addr: &str, fs: Vec<std::sync::Arc<Front>>) -> Result<()> {
    let ln = std::net::TcpListener::bind(addr)?; let fs = std::sync::Arc::new(fs);
    std::thread::spawn(move || { let mut id = 0u64; for s in ln.incoming() { let Ok(s) = s else { continue }; id += 1; let fs = fs.clone(); std::thread::spawn(move || handle_many(&fs, s, id)); } });
    Ok(())
}
pub fn recv_all(rx: &Receiver<Job>, block: bool) -> Vec<Job> {
    let mut v = vec![]; if block { if let Ok(j) = rx.recv() { v.push(j) } } while let Ok(j) = rx.try_recv() { v.push(j) } v
}

/// temperature / top-p sampling on host logits (temperature <= 0: argmax)
pub fn sample(lg: &[f32], temp: f32, top_p: f32, rng: &mut u64) -> u32 {
    let am = lg.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    if temp <= 0.0 { return am.0 as u32 }
    let mut idx: Vec<usize> = (0..lg.len()).collect();
    let k = 256.min(lg.len()); idx.select_nth_unstable_by(k - 1, |a, b| lg[*b].total_cmp(&lg[*a])); idx.truncate(k);
    idx.sort_by(|a, b| lg[*b].total_cmp(&lg[*a]));
    let p: Vec<f64> = idx.iter().map(|&i| (((lg[i] - am.1) / temp) as f64).exp()).collect();
    let tot: f64 = p.iter().sum(); let mut cut = idx.len(); let mut acc = 0.0;
    for (n, v) in p.iter().enumerate() { acc += v / tot; if acc >= top_p as f64 { cut = n + 1; break } }
    let tot: f64 = p[..cut].iter().sum();
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    let mut r = (*rng >> 11) as f64 / (1u64 << 53) as f64 * tot;
    for n in 0..cut { r -= p[n]; if r <= 0.0 { return idx[n] as u32 } }
    idx[cut - 1] as u32
}
