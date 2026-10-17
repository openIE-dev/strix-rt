//! Laya behind the System One API: a port of laya's build_head/build_sequence (sequential option layout, one
//! <= max_len window, as every published checkpoint uses), render_options, per-bucket temperatures (clamped to
//! [0.5, 5] like clamp_temperature) and _decode_answers. Answers keep Laya's own shape (a superset of TypeSafe's).
use crate::{hftok::HfTok, json::{self, J}};
use anyhow::{anyhow, bail, Result};

pub struct LayaS1 { pub m: laya::Laya, pub tok: HfTok, pub temp: [f32; 3], pub tbo: Vec<(String, f32)>, pub max_len: usize, pub head_max: usize, cls: u32, sep: u32, mask: u32 }
pub struct Lq { pub id: String, pub t: &'static str, pub ins: String, pub opts: Vec<String>, pub keys: Vec<String>, pub legend: Vec<String> }
pub struct Row { pub ids: Vec<u32>, pub markers: Vec<usize>, pub state_used: usize }

fn clamp_t(x: f64) -> f32 { if x.is_finite() { x.clamp(0.5, 5.0) as f32 } else { 1.0 } }
fn fnum(j: &J) -> Option<f64> { match j { J::Num(t) => t.parse().ok(), _ => None } }
fn r4(x: f64) -> J { J::num((x * 1e4).round() / 1e4) }

/// Python json.dumps(v, ensure_ascii=False) with the default ", " / ": " separators
pub fn dump_py(v: &J) -> String {
    match v {
        J::Null => "null".into(), J::Bool(b) => b.to_string(),
        J::Num(t) => if t.contains(['.', 'e', 'E']) { let x: f64 = t.parse().unwrap_or(0.0); let s = format!("{x:?}"); if s.contains("e") && !s.contains('.') { s.replace('e', ".0e") } else { s } } else { t.clone() },
        J::Str(s) => { let mut o = String::from("\""); for ch in s.chars() { match ch { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"), '\u{c}' => o.push_str("\\f"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } } o.push('"'); o }
        J::Arr(xs) => format!("[{}]", xs.iter().map(dump_py).collect::<Vec<_>>().join(", ")),
        J::Obj(kv) => format!("{{{}}}", kv.iter().map(|(k, x)| format!("{}: {}", dump_py(&J::Str(k.clone())), dump_py(x))).collect::<Vec<_>>().join(", ")),
    }
}
pub fn render_criterion(v: &J) -> String { match v { J::Str(s) => s.clone(), x => dump_py(x) } }
fn empty(v: Option<&J>) -> bool { matches!(v, None | Some(J::Null)) || matches!(v, Some(J::Str(s)) if s.is_empty()) }

/// Agent._to_internal + render_options
pub fn questions(req: &J) -> Result<Vec<Lq>> {
    let qs = match req.get("questions") { Some(J::Obj(v)) if !v.is_empty() => v, _ => bail!("questions must be a non-empty object") };
    let mut out = vec![];
    for (id, q) in qs {
        let ins = match q.get("instructions") { Some(J::Str(s)) => s.clone(), Some(x) => dump_py(x), None => bail!("question {id:?}: instructions are required") };
        let crit = q.get("criteria");
        let (t, keys, opts, legend): (&'static str, Vec<String>, Vec<String>, Vec<String>) = match q.get("type").and_then(|t| t.as_str()) {
            Some("choice") => { let kv: Vec<(String, Option<J>)> = match crit { Some(J::Obj(kv)) if !kv.is_empty() => kv.iter().map(|(k, v)| (k.clone(), Some(v.clone()))).collect(),
                    Some(J::Arr(xs)) if !xs.is_empty() => xs.iter().map(|x| (render_criterion(x), None)).collect(), _ => bail!("question {id:?}: a choice question needs criteria") };
                ("choice", kv.iter().map(|x| x.0.clone()).collect(), kv.iter().map(|(k, v)| if empty(v.as_ref()) { k.clone() } else { format!("{k}: {}", render_criterion(v.as_ref().unwrap())) }).collect(), vec![]) }
            Some("score") => { let xs = match crit { Some(J::Arr(xs)) if !xs.is_empty() => xs, _ => bail!("question {id:?}: a score question needs at least one level") };
                if xs.iter().any(|x| matches!(x, J::Null)) { bail!("question {id:?}: score levels must not be null") }
                ("score", (0..xs.len()).map(|i| i.to_string()).collect(), xs.iter().enumerate().map(|(i, c)| format!("level {i}: {}", render_criterion(c))).collect(), xs.iter().map(render_criterion).collect()) }
            Some("noul") => {
                let g = |k: &str| -> Option<J> { match crit { Some(J::Obj(kv)) => kv.iter().find(|(a, _)| a.to_lowercase() == k).map(|x| x.1.clone()), _ => None } };
                let lab = |k: &str| -> Result<String> { match q.get("labels") { None => Ok(k.into()), Some(l) => { let s = l.get(k).and_then(|x| x.as_str()).map(|s| s.trim().to_string()).unwrap_or_default();
                    if s.is_empty() { bail!("noul labels must map exactly 'false' and 'true' to distinct non-empty strings") } Ok(s) } } };
                let (fl, tl) = (lab("false")?, lab("true")?); if fl == tl { bail!("noul labels must be distinct") }
                let (fc, tc) = (g("false"), g("true"));
                ("noul", vec!["false".into(), "true".into()], vec![format!("{fl}: {}", if empty(fc.as_ref()) { "no, the statement does not hold".into() } else { render_criterion(fc.as_ref().unwrap()) }),
                    format!("{tl}: {}", if empty(tc.as_ref()) { "yes, the statement holds".into() } else { render_criterion(tc.as_ref().unwrap()) })], vec![]) }
            t => bail!("question {id:?}: unknown type {t:?}"),
        };
        out.push(Lq { id: id.clone(), t, ins, opts, keys, legend });
    }
    Ok(out)
}

impl LayaS1 {
    pub fn load(gguf: &str, tokenizer: &str, cfg: &str) -> Result<Self> {
        let c = json::parse(&std::fs::read_to_string(cfg)?)?;
        let mut temp = [1f32; 3]; if let Some(J::Arr(v)) = c.get("temperature") { for (i, x) in v.iter().take(3).enumerate() { temp[i] = clamp_t(fnum(x).unwrap_or(1.0)) } }
        let tbo = match c.get("temperature_by_options") { Some(J::Obj(kv)) => kv.iter().map(|(k, v)| (k.clone(), clamp_t(fnum(v).unwrap_or(1.0)))).collect(), _ => vec![] };
        let max_len = c.get("max_len").and_then(fnum).unwrap_or(512.0) as usize; let head_max = c.get("head_max_len").and_then(fnum).unwrap_or(192.0) as usize;
        let tok = HfTok::load(tokenizer)?; let id = |s: &str| tok.id(s).ok_or_else(|| anyhow!("tokenizer has no {s}"));
        let (cls, sep, mask) = (id("[CLS]")?, id("[SEP]")?, id("[MASK]")?);
        Ok(LayaS1 { m: laya::Laya::load(gguf, max_len as u32)?, tok, temp, tbo, max_len, head_max, cls, sep, mask })
    }
    pub fn state_ids(&self, state: &J) -> Vec<u32> { let s = match state { J::Str(s) => s.clone(), x => dump_py(x) }; self.tok.encode(&s.replace("[MASK]", " ")) }
    /// build_head + build_sequence
    pub fn row(&self, q: &Lq, st: &[u32]) -> Row {
        let mut head = self.tok.encode(&format!("{} question: {}", q.t, q.ins.replace("[MASK]", " ")));
        let mut opt: Vec<Vec<u32>> = q.opts.iter().map(|o| { let mut v = vec![self.mask]; v.extend(self.tok.encode(&format!(" {}", o.replace("[MASK]", " "))).into_iter().take(48)); v }).collect();
        let hm = self.head_max as i64; let mut budget = hm - opt.iter().map(|o| o.len() as i64).sum::<i64>();
        if budget < 16 { let per = 4.max((hm - 16) / (opt.len().max(1) as i64)) as usize; for o in &mut opt { o.truncate(per) } budget = hm - opt.iter().map(|o| o.len() as i64).sum::<i64>(); }
        head.truncate(8.max(budget) as usize);
        let mut ids = vec![self.cls]; ids.extend(head); ids.push(self.sep);
        let mut markers = vec![]; for o in opt { markers.push(ids.len()); ids.extend(o) } ids.push(self.sep);
        let room = (self.max_len as i64 - ids.len() as i64 - 1).max(0) as usize; let used = st.len().min(room);
        ids.extend(&st[..used]); ids.push(self.sep); ids.truncate(self.max_len); markers.retain(|&m| m < self.max_len);
        Row { ids, markers, state_used: used }
    }
    /// Returns (answers, usage, GPU ms)
    pub fn system_one(&self, req: &J) -> Result<(J, J, f64)> {
        let state = req.get("state").ok_or_else(|| anyhow!("state is required"))?;
        let qs = questions(req)?; let st = self.state_ids(state);
        let (mut answers, mut gpu, mut input, mut dropped, mut tq) = (vec![], 0.0, 0usize, 0usize, vec![]);
        for q in &qs {
            let r = self.row(q, &st); input += r.ids.len();
            if r.state_used < st.len() { dropped = dropped.max(st.len() - r.state_used); tq.push(J::Str(q.id.clone())) }
            let qt = match q.t { "choice" => 0, "score" => 1, _ => 2 };
            let o = self.m.forward(&r.ids, qt as u32, &r.markers, 0)?; gpu += o.gpu_ms;
            let k = r.markers.len();
            let size = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
            let bucket = format!("{}:{size}", q.t); let t = self.tbo.iter().find(|x| x.0 == bucket).map(|x| x.1).unwrap_or(self.temp[qt]);
            let z: Vec<f64> = o.logits[..k].iter().map(|&x| x as f64 / t as f64).collect(); let mx = z.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = z.iter().map(|x| (x - mx).exp()).collect(); let sm: f64 = e.iter().sum(); let p: Vec<f64> = e.iter().map(|x| x / sm).collect();
            let ans_conf = if k < 1 { 1.0 } else { p.iter().cloned().fold(0.0, f64::max).clamp(0.0, 1.0) };
            let ent_conf = if k < 2 { 1.0 } else { (1.0 - -p.iter().map(|&x| x * x.clamp(1e-12, 1.0).ln()).sum::<f64>() / (k as f64).ln()).clamp(0.0, 1.0) };
            let (a0, a1) = (o.act[0] as f64, o.act[1] as f64); let act = 1.0 / (1.0 + (a1 - a0).exp());
            let ext = J::obj(vec![("act_probability", r4(act))]);
            let am = (0..k).fold(0, |m, i| if p[i] > p[m] { i } else { m });
            let probs = |keys: &[String]| J::Obj(keys.iter().zip(&p).map(|(kk, v)| (kk.clone(), r4(*v))).collect());
            let a = match q.t {
                "choice" => J::obj(vec![("type", J::Str("choice".into())), ("choice", J::Str(q.keys[am].clone())), ("probabilities", probs(&q.keys)), ("confidence", r4(ent_conf)), ("answer_confidence", r4(ans_conf)), ("action", ext)]),
                "score" => J::obj(vec![("type", J::Str("score".into())), ("score", r4(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum())),
                    ("legend", J::Obj(q.keys.iter().zip(&q.legend).map(|(kk, v)| (kk.clone(), J::Str(v.clone()))).collect())), ("probabilities", probs(&q.keys)), ("confidence", r4(ent_conf)), ("answer_confidence", r4(ans_conf)), ("action", ext)]),
                _ => { let p1 = p.get(1).copied().unwrap_or(0.0); J::obj(vec![("type", J::Str("noul".into())), ("noul", r4(p1)), ("confidence", r4(p1.max(1.0 - p1))), ("answer_confidence", r4(ans_conf)), ("action", ext)]) }
            };
            answers.push((q.id.clone(), a));
        }
        let usage = J::obj(vec![("input_tokens", J::num(input as f64)), ("output_tokens", J::num(0.0)), ("state_tokens", J::num(st.len() as f64)), ("state_tokens_dropped", J::num(dropped as f64)),
            ("truncated", J::Bool(dropped > 0)), ("truncated_questions", J::Arr(tq))]);
        Ok((J::Obj(answers), usage, gpu))
    }
}
