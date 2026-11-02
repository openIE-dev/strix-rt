//! OpenAI Decisions API (POST /v1/decisions) <-> TypeSafe System One. Request: {model, input, questions:[...]}, where input is a
//! string, a JSON value, or a list of messages/parts ({content: string | [{type input_text, text} | {type input_json, data}]});
//! questions are {type predicate, name?, instructions} | {type choice, name?, instructions, choices:[{value, description?}]}
//! | {type score, name?, instructions, levels:[string | {value|label, description?}]}. Answers come back in question order:
//! predicate {type, name, probability}, choice {type, name, choice, confidence, probabilities:[{value, probability}]},
//! score {type, name, score, confidence, probabilities:[{level, probability}]}. Images are refused (text-only models).
use crate::json::J;
use anyhow::{bail, Result};

pub struct Meta { pub kinds: Vec<(String, Option<String>, Vec<String>)> }

fn text_of(v: &J, out: &mut Vec<J>) -> Result<()> {
    match v {
        J::Str(_) => out.push(v.clone()),
        J::Arr(xs) => for x in xs { text_of(x, out)? },
        J::Obj(_) => match v.get("type").and_then(|t| t.as_str()) {
            Some("input_text") => out.push(v.get("text").cloned().unwrap_or(J::Str(String::new()))),
            Some("input_json") => out.push(v.get("data").cloned().unwrap_or(J::Null)),
            Some("input_image") => bail!("images are not supported by this text-only model"),
            _ => match v.get("content") { Some(c) => text_of(c, out)?, None => out.push(v.clone()) },
        },
        other => out.push(other.clone()),
    }
    Ok(())
}

pub fn to_systemone(req: &J) -> Result<(J, Meta)> {
    let input = req.get("input").ok_or_else(|| anyhow::anyhow!("input is required"))?;
    let mut parts = vec![]; text_of(input, &mut parts)?;
    let state = if parts.len() == 1 { parts.remove(0) } else if parts.iter().all(|p| matches!(p, J::Str(_))) {
        J::Str(parts.iter().map(|p| p.as_str().unwrap().to_string()).collect::<Vec<_>>().join("\n\n")) } else { J::Arr(parts) };
    let qs = match req.get("questions") { Some(J::Arr(v)) if !v.is_empty() && v.len() <= 400 => v, _ => bail!("questions must be an array of 1..400 questions") };
    let mut s1q = vec![]; let mut kinds = vec![];
    for (i, q) in qs.iter().enumerate() {
        let ty = q.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let name = q.get("name").and_then(|t| t.as_str()).map(|s| s.to_string());
        let mut instr = vec![]; if let Some(x) = q.get("instructions") { text_of(x, &mut instr)? }
        let instructions = if instr.len() == 1 { instr.remove(0) } else { J::Str(instr.iter().map(|p| match p { J::Str(s) => s.clone(), o => o.dump() }).collect::<Vec<_>>().join("\n")) };
        let id = format!("q{i}");
        let (s1, vals) = match ty {
            "predicate" | "noul" => (J::obj(vec![("type", J::Str("noul".into())), ("instructions", instructions)]), vec![]),
            "choice" => { let cs = match q.get("choices") { Some(J::Arr(c)) if !c.is_empty() => c, _ => bail!("question {i}: choices must be a non-empty array") };
                let mut kv = vec![]; let mut vals = vec![];
                for c in cs { let (v, d) = match c { J::Str(s) => (s.clone(), J::Str(String::new())), _ => (c.get("value").and_then(|x| x.as_str()).ok_or_else(|| anyhow::anyhow!("question {i}: choice without value"))?.to_string(), c.get("description").cloned().unwrap_or(J::Str(String::new()))) };
                    vals.push(v.clone()); kv.push((v, d)); }
                (J::obj(vec![("type", J::Str("choice".into())), ("instructions", instructions), ("criteria", J::Obj(kv))]), vals) }
            "score" => { let ls = match q.get("levels") { Some(J::Arr(c)) if !c.is_empty() => c, _ => bail!("question {i}: levels must be a non-empty array") };
                let mut lv = vec![]; let mut vals = vec![];
                for l in ls { let (v, d) = match l { J::Str(s) => (s.clone(), s.clone()), _ => { let v = l.get("value").or(l.get("label")).and_then(|x| x.as_str()).unwrap_or("").to_string();
                    let d = l.get("description").and_then(|x| x.as_str()).map(|d| if v.is_empty() { d.to_string() } else { format!("{v}: {d}") }).unwrap_or(v.clone()); (v, d) } };
                    vals.push(v); lv.push(J::Str(d)); }
                (J::obj(vec![("type", J::Str("score".into())), ("instructions", instructions), ("criteria", J::Arr(lv))]), vals) }
            _ => bail!("question {i}: type must be predicate, choice or score"),
        };
        s1q.push((id, s1)); kinds.push((ty.to_string(), name, vals));
    }
    let mut r = vec![("state", state), ("questions", J::Obj(s1q))];
    if let Some(m) = req.get("model") { r.push(("model", m.clone())) }
    Ok((J::obj(r), Meta { kinds }))
}

fn r4(x: f64) -> J { J::num((x * 1e4).round() / 1e4) }
fn f(j: Option<&J>) -> f64 { match j { Some(J::Num(s)) => s.parse().unwrap_or(f64::NAN), _ => f64::NAN } }

pub fn from_systemone(r: &J, meta: &Meta, req: &J, t0: std::time::Instant) -> Result<J> {
    let ans = r.get("answers").ok_or_else(|| anyhow::anyhow!("no answers"))?; let mut out = vec![];
    for (i, (ty, name, vals)) in meta.kinds.iter().enumerate() {
        let a = ans.get(&format!("q{i}")).ok_or_else(|| anyhow::anyhow!("missing answer {i}"))?;
        let mut o: Vec<(&str, J)> = vec![("type", J::Str(if ty == "noul" { "predicate".into() } else { ty.clone() }))];
        if let Some(n) = name { o.push(("name", J::Str(n.clone()))) }
        let probs = |a: &J, keys: Vec<String>| -> Vec<f64> { keys.iter().map(|k| f(a.get("probabilities").and_then(|p| p.get(k)))).collect() };
        match ty.as_str() {
            "predicate" | "noul" => o.push(("probability", r4(f(a.get("noul"))))),
            "choice" => { let p = probs(a, vals.clone());
                o.push(("choice", a.get("choice").cloned().unwrap_or(J::Null))); o.push(("confidence", r4(f(a.get("confidence")))));
                o.push(("probabilities", J::Arr(vals.iter().zip(&p).map(|(v, x)| J::obj(vec![("value", J::Str(v.clone())), ("probability", r4(*x))])).collect()))) }
            _ => { let p = probs(a, (0..vals.len()).map(|k| k.to_string()).collect());
                o.push(("score", r4(f(a.get("score"))))); o.push(("confidence", r4(f(a.get("confidence")))));
                o.push(("probabilities", J::Arr(vals.iter().enumerate().zip(&p).map(|((k, v), x)| J::obj(vec![("level", J::num(k as f64)), ("value", J::Str(v.clone())), ("probability", r4(*x))])).collect()))) }
        }
        out.push(J::obj(o));
    }
    let id = format!("dec_{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    Ok(J::obj(vec![("id", J::Str(id)), ("object", J::Str("decision".into())),
        ("model", r.get("model").cloned().or(req.get("model").cloned()).unwrap_or(J::Null)), ("answers", J::Arr(out)),
        ("usage", r.get("usage").cloned().unwrap_or(J::Null)), ("latency_ms", J::num((t0.elapsed().as_secs_f64() * 1e4).round() / 10.0))]))
}

#[cfg(test)]
mod tests {
    use super::*; use crate::json::parse;
    #[test] fn roundtrip() {
        let req = parse(r#"{"model":"kev-0.8b","input":[{"role":"user","content":[{"type":"input_text","text":"Ticket: refund please"}]}],
          "questions":[{"type":"predicate","name":"urgent","instructions":"Is it urgent?"},
                       {"type":"choice","name":"team","instructions":"Route","choices":[{"value":"billing","description":"money"},{"value":"tech"}]},
                       {"type":"score","instructions":"Anger","levels":["calm",{"value":"angry","description":"very"}]}]}"#).unwrap();
        let (s1, m) = to_systemone(&req).unwrap(); let d = s1.dump();
        assert!(d.contains(r#""state":"Ticket: refund please""#), "{d}");
        assert!(d.contains(r#""q0":{"type":"noul""#) && d.contains(r#""criteria":{"billing":"money","tech":""}"#) && d.contains(r#""criteria":["calm","angry: very"]"#), "{d}");
        let r = parse(r#"{"model":"kev-0.8b","answers":{"q0":{"type":"noul","noul":0.8},"q1":{"type":"choice","choice":"billing","confidence":0.6,"probabilities":{"billing":0.8,"tech":0.2}},
          "q2":{"type":"score","score":0.3,"probabilities":{"0":0.7,"1":0.3},"confidence":0.4}},"usage":{"input_tokens":5,"output_tokens":3}}"#).unwrap();
        let o = from_systemone(&r, &m, &req, std::time::Instant::now()).unwrap().dump();
        assert!(o.contains(r#""object":"decision""#) && o.contains(r#"{"type":"predicate","name":"urgent","probability":0.8}"#), "{o}");
        assert!(o.contains(r#"{"value":"tech","probability":0.2}"#) && o.contains(r#"{"level":1,"value":"angry","probability":0.3}"#), "{o}");
        assert!(to_systemone(&parse(r#"{"input":[{"type":"input_image"}],"questions":[{"type":"predicate","instructions":"x"}]}"#).unwrap()).is_err());
        assert!(to_systemone(&parse(r#"{"input":"x","questions":[]}"#).unwrap()).is_err());
    }
}
