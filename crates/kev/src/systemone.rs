//! TypeSafe System One request/response mapping (port of kev/api.py): noul -> [no, yes], choice -> criteria names
//! ("name" or "name: desc"), score -> ordered level descriptions; one pointer question each.
use crate::json::J;
use anyhow::{bail, Result};

/// Python str() of a JSON scalar as kev's render() sees it after json.loads
fn scalar(n: &J) -> String { match n { J::Null => String::new(), J::Bool(b) => (if *b { "True" } else { "False" }).into(), J::Str(s) => s.clone(),
    J::Num(t) => if t.contains(['.', 'e', 'E']) { let x: f64 = t.parse().unwrap_or(0.0); let s = format!("{x:?}"); s } else { t.trim_start_matches('+').to_string() }, _ => unreachable!() } }

pub fn render(v: &J, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        J::Arr(xs) => xs.iter().map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start())).collect::<Vec<_>>().join("\n"),
        J::Obj(kv) => kv.iter().map(|(k, x)| match x { J::Obj(_) | J::Arr(_) => format!("{pad}{k}:\n{}", render(x, indent + 1)), _ => format!("{pad}{k}: {}", render(x, 0)) }).collect::<Vec<_>>().join("\n"),
        s => scalar(s),
    }
}
fn option_text(name: &str, desc: Option<&J>) -> String { match desc { None | Some(J::Null) => name.into(), Some(J::Str(s)) if s.is_empty() => name.into(), Some(d) => format!("{name}: {}", render(d, 0)) } }

pub struct Q { pub id: String, pub ty: String, pub keys: Vec<String>, pub legend: Vec<String>, pub instr: String, pub options: Vec<String> }

pub fn to_record(req: &J) -> Result<(String, Vec<Q>)> {
    let state = req.get("state").ok_or_else(|| anyhow::anyhow!("state is required"))?;
    let qs = match req.get("questions") { Some(J::Obj(v)) if !v.is_empty() => v, _ => bail!("questions must be a non-empty object") };
    let mut out = vec![];
    for (id, q) in qs {
        let ty = q.get("type").and_then(|t| t.as_str()).unwrap_or("").to_string();
        let instr = q.get("instructions").map(|x| render(x, 0)).unwrap_or_default();
        let crit = q.get("criteria");
        let (keys, options, legend): (Vec<String>, Vec<String>, Vec<String>) = match (ty.as_str(), crit) {
            ("noul", c) => { let g = |k: &str| c.and_then(|c| c.get(k)); (vec!["false".into(), "true".into()], vec![option_text("no", g("false")), option_text("yes", g("true"))], vec![]) }
            ("choice", Some(J::Obj(kv))) if !kv.is_empty() && kv.len() <= 255 => (kv.iter().map(|x| x.0.clone()).collect(), kv.iter().map(|(k, v)| option_text(k, Some(v))).collect(), vec![]),
            ("score", Some(J::Arr(xs))) if !xs.is_empty() && xs.len() <= 255 => { let o: Vec<String> = xs.iter().map(|x| render(x, 0)).collect(); ((0..xs.len()).map(|i| i.to_string()).collect(), o.clone(), o) }
            ("choice", _) => bail!("question {id:?}: choice criteria must be an object of 1..255 options"),
            ("score", _) => bail!("question {id:?}: score criteria must be a list of 1..255 levels"),
            _ => bail!("question {id:?}: unknown type {ty:?}"),
        };
        out.push(Q { id: id.clone(), ty, keys, legend, instr, options });
    }
    Ok((render(state, 0), out))
}

fn r4(x: f64) -> J { J::num((x * 1e4).round() / 1e4) }
fn norm(p: &[f32]) -> Vec<f64> { let t: f64 = p.iter().map(|&x| x as f64).sum(); if t == 0.0 { vec![1.0 / p.len() as f64; p.len()] } else { p.iter().map(|&x| x as f64 / t).collect() } }
pub fn choice_confidence(p: &[f32]) -> f64 { let k = p.len() as f64; if p.len() == 1 { return 1.0 } let q = norm(p); (q.iter().cloned().fold(0.0, f64::max) - 1.0 / k) / (1.0 - 1.0 / k) }
pub fn score_confidence(p: &[f32]) -> f64 { let l = p.len(); if l == 1 { return 1.0 } let q = norm(p);
    let mode = (0..l).fold(0, |m, i| if q[i] > q[m] { i } else { m }); let c = (l as f64 - 1.0) / 2.0;
    let d = (0..l).map(|i| (i as f64 - c).abs()).sum::<f64>() / l as f64; (1.0 - (0..l).map(|i| q[i] * (i as f64 - mode as f64).abs()).sum::<f64>() / d).max(0.0) }

pub fn to_answers(probs: &[Vec<f32>], qs: &[Q]) -> J {
    J::Obj(probs.iter().zip(qs).map(|(p, q)| {
        let a = match q.ty.as_str() {
            "noul" => J::obj(vec![("type", J::Str("noul".into())), ("noul", r4(p[1] as f64))]),
            "choice" => { let am = (0..p.len()).fold(0, |m, i| if p[i] > p[m] { i } else { m });
                J::obj(vec![("type", J::Str("choice".into())), ("choice", J::Str(q.keys[am].clone())), ("confidence", r4(choice_confidence(p))),
                    ("probabilities", J::Obj(q.keys.iter().zip(p).map(|(k, v)| (k.clone(), r4(*v as f64))).collect()))]) }
            _ => J::obj(vec![("type", J::Str("score".into())), ("score", r4(p.iter().enumerate().map(|(i, v)| i as f64 * *v as f64).sum())),
                    ("legend", J::Obj(q.keys.iter().zip(&q.legend).map(|(k, v)| (k.clone(), J::Str(v.clone()))).collect())),
                    ("probabilities", J::Obj(q.keys.iter().zip(p).map(|(k, v)| (k.clone(), r4(*v as f64))).collect())), ("confidence", r4(score_confidence(p)))]),
        };
        (q.id.clone(), a) }).collect())
}
