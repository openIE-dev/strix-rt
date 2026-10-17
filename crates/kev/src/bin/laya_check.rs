//! laya_check <laya.gguf> <tokenizer.json> <rl_agent_config.json> <golden.json> [texts.txt]: compares the Rust System One
//! path (tokenizer, sequence layout, temperatures, answers) with laya.Agent.system_one results captured by ref_laya.py.
use kev::json::{self, J};
fn leaves(a: &J, b: &J, path: &str, worst: &mut (f64, String), miss: &mut Vec<String>) {
    match (a, b) {
        (J::Obj(x), J::Obj(_)) => for (k, v) in x { match b.get(k) { Some(w) => leaves(v, w, &format!("{path}.{k}"), worst, miss), None => miss.push(format!("{path}.{k} (extra)")) } },
        (J::Num(x), J::Num(y)) => { let d = (x.parse::<f64>().unwrap() - y.parse::<f64>().unwrap()).abs(); if d > worst.0 { *worst = (d, path.into()) } }
        (x, y) => if x.dump() != y.dump() { miss.push(format!("{path}: {} vs {}", x.dump(), y.dump())) },
    }
}
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let s1 = kev::laya_s1::LayaS1::load(&a[1], &a[2], &a[3])?;
    let g = json::parse(&std::fs::read_to_string(&a[4])?)?;
    let (J::Arr(states), Some(qs), J::Arr(res)) = (g.get("states").unwrap().clone(), g.get("questions"), g.get("results").unwrap().clone()) else { anyhow::bail!("golden") };
    for (i, (st, r)) in states.iter().zip(&res).enumerate() {
        let req = J::obj(vec![("state", st.clone()), ("questions", qs.clone())]);
        let t = std::time::Instant::now(); let (ans, usage, gpu) = s1.system_one(&req)?; let ms = t.elapsed().as_secs_f64() * 1e3;
        let (mut worst, mut miss) = ((0.0, String::new()), vec![]);
        leaves(&ans, r.get("answers").unwrap(), "answers", &mut worst, &mut miss); leaves(&usage, r.get("usage").unwrap(), "usage", &mut worst, &mut miss);
        println!("state {i}: max|d| {:.4} at {} | mismatches {:?} | {ms:.1} ms wall, {gpu:.1} ms GPU", worst.0, worst.1, miss);
    }
    if let Some(p) = a.get(5) { for l in std::fs::read_to_string(p)?.lines() { println!("{}", s1.tok.encode(&l.replace("\\n", "\n").replace("\\t", "\t")).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")) } }
    Ok(())
}
