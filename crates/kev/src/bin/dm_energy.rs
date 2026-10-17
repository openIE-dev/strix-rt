//! dm_energy kev <gguf> <head.bin> <golden.bin> <q_token_id> <N iters> [state|full] | dm_energy laya <gguf> <tok.json> <cfg.json> <golden.json> <N iters>
//! Loops N decision workloads (kev: set_state(STATE tokens) [+ 8-question one-pass]; laya: system_one on golden states) for differential
//! energy under the energy recorder (tools/comp/dm_energy.sh). Prints wall time per iteration.
use anyhow::Result;
use kev::{Branch, Kev};
use kev::json::{self, J};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "laya" {
        let s1 = kev::laya_s1::LayaS1::load(&a[2], &a[3], &a[4])?;
        let g = json::parse(&std::fs::read_to_string(&a[5])?)?;
        let n: usize = a[6].parse()?;
        let (J::Arr(states), Some(qs)) = (g.get("states").unwrap().clone(), g.get("questions")) else { anyhow::bail!("golden") };
        let reqs: Vec<J> = states.iter().map(|st| J::obj(vec![("state", st.clone()), ("questions", qs.clone())])).collect();
        let nq = match qs { J::Arr(v) => v.len(), _ => 0 };
        s1.system_one(&reqs[0])?; // warm-up
        let t = std::time::Instant::now();
        for i in 0..n { s1.system_one(&reqs[i % reqs.len()])?; }
        let s = t.elapsed().as_secs_f64();
        println!("DM laya: {n} requests x {nq} questions in {s:.2} s ({:.1} ms/request)", s * 1e3 / n as f64);
        return Ok(());
    }
    let qid: u32 = a[5].parse()?; let n: usize = a[6].parse()?; let full = a.get(7).map(|s| s != "state").unwrap_or(true);
    let state_n = prefill::env("STATE", 2048) as usize;
    let b = std::fs::read(&a[4])?; let mut p = 0usize;
    let rdu = |p: &mut usize, n: usize| -> Vec<u32> { let v = b[*p..*p + 4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(); *p += 4 * n; v };
    let nq = rdu(&mut p, 1)[0] as usize;
    let mut kv = Kev::load(&a[2], &a[3], (state_n + 2048) as u32, 512)?;
    let d = kv.m.cfg.dim as usize;
    p = 4; let mut qs = vec![];
    for _ in 0..nq { let n = rdu(&mut p, 1)[0] as usize; let ids = rdu(&mut p, n); let h = rdu(&mut p, 2); let (dc, k) = (h[0] as usize, h[1] as usize);
        let opts: Vec<usize> = rdu(&mut p, k).into_iter().map(|x| x as usize).collect(); p += 4 * (k + k + d + k * d); qs.push((ids, dc, opts)); }
    let mut x = 7u64; let state: Vec<u32> = (0..state_n).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); 1000 + ((x >> 33) % 60000) as u32 }).collect();
    let br = qs[1].0.iter().position(|&t| t == qid).unwrap();
    let tmpl = Branch { ids: qs[1].0[br..].to_vec(), decide: qs[1].1 - br, opts: qs[1].2.iter().map(|o| o - br).collect() };
    let brs = vec![tmpl; 8];
    kv.set_state(&state)?; kv.branches(state.len(), &brs)?; // warm-up
    let t = std::time::Instant::now();
    for _ in 0..n { kv.set_state(&state)?; if full { kv.branches(state.len(), &brs)?; } }
    let s = t.elapsed().as_secs_f64();
    println!("DM kev {}: {n} iters ({state_n}-tok state{}) in {s:.2} s ({:.1} ms/iter)", if full { "full" } else { "state" }, if full { " + 8 questions" } else { "" }, s * 1e3 / n as f64);
    Ok(())
}
