//! beam DIR ids_csv [n_new] -- greedy decode; prints per-step top-5 (softcapped logit, logprob), timing and stream stats.
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: beam DIR ids_csv [n_new]"); std::process::exit(2) }
    let ids: Vec<u32> = a[2].split(',').map(|s| s.trim().parse().unwrap()).collect();
    let n_new: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ctxn: usize = std::env::var("BEAM_MAXCTX").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let m = beam::Model::load(&a[1], ctxn)?;
    if let Ok(path) = std::env::var("BEAM_WARMSEQ") { // warm the expert cache on held-out sequences (first BEAM_WARMT tokens of each), then reset stats
        let wt: usize = std::env::var("BEAM_WARMT").ok().and_then(|v| v.parse().ok()).unwrap_or(24); let me = a[2].clone(); let t0 = std::time::Instant::now();
        for line in std::fs::read_to_string(path)?.lines() { let f: Vec<&str> = line.split('\t').collect(); if f[1].starts_with(&me) || me.starts_with(f[1]) { continue }
            let s: Vec<u32> = f[1].split(',').map(|x| x.parse().unwrap()).collect(); for pos in 0..wt.min(s.len()) { m.step(s[pos], pos)?; } }
        eprintln!("[warm] {:.1}s {}", t0.elapsed().as_secs_f64(), m.cache_report()); m.reset_stats();
    }
    if let Ok(from) = std::env::var("BEAM_FORCE") { // teacher-forced: ids is the whole sequence; print top-20 logprobs for every position >= FROM
        let from: usize = from.parse().unwrap_or(0);
        for pos in 0..ids.len() - 1 { m.step(ids[pos], pos)?; if pos >= from { let lse = m.logsumexp(); let top = m.top(20);
            println!("F {pos} {}", top.iter().map(|(i, l)| format!("{i}:{:.4}", *l as f64 - lse)).collect::<Vec<_>>().join(" ")); } }
        if let Ok(p) = std::env::var("BEAM_ACTOUT") { m.dump_act(&p)?; }
        let s = m.stats.borrow().clone(); println!("forced total {:.2}s disk {:.2}s bytes {:.1} GB", s.wall_s, s.disk_s, s.bytes as f64 / 1e9); return Ok(());
    }
    let mut cur = ids.clone(); let mut out = vec![]; let t = std::time::Instant::now();
    let mut tpos = vec![];
    let mut start = 0;
    if std::env::var("BEAM_PREFILL").is_ok() {
        let ts = std::time::Instant::now(); let mut p = 0;
        while p < ids.len() { let e = (p + beam::PF_T).min(ids.len()); m.prefill_chunk(&ids[p..e], p)?; p = e; }
        let lse = m.logsumexp(); let top = m.top(5);
        println!("pos {} top5 {}", ids.len() - 1, top.iter().map(|(i, l)| format!("{i}:{l:.4}:{:.4}", *l as f64 - lse)).collect::<Vec<_>>().join(" "));
        println!("prefill {} tokens {:.2}s", ids.len(), ts.elapsed().as_secs_f64());
        let nx = top[0].0; out.push(nx); cur.push(nx); start = ids.len();
    }
    for pos in start..ids.len() + n_new - 1 {
        let ts = std::time::Instant::now();
        m.step(cur[pos], pos)?;
        if pos + 1 >= ids.len() {
            let lse = m.logsumexp(); let top = m.top(5);
            println!("pos {pos} top5 {}", top.iter().map(|(i, l)| format!("{i}:{l:.4}:{:.4}", *l as f64 - lse)).collect::<Vec<_>>().join(" "));
            let nx = top[0].0; out.push(nx); if pos + 1 >= cur.len() { cur.push(nx); }
        }
        tpos.push(ts.elapsed().as_secs_f64());
    }
    let s = m.stats.borrow().clone(); println!("{}", m.cache_report());
    println!("generated {:?}", out);
    println!("per-step wall s: {}", tpos.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "));
    println!("total {:.2}s  gpu {:.2}s  disk(stream) {:.2}s ({:.2} GB, {:.2} GB/s)  steps {}", t.elapsed().as_secs_f64(), s.gpu_s, s.disk_s, s.bytes as f64 / 1e9, s.bytes as f64 / 1e9 / s.disk_s.max(1e-9), tpos.len());
    { let p = m.pfs.borrow(); if p.tot[1] > 0 { println!("prefetch hit-rate (top-{} predicted, 8 actual): {}", std::env::var("BEAM_PF_TOP").unwrap_or("12".into()), (1..4).filter(|&d| p.tot[d] > 0).map(|d| format!("d{d}={:.2}", p.hit[d] as f64 / p.tot[d] as f64)).collect::<Vec<_>>().join(" ")); } }
    if std::env::var("BEAM_TRACE").is_ok() { for (l, sel) in m.trace.borrow().iter() { println!("route {l} {:?}", sel); } }
    Ok(())
}
