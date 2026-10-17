//! kev_parity <merged.gguf> <head.bin> <golden.bin> <q_token_id>: Rust/Vulkan Kev (shared state prefix + GPU head) vs the
//! PyTorch reference (ref/ref_kev.py, fp32 CPU, full row per question). BENCH_STATE=N adds a latency run: N-token state, 8 questions.
use anyhow::Result;
use kev::{Branch, Kev};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let qid: u32 = a[4].parse()?;
    let b = std::fs::read(&a[3])?; let mut p = 0usize;
    let rdu = |p: &mut usize, n: usize| -> Vec<u32> { let v = b[*p..*p + 4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(); *p += 4 * n; v };
    let rdf = |p: &mut usize, n: usize| -> Vec<f32> { let v = b[*p..*p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(); *p += 4 * n; v };
    let nq = rdu(&mut p, 1)[0] as usize;
    let bench = prefill::env("BENCH_STATE", 0) as usize;
    let mut kv = Kev::load(&a[1], &a[2], (bench + 2048) as u32, 512)?;
    let d = kv.m.cfg.dim as usize;
    // proper parse now that d is known
    p = 4; let mut qs = vec![];
    for _ in 0..nq { let n = rdu(&mut p, 1)[0] as usize; let ids = rdu(&mut p, n); let h = rdu(&mut p, 2); let (dc, k) = (h[0] as usize, h[1] as usize);
        let opts: Vec<usize> = rdu(&mut p, k).into_iter().map(|x| x as usize).collect();
        let (gp, gz, ghd, gho) = (rdf(&mut p, k), rdf(&mut p, k), rdf(&mut p, d), rdf(&mut p, k * d)); qs.push((ids, dc, opts, gp, gz, ghd, gho)); }
    assert_eq!(p, b.len(), "golden.bin layout (hidden size {d})");
    let cos = |x: &[f32], y: &[f32]| { let dt = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| *p as f64 * *q as f64).sum::<f64>(); dt(x, y) / (dt(x, x) * dt(y, y)).sqrt() };
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    let (mut worst_p, mut worst_cos, mut worst_head, mut agree) = (0f32, 1f64, 0f32, 0);
    let mut cur_state: Vec<u32> = vec![];
    for (i, (ids, dc, opts, gp, _gz, ghd, gho)) in qs.iter().enumerate() {
        let ls = ids.iter().position(|&t| t == qid).expect("no <q> token");
        let t = std::time::Instant::now(); let mut st_ms = 0.0;
        if ids[..ls] != cur_state[..] { kv.set_state(&ids[..ls])?; cur_state = ids[..ls].to_vec(); st_ms = t.elapsed().as_secs_f64() * 1e3; }
        let br = Branch { ids: ids[ls..].to_vec(), decide: dc - ls, opts: opts.iter().map(|o| o - ls).collect() };
        let t1 = std::time::Instant::now(); let z = kv.branch(ls, &br)?; let br_ms = t1.elapsed().as_secs_f64() * 1e3;
        let mut rows = vec![br.decide]; rows.extend_from_slice(&br.opts); let h = kv.hidden_rows(&rows)?;
        let zc = kv.head.logits(&h[0], &h[1..]); let dh = z.iter().zip(&zc).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let pr = kev::softmax(&z);
        let mut c = cos(&h[0], ghd); for j in 0..opts.len() { c = c.min(cos(&h[1 + j], &gho[j * d..(j + 1) * d])); }
        let dp = pr.iter().zip(gp).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        if am(&pr) == am(gp) { agree += 1 }
        worst_p = worst_p.max(dp); worst_cos = worst_cos.min(c); worst_head = worst_head.max(dh);
        println!("q{i}: state {ls} + branch {} tok: hidden cos {c:.6}, max|dprob| {dp:.4}, GPU-vs-CPU head max|dlogit| {dh:.2e}, argmax {} vs ref {} | state {st_ms:.1} ms, branch+head {br_ms:.1} ms\n   rust {pr:.4?}\n   ref  {gp:.4?}", br.ids.len(), am(&pr), am(gp));
    }
    // all questions of a document in one pass
    let mut mworst = 0f32; let mut magree = 0; let mut i = 0;
    while i < qs.len() {
        let ls = qs[i].0.iter().position(|&t| t == qid).unwrap(); let mut j = i; while j < qs.len() && qs[j].0[..] .starts_with(&qs[i].0[..ls]) && qs[j].0.iter().position(|&t| t == qid) == Some(ls) { j += 1 }
        kv.set_state(&qs[i].0[..ls])?;
        let brs: Vec<Branch> = qs[i..j].iter().map(|q| Branch { ids: q.0[ls..].to_vec(), decide: q.1 - ls, opts: q.2.iter().map(|o| o - ls).collect() }).collect();
        let t = std::time::Instant::now(); let zs = kv.branches(ls, &brs)?; let ms = t.elapsed().as_secs_f64() * 1e3;
        for (z, q) in zs.iter().zip(&qs[i..j]) { let pr = kev::softmax(z); let dp = pr.iter().zip(&q.3).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max); mworst = mworst.max(dp); if am(&pr) == am(&q.3) { magree += 1 } }
        println!("one-pass: questions {i}..{j} ({} branches) {ms:.1} ms", j - i); i = j;
    }
    println!("ONE-PASS {nq} questions: argmax agree {magree}/{nq}, worst max|dprob| {mworst:.4}");
    println!("SUMMARY {nq} questions: argmax agree {agree}/{nq}, worst max|dprob| {worst_p:.4}, worst hidden cos {worst_cos:.6}, worst GPU-head dlogit {worst_head:.2e}");
    if bench > 0 {
        let mut x = 7u64; let mut rnd = |n: usize| -> Vec<u32> { (0..n).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); 1000 + ((x >> 33) % 60000) as u32 }).collect() };
        let state = rnd(bench); let br = qs[1].0.iter().position(|&t| t == qid).unwrap();
        let tmpl = Branch { ids: qs[1].0[br..].to_vec(), decide: qs[1].1 - br, opts: qs[1].2.iter().map(|o| o - br).collect() };
        let brs = vec![tmpl; 8]; let _ = rnd(1);
        for rep in 0..3 {
            let t = std::time::Instant::now(); kv.set_state(&state)?; let ts = t.elapsed().as_secs_f64() * 1e3;
            let t1 = std::time::Instant::now(); let one = kv.branches(state.len(), &brs)?; let t1p = t1.elapsed().as_secs_f64() * 1e3;
            let t1 = std::time::Instant::now(); let mut seq = vec![]; for b in &brs { seq.push(kv.branch(state.len(), b)?); } let tb = t1.elapsed().as_secs_f64() * 1e3;
            let dz = one.iter().flatten().zip(seq.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("BENCH rep {rep}: 8 questions one pass {t1p:.1} ms ({:.1} ms/question) vs one at a time {tb:.1} ms; max|dlogit| one-pass vs sequential {dz:.2e}", t1p / 8.0);
            println!("BENCH rep {rep}: state {bench} tok {ts:.1} ms ({:.0} tok/s), 8 questions x {} tok {tb:.1} ms ({:.1} ms/question), total {:.1} ms", bench as f64 / ts * 1e3, brs[0].ids.len(), tb / 8.0, ts + tb);
        }
    }
    Ok(())
}
