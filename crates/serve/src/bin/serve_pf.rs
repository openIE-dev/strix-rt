//! serve_pf <model.gguf> [L..]: prefill wall time per prompt length (3 reps each, distinct random prompts), as the server calls it.
//! Also times prefill::qwen35::multi::prefill_pass: one 512-row segment, 2 x 256, 4 x 128 (packed), into slots.
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ls: Vec<usize> = if a.len() > 2 { a[2..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![512, 600] };
    let m = qwen38::Model::load(&a[1], 2048)?;
    let pf = prefill::Pf::new(&m.ctx)?;
    let q = prefill::qwen35::Q35::new(&m, prefill::env("PF_M", 1024))?;
    let qt = prefill::qwen35::Q35::new(&m, 512)?;
    let mut x = 7u64;
    let mut rp = |l: usize| -> Vec<u32> { (0..l).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); 1000 + ((x >> 33) % 60000) as u32 }).collect() };
    for &l in &ls { for r in 0..3 { let p = rp(l);
        let t = std::time::Instant::now(); prefill::qwen35::prefill(&pf, &m, &q, &p)?;
        println!("prefill L={l} rep {r}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3); } }
    let slots: Vec<_> = (0..4).map(|_| m.new_slot()).collect::<Result<_>>()?;
    for &(k, len) in &[(1usize, 512usize), (2, 256), (4, 128), (2, 512), (4, 256)] {
        for r in 0..3 { let ps: Vec<Vec<u32>> = (0..k).map(|_| rp(len)).collect();
            for s in &slots { m.slot_reset(s) }
            let segs: Vec<_> = ps.iter().zip(&slots).map(|(p, s)| prefill::qwen35::multi::Seg { toks: p, pos0: 0, st: [&s.kc, &s.vc, &s.cst, &s.sst], last: true }).collect();
            let t = std::time::Instant::now(); prefill::qwen35::multi::prefill_pass(&pf, &m, &q, &qt, &segs)?;
            println!("pass {k} x {len}: rep {r}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3); } }
    Ok(())
}
