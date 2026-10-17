//! s1-steps <model.gguf> [maxm] [pos]: GPU time of one batched decode step (record_multi) for m = 1..maxm columns,
//! LFM2.5 (lfm2moe) or MiniCPM-V (qwen35moe); state slots are fresh (zero KV), positions = pos.
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let maxm: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8); let pos: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(600);
    let arch = { let g = gguf::Gguf::open(&a[1])?; g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default() };
    std::env::set_var("NCOL", maxm.to_string());
    if arch == "lfm2moe" {
        let mut m = lfm2::Model::load(&a[1], 2048)?;
        let sts: Vec<_> = (0..maxm).map(|_| m.new_st()).collect::<Result<_>>()?;
        let mu = m.new_multi(maxm)?;
        for n in 1..=maxm {
            for j in 0..n as usize { let w = unsafe { std::slice::from_raw_parts_mut(mu.a[j].ptr as *mut u32, 256) }; w[160] = pos; }
            let sl: Vec<_> = sts[..n as usize].iter().collect();
            let mut err = Ok(()); let t = std::time::Instant::now(); let cb = m.ctx.record_raw(|cb| { err = m.record_multi(cb, &sl, &mu); })?; err?; let rec = t.elapsed().as_secs_f64();
            m.ctx.submit_repeated(cb, 1)?; let mut best = f64::MAX; for _ in 0..5 { best = best.min(m.ctx.submit_repeated(cb, 1)?); }
            println!("lfm m={n}: {:.2} ms/step GPU -> {:.1} tok/s (record {:.1} ms)", best * 1e3, n as f64 / best, rec * 1e3);
        }
    } else {
        let mut m = mcv::llm::Llm::load(&a[1], 2048)?;
        let sts: Vec<_> = (0..maxm).map(|_| m.new_st()).collect::<Result<_>>()?;
        let mu = m.new_multi()?; let au = m.au_all();
        for j in 0..maxm as usize { au[mcv::llm::MPOSW as usize + j] = pos; au[mcv::llm::MTOKW as usize + j] = 1000 + j as u32; au[mcv::llm::MRW as usize + 4 * j + 3] = 2; }
        for n in 1..=maxm {
            let sl: Vec<_> = sts[..n as usize].iter().collect();
            let mut err = Ok(()); let t = std::time::Instant::now(); let cb = m.ctx.record_raw(|cb| { err = m.record_multi(cb, &sl, &mu); })?; err?; let rec = t.elapsed().as_secs_f64();
            m.ctx.submit_repeated(cb, 1)?; let mut best = f64::MAX; for _ in 0..5 { best = best.min(m.ctx.submit_repeated(cb, 1)?); }
            println!("mcv m={n}: {:.2} ms/step GPU -> {:.1} tok/s (record {:.1} ms)", best * 1e3, n as f64 / best, rec * 1e3);
        }
    }
    Ok(())
}
