//! serve-prof <model.gguf>: per-op GPU time of one decode step (LFM2.5 or MiniCPM-V LLM), grouped by op label,
//! to see how much of the step is shareable dense GEMV vs per-sequence work (attention, state, MoE experts).
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let arch = { let g = gguf::Gguf::open(&a[1])?; g.get("general.architecture").ok().and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default() };
    let p = if arch == "lfm2moe" { lfm2::Model::load(&a[1], 2048)?.profile()? } else { mcv::llm::Llm::load(&a[1], 2048)?.profile()? };
    let mut agg: Vec<(&str, f64)> = vec![];
    for (l, t) in p { if let Some(e) = agg.iter_mut().find(|e| e.0 == l) { e.1 += t } else { agg.push((l, t)) } }
    let tot: f64 = agg.iter().map(|e| e.1).sum();
    for (l, t) in &agg { println!("{l:10} {:7.3} ms {:5.1}%", t * 1e3, 100.0 * t / tot); }
    println!("total {:.3} ms", tot * 1e3);
    Ok(())
}
