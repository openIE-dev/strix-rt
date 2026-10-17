//! NPU context-switch probe: npu_run time for split linears run back-to-back vs alternating (`npu_alt [m]`).
use npu::split::{build_dir, SplitLinear};
fn main() -> anyhow::Result<()> {
    let m: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(512);
    let ctx = vkrt::Ctx::new()?;
    let shapes = [(5120usize, 6144usize, 3584usize), (5120, 10240, 5632), (5120, 17408, if m >= 2048 { 9728 } else { 9216 })];
    let sg = vec![1f32; 5120];
    let mut ls = vec![]; let mut ws = vec![];
    for &(k, n, nn) in &shapes { let l = SplitLinear::new(&ctx, m, k, n, nn, &build_dir(m, k, nn), Some(&sg))?;
        ws.push(l.load_npu(&ctx, &vec![0u8; n * k / 128 * 34])?); ls.push(l); }
    for i in 0..ls.len() { ls[i].npu_run(&ws[i])?; }
    for i in 0..ls.len() { let mut t = 0.0; for _ in 0..10 { t += ls[i].npu_run(&ws[i])?; } println!("same   {:?}: {:.3} ms", shapes[i], t * 100.0); }
    let mut t = vec![0.0; ls.len()];
    for _ in 0..10 { for i in 0..ls.len() { t[i] += ls[i].npu_run(&ws[i])?; } }
    for i in 0..ls.len() { println!("alter  {:?}: {:.3} ms", shapes[i], t[i] * 100.0); }
    Ok(())
}
