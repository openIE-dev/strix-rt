//! npu_resid <out.bin> [T=512]   env DBG_LAYERS=L (read by qwen38 when recording its decode step)
//! Dumps the real residual stream of Bonsai 2 (PQ2_0) after L layers for T prompt tokens, using crates/qwen38's
//! token-by-token decode (read-only use): after each step, Model::x holds the token's residual. Writes f32 [T][dim].
use std::io::Write;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let h = std::env::var("HOME")?;
    let m = qwen38::Model::load(&format!("{h}/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf"), (t + 8) as u32)?;
    let d = m.cfg.dim as usize;
    let mut x = 12345u64;
    let prompt: Vec<u32> = (0..t).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); 1000 + ((x >> 33) % 60000) as u32 }).collect();
    let toks = unsafe { std::slice::from_raw_parts_mut(m.toks.ptr as *mut u32, m.toks.size as usize / 4) };
    let au = unsafe { std::slice::from_raw_parts_mut(m.aux.ptr as *mut u32, 1024) };
    m.cst.bytes_mut().fill(0); m.sst.bytes_mut().fill(0);
    toks[..t].copy_from_slice(&prompt); au[160] = 0; au[161] = t as u32; au[162] = 0;
    let mut out = Vec::with_capacity(t * d * 4);
    let mut amax = 0f32; let mut rms_sum = 0f64;
    for _ in 0..t { m.ctx.submit_repeated(m.step, 1)?; let v = &m.x.as_f32()[..d];
        for &f in v { out.extend_from_slice(&f.to_le_bytes()); amax = amax.max(f.abs()); }
        rms_sum += (v.iter().map(|f| (*f as f64).powi(2)).sum::<f64>() / d as f64).sqrt(); }
    std::fs::File::create(&a[1])?.write_all(&out)?;
    println!("wrote {} tokens x {d}: residual max|x| {amax:.1}, mean token rms {:.3} (DBG_LAYERS={})", t, rms_sum / t as f64, std::env::var("DBG_LAYERS").unwrap_or_default());
    Ok(())
}
