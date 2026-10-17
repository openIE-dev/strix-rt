use anyhow::Result;
use gguf::{dequant::dequant_row, GgmlType, Gguf, TensorInfo};
use vkrt::{push_u32, shaders, Ctx};

fn lcg(s: &mut u64) -> f32 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }

fn cpu_gemv(t: &TensorInfo, data: &[u8], x: &[f32]) -> Vec<f32> {
    let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
    let rb = t.row_bytes();
    let mut y = vec![0f32; n];
    let nt = 16; let chunk = n.div_ceil(nt);
    std::thread::scope(|s| {
        for (ci, yc) in y.chunks_mut(chunk).enumerate() {
            s.spawn(move || { let mut buf = vec![0f32; k];
                for (i, yv) in yc.iter_mut().enumerate() { let r = ci * chunk + i;
                    dequant_row(t.ty, &data[r * rb..(r + 1) * rb], &mut buf);
                    *yv = buf.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32; } });
        }
    });
    y
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| format!("{}/models/q7.gguf", std::env::var("HOME").unwrap()));
    let iters: u32 = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let g = Gguf::open(&path)?;
    let mut hist = std::collections::BTreeMap::new();
    for t in &g.tensors { *hist.entry(format!("{:?}", t.ty)).or_insert(0usize) += 1; }
    println!("gguf v{} arch={:?} tensors={} kv={} align={} data_off={} types={:?}", g.version, g.get("general.architecture").ok().and_then(|v| v.as_str()), g.tensors.len(), g.kv.len(), g.alignment, g.data_offset, hist);
    let ctx = Ctx::new()?;
    println!("{:#?}", ctx.info);

    let shapes = [("ffn_gate/up", 3584u64, 18944u64), ("ffn_down", 18944, 3584), ("attn_q/o", 3584, 3584), ("token_embd(lm-head shape)", 3584, 152064)];
    // (local_size, rows_per_wg, required subgroup)
    let cfgs: &[(u32, u32, Option<u32>)] = &[(32, 1, Some(32)), (64, 1, Some(32)), (64, 1, Some(64)), (128, 1, Some(64)), (64, 2, Some(64)), (64, 4, Some(64)), (64, 1, None)];
    let kernels: Vec<_> = cfgs.iter().map(|&(ls, nr, sg)| ctx.kernel(shaders::GEMV_Q4K, 3, 12, &[ls, nr], sg).map(|k| (ls, nr, sg, k))).collect::<Result<_>>()?;
    println!("\n| shape (N x K) | tensor | cfg ls/nr/sg | MiB | copies | us/iter | GB/s | % of 256 | max_abs | max_abs/max|ref| |");
    for (label, k, n) in shapes {
        let Some(t) = g.tensors.iter().find(|t| t.ty == GgmlType::Q4K && t.dims == [k, n]) else { println!("skip {label}: no Q4_K tensor [{k},{n}]"); continue };
        let data = g.data(t); let size = data.len() as u64;
        let copies = ((256u64 << 20).div_ceil(size)).clamp(1, 32);
        let wb = ctx.buffer(size * copies)?;
        for c in 0..copies as usize { wb.bytes_mut()[c * size as usize..(c + 1) * size as usize].copy_from_slice(data); }
        let xb = ctx.buffer(k * 4)?; let yb = ctx.buffer(n * 4)?;
        let mut seed = 42u64; for v in xb.as_f32_mut() { *v = lcg(&mut seed); }
        let x = xb.as_f32().to_vec();
        let r = cpu_gemv(t, data, &x);
        let rmax = r.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (ls, nr, sg, kern) in &kernels {
            let ds = ctx.bind(kern, &[&wb, &xb, &yb])?;
            let groups = (n as u32).div_ceil(*nr);
            yb.as_f32_mut().fill(f32::NAN);
            ctx.run(|cb| ctx.dispatch(cb, kern, ds, &push_u32(&[n as u32, k as u32, 0]), groups))?;
            let maxabs = yb.as_f32().iter().zip(&r).fold(0f32, |m, (a, b)| if a.is_nan() { f32::INFINITY } else { m.max((a - b).abs()) });
            let run = |it: u32| ctx.run(|cb| for i in 0..it { let off = ((i as u64 % copies) * size / 4) as u32;
                ctx.dispatch(cb, kern, ds, &push_u32(&[n as u32, k as u32, off]), groups); ctx.barrier(cb); });
            run(10)?;
            let secs = run(iters)?;
            let per = secs / iters as f64;
            let bytes = size + k * 4 + n * 4;
            let gbs = bytes as f64 / per / 1e9;
            println!("| {n} x {k} ({label}) | {} | {ls}/{nr}/{} | {:.1} | {copies} | {:.1} | {:.1} | {:.1}% | {:.3e} | {:.3e} |", t.name, sg.map(|s| s.to_string()).unwrap_or("auto".into()), size as f64 / 1048576.0, per * 1e6, gbs, gbs / 256.0 * 100.0, maxabs, maxabs / rmax);
        }
    }
    Ok(())
}
