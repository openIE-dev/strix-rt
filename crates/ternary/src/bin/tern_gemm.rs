//! Batched ternary GEMM (prompt processing) benchmark: int8 activations x ternary weights decoded to int8,
//! KHR cooperative matrix s8*s8->s32. Measures quantx + gemm separately.
//! usage: tern_gemm [M=512]   env ITERS (default 20)
use anyhow::Result;
use ternary::{dequant_row, gg::Gg, shaders, Fmt};
use vkrt::{push_u32, Ctx};
fn lcg(s: &mut u64) -> f32 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }
fn main() -> Result<()> {
    let m: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(512);
    let iters: u32 = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let h = std::env::var("HOME")?;
    let ctx = Ctx::new()?;
    let kq = ctx.kernel(shaders::QUANTX, 3, 8, &[], Some(32))?;
    println!("M={m} iters={iters}\n| fmt | tensor | N x K | gemm us | TOPS (2MNK) | quantx us | weight GB/s-equiv | max_rel vs int8-x ref | max_rel vs fp32-x ref |\n|---|---|---|---|---|---|---|---|---|");
    for (file, spv) in [("Ternary-Bonsai-2-27B-PQ2_0.gguf", shaders::GEMM_PQ2), ("Ternary-Bonsai-2-27B-PTQ1_0.gguf", shaders::GEMM_PTQ1)] {
        let g = Gg::open(&format!("{h}/models/bonsai2/{file}"))?;
        let kg = ctx.kernel(spv, 4, 16, &[], Some(32))?;
        for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.attn_qkv.weight", "blk.3.attn_output.weight"] {
            let t = g.tensor(name).unwrap(); let f = Fmt::from_ggml(t.ty).unwrap(); let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
            let data = g.data(t); let size = data.len() as u64; let stride = size.div_ceil(256) * 256; let copies = ((256u64 << 20).div_ceil(size)).clamp(1, 16);
            let wb = ctx.buffer(stride * copies)?; for c in 0..copies as usize { wb.bytes_mut()[c * stride as usize..c * stride as usize + size as usize].copy_from_slice(data); }
            let xb = ctx.buffer((m * k * 4) as u64)?; let xq = ctx.buffer((m * k) as u64)?; let xs = ctx.buffer((m * k / 128 * 4) as u64)?; let yb = ctx.buffer((m * n * 4) as u64)?;
            let mut s = 11u64; for v in xb.as_f32_mut() { *v = lcg(&mut s); }
            let dq = ctx.bind(&kq, &[&xb, &xq, &xs])?; let dg = ctx.bind(&kg, &[&wb, &xq, &xs, &yb])?;
            let gq = m as u32; /* one WG per token */ let gg = ((m / 64) * (n / 64)) as u32;
            let pg = |c: u64| push_u32(&[m as u32, n as u32, k as u32, (c * stride) as u32]);
            ctx.run(|cb| { ctx.dispatch(cb, &kq, dq, &push_u32(&[m as u32, k as u32]), gq); ctx.barrier(cb); ctx.dispatch(cb, &kg, dg, &pg(0), gg); })?;
            // references on a token subset
            let xf = xb.as_f32(); let xqv: Vec<i8> = xq.bytes_mut().iter().map(|b| *b as i8).collect(); let xsv = xs.as_f32(); let yv = yb.as_f32();
            let toks = [0usize, 1, 63, 64, 130, m / 2, m - 1];
            let rb = f.row_bytes(k);
            let wrows: Vec<Vec<f32>> = { let mut v = vec![vec![0f32; k]; n]; std::thread::scope(|sc| { for (ci, ch) in v.chunks_mut(n.div_ceil(32)).enumerate() { sc.spawn(move || for (i, r) in ch.iter_mut().enumerate() { let row = ci * n.div_ceil(32) + i; dequant_row(f, &data[row * rb..(row + 1) * rb], r); }); } }); v };
            let (mut e1, mut e2, mut r1, mut r2) = (0f32, 0f32, 0f32, 0f32);
            for &tk in &toks { let xr = &xf[tk * k..(tk + 1) * k]; let xqd: Vec<f32> = (0..k).map(|i| xqv[tk * k + i] as f32 * xsv[tk * (k / 128) + i / 128]).collect();
                let res: Vec<(f64, f64)> = { let mut v = vec![(0f64, 0f64); n]; let (wr, xq2) = (&wrows, &xqd); std::thread::scope(|sc| { for (ci, ch) in v.chunks_mut(n.div_ceil(32)).enumerate() { sc.spawn(move || for (i, o) in ch.iter_mut().enumerate() { let j = ci * n.div_ceil(32) + i;
                    *o = (wr[j].iter().zip(xq2).map(|(w, x)| *w as f64 * *x as f64).sum(), wr[j].iter().zip(xr).map(|(w, x)| *w as f64 * *x as f64).sum()); }); } }); v };
                for j in 0..n { let (a, b) = res[j]; let gy = yv[tk * n + j]; e1 = e1.max((gy - a as f32).abs()); e2 = e2.max((gy - b as f32).abs()); r1 = r1.max(a.abs() as f32); r2 = r2.max(b.abs() as f32); } }
            let tq = ctx.run(|cb| for _ in 0..iters { ctx.dispatch(cb, &kq, dq, &push_u32(&[m as u32, k as u32]), gq); ctx.barrier(cb); })? / iters as f64;
            let tg = ctx.run(|cb| for i in 0..iters { ctx.dispatch(cb, &kg, dg, &pg(i as u64 % copies), gg); ctx.barrier(cb); })? / iters as f64;
            println!("| {} | {name} | {n}x{k} | {:.1} | {:.2} | {:.1} | {:.1} | {:.2e} | {:.2e} |", f.name(), tg * 1e6, 2.0 * (m * n * k) as f64 / tg / 1e12, tq * 1e6, size as f64 / tg / 1e9, e1 / r1, e2 / r2);
            for b in [wb, xb, xq, xs, yb] { ctx.free_buffer(b); }
        }
    }
    Ok(())
}
