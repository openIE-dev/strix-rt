//! npu_bonsai M layer nn_gu dir_gu nn_down dir_down [iters]
//! Real Bonsai 2 PQ2_0 ffn_gate/up/down of one layer through npu::split::SplitLinear (fused GPU rotate+quant,
//! concurrent NPU/GPU split GEMM, one shared output), compared against the GPU-only ternary path (same API, nn = 0)
//! and an f32 reference. Activations: real token embeddings (inverse-rotated per the GGUF), RMS-normed with the
//! layer's post_attention_norm -> gate/up input; down input = silu(gate) * up from the GPU-only outputs.
use npu::split::SplitLinear; use std::time::Instant;
use ternary::{dequant_row, gg::Gg, Fmt};
fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }
fn fwht(v: &mut [f64]) { for b in v.chunks_mut(1024) { let mut h = 1; while h < 1024 { for i in (0..1024).step_by(2 * h) { for j in i..i + h { let (a, c) = (b[j], b[j + h]); b[j] = a + c; b[j + h] = a - c; } } h *= 2; } for x in b.iter_mut() { *x /= 32.0; } } }
fn med(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
struct Res { y: Vec<f32>, t: f64, parts: npu::split::SplitTimes }
fn run(ctx: &vkrt::Ctx, w: &[u8], m: usize, k: usize, n: usize, nn: usize, dir: &str, signs: &[f32], x: &[f32], iters: usize) -> anyhow::Result<Res> {
    let sl = SplitLinear::new(ctx, m, k, n, nn, dir, Some(signs))?; let sw = sl.load(ctx, w)?;
    let xf16 = std::env::var("XF16").is_ok();
    let xb = if xf16 { let b = ctx.buffer((m * k * 2) as u64)?; let d = b.bytes_mut(); for (i, v) in x.iter().enumerate() { d[2 * i..2 * i + 2].copy_from_slice(&half::f16::from_f32(*v).to_le_bytes()); } b }
             else { let b = ctx.buffer((m * k * 4) as u64)?; b.as_f32_mut()[..m * k].copy_from_slice(x); b };
    let yb = ctx.buffer_readback((m * n * 4) as u64)?;
    for _ in 0..3 { (if xf16 { sl.forward_f16(ctx, &sw, &xb, k, &yb, n, 0)? } else { sl.forward(ctx, &sw, &xb, k, &yb, n, 0)? }); }
    let y = yb.as_f32()[..m * n].to_vec();
    let mut ts = vec![]; let mut ps = vec![];
    for _ in 0..iters { let p = (if xf16 { sl.forward_f16(ctx, &sw, &xb, k, &yb, n, 0)? } else { sl.forward(ctx, &sw, &xb, k, &yb, n, 0)? }); ts.push(p.total); ps.push(p); }
    let mut tt = ts.clone(); let t = med(&mut tt);
    let mut parts = npu::split::SplitTimes::default();
    for f in [0usize, 1, 2, 3, 4] { let mut v: Vec<f64> = ps.iter().map(|p| [p.pre, p.gemm, p.npu, p.gpu, p.post][f]).collect(); let md = med(&mut v);
        match f { 0 => parts.pre = md, 1 => parts.gemm = md, 2 => parts.npu = md, 3 => parts.gpu = md, _ => parts.post = md } }
    ctx.free_buffer(xb); ctx.free_buffer(yb); sl.free(ctx, sw);
    Ok(Res { y, t, parts })
}
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect(); let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (m, layer, nn_gu, dir_gu, nn_dn, dir_dn) = (p(1), p(2), p(3), a[4].clone(), p(5), a[6].clone());
    let iters = if a.len() > 7 { p(7) } else { 20 };
    let h = std::env::var("HOME")?;
    let g = Gg::open(&format!("{h}/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf"))?;
    let sv: Vec<f32> = match g.kv.get("prism.hadamard.sign_values") { Some(gguf::Value::Arr(v)) => v.iter().map(|x| match x { gguf::Value::I32(i) => *i as f32, _ => 0.0 }).collect(), _ => anyhow::bail!("no signs") };
    let (s5120, s17408) = (&sv[0..5120], &sv[5120 + 6144..5120 + 6144 + 17408]);
    let d = 5120usize; let ff = 17408usize;
    // activations
    let te = g.tensor("token_embd.weight").unwrap(); let ted = g.data(te); let rb = d / 128 * 34;
    let nw_t = g.tensor(&format!("blk.{layer}.post_attention_norm.weight")).unwrap();
    let nw: Vec<f32> = g.data(nw_t).chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let nmean = nw.iter().sum::<f32>() / d as f32;
    let mut s = 1234u64; let mut x = vec![0f32; m * d]; let mut z = vec![0f32; d];
    let xfile = std::env::var("XFILE").ok();
    if let Some(f) = &xfile { // real residual stream (npu_resid dump), RMS-normed with this layer's post_attention_norm
        let raw = std::fs::read(f)?; let rv: Vec<f32> = raw.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        anyhow::ensure!(rv.len() >= m * d, "XFILE has fewer than M tokens");
        for r in 0..m { let row = &rv[r * d..(r + 1) * d]; let rms = (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / d as f64 + 1e-6).sqrt();
            for i in 0..d { x[r * d + i] = (row[i] as f64 / rms) as f32 * nw[i]; } }
    }
    for r in 0..if xfile.is_some() { 0 } else { m } { let id = (lcg(&mut s) % 150000) as usize; dequant_row(Fmt::Pq2_0, &ted[id * rb..(id + 1) * rb], &mut z);
        let mut hv: Vec<f64> = z.iter().map(|v| *v as f64).collect(); fwht(&mut hv); for i in 0..d { hv[i] *= s5120[i] as f64; }
        let rms = (hv.iter().map(|v| v * v).sum::<f64>() / d as f64 + 1e-6).sqrt();
        for i in 0..d { x[r * d + i] = (hv[i] / rms) as f32 * nw[i]; } }
    println!("M={m} layer {layer}: activations = {} RMS-normed with post_attention_norm (mean weight {nmean:.3}); x amax/rms {:.1}; input {}; iters {iters}", xfile.as_deref().unwrap_or("real embeddings (inverse-rotated)"), x.iter().fold(0f32, |a, v| a.max(v.abs())) / (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt() as f32, if std::env::var("XF16").is_ok() { "f16" } else { "f32" });
    let ctx = vkrt::Ctx::new()?;
    let mut ys_gpu: Vec<Vec<f32>> = vec![];
    println!("| op | NxK | split NPU/GPU | fused e2e ms | TOPS | pre (rot+quant) | split GEMM (NPU / GPU) | dequant | GPU-only e2e ms | TOPS | speedup | relRMS vs f32: fused NPU cols / fused GPU cols / GPU-only / exact-g128 emu (NPU cols) | fused vs GPU-only rel diff |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    // env OPS="blk.L.name:nn,..." (any PQ2_0 weight, NPU build from npu::split::build_dir); default: this layer's FFN
    let ops: Vec<(String, usize, String)> = match std::env::var("OPS") {
        Ok(v) => v.split(',').map(|e| { let (nm, nn) = e.split_once(':').unwrap(); (nm.to_string(), nn.parse().unwrap(), String::new()) }).collect(),
        Err(_) => vec![(format!("blk.{layer}.ffn_gate"), nn_gu, dir_gu.clone()), (format!("blk.{layer}.ffn_up"), nn_gu, dir_gu.clone()), (format!("blk.{layer}.ffn_down"), nn_dn, dir_dn.clone())] };
    let s6144 = &sv[5120..5120 + 6144];
    for (opfull, nn, dir0) in &ops {
        let op = opfull.rsplit('.').next().unwrap();
        let t = g.tensor(&format!("{opfull}.weight")).unwrap(); let (k, n) = (t.dims[0] as usize, t.dims[1] as usize); let w = g.data(t);
        let nn = *nn; let dir_s = if dir0.is_empty() { npu::split::build_dir(m, k, nn) } else { dir0.clone() }; let dir = dir_s.as_str();
        let sg = match k { 5120 => s5120, 6144 => s6144, _ => s17408 };
        let xin: Vec<f32> = if k == 6144 { (0..m * k).map(|i| x[(i / k) * d + (i % k) % d]).collect() } else if op == "ffn_down" { let (gt, up) = (&ys_gpu[0], &ys_gpu[1]); (0..m * ff).map(|i| { let v = gt[i]; v / (1.0 + (-v).exp()) * up[i] }).collect() } else { x.clone() };
        let fu = run(&ctx, w, m, k, n, nn, dir, sg, &xin, iters)?;
        let op = opfull.as_str();
        let go = run(&ctx, w, m, k, n, 0, dir, sg, &xin, iters)?;
        // references on samples
        let rbk = k / 128 * 34; let mut sm = 77u64;
        let (mut e_fn, mut r_fn, mut e_fg, mut r_fg, mut e_go, mut r_go, mut e_ex, mut dd, mut rr) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let emu = std::env::var("EMU").is_ok(); let mut ev = [0f64; 5];
        let rows: Vec<usize> = (0..24).map(|_| lcg(&mut sm) as usize % m).collect();
        let cols: Vec<usize> = (0..160).map(|i| if i < 80 && nn > 0 { lcg(&mut sm) as usize % nn } else { nn + lcg(&mut sm) as usize % (n - nn) }).collect();
        let wrows: Vec<Vec<f32>> = cols.iter().map(|&c| { let mut v = vec![0f32; k]; dequant_row(Fmt::Pq2_0, &w[c * rbk..(c + 1) * rbk], &mut v); v }).collect();
        let dsc: Vec<Vec<f32>> = cols.iter().map(|&c| (0..k / 128).map(|gi| ternary::f16(&w[c * rbk..], gi * 34)).collect()).collect();
        for &r in &rows {
            let mut xr: Vec<f64> = (0..k).map(|i| xin[r * k + i] as f64 * sg[i] as f64).collect(); fwht(&mut xr);
            let amax = xr.iter().fold(0f64, |a, v| a.max(v.abs())); let sx = amax / 127.0;
            let xq: Vec<f64> = xr.iter().map(|v| (v / sx).round().clamp(-127.0, 127.0)).collect();
            for (ci, &c) in cols.iter().enumerate() {
                let rf: f64 = (0..k).map(|i| xr[i] * wrows[ci][i] as f64).sum();
                let yf = fu.y[r * n + c] as f64; let yg = go.y[r * n + c] as f64;
                if c < nn && emu { // act-quant variants on the folded NPU weights (CPU emulation)
                    let smax = dsc[ci].iter().fold(0f32, |a, v| a.max(v.abs())) as f64 / 127.0;
                    let wf: Vec<f64> = (0..k).map(|i| { let dg = dsc[ci][i / 128] as f64; if dg != 0.0 { (wrows[ci][i] as f64 / dg).round() * (dg / smax).round() * smax } else { 0.0 } }).collect();
                    for (vi, clip) in [1.0f64, 0.8, 0.65, 0.5].iter().enumerate() { let sc = amax * clip / 127.0;
                        let y: f64 = (0..k).map(|i| (xr[i] / sc).round().clamp(-127.0, 127.0) * sc * wf[i]).sum(); ev[vi] += (y - rf).powi(2); }
                    let y: f64 = (0..k / 128).map(|g| { let ga = xr[g * 128..(g + 1) * 128].iter().fold(0f64, |a, v| a.max(v.abs())); let sc = if ga > 0.0 { ga / 127.0 } else { 1.0 };
                        (g * 128..(g + 1) * 128).map(|i| (xr[i] / sc).round() * sc * wf[i]).sum::<f64>() }).sum(); ev[4] += (y - rf).powi(2); }
                if c < nn { e_fn += (yf - rf).powi(2); r_fn += rf * rf;
                    let ex: f64 = (0..k).map(|i| xq[i] * sx * if dsc[ci][i / 128] != 0.0 { (wrows[ci][i] / dsc[ci][i / 128]).round() as f64 * dsc[ci][i / 128] as f64 } else { 0.0 }).sum();
                    e_ex += (ex - rf).powi(2); }
                else { e_fg += (yf - rf).powi(2); r_fg += rf * rf; }
                e_go += (yg - rf).powi(2); r_go += rf * rf; dd += (yf - yg).powi(2); rr += yg * yg;
            }
        }
        let q = |e: f64, r: f64| if r > 0.0 { format!("{:.4}", (e / r).sqrt()) } else { "-".into() };
        let ops = 2.0 * (m * k * n) as f64;
        println!("| {op} | {n}x{k} | {nn}/{} | {:.3} | {:.2} | {:.3} | {:.3} ({:.3} / {:.3}) | {:.3} | {:.3} | {:.2} | {:.2}x | {} / {} / {} / {} | {} |", n - nn, fu.t * 1e3, ops / fu.t / 1e12,
            fu.parts.pre * 1e3, fu.parts.gemm * 1e3, fu.parts.npu * 1e3, fu.parts.gpu * 1e3, fu.parts.post * 1e3, go.t * 1e3, ops / go.t / 1e12, go.t / fu.t,
            q(e_fn, r_fn), q(e_fg, r_fg), q(e_go, r_go), q(e_ex, r_fn), q(dd, rr));
        if emu { println!("  {op} NPU-cols emulation relRMS (folded weights): per-token amax {} | clip 0.8 {} | clip 0.65 {} | clip 0.5 {} | per-128-group act scales {}", q(ev[0], r_fn), q(ev[1], r_fn), q(ev[2], r_fn), q(ev[3], r_fn), q(ev[4], r_fn)); }
        if ys_gpu.len() < 2 { ys_gpu.push(go.y); }
        let _ = Instant::now();
    }
    Ok(())
}
