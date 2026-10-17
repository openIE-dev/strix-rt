//! Ternary GEMV + FWHT benchmark on real tensors. Cache-defeating: weight copies totalling >= 256 MiB,
//! each dispatch reads the next copy (same method as crates/bench).
//! usage: tern_bench [pq2|ptq1|kitten|kitten_st|parakeet|fwht|all]   env ITERS (default 100), SG (64), SWEEP=1 (all variants)
use anyhow::Result;
use ternary::{dequant_row, fwht_ref, gg::Gg, shaders, st::St, thrush_row, thrush_row_flat, Fmt};
use vkrt::{push_u32, Buffer, Ctx};

fn lcg(s: &mut u64) -> f32 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }

/// weights source: either a ggml-block tensor or thrush rows + f32 scales
struct Wt { label: String, fmt: Fmt, n: usize, k: usize, data: Vec<u8>, scales: Vec<f32>, flat: bool }

fn cpu_ref(w: &Wt, x: &[f32]) -> Vec<f32> {
    let (n, k) = (w.n, w.k); let mut y = vec![0f32; n]; let nt = 32; let chunk = n.div_ceil(nt);
    std::thread::scope(|s| { for (ci, yc) in y.chunks_mut(chunk).enumerate() { s.spawn(move || { let mut buf = vec![0f32; k]; let g = k.div_ceil(128);
        for (i, yv) in yc.iter_mut().enumerate() { let r = ci * chunk + i;
            match w.fmt { Fmt::Thrush5 => if w.flat { thrush_row_flat(&w.data, r, k, &w.scales[r * g..(r + 1) * g], &mut buf) } else { let rb = k.div_ceil(5); thrush_row(&w.data[r * rb..(r + 1) * rb], &w.scales[r * g..(r + 1) * g], &mut buf) },
                f => { let rb = f.row_bytes(k); dequant_row(f, &w.data[r * rb..(r + 1) * rb], &mut buf) } }
            *yv = buf.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32; } }); } });
    y
}

fn bench_gemv(ctx: &Ctx, w: &Wt, iters: u32, sweep: bool, sg: u32, out: &mut Vec<String>) -> Result<()> {
    let size = w.data.len() as u64; let al = 256u64; let stride = size.div_ceil(al) * al;
    let copies = ((256u64 << 20).div_ceil(size)).clamp(1, 4096);
    let wb = ctx.buffer(stride * copies)?;
    for c in 0..copies as usize { wb.bytes_mut()[c * stride as usize..c * stride as usize + size as usize].copy_from_slice(&w.data); }
    let sbytes = (w.scales.len() * 4) as u64; let scopies = if sbytes > 0 { copies } else { 1 };
    let sb = ctx.buffer((sbytes.max(16)) * scopies)?;
    for c in 0..scopies as usize { if sbytes > 0 { sb.as_f32_mut()[c * w.scales.len()..(c + 1) * w.scales.len()].copy_from_slice(&w.scales); } }
    let xb = ctx.buffer((w.k as u64 * 4).max(16) + 64)?; let yb = ctx.buffer(w.n as u64 * 4)?;
    let mut seed = 7u64; for v in xb.as_f32_mut() { *v = lcg(&mut seed); }
    let x = xb.as_f32()[..w.k].to_vec(); let r = cpu_ref(w, &x);
    let rmax = r.iter().fold(0f32, |m, v| m.max(v.abs()));
    let variants: Vec<(u32, u32)> = if sweep { [4u32, 8, 16, 32].iter().flat_map(|&t| [1u32, 2, 4].map(move |nr| (t, nr))).collect() } else { vec![(8, 2), (16, 2), (32, 1)] };
    let mut best: Option<(f64, String)> = None;
    for (t, nr) in variants {
        let kern = ctx.kernel(shaders::gemv(w.fmt.spv_tag(), t, nr), 4, 24, &[], Some(sg))?;
        let ds = ctx.bind(&kern, &[&wb, &xb, &yb, &sb])?;
        let rpw = (64 / t) * nr; let groups = (w.n as u32).div_ceil(rpw);
        let push = |c: u64| push_u32(&[w.n as u32, w.k as u32, (c * stride) as u32, 0, (c as usize * w.scales.len()) as u32, w.flat as u32]);
        yb.as_f32_mut().fill(f32::NAN);
        ctx.run(|cb| ctx.dispatch(cb, &kern, ds, &push(0), groups))?;
        let (mut maxabs, mut maxrel) = (0f32, 0f32);
        for (a, b) in yb.as_f32().iter().zip(&r) { let d = if a.is_nan() { f32::INFINITY } else { (a - b).abs() }; maxabs = maxabs.max(d); }
        maxrel = maxrel.max(maxabs / rmax);
        let run = |it: u32| ctx.run(|cb| for i in 0..it { ctx.dispatch(cb, &kern, ds, &push(i as u64 % copies), groups); ctx.barrier(cb); });
        run(10)?; let per = run(iters)? / iters as f64;
        let bytes = size + sbytes + w.k as u64 * 4 + w.n as u64 * 4; let gbs = bytes as f64 / per / 1e9;
        let wps = (w.n * w.k) as f64 / per / 1e9;
        let line = format!("| {} | {} | {}x{} | T{t}/R{nr} | {:.1} | {copies} | {:.1} | {:.1} | {:.1}% | {:.0} | {:.2e} | {:.2e} |", w.label, w.fmt.name(), w.n, w.k, size as f64 / 1048576.0, per * 1e6, gbs, gbs / 256.0 * 100.0, wps, maxabs, maxrel);
        println!("{line}");
        if best.as_ref().map(|b| gbs > b.0).unwrap_or(true) { best = Some((gbs, line)); }
        unsafe { ctx.device.destroy_pipeline(kern.pipeline, None); }
    }
    out.push(best.unwrap().1);
    for b in [wb, sb, xb, yb] { ctx.free_buffer(b); }
    Ok(())
}

fn ggml_wt(g: &Gg, name: &str, label: &str) -> Wt { let t = g.tensor(name).unwrap(); Wt { label: label.into(), fmt: Fmt::from_ggml(t.ty).unwrap(), n: t.dims[1] as usize, k: t.dims[0] as usize, data: g.data(t).to_vec(), scales: vec![], flat: false } }

fn bench_fwht(ctx: &Ctx, g: &Gg, iters: u32, out: &mut Vec<String>) -> Result<()> {
    // signs from GGUF KV: prism.hadamard.sign_widths / sign_values (concatenated per width)
    let widths: Vec<usize> = match g.kv.get("prism.hadamard.sign_widths") { Some(gguf::Value::Arr(a)) => a.iter().map(|v| match v { gguf::Value::I32(x) => *x as usize, _ => 0 }).collect(), _ => vec![] };
    let vals: Vec<f32> = match g.kv.get("prism.hadamard.sign_values") { Some(gguf::Value::Arr(a)) => a.iter().map(|v| match v { gguf::Value::I32(x) => *x as f32, _ => 0.0 }).collect(), _ => vec![] };
    let kern = ctx.kernel(shaders::FWHT, 3, 28, &[], None)?;
    let mut off = 0;
    for &k in &widths {
        let signs = &vals[off..off + k]; off += k;
        // ssm_out (K=6144) needs the GDN v-grouped permutation: hd=128 (6144/48), nk=16, rep=3
        let perm = if k == 6144 { Some((128usize, 16usize, 3usize)) } else { None };
        let copies = 64u64; // x is tiny; cycle 64 copies anyway
        let xb = ctx.buffer(k as u64 * 4 * copies)?; let yb = ctx.buffer(k as u64 * 4 * copies)?; let sb = ctx.buffer(k as u64 * 4)?;
        sb.as_f32_mut().copy_from_slice(signs);
        let mut seed = 3u64; for v in xb.as_f32_mut() { *v = lcg(&mut seed); }
        let x = xb.as_f32()[..k].to_vec(); let r = fwht_ref(&x, Some(signs), 1024, perm);
        let ds = ctx.bind(&kern, &[&xb, &yb, &sb])?;
        let (hd, nk, rep) = perm.unwrap_or((1, 1, 1));
        let push = |c: u64| push_u32(&[k as u32, (c * k as u64) as u32, (c * k as u64) as u32, 1, hd as u32, nk as u32, rep as u32]);
        ctx.run(|cb| ctx.dispatch(cb, &kern, ds, &push(0), (k / 1024) as u32))?;
        let maxabs = yb.as_f32()[..k].iter().zip(&r).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let rmax = r.iter().fold(0f32, |m, v| m.max(v.abs()));
        let run = |it: u32| ctx.run(|cb| for i in 0..it { ctx.dispatch(cb, &kern, ds, &push(i as u64 % copies), (k / 1024) as u32); ctx.barrier(cb); });
        run(10)?; let per = run(iters)? / iters as f64;
        // round trip: inverse transform (signs after H) of the forward output must return x (no-perm widths)
        let mut rt = String::new();
        if perm.is_none() { let zb = ctx.buffer(k as u64 * 4)?; let ds2 = ctx.bind(&kern, &[&yb, &zb, &sb])?;
            ctx.run(|cb| { ctx.dispatch(cb, &kern, ds, &push(0), (k / 1024) as u32); ctx.barrier(cb); ctx.dispatch(cb, &kern, ds2, &push_u32(&[k as u32, 0, 0, 2, 1, 1, 1]), (k / 1024) as u32); })?;
            let e = zb.as_f32().iter().zip(&x).fold(0f32, |m, (a, b)| m.max((a - b).abs())); rt = format!(" | inverse round-trip max_abs {e:.2e}"); ctx.free_buffer(zb); }
        let line = format!("| fwht K={k}{} | {} WGs | {:.2} us | max_abs {:.2e} | max_rel {:.2e}{rt} |", if perm.is_some() { " +GDN perm" } else { "" }, k / 1024, per * 1e6, maxabs, maxabs / rmax);
        println!("{line}"); out.push(line);
        for b in [xb, yb, sb] { ctx.free_buffer(b); }
    }
    Ok(())
}

fn main() -> Result<()> {
    let what = std::env::args().nth(1).unwrap_or("all".into());
    let iters: u32 = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let sg: u32 = std::env::var("SG").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let sweep = std::env::var("SWEEP").is_ok();
    let h = std::env::var("HOME")?;
    let ctx = Ctx::new()?;
    println!("device {} / {} subgroup={sg} iters={iters}", ctx.info.name, ctx.info.driver_info);
    println!("| tensor | fmt | N x K | cfg | MiB | copies | us | GB/s | % of 256 | Gweights/s | max_abs | max_abs/max|ref| |\n|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut best = vec![];
    let bonsai = [("blk.0.attn_qkv.weight", "attn_qkv (GDN in)"), ("blk.0.attn_gate.weight", "attn_gate (GDN z)"), ("blk.0.ssm_out.weight", "ssm_out"), ("blk.3.attn_q.weight", "attn_q (+gate)"), ("blk.3.attn_k.weight", "attn_k/v"), ("blk.3.attn_output.weight", "attn_output"),
                  ("blk.0.ffn_gate.weight", "ffn_gate/up"), ("blk.0.ffn_down.weight", "ffn_down"), ("output.weight", "lm_head")];
    for (tag, file) in [("pq2", "Ternary-Bonsai-2-27B-PQ2_0.gguf"), ("ptq1", "Ternary-Bonsai-2-27B-PTQ1_0.gguf")] {
        if what != tag && what != "all" { continue }
        let g = Gg::open(&format!("{h}/models/bonsai2/{file}"))?;
        for (n, l) in bonsai { bench_gemv(&ctx, &ggml_wt(&g, n, l), iters, sweep, sg, &mut best)?; }
    }
    if what == "kitten" || what == "all" {
        let g = Gg::open(&format!("{h}/models/kitten2/model-tq2_1.gguf"))?;
        for (n, l) in [("blk.0.attn_q.weight", "kitten attn_q/o"), ("blk.0.attn_k.weight", "kitten attn_k/v"), ("blk.0.ffn_up.weight", "kitten ffn_gate/up"), ("blk.0.ffn_down.weight", "kitten ffn_down")] { bench_gemv(&ctx, &ggml_wt(&g, n, l), iters, sweep, sg, &mut best)?; }
    }
    if what == "kitten_st" || what == "all" {
        let s = St::open(&format!("{h}/models/kitten2/model-ternary.safetensors"))?;
        for (n, l) in [("model.layers.0.mlp.up_proj.weight", "kitten-st ffn_up"), ("model.layers.0.mlp.down_proj.weight", "kitten-st ffn_down")] {
            let sc = s.get(&format!("{n}::scale")).unwrap();
            best_push(&ctx, Wt { label: l.into(), fmt: Fmt::Thrush5, n: sc.shape[0], k: sc.shape[1] * 128, data: s.data(s.get(&format!("{n}::codes")).unwrap()).to_vec(), scales: s.f32s(sc), flat: true }, iters, sweep, sg, &mut best)?;
        }
    }
    if what == "parakeet" || what == "all" {
        let s = St::open(&format!("{h}/models/parakeet-redux/model.safetensors"))?;
        for (n, l) in [("encoder.layers.0.feed_forward1.linear1", "parakeet ff linear1"), ("encoder.layers.0.feed_forward1.linear2", "parakeet ff linear2"), ("encoder.layers.0.self_attn.q_proj", "parakeet q_proj")] {
            let q = s.get(&format!("{n}.qweight")).unwrap(); let sc = s.get(&format!("{n}.scales")).unwrap();
            let k = (sc.shape[1] * 128).min(q.shape[1] * 5);
            best_push(&ctx, Wt { label: l.into(), fmt: Fmt::Thrush5, n: q.shape[0], k, data: s.data(q).to_vec(), scales: s.f32s(sc), flat: false }, iters, sweep, sg, &mut best)?;
        }
    }
    if what == "fwht" || what == "all" { let g = Gg::open(&format!("{h}/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf"))?; bench_fwht(&ctx, &g, iters, &mut best)?; }
    println!("\n### best config per tensor\n");
    for b in best { println!("{b}"); }
    Ok(())
}
fn best_push(ctx: &Ctx, w: Wt, iters: u32, sweep: bool, sg: u32, best: &mut Vec<String>) -> Result<()> { bench_gemv(ctx, &w, iters, sweep, sg, best) }
#[allow(dead_code)] fn _b(_: &Buffer) {}
