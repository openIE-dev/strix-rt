//! Small-input kernel tests for nemotron35 (no full-model load, sub-second GPU work):
//! `n35_kt <model.gguf> [layer=1]`: Q4_0 lgemv (latent down/up), router (512 experts, top-22, NaN input), expert up/down vs CPU.
use anyhow::{bail, Result};
use half::f16;
use vkrt::{push_u32, Ctx};
macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
fn deq_q40(b: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (bi, blk) in b.chunks_exact(18).enumerate().take(n / 32) { let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for j in 0..16 { o[bi * 32 + j] = d * ((blk[2 + j] & 15) as f32 - 8.0); o[bi * 32 + 16 + j] = d * ((blk[2 + j] >> 4) as f32 - 8.0); } }
    o
}
struct Rng(u64);
impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 } }
fn rel(a: &[f32], b: &[f32]) -> f32 { let (mut e, mut n) = (0f64, 0f64); for i in 0..a.len() { e += ((a[i] - b[i]) as f64).powi(2); n += (b[i] as f64).powi(2) } (e / n.max(1e-30)).sqrt() as f32 }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = gguf::Gguf::open(&a[1])?; let l: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let ctx = Ctx::new()?; let ds = rtcore::DsCache::default();
    let moe = [ctx.kernel(spv!("n35_moe"), 5, 24, &[0], None)?, ctx.kernel(spv!("n35_moe"), 5, 24, &[1], None)?, ctx.kernel(spv!("n35_moe"), 5, 24, &[2], None)?];
    let router = ctx.kernel(spv!("n35_router"), 4, 28, &[0], Some(32))?; let router_pre = ctx.kernel(spv!("n35_router"), 4, 28, &[1], Some(32))?; let rlog = ctx.kernel(spv!("n35_rlog"), 3, 12, &[], Some(32))?;
    let t = |n: &str| -> Result<&gguf::TensorInfo> { g.tensor(&format!("blk.{l}.{n}")).ok_or_else(|| anyhow::anyhow!("missing {n}")) };
    let up = |bytes: &[u8]| -> Result<vkrt::Buffer> { let b = ctx.buffer(bytes.len().max(256) as u64)?; b.bytes_mut()[..bytes.len()].copy_from_slice(bytes); Ok(b) };
    let fb = |v: &[f32]| -> Result<vkrt::Buffer> { let b = ctx.buffer((v.len().max(64) * 4) as u64)?; b.as_f32_mut()[..v.len()].copy_from_slice(v); Ok(b) };
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let dummy = ctx.buffer(4096)?; let aux = ctx.buffer(512 * 4)?;
    // 1. lgemv Q4_0 (latent down: 4096 -> 1024, latent up: 1024 -> 4096)
    for name in ["ffn_latent_down.weight", "ffn_latent_up.weight"] {
        let ti = t(name)?; if ti.ty != gguf::GgmlType::Q4_0 { bail!("{name} is {:?}", ti.ty) }
        let (k, n) = (ti.dims[0] as usize, ti.dims[1] as usize); let w = up(g.data(ti))?;
        let x: Vec<f32> = (0..k).map(|_| rng.f()).collect(); let xb = fb(&x)?; let yb = ctx.buffer((n * 4) as u64)?;
        let d = ds.get(&ctx, &moe[2], &[&w, &xb, &yb, &dummy, &aux])?;
        ctx.run(|cb| ctx.dispatch(cb, &moe[2], d, &push_u32(&[0, 0, n as u32, k as u32, 16, 0]), (n as u32).div_ceil(8)))?;
        let wd = deq_q40(g.data(ti), k * n); let r: Vec<f32> = (0..n).map(|i| (0..k).map(|j| wd[i * k + j] * x[j]).sum()).collect();
        println!("lgemv {name} [{n}x{k}] Q4_0: rel err {:.2e}  y[..3] gpu {:?} cpu {:?}", rel(&yb.as_f32()[..n], &r), &yb.as_f32()[..3], &r[..3]);
    }
    // 2. router
    let (gw, gb) = (t("ffn_gate_inp.weight")?, t("exp_probs_b.bias")?);
    let (k, ne) = (gw.dims[0] as usize, gw.dims[1] as usize);
    let wf: Vec<f32> = g.data(gw).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let bf: Vec<f32> = g.data(gb).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut fv = wf.clone(); let boff = fv.len(); fv.extend(&bf); let fbuf = fb(&fv)?;
    let (topk, scale) = (22usize, 5.0f32);
    let mut ids = vec![]; let mut wts = vec![];
    for case in [1, 0] {
        let x: Vec<f32> = (0..k).map(|_| if case == 1 { f32::NAN } else { rng.f() * 0.05 }).collect(); let xb = fb(&x)?;
        let prb = ctx.buffer(4096)?;
        if case == 0 {   // two-kernel path used by the model: n35_rlog (logits, 64 workgroups) then n35_router PRE=1
            let d0 = ds.get(&ctx, &rlog, &[&fbuf, &xb, &prb])?; let d1 = ds.get(&ctx, &router_pre, &[&fbuf, &xb, &aux, &prb])?;
            ctx.run(|cb| { ctx.dispatch(cb, &rlog, d0, &push_u32(&[0, k as u32, ne as u32]), (ne as u32).div_ceil(8)); ctx.barrier(cb);
                ctx.dispatch(cb, &router_pre, d1, &push_u32(&[0, boff as u32, k as u32, ne as u32, topk as u32, 1, scale.to_bits()]), 1); })?;
        } else {
            let d = ds.get(&ctx, &router, &[&fbuf, &xb, &aux, &prb])?;
            ctx.run(|cb| ctx.dispatch(cb, &router, d, &push_u32(&[0, boff as u32, k as u32, ne as u32, topk as u32, 1, scale.to_bits()]), 1))?;
        }
        let au = unsafe { std::slice::from_raw_parts(aux.ptr as *const u32, 512) };
        let gi: Vec<u32> = au[256..256 + topk].to_vec(); let gwt: Vec<f32> = au[320..320 + topk].iter().map(|&v| f32::from_bits(v)).collect();
        if case == 1 { println!("router NaN input: ids all < {ne}: {} (ids[..4] {:?})", gi.iter().all(|&i| (i as usize) < ne), &gi[..4]); continue }
        let p: Vec<f32> = (0..ne).map(|e| 1.0 / (1.0 + (-(0..k).map(|j| wf[e * k + j] * x[j]).sum::<f32>()).exp())).collect();
        let mut o: Vec<usize> = (0..ne).collect(); o.sort_by(|&a, &b| (p[b] + bf[b]).partial_cmp(&(p[a] + bf[a])).unwrap().then(a.cmp(&b)));
        let sel = &o[..topk]; let s: f32 = sel.iter().map(|&e| p[e]).sum();
        let cw: Vec<f32> = sel.iter().map(|&e| p[e] / s * scale).collect();
        println!("router: ids match {} | weights rel err {:.2e} | gpu ids[..6] {:?} cpu {:?}", gi.iter().zip(sel).all(|(a, b)| *a as usize == *b), rel(&gwt, &cw), &gi[..6], &sel[..6]);
        ids = gi; wts = gwt;
    }
    // 3. expert up (relu^2) and down for the routed ids
    let (ue, de) = (t("ffn_up_exps.weight")?, t("ffn_down_exps.weight")?);
    let (lk, ff) = (ue.dims[0] as usize, ue.dims[1] as usize); let ub = up(g.data(ue))?; let db = up(g.data(de))?;
    let xl: Vec<f32> = (0..lk).map(|_| rng.f()).collect(); let xb = fb(&xl)?; let hm = ctx.buffer((topk * ff * 4) as u64)?; let yb = ctx.buffer((lk * 4) as u64)?;
    let (us, dsz) = ((g.data(ue).len() / 512) as u32, (g.data(de).len() / 512) as u32);
    let d1 = ds.get(&ctx, &moe[0], &[&ub, &xb, &hm, &dummy, &aux])?; let d2 = ds.get(&ctx, &moe[1], &[&db, &hm, &yb, &dummy, &aux])?;
    ctx.run(|cb| { ctx.dispatch3(cb, &moe[0], d1, &push_u32(&[0, us, ff as u32, lk as u32, 16, topk as u32]), (ff as u32).div_ceil(8), topk as u32, 1); ctx.barrier(cb);
        ctx.dispatch(cb, &moe[1], d2, &push_u32(&[0, dsz, lk as u32, ff as u32, 16, topk as u32]), (lk as u32).div_ceil(8)); })?;
    let mut hr = vec![0f32; topk * ff]; let mut yr = vec![0f32; lk];
    for (s, &e) in ids.iter().enumerate() {
        let wu = deq_q40(&g.data(ue)[e as usize * us as usize..(e as usize + 1) * us as usize], ff * lk);
        for r in 0..ff { let v: f32 = (0..lk).map(|j| wu[r * lk + j] * xl[j]).sum(); hr[s * ff + r] = v.max(0.0).powi(2); }
        let wd = deq_q40(&g.data(de)[e as usize * dsz as usize..(e as usize + 1) * dsz as usize], lk * ff);
        for r in 0..lk { yr[r] += wts[s] * (0..ff).map(|j| wd[r * ff + j] * hr[s * ff + j]).sum::<f32>(); }
    }
    println!("expert up relu^2 [22x{ff}]: rel err {:.2e} | expert down -> latent [{lk}]: rel err {:.2e}", rel(&hm.as_f32()[..topk * ff], &hr), rel(&yb.as_f32()[..lk], &yr));
    Ok(())
}
