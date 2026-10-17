//! fa_check <n_head> <n_kv> <M> <pos0> [vscale]: pf_attn vs pf_attnfa on random data (head dim 256, gated q)
use anyhow::Result;
use prefill::Pf;
fn main() -> Result<()> {
    let a: Vec<u32> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (nh, nkv, m, pos0) = (a[0], a[1], a[2], a[3]);
    let vs = std::env::args().nth(5).and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
    let ctx = vkrt::Ctx::new()?; let pf = Pf::new(&ctx)?;
    // FA_D=64: attn64 / attnfa64 (no gate, q stride 64); else head dim 256 gated
    let d64 = prefill::env("FA_D", 256) == 64;
    let d = if d64 { 64u32 } else { 256 }; let ldp = if d64 { nh * d + 2 * nkv * d } else { nh * 2 * d + 2 * nkv * d }; let lda = nh * d; let np = pos0 + m;
    let mut x = 777u64; let mut rnd = || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 };
    let q = ctx.buffer((m * ldp) as u64 * 4)?; let kc = ctx.buffer((np * nkv * d) as u64 * 2)?; let vc = ctx.buffer((np * nkv * d) as u64 * 2)?;
    let o0 = ctx.buffer((m * lda) as u64 * 2)?; let o1 = ctx.buffer((m * lda) as u64 * 2)?;
    unsafe {
        let qs = std::slice::from_raw_parts_mut(q.ptr as *mut f32, (m * ldp) as usize); for v in qs.iter_mut() { *v = rnd() * 4.0; }
        let ks = std::slice::from_raw_parts_mut(kc.ptr as *mut half::f16, (np * nkv * d) as usize); for v in ks.iter_mut() { *v = half::f16::from_f32(rnd() * 4.0); }
        let vv = std::slice::from_raw_parts_mut(vc.ptr as *mut half::f16, (np * nkv * d) as usize); for v in vv.iter_mut() { *v = half::f16::from_f32(rnd() * vs); }
    }
    let push: Vec<u32> = if d64 { vec![nh, nkv, 0, ldp, pos0, (1.0f32 / 8.0).to_bits(), lda, m] } else { vec![nh, nkv, 0, 2 * d, pos0, (1.0f32 / 16.0).to_bits(), ldp, lda, m] };
    for (fa, o) in [(0, &o0), (1, &o1)] {
        std::env::set_var("PF_FA", fa.to_string()); std::env::set_var("PF_FA64", fa.to_string());
        let cb = ctx.record_raw(|cb| { pf.run(&ctx, if d64 { "attn64" } else { "attn" }, &[&q, &kc, &vc, o], &push, m.div_ceil(16) * nh, cb).unwrap(); })?;
        let t = ctx.submit_repeated(cb, 1)?; println!("FA={fa}: {:.3} ms", t * 1e3);
    }
    let (r0, r1) = unsafe { (std::slice::from_raw_parts(o0.ptr as *const half::f16, (m * lda) as usize), std::slice::from_raw_parts(o1.ptr as *const half::f16, (m * lda) as usize)) };
    let (mut me, mut at, mut nan) = (0f32, 0usize, 0usize);
    for i in 0..r0.len() { let (u, v) = (r0[i].to_f32(), r1[i].to_f32()); if !v.is_finite() { nan += 1; } let e = (u - v).abs(); if e > me { me = e; at = i; } }
    let lr = lda as usize;
    println!("max abs err {me:.5} at row {} head {} dim {} | nonfinite {nan}", at / lr, (at % lr) / d as usize, at % d as usize);
    Ok(())
}
