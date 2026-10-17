//! Coop-matrix peak probe: f16->f32 vs s8->s32 16x16x16 MMA rate (register-only).
use vkrt::{push_u32, Ctx};
fn main() -> anyhow::Result<()> {
    let ctx = Ctx::new()?;
    let groups = 40 * 16; let iters = 4096u32;
    let y = ctx.buffer(groups as u64 * 8 * 256 * 4)?;
    for (n, s) in [("f16", prefill::spv::PEAK_F16), ("s8", prefill::spv::PEAK_S8)] {
        let k = ctx.kernel(s, 1, 4, &[], Some(32))?; let ds = ctx.bind(&k, &[&y])?;
        ctx.run(|cb| ctx.dispatch(cb, &k, ds, &push_u32(&[16]), groups))?;
        let t = ctx.run(|cb| ctx.dispatch(cb, &k, ds, &push_u32(&[iters]), groups))?;
        let ops = groups as f64 * 8.0 * 8.0 * iters as f64 * 2.0 * 16.0 * 16.0 * 16.0;
        println!("{n}: {:.1} T(FL)OPS", ops / t / 1e12);
    }
    Ok(())
}
