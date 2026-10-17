//! PiFlow scheduler (diffusers PiflowScheduler, grid DX policy) on the CPU, f32 like the reference.
pub const SHIFT: f32 = 5.0; pub const EPS: f32 = 1e-6; pub const SUBSTEPS: f32 = 128.0; pub const FINAL: f32 = 0.5;
pub fn shift_t(t: f32) -> f32 { SHIFT * t / (1.0 + (SHIFT - 1.0) * t) }
fn unwarp(s: f32) -> f32 { s / (SHIFT + (1.0 - SHIFT) * s) }
/// (raw timesteps, sigmas) for `steps` model evaluations.
pub fn schedule(steps: usize) -> (Vec<f32>, Vec<f32>) {
    let seg = 1.0 / (steps as f32 - (1.0 - FINAL));
    let raw: Vec<f32> = (0..steps).map(|i| 1.0 - i as f32 * seg).collect();
    let sig = raw.iter().map(|&r| shift_t(r)).collect(); (raw, sig)
}
/// One scheduler step in place: x [n][c] (f32), grid out [n][NG*c] (grid-major within a row: g*c + ch).
pub fn step(x: &mut [f32], out: &[f32], c: usize, ng: usize, raw: &[f32], sig: &[f32], i: usize) {
    let raw_src = raw[i]; let raw_dst = if i + 1 < raw.len() { raw[i + 1] } else { EPS }; let sigma_src = sig[i];
    let seg = raw_src - raw_dst;
    let rs = unwarp(sigma_src); let rd = (rs - seg).max(0.0); let segsize = (rs - rd).max(EPS);
    let nsub = ((seg * SUBSTEPS).round() as i64).max(1) as usize; let h = seg / nsub as f32;
    let n = x.len() / c;
    // per-substep scalars
    let mut sched = vec![]; let (mut r, mut s) = (raw_src, sigma_src);
    for _ in 0..nsub { let rm = (r - h).max(0.0); let sm = shift_t(rm); sched.push((s, sm)); r = rm; s = sm; }
    for e in 0..n { for ch in 0..c {
        let xi = e * c + ch; let mut xt = x[xi];
        let x0g = |g: usize| x[xi] - sigma_src * out[e * ng * c + g * c + ch];
        let x0: Vec<f32> = (0..ng).map(x0g).collect();
        for &(st, sm) in &sched {
            let tt = ((unwarp(st) - rd) / segsize).clamp(0.0, 1.0) * (ng - 1) as f32;
            let t0 = (tt.floor() as i64).clamp(0, ng as i64 - 2) as usize; let t1 = t0 + 1;
            let x0i = (t1 as f32 - tt) * x0[t0] + (tt - t0 as f32) * x0[t1];
            let v = (xt - x0i) / st.max(EPS);
            xt += v * (sm - st);
        }
        x[xi] = xt; } }
}
