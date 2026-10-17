//! UniPC multistep (bh2, order 2, predict_x0, flow_prediction, lower_order_final) with the Karras flow sigmas that
//! Cosmos3-Nano's scheduler_config selects (use_karras_sigmas + use_flow_sigmas: sigma_min 0.147, sigma_max 200, rho 7,
//! s -> s / (s + 1), final sigma 0). Port of diffusers UniPCMultistepScheduler for exactly this configuration.
pub fn karras_flow_sigmas(steps: usize, smin: f64, smax: f64) -> Vec<f64> {
    let (a, b) = (smax.powf(1.0 / 7.0), smin.powf(1.0 / 7.0));
    let mut s: Vec<f64> = (0..steps).map(|i| { let r = if steps > 1 { i as f64 / (steps - 1) as f64 } else { 0.0 }; (a + r * (b - a)).powi(7) })
        .map(|s| s / (s + 1.0)).collect();
    s.push(0.0); s
}
/// diffusers casts timesteps (= sigma * 1000 in f64) to int64 and keeps sigmas as float32.
pub fn karras_flow(steps: usize, smin: f64, smax: f64) -> (Vec<f64>, Vec<f64>) {
    let s = karras_flow_sigmas(steps, smin, smax); let ts = s[..steps].iter().map(|v| (v * 1000.0).trunc()).collect();
    (s.iter().map(|v| *v as f32 as f64).collect(), ts)
}
pub struct UniPc { pub sigmas: Vec<f64>, outs: Vec<Vec<f32>>, last: Option<Vec<f32>>, lower: usize, order: usize, pub i: usize }
fn lam(s: f64) -> f64 { (1.0 - s).ln() - s.ln() }
impl UniPc {
    pub fn new(sigmas: Vec<f64>) -> Self { UniPc { sigmas, outs: vec![], last: None, lower: 0, order: 1, i: 0 } }
    /// One scheduler step: velocity v at the current sigma, sample x -> x at the next sigma.
    pub fn step(&mut self, v: &[f32], x: &[f32]) -> Vec<f32> {
        let n = self.sigmas.len() - 1; let i = self.i; let s = &self.sigmas;
        let conv: Vec<f32> = x.iter().zip(v).map(|(a, b)| a - s[i] as f32 * b).collect();
        let mut sample = x.to_vec();
        if i > 0 { if let Some(last) = &self.last {
            // corrector (uni_c): sigma_t = s[i], s0 = s[i-1], m0 = previous model output
            let (st, s0) = (s[i], s[i - 1]); let (at, a0) = (1.0 - st, 1.0 - s0); let _ = a0;
            let h = lam(st) - lam(s0); let hh = -h; let hp1 = hh.exp_m1(); let bh = hh.exp_m1();
            let m0 = self.outs.last().unwrap();
            let mut hk = hp1 / hh - 1.0; let mut fact = 1.0; let mut b = vec![];
            for k in 1..=self.order { b.push(hk * fact / bh); fact *= (k + 1) as f64; hk = hk / hh - 1.0 / fact; }
            let (rho_d, rho_t, d1): (f64, f64, Option<Vec<f32>>) = if self.order == 1 { (0.0, 0.5, None) } else {
                let rk = (lam(s[i - 2]) - lam(s0)) / h; let mi = &self.outs[self.outs.len() - 2];
                // R = [[1, 1], [rk, 1]], solve R rho = b
                let det = 1.0 - rk; let r0 = (b[0] - b[1]) / det; let r1 = (b[1] - rk * b[0]) / det;
                (r0, r1, Some(mi.iter().zip(m0).map(|(a, c)| ((a - c) as f64 / rk) as f32).collect())) };
            for j in 0..sample.len() {
                let xt_ = st / s0 * last[j] as f64 - at * hp1 * m0[j] as f64;
                let corr = d1.as_ref().map_or(0.0, |d| rho_d * d[j] as f64);
                sample[j] = (xt_ - at * bh * (corr + rho_t * (conv[j] - m0[j]) as f64)) as f32;
            } } }
        self.outs.push(conv); if self.outs.len() > 2 { self.outs.remove(0); }
        self.order = 2usize.min(n - i).min(self.lower + 1);
        self.last = Some(sample.clone());
        // predictor (uni_p)
        let (st, s0) = (s[i + 1], s[i]); let at = 1.0 - st;
        let h = lam(st) - lam(s0); let hh = -h; let hp1 = hh.exp_m1(); let bh = hh.exp_m1();
        let m0 = self.outs.last().unwrap();
        let d1: Option<Vec<f32>> = if self.order == 2 { let rk = (lam(s[i - 1]) - lam(s0)) / h; let mi = &self.outs[0];
            Some(mi.iter().zip(m0).map(|(a, c)| ((a - c) as f64 / rk) as f32).collect()) } else { None };
        let ratio = if st == 0.0 { 0.0 } else { st / s0 };
        let out = (0..sample.len()).map(|j| { let xt_ = ratio * sample[j] as f64 - at * hp1 * m0[j] as f64;
            let pr = d1.as_ref().map_or(0.0, |d| 0.5 * d[j] as f64); (xt_ - at * bh * pr) as f32 }).collect();
        if self.lower < 2 { self.lower += 1 } self.i += 1; out
    }
}
