//! YuE2-Vae Oobleck decoder (48 kHz stereo, 1920x upsampling): conv k7 -> 6 x [SnakeBeta, ConvTranspose(2s, s),
//! 3 residual units (SnakeBeta, conv k7 dil 1/3/9, SnakeBeta, conv k1)] -> SnakeBeta -> conv k7 (no bias).
//! Time-major f32 activations [T][C]. Every conv is snake+im2col (f16, K = (ci, tap)) + prefill coop-matrix GEMM; a
//! ConvTranspose is s polyphase 2-tap convs whose GEMM writes the strided output rows (ldy = s*C, column r*C).
//! Weight norm folded on the host; weights repacked once to f16 [Cout][K] in one GPU arena. Tiled like the reference
//! decode_tiled (core 1024 latent frames, halo 16), so songs up to 1024 frames are one exact full decode.
use crate::Env;
use anyhow::Result;
use gguf::GgmlType;
use half::f16;
use prefill::GemmOp;
use qimage::st;
use rtcore::WRef;
use vkrt::{Buffer, Ctx};

struct Conv { w: WRef, cin: u32, cout: u32, k: u32, dil: u32, bias: Option<u32>, snake: Option<(u32, u32)> }
struct ConvT { ph: Vec<(WRef, i32, i32)>, s: u32, p: u32, cin: u32, cout: u32, bias: u32, snake: (u32, u32) }
enum Op { Conv(Conv), ConvT(ConvT), Res(Conv, Conv) }
pub struct Vae { arena: Vec<Buffer>, params: Buffer, ops: Vec<Op>, pub core: usize, pub halo: usize }

impl Vae {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let m = st::Model::open(ctx, dir, &|_| false)?;
        let mut wv: Vec<f16> = vec![]; let mut pv: Vec<f32> = vec![];
        let pp = |pv: &mut Vec<f32>, v: &[f32]| -> u32 { let o = pv.len() as u32; pv.extend_from_slice(v); while pv.len() % 64 != 0 { pv.push(0.0) } o };
        let snake = |pv: &mut Vec<f32>, pre: &str| -> Result<(u32, u32)> {
            let a: Vec<f32> = m.f32(&format!("{pre}.alpha"))?.iter().map(|x| x.exp()).collect(); let b: Vec<f32> = m.f32(&format!("{pre}.beta"))?.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect();
            Ok((pp(pv, &a), pp(pv, &b))) };
        // weight-normed tensor -> (dense weight, shape)
        let wn = |pre: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let g = m.f32(&format!("{pre}.weight_g"))?; let v = m.f32(&format!("{pre}.weight_v"))?; let sh = m.get(&format!("{pre}.weight_v"))?.shape.clone();
            let per = v.len() / sh[0]; let mut w = v.clone();
            for o in 0..sh[0] { let n = v[o * per..(o + 1) * per].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32; for x in &mut w[o * per..(o + 1) * per] { *x *= g[o] / n } }
            Ok((w, sh)) };
        let mut add_w = |wv: &mut Vec<f16>, rows: &[f32], n: u32, k: u32| -> WRef { while wv.len() % 64 != 0 { wv.push(f16::ZERO) } let off = wv.len() as u64 * 2; wv.extend(rows.iter().map(|x| f16::from_f32(*x)));
            WRef { arena: 0, off, ty: GgmlType::F16, k, n, e: 1, bytes: n as u64 * k as u64 * 2 } };
        let conv = |wv: &mut Vec<f16>, pv: &mut Vec<f32>, add_w: &mut dyn FnMut(&mut Vec<f16>, &[f32], u32, u32) -> WRef, pre: &str, dil: u32, sn: Option<(u32, u32)>, bias: bool| -> Result<Conv> {
            let (w, sh) = wn(pre)?; let (cout, cin, k) = (sh[0] as u32, sh[1] as u32, sh[2] as u32);
            let b = if bias { Some(pp(pv, &m.f32(&format!("{pre}.bias"))?)) } else { None };
            Ok(Conv { w: add_w(wv, &w, cout, cin * k), cin, cout, k, dil, bias: b, snake: sn }) };
        let mut ops = vec![];
        ops.push(Op::Conv(conv(&mut wv, &mut pv, &mut add_w, "decoder.layers.0", 1, None, true)?));
        let strides = [6u32, 5, 4, 4, 2, 2];
        for (bi, &s) in strides.iter().enumerate() { let pre = format!("decoder.layers.{}", bi + 1);
            let sn = snake(&mut pv, &format!("{pre}.layers.0"))?;
            let (w, sh) = wn(&format!("{pre}.layers.1"))?; let (cin, cout, k) = (sh[0], sh[1], sh[2]); assert_eq!(k as u32, 2 * s);
            let p = s.div_ceil(2); let mut ph = vec![];
            for r in 0..s { let q = (r + p) as i32; let taps: Vec<(i32, i32)> = (-1..=1).filter_map(|u| { let kk = s as i32 * u + q; (kk >= 0 && kk < 2 * s as i32).then_some((u, kk)) }).collect();
                assert_eq!(taps.len(), 2);
                let mut wr = vec![0f32; cout * cin * 2];
                for co in 0..cout { for ci in 0..cin { for (t, &(_, kk)) in taps.iter().enumerate() { wr[co * cin * 2 + ci * 2 + t] = w[(ci * cout + co) * k + kk as usize] } } }
                ph.push((add_w(&mut wv, &wr, cout as u32, cin as u32 * 2), taps[0].0, taps[1].0)); }
            let bias = pp(&mut pv, &m.f32(&format!("{pre}.layers.1.bias"))?);
            ops.push(Op::ConvT(ConvT { ph, s, p, cin: cin as u32, cout: cout as u32, bias, snake: sn }));
            for (ri, dil) in [(2, 1u32), (3, 3), (4, 9)] { let rp = format!("{pre}.layers.{ri}.layers");
                let s0 = snake(&mut pv, &format!("{rp}.0"))?; let c7 = conv(&mut wv, &mut pv, &mut add_w, &format!("{rp}.1"), dil, Some(s0), true)?;
                let s2 = snake(&mut pv, &format!("{rp}.2"))?; let c1 = conv(&mut wv, &mut pv, &mut add_w, &format!("{rp}.3"), 1, Some(s2), true)?;
                ops.push(Op::Res(c7, c1)); } }
        let sf = snake(&mut pv, "decoder.layers.7")?;
        ops.push(Op::Conv(conv(&mut wv, &mut pv, &mut add_w, "decoder.layers.8", 1, Some(sf), false)?));
        let arena = ctx.buffer(wv.len() as u64 * 2 + 4096)?; unsafe { std::ptr::copy_nonoverlapping(wv.as_ptr() as *const u8, arena.ptr, wv.len() * 2) }
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        Ok(Vae { arena: vec![arena], params, ops, core: 1024, halo: 16 })
    }

    fn out_len(&self, t: usize) -> usize { let mut l = t; for op in &self.ops { if let Op::ConvT(c) = op { l = l * c.s as usize + c.s as usize - 2 * c.p as usize } } l }

    /// Decode one tile of latents [F][64] -> interleaved stereo [Lout][2] f32 (full natural length).
    fn decode_tile(&self, e: &Env, lat: &[f32]) -> Result<(Vec<f32>, f64)> {
        let f = lat.len() / 64; let ctx = e.ctx;
        // sizes
        let (mut t, mut maxx, mut maxa) = (f as u64, 0u64, 0u64);
        for op in &self.ops { match op {
            Op::Conv(c) => { maxa = maxa.max((t + 256) * (c.cin * c.k) as u64); maxx = maxx.max(t * c.cout as u64) }
            Op::ConvT(c) => { maxa = maxa.max((t + 256) * (c.cin * 2) as u64); t = t * c.s as u64 + c.s as u64; maxx = maxx.max(t * c.cout as u64) }
            Op::Res(c7, _) => { maxa = maxa.max((t + 256) * (c7.cin * 7) as u64); maxx = maxx.max(t * c7.cout as u64) } } }
        let xa = ctx.buffer(maxx * 4 + 4096)?; let xb = ctx.buffer(maxx * 4 + 4096)?; let hb = ctx.buffer(maxx * 4 + 4096)?; let a = ctx.buffer(maxa * 2 + 4096)?;
        xa.as_f32_mut()[..lat.len()].copy_from_slice(lat);
        let pr = &self.params; let ar = &self.arena[0]; let mut err = Ok(()); let mut outl = 0usize;
        let none = 0xffff_ffffu32;
        let g = ctx.run(|cb| { err = (|| -> Result<()> {
            let (mut cur, mut oth) = (&xa, &xb); let mut t = f as u32;
            let gemm = |w: WRef, a_: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32, ycol: u32, acc: bool| -> Result<()> {
                e.pf.gemm(ctx, cb, ar, pr, &GemmOp { w, row0: 0, rows: w.n, a: a_, lda, a_row0: 0, m, y, ldy, y_col: ycol, acc })?; ctx.barrier(cb); Ok(()) };
            let im2col = |x: &Buffer, c: &Conv, t: u32| -> Result<()> { let (sa, sb) = c.snake.unwrap_or((none, none)); let ck = c.cin * c.k;
                e.misc(cb, &[x, x, &a, pr], &[1, t * ck, c.cin, c.k, c.dil, c.dil * (c.k - 1) / 2, sa, sb, t, ck, 0, 0]) };
            for op in &self.ops { match op {
                Op::Conv(c) => { im2col(cur, c, t)?; gemm(c.w, &a, c.cin * c.k, t, oth, c.cout, 0, false)?;
                    if let Some(b) = c.bias { e.misc(cb, &[oth, oth, &a, pr], &[2, t * c.cout, c.cout, 0, b, 0, 0, 0, 0, 0, 0, 0])?; }
                    std::mem::swap(&mut cur, &mut oth); }
                Op::ConvT(c) => { let lout = t * c.s + c.s - 2 * c.p;
                    for (r, &(w, u0, u1)) in c.ph.iter().enumerate() {
                        e.misc(cb, &[cur, cur, &a, pr], &[3, t * c.cin * 2, c.cin, 2, 0, 0, c.snake.0, c.snake.1, t, c.cin * 2, u0 as u32, u1 as u32])?;
                        gemm(w, &a, c.cin * 2, t, oth, c.s * c.cout, r as u32 * c.cout, false)?; }
                    e.misc(cb, &[oth, oth, &a, pr], &[2, lout * c.cout, c.cout, 0, c.bias, 0, 0, 0, 0, 0, 0, 0])?;
                    std::mem::swap(&mut cur, &mut oth); t = lout; }
                Op::Res(c7, c1) => { im2col(cur, c7, t)?; gemm(c7.w, &a, c7.cin * 7, t, &hb, c7.cout, 0, false)?;
                    e.misc(cb, &[&hb, &hb, &a, pr], &[2, t * c7.cout, c7.cout, 0, c7.bias.unwrap(), 0, 0, 0, 0, 0, 0, 0])?;
                    im2col(&hb, c1, t)?; gemm(c1.w, &a, c1.cin, t, cur, c1.cout, 0, true)?;
                    e.misc(cb, &[cur, cur, &a, pr], &[2, t * c1.cout, c1.cout, 0, c1.bias.unwrap(), 0, 0, 0, 0, 0, 0, 0])?; }
            } }
            outl = t as usize; if !std::ptr::eq(cur, &xa) { /* result in xb */ }
            Ok(()) })(); })?;
        err?;
        // the op count is even/odd fixed: find which buffer holds the result by replaying the swaps
        let swaps = self.ops.iter().filter(|o| !matches!(o, Op::Res(..))).count();
        let res = if swaps % 2 == 0 { &xa } else { &xb };
        let out = res.as_f32()[..outl * 2].to_vec();
        for b in [xa, xb, hb, a] { ctx.free_buffer(b) }
        Ok((out, g))
    }

    /// Full song: tiles of `core` frames with `halo` context, cropped exactly like decode_tiled. Returns ([N][2], gpu s).
    pub fn decode(&self, e: &Env, lat: &[f32]) -> Result<(Vec<f32>, f64)> {
        let f = lat.len() / 64; let total = self.out_len(f); let ratio = 1920usize;
        let mut out = vec![0f32; total * 2]; let mut g = 0.0;
        let mut start = 0; while start < f {
            let end = (start + self.core).min(f); let left = start.saturating_sub(self.halo); let right = (end + self.halo).min(f);
            let (tile, gt) = self.decode_tile(e, &lat[left * 64..right * 64])?; g += gt;
            let (os, oe) = (start * ratio, (end * ratio).min(total)); let cs = (start - left) * ratio;
            out[os * 2..oe * 2].copy_from_slice(&tile[cs * 2..(cs + oe - os) * 2]);
            start = end; }
        Ok((out, g))
    }
}
