//! Kandinsky 6 audio decode: MMAudio VAE decoder (latents [40][L] -> log-mel [128][2L]) and the BigVGAN-v2 vocoder
//! (mel -> 44.1 kHz waveform, x512). Channel-last f32 activations; every conv is a 1D im2col (f16, K = k*Cin padded to 64)
//! followed by prefill's GEMM, transposed convs are GEMM + col2im, anti-aliased SnakeBeta is one fused kernel.
//! The VAE's mid attention (88 tokens, 1 head, d 2048) runs on the CPU.
use crate::Kern;
use anyhow::Result;
use gguf::GgmlType;
use half::f16;
use prefill::{GemmOp, Pf};
use qimage::st;
use rtcore::WRef;
use vkrt::{push_u32, Buffer, Ctx};

#[derive(Clone, Copy)] pub struct C1 { w: usize, b: u32, cin: u32, cout: u32, k: u32, dil: u32, pad: u32, kpad: u32, tr: u32 }
pub struct Audio { ws: Vec<(Buffer, WRef)>, pub p: Buffer, pv: Vec<f32>, zero: u32, mean: Vec<f32>, std: Vec<f32>, pub load_s: f64,
    dec: std::collections::HashMap<String, C1>, voc: std::collections::HashMap<String, C1>, snake: std::collections::HashMap<String, (u32, u32)> }

fn kaiser(cutoff: f64, hw: f64, n: usize) -> Vec<f32> {
    let half = n / 2; let att = 2.285 * (half as f64 - 1.0) * std::f64::consts::PI * 4.0 * hw + 7.95;
    let beta = if att > 50.0 { 0.1102 * (att - 8.7) } else if att >= 21.0 { 0.5842 * (att - 21.0).powf(0.4) + 0.07886 * (att - 21.0) } else { 0.0 };
    let i0 = |x: f64| { let mut s = 1.0; let mut t = 1.0; for k in 1..50 { t *= (x / 2.0) / k as f64; s += t * t; } s };
    let mut f: Vec<f64> = (0..n).map(|i| {
        let w = i0(beta * (1.0 - (2.0 * i as f64 / (n - 1) as f64 - 1.0).powi(2)).sqrt()) / i0(beta);
        let t = if n % 2 == 0 { i as f64 - half as f64 + 0.5 } else { i as f64 - half as f64 };
        let a = 2.0 * cutoff * t; let sinc = if a == 0.0 { 1.0 } else { (std::f64::consts::PI * a).sin() / (std::f64::consts::PI * a) };
        2.0 * cutoff * w * sinc }).collect();
    let s: f64 = f.iter().sum(); for v in f.iter_mut() { *v /= s } f.iter().map(|&v| v as f32).collect()
}
fn put(pv: &mut Vec<f32>, x: &[f32]) -> u32 { let o = pv.len() as u32; pv.extend_from_slice(x); while pv.len() % 64 != 0 { pv.push(0.0) } o }

impl Audio {
    pub fn load(ctx: &Ctx, vae_dir: &str, voc_dir: &str) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let mut pv = kaiser(0.25, 0.3, 12); pv.extend(kaiser(0.25, 0.3, 12)); while pv.len() % 64 != 0 { pv.push(0.0) }
        let zero = put(&mut pv, &vec![0.0; 2048]);
        let mut ws = vec![];
        let mut conv = |ws: &mut Vec<(Buffer, WRef)>, pv: &mut Vec<f32>, s: &st::Model, key: &str, dil: u32, pad: Option<u32>, gain: f32, tr: u32| -> Result<C1> {
            let t = s.get(&format!("{key}.weight"))?; let sh = t.shape.clone(); let w = s.f32(&format!("{key}.weight"))?;
            let (cin, cout, k) = if tr > 0 { (sh[0], sh[1], sh[2]) } else { (sh[1], sh[0], sh[2]) };
            let (rows, kk) = if tr > 0 { (cout * k, cin) } else { (cout, k * cin) }; let kpad = kk.div_ceil(64) * 64;
            let buf = ctx.buffer((rows * kpad * 2) as u64)?;
            { let o = unsafe { std::slice::from_raw_parts_mut(buf.ptr as *mut f16, rows * kpad) }; o.fill(f16::ZERO);
              if tr > 0 { for ci in 0..cin { for co in 0..cout { for j in 0..k { o[(co * k + j) * kpad + ci] = f16::from_f32(w[(ci * cout + co) * k + j] * gain) } } } }
              else { for co in 0..cout { for ci in 0..cin { for j in 0..k { o[co * kpad + j * cin + ci] = f16::from_f32(w[(co * cin + ci) * k + j] * gain) } } } } }
            ws.push((buf, WRef { arena: 0, off: 0, ty: GgmlType::F16, k: kpad as u32, n: rows as u32, e: 1, bytes: (rows * kpad * 2) as u64 }));
            let b = match s.get(&format!("{key}.bias")) { Ok(_) => put(pv, &s.f32(&format!("{key}.bias"))?), Err(_) => zero };
            let pad = pad.unwrap_or(((k as u32 - 1) * dil) / 2);
            Ok(C1 { w: ws.len() - 1, b, cin: cin as u32, cout: cout as u32, k: k as u32, dil, pad, kpad: kpad as u32, tr })
        };
        let sv = st::Model::open(ctx, vae_dir, &|k| k.starts_with("vae."))?;
        let mut dec = std::collections::HashMap::new();
        let gain = sv.f32("vae.decoder.learnable_gain")?[0] + 1.0;
        let mut keys: Vec<String> = sv.t.keys().filter(|k| k.starts_with("vae.decoder.") && k.ends_with(".weight")).cloned().collect(); keys.sort();
        for k in keys { let base = k.strip_suffix(".weight").unwrap().to_string();
            let g = if base.ends_with("conv_out") { gain } else { 1.0 };
            dec.insert(base.trim_start_matches("vae.decoder.").to_string(), conv(&mut ws, &mut pv, &sv, &base, 1, None, g, 0)?); }
        let mean = sv.f32("vae.data_mean")?; let std = sv.f32("vae.data_std")?;
        let so = st::Model::open(ctx, voc_dir, &|_| true)?;
        let mut voc = std::collections::HashMap::new(); let mut snake = std::collections::HashMap::new();
        let rates = [8u32, 4, 2, 2, 2, 2]; let dils = [1u32, 3, 5];
        voc.insert("conv_pre".into(), conv(&mut ws, &mut pv, &so, "conv_pre", 1, None, 1.0, 0)?);
        voc.insert("conv_post".into(), conv(&mut ws, &mut pv, &so, "conv_post", 1, None, 1.0, 0)?);
        for (i, &r) in rates.iter().enumerate() { let c = conv(&mut ws, &mut pv, &so, &format!("ups.{i}.0"), r, None, 1.0, r)?;
            let kk = c.k; voc.insert(format!("ups.{i}"), C1 { pad: (kk - r) / 2, ..c }); }
        for b in 0..18 { for (j, &d) in dils.iter().enumerate() {
            voc.insert(format!("rb.{b}.c1.{j}"), conv(&mut ws, &mut pv, &so, &format!("resblocks.{b}.convs1.{j}"), d, None, 1.0, 0)?);
            voc.insert(format!("rb.{b}.c2.{j}"), conv(&mut ws, &mut pv, &so, &format!("resblocks.{b}.convs2.{j}"), 1, None, 1.0, 0)?); }
            for a in 0..6 { let pre = format!("resblocks.{b}.activations.{a}.act");
                let al: Vec<f32> = so.f32(&format!("{pre}.alpha"))?.iter().map(|v| v.exp()).collect();
                let be: Vec<f32> = so.f32(&format!("{pre}.beta"))?.iter().map(|v| 1.0 / (v.exp() + 1e-9)).collect();
                let (x, y) = (put(&mut pv, &al), put(&mut pv, &be)); snake.insert(format!("rb.{b}.a.{a}"), (x, y)); } }
        { let al: Vec<f32> = so.f32("activation_post.act.alpha")?.iter().map(|v| v.exp()).collect();
          let be: Vec<f32> = so.f32("activation_post.act.beta")?.iter().map(|v| 1.0 / (v.exp() + 1e-9)).collect();
          let (x, y) = (put(&mut pv, &al), put(&mut pv, &be)); snake.insert("post".into(), (x, y)); }
        // scratch region for row norms (VAE normalize), up to 4096 rows
        let p = ctx.buffer(((pv.len() + 8192) * 4) as u64)?; p.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        Ok(Audio { ws, p, pv, zero, mean, std, load_s: t0.elapsed().as_secs_f64(), dec, voc, snake })
    }
}

pub struct ARun<'a> { ctx: &'a Ctx, pf: &'a Pf, kn: &'a Kern, a: &'a Audio, col: Buffer, y: Buffer, dummy: Buffer, pub gpu: f64, nrm: u32 }
impl<'a> ARun<'a> {
    pub fn new(ctx: &'a Ctx, pf: &'a Pf, kn: &'a Kern, a: &'a Audio) -> Result<Self> {
        Ok(ARun { ctx, pf, kn, a, col: ctx.buffer(256 << 20)?, y: ctx.buffer(128 << 20)?, dummy: ctx.buffer(4096)?, gpu: 0.0, nrm: a.pv.len() as u32 })
    }
    fn k(&self, cb: ash::vk::CommandBuffer, x: &Buffer, o: &Buffer, push: [u32; 12]) -> Result<()> {
        let kk = &self.kn.k["k6a1d"]; let ds = self.pf.ds.get(self.ctx, kk, &[x, &self.y, &self.col, o, &self.a.p])?;
        self.ctx.dispatch(cb, kk, ds, &push_u32(&push), push[1].div_ceil(256).clamp(1, 65535)); self.ctx.barrier(cb); Ok(())
    }
    /// o = ca*o (if acc) + cb*conv(pre(x)); x has length l (input domain). Returns output length.
    pub fn conv(&mut self, c: &C1, x: &Buffer, l: u32, o: &Buffer, pre: u32, acc: bool, ca: f32, cb_: f32, clamp: bool) -> Result<u32> {
        let (wb, wr) = (&self.a.ws[c.w].0, self.a.ws[c.w].1); let nrm = self.nrm;
        let mut err = Ok(()); let mut lout = 0;
        self.gpu += self.ctx.run(|cb| { err = (|| -> Result<()> {
            if c.tr > 0 {
                self.k(cb, x, &self.dummy, [0, l * c.kpad, c.cin, l, c.kpad, 1, 1, 0, 0, 0, 0, 0])?;
                self.pf.gemm(self.ctx, cb, wb, &self.a.p, &GemmOp { w: wr, row0: 0, rows: wr.n, a: &self.col, lda: c.kpad, a_row0: 0, m: l, y: &self.y, ldy: wr.n, y_col: 0, acc: false })?; self.ctx.barrier(cb);
                lout = (l - 1) * c.dil - 2 * c.pad + c.k;
                self.k(cb, x, o, [5, lout * c.cout, c.cout, l, 0, c.k, c.dil, c.pad, 0, c.b, 0, 0])
            } else {
                let up2 = pre & 4 != 0; let lin = if up2 { 2 * l } else { l }; lout = lin + 2 * c.pad - c.dil * (c.k - 1);
                if pre & 2 != 0 { self.k(cb, x, &self.dummy, [3, l, c.cin, 0, 0, 0, 0, 0, 0, nrm, 0, 0])?; }
                self.k(cb, x, &self.dummy, [0, lout * c.kpad, c.cin, l, c.kpad, c.k, c.dil, c.pad, pre, nrm, 0, 0])?;
                self.pf.gemm(self.ctx, cb, wb, &self.a.p, &GemmOp { w: wr, row0: 0, rows: wr.n, a: &self.col, lda: c.kpad, a_row0: 0, m: lout, y: &self.y, ldy: c.cout, y_col: 0, acc: false })?; self.ctx.barrier(cb);
                self.k(cb, x, o, [1, lout * c.cout, c.cout, 0, ca.to_bits(), cb_.to_bits(), 0, 0, acc as u32 | (clamp as u32) << 1, c.b, 0, 0])
            }
        })(); })?; err?; Ok(lout)
    }
    fn act(&mut self, name: &str, x: &Buffer, l: u32, c: u32, o: &Buffer) -> Result<()> {
        let (a, b) = self.a.snake[name]; let mut err = Ok(());
        self.gpu += self.ctx.run(|cb| { err = self.k(cb, x, o, [4, l * c, c, l, 0, 0, 0, 0, 0, a, b, 0]); })?; err
    }
    fn copy(&mut self, x: &Buffer, o: &Buffer, n: u32) -> Result<()> {   // o = x via store mode with y = x? (CPU copy, unified memory)
        let _ = &self.ctx; o.as_f32_mut()[..n as usize].copy_from_slice(&x.as_f32()[..n as usize]); Ok(())
    }
}

fn rd(b: &Buffer, n: usize) -> Vec<f32> { b.as_f32()[..n].to_vec() }
fn dmp(name: &str, b: &Buffer, l: u32, c: u32) { if let Ok(d) = std::env::var("K6_ADUMP") { let _ = crate::npy::write(&format!("{d}/a_{name}.npy"), &rd(b, (l * c) as usize), &[l as usize, c as usize]); } }

impl Audio {
    /// z [40][L] (already divided by the scaling factor) -> (mel [128][2L], wav, gpu seconds)
    pub fn decode(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, z: &[f32], l: u32) -> Result<(Vec<f32>, Vec<f32>, f64)> {
        let mut r = ARun::new(ctx, pf, kn, self)?;
        let nmax = (l as u64 * 2 * 512 * 64).max(l as u64 * 2 * 2048 * 3) * 4; // f32 bytes
        let b: Vec<Buffer> = (0..5).map(|_| ctx.buffer(nmax)).collect::<Result<_>>()?;
        { let x = b[0].as_f32_mut(); for p in 0..l as usize { for c in 0..40 { x[p * 40 + c] = z[c * l as usize + p] } } }
        let sq = (0.58f32).sqrt(); let (c7, c3) = (0.7 / sq, 0.3 / sq);
        let d = |n: &str| self.dec[n];
        // conv_in
        let mut len = r.conv(&d("conv_in"), &b[0], l, &b[1], 0, false, 0.0, 1.0, false)?; let mut ch = d("conv_in").cout; let mut xi = 1; dmp("conv_in", &b[xi], len, ch);
        let resnet = |r: &mut ARun, pre: &str, xi: usize, len: u32, ch: u32| -> Result<(usize, u32)> {
            let c1 = self.dec[&format!("{pre}.conv1")]; let c2 = self.dec[&format!("{pre}.conv2")];
            let (h, o) = ((xi + 1) % 5, (xi + 2) % 5);
            // x_n = normalize(x); h = conv1(mp_silu(x_n)); h2 = conv2(mp_silu(h)); out = mp_sum(shortcut(x_n), h2, 0.3)
            r.conv(&c1, &b[xi], len, &b[h], 3, false, 0.0, 1.0, false)?;
            if let Some(sc) = self.dec.get(&format!("{pre}.nin_shortcut")) { r.conv(sc, &b[xi], len, &b[o], 2, false, 0.0, 1.0, false)?; }
            else { // o = normalize(x) on the CPU (L <= 176)
                let x = rd(&b[xi], (len * ch) as usize); let ov = b[o].as_f32_mut();
                for p in 0..len as usize { let row = &x[p * ch as usize..(p + 1) * ch as usize]; let s: f32 = row.iter().map(|v| v * v).sum();
                    let sc = 1.0 / (1e-4 + (s / ch as f32).sqrt()); for c in 0..ch as usize { ov[p * ch as usize + c] = row[c] * sc } } }
            r.conv(&c2, &b[h], len, &b[o], 1, true, c7, c3, true)?;
            Ok((o, c2.cout))
        };
        let (n, c) = resnet(&mut r, "mid.block_1", xi, len, ch)?; xi = n; ch = c; dmp("mid_b1", &b[xi], len, ch);
        { // attention on the CPU: qkv = 1x1 conv (GPU), heads 1, channel index = d*3 + s, normalize q/k/v over d
            let qkv = self.dec["mid.attn_1.qkv"]; let o = (xi + 1) % 5; r.conv(&qkv, &b[xi], len, &b[o], 0, false, 0.0, 1.0, false)?;
            let y = rd(&b[o], (len * 3 * ch) as usize); let (n, dd) = (len as usize, ch as usize);
            let get = |p: usize, s: usize| -> Vec<f32> { let v: Vec<f32> = (0..dd).map(|d| y[p * 3 * dd + d * 3 + s]).collect(); let nn: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt(); let sc = 1.0 / (1e-4 + nn / (dd as f32).sqrt()); v.iter().map(|a| a * sc).collect() };
            let q: Vec<Vec<f32>> = (0..n).map(|p| get(p, 0)).collect(); let k: Vec<Vec<f32>> = (0..n).map(|p| get(p, 1)).collect(); let v: Vec<Vec<f32>> = (0..n).map(|p| get(p, 2)).collect();
            let at = (xi + 2) % 5; let ov = b[at].as_f32_mut(); let sc = 1.0 / (dd as f32).sqrt();
            for i in 0..n { let s: Vec<f32> = (0..n).map(|j| q[i].iter().zip(&k[j]).map(|(a, b)| a * b).sum::<f32>() * sc).collect();
                let m = s.iter().cloned().fold(f32::MIN, f32::max); let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect(); let z: f32 = e.iter().sum();
                for d in 0..dd { ov[i * dd + d] = (0..n).map(|j| e[j] * v[j][d]).sum::<f32>() / z } }
            // x = mp_sum(x, proj_out(att), 0.3)
            r.conv(&self.dec["mid.attn_1.proj_out"], &b[at], len, &b[xi], 0, true, c7, c3, false)?;
        }
        dmp("mid_attn", &b[xi], len, ch);
        let (n, c) = resnet(&mut r, "mid.block_2", xi, len, ch)?; xi = n; ch = c; dmp("mid_b2", &b[xi], len, ch);
        for level in (0..3).rev() { for blk in 0..3 { let (n, c) = resnet(&mut r, &format!("up.{level}.block.{blk}"), xi, len, ch)?; xi = n; ch = c; }
            if level == 1 { let o = (xi + 1) % 5; len = r.conv(&self.dec["up.1.upsample.conv"], &b[xi], len, &b[o], 4, false, 0.0, 1.0, false)?; xi = o; }
            dmp(&format!("up{level}"), &b[xi], len, ch); }
        let o = (xi + 1) % 5; let co = self.dec["conv_out"]; r.conv(&co, &b[xi], len, &b[o], 1, false, 0.0, 1.0, false)?;
        let nm = len as usize; let raw = rd(&b[o], nm * 128);
        let mut mel = vec![0f32; nm * 128]; for p in 0..nm { for m in 0..128 { mel[m * nm + p] = raw[p * 128 + m] * self.std[m] + self.mean[m] } }
        let melc: Vec<f32> = raw.iter().enumerate().map(|(i, v)| v * self.std[i % 128] + self.mean[i % 128]).collect(); // channel-last mel
        // ---- vocoder ----
        let vb: Vec<Buffer> = { let n = (nm as u64 * 512 * 48).max(nm as u64 * 8 * 768) * 4 + 4096; (0..6).map(|_| ctx.buffer(n)).collect::<Result<_>>()? };
        let melc = match std::env::var("K6_MEL") { Ok(p) => { let (m, _) = crate::npy::read(&p)?; (0..nm * 128).map(|i| m[(i % 128) * nm + i / 128]).collect() } Err(_) => melc };
        vb[0].as_f32_mut()[..melc.len()].copy_from_slice(&melc);
        let v = |n: &str| self.voc[n];
        let mut l = r.conv(&v("conv_pre"), &vb[0], nm as u32, &vb[1], 0, false, 0.0, 1.0, false)?; let mut c = v("conv_pre").cout; dmp("conv_pre", &vb[1], l, c);
        // buffers: 1 = hs, 2 = sum, 3 = xb, 4 = act tmp, 5 = conv1 out
        for s in 0..6 { let u = v(&format!("ups.{s}")); l = r.conv(&u, &vb[1], l, &vb[2], 0, false, 0.0, 1.0, false)?; c = u.cout;
            r.copy(&vb[2], &vb[1], l * c)?; dmp(&format!("ups{s}"), &vb[1], l, c);
            for kb in 0..3 { let bi = s * 3 + kb; r.copy(&vb[1], &vb[3], l * c)?;
                for j in 0..3 { r.act(&format!("rb.{bi}.a.{}", 2 * j), &vb[3], l, c, &vb[4])?;
                    r.conv(&v(&format!("rb.{bi}.c1.{j}")), &vb[4], l, &vb[5], 0, false, 0.0, 1.0, false)?;
                    r.act(&format!("rb.{bi}.a.{}", 2 * j + 1), &vb[5], l, c, &vb[4])?;
                    r.conv(&v(&format!("rb.{bi}.c2.{j}")), &vb[4], l, &vb[3], 0, true, 1.0, 1.0, false)?; }
                // sum (vb[2]) accumulates blocks / 3
                let n = (l * c) as usize; let xb = vb[3].as_f32().to_vec(); let sm = vb[2].as_f32_mut();
                for i in 0..n { sm[i] = if kb == 0 { xb[i] / 3.0 } else { sm[i] + xb[i] / 3.0 } } }
            r.copy(&vb[2], &vb[1], l * c)?; dmp(&format!("stage{s}"), &vb[1], l, c); }
        r.act("post", &vb[1], l, c, &vb[4])?;
        r.conv(&v("conv_post"), &vb[4], l, &vb[5], 0, false, 0.0, 1.0, false)?;
        let wav: Vec<f32> = rd(&vb[5], l as usize).iter().map(|x| x.clamp(-1.0, 1.0)).collect();
        let g = r.gpu; for x in b { ctx.free_buffer(x) } for x in vb { ctx.free_buffer(x) }
        Ok((mel, wav, g))
    }
}
