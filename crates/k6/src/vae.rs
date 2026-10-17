//! HunyuanVideo causal 3D VAE decoder (AutoencoderKLHunyuanVideo, Kandinsky 6 video VAE) on strix-rt.
//! Channel-last f16 activations [t][y][x][C] (a full 49x320x576x128 volume is 2.3 GB, under the 4 GB buffer range;
//! the reference runs this VAE in bf16). 3x3x3 causal convs (replicate pad, time front 2) are im2col (f16, K ordered
//! (kt, ky, kx, c) padded to 64) with nearest upsampling folded into the gather, then prefill's coop-matrix GEMM in
//! pixel chunks, one GPU submission per chunk. GroupNorm = per-workgroup partial sums + finalize + apply(+SiLU).
//! Mid-block attention (1 head, d 512, causal over frames) = per-frame QK^T GEMM, row softmax, PV GEMM.
use crate::{npy, Kern};
use anyhow::Result;
use gguf::GgmlType;
use half::f16;
use prefill::{GemmOp, Pf};
use qimage::st;
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{push_u32, Buffer, Ctx};

#[derive(Clone, Copy)] struct Conv { w: usize, b: u32, cin: u32, cout: u32, k1: bool, kpad: u32 }
pub struct Vae { ws: Vec<(Buffer, WRef)>, p: Buffer, conv: HashMap<String, Conv>, gn: HashMap<String, u32>, lin: HashMap<String, Conv>, zero: u32, pq: (Vec<f32>, Vec<f32>), pub scale: f32, pub load_s: f64 }
const P: &str = "decoder.";
fn envu(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

impl Vae {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let s = st::Model::open(ctx, dir, &|_| false)?;
        let mut pv: Vec<f32> = vec![]; let put = |pv: &mut Vec<f32>, x: Vec<f32>| -> u32 { let o = pv.len() as u32; pv.extend(x); while pv.len() % 64 != 0 { pv.push(0.0) } o };
        let mut ws = vec![]; let mut conv = HashMap::new(); let mut gn = HashMap::new(); let mut lin = HashMap::new();
        let mut keys: Vec<&String> = s.t.keys().filter(|k| k.starts_with(P)).collect(); keys.sort();
        for k in keys {
            if let Some(base) = k.strip_suffix(".conv.weight") {
                let t = s.get(k)?; let (cout, cin, kt) = (t.shape[0], t.shape[1], t.shape[2]); let k1 = kt == 1;
                let kk = if k1 { cin } else { 27 * cin }; let kpad = kk.div_ceil(64) * 64;
                let w = s.f32(k)?; let buf = ctx.buffer((cout * kpad * 2) as u64)?;
                { let o = unsafe { std::slice::from_raw_parts_mut(buf.ptr as *mut f16, cout * kpad) };
                  for co in 0..cout { for i in 0..kpad { o[co * kpad + i] = f16::ZERO }
                    if k1 { for ci in 0..cin { o[co * kpad + ci] = f16::from_f32(w[co * cin + ci]) } }
                    else { for ci in 0..cin { for tap in 0..27 { o[co * kpad + tap * cin + ci] = f16::from_f32(w[(co * cin + ci) * 27 + tap]) } } } } }
                let wr = WRef { arena: 0, off: 0, ty: GgmlType::F16, k: kpad as u32, n: cout as u32, e: 1, bytes: (cout * kpad * 2) as u64 };
                ws.push((buf, wr)); let b = put(&mut pv, s.f32(&format!("{base}.conv.bias"))?);
                conv.insert(base.to_string(), Conv { w: ws.len() - 1, b, cin: cin as u32, cout: cout as u32, k1, kpad: kpad as u32 });
            } else if (k.contains("norm") ) && k.ends_with(".weight") {
                let base = k.strip_suffix(".weight").unwrap(); let mut g = s.f32(k)?; g.extend(s.f32(&format!("{base}.bias"))?);
                gn.insert(base.to_string(), put(&mut pv, g));
            } else if k.contains("attentions") && k.ends_with(".weight") {
                let base = k.strip_suffix(".weight").unwrap(); let t = s.get(k)?; let (n, kk) = (t.shape[0], t.shape[1]); let w = s.f32(k)?;
                let buf = ctx.buffer((n * kk * 2) as u64)?; { let o = unsafe { std::slice::from_raw_parts_mut(buf.ptr as *mut f16, n * kk) }; for i in 0..n * kk { o[i] = f16::from_f32(w[i]) } }
                ws.push((buf, WRef { arena: 0, off: 0, ty: GgmlType::F16, k: kk as u32, n: n as u32, e: 1, bytes: (n * kk * 2) as u64 }));
                let b = put(&mut pv, s.f32(&format!("{base}.bias"))?);
                lin.insert(base.to_string(), Conv { w: ws.len() - 1, b, cin: kk as u32, cout: n as u32, k1: true, kpad: kk as u32 });
            }
        }
        let zero = put(&mut pv, vec![0.0; 1024]);
        let p = ctx.buffer(pv.len() as u64 * 4)?; p.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
        let scale = cfg["scaling_factor"].as_f64().unwrap_or(0.476986) as f32;
        Ok(Vae { ws, p, conv, gn, lin, zero, pq: (s.f32("post_quant_conv.weight")?, s.f32("post_quant_conv.bias")?), scale, load_s: t0.elapsed().as_secs_f64() })
    }
}

struct Vol { t: u32, h: u32, w: u32, c: u32 }
impl Vol { fn px(&self) -> u32 { self.t * self.h * self.w } fn n(&self) -> u32 { self.px() * self.c } }
pub struct Dec<'a> { ctx: &'a Ctx, pf: &'a Pf, kn: &'a Kern, v: &'a Vae, col: Buffer, ych: Buffer, part: Buffer, stats: Buffer, pub gpu: f64, colb: u64 }
impl<'a> Dec<'a> {
    fn disp(&self, cb: ash::vk::CommandBuffer, name: &str, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<()> {
        let k = &self.kn.k[name]; let ds = self.pf.ds.get(self.ctx, k, bufs)?; self.ctx.dispatch(self.cb_fix(cb), k, ds, &push_u32(push), groups.max(1)); self.ctx.barrier(cb); Ok(())
    }
    fn cb_fix(&self, cb: ash::vk::CommandBuffer) -> ash::vk::CommandBuffer { cb }
    fn ew(&self, cb: ash::vk::CommandBuffer, push: [u32; 12], x: &Buffer, y: &Buffer, a: &Buffer, b: &Buffer) -> Result<()> {
        let n = push[1]; self.disp(cb, "k6vae", &[x, y, a, b, &self.v.p, &self.stats], &push, n.div_ceil(256).min(65535))
    }
    fn gemm(&self, cb: ash::vk::CommandBuffer, wb: &Buffer, w: WRef, a: &Buffer, lda: u32, a_row0: u32, m: u32, y: &Buffer, ldy: u32) -> Result<()> {
        self.pf.gemm(self.ctx, cb, wb, &self.v.p, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0, m, y, ldy, y_col: 0, acc: false })?; self.ctx.barrier(cb); Ok(())
    }
    /// dst = conv(src) (+ dst if acc); src volume `iv`, output volume upsampled by (ut, us).
    fn conv(&mut self, name: &str, src: &Buffer, iv: &Vol, ut: bool, us: bool, dst: &Buffer, acc: bool) -> Result<Vol> {
        let c = self.v.conv[&format!("{P}{name}")]; assert_eq!(c.cin, iv.c, "{name}");
        let ov = Vol { t: if ut { 2 * iv.t - 1 } else { iv.t }, h: if us { 2 * iv.h } else { iv.h }, w: if us { 2 * iv.w } else { iv.w }, c: c.cout };
        let npx = ov.px(); let chunk = ((self.colb / (c.kpad as u64 * 2)) as u32 / 256 * 256).max(256).min(envu("K6_VAE_CHUNK", 1 << 30) as u32);
        let (wb, wr) = (&self.v.ws[c.w].0, self.v.ws[c.w].1);
        let mut p0 = 0;
        while p0 < npx { let m = chunk.min(npx - p0);
            let mut err = Ok(());
            self.gpu += self.ctx.run(|cb| { err = (|| -> Result<()> {
                self.ew(cb, [0, m * c.kpad, iv.c, iv.t, iv.h, iv.w, p0, c.kpad, c.k1 as u32, ut as u32, us as u32, 0], src, &self.ych, src, &self.col)?;
                self.gemm(cb, wb, wr, &self.col, c.kpad, 0, m, &self.ych, c.cout)?;
                self.ew(cb, [1, m * c.cout, c.cout, 0, 0, 0, p0, 0, acc as u32, 0, 0, c.b], dst, &self.ych, src, &self.col)
            })(); })?; err?;
            p0 += m; }
        Ok(ov)
    }
    /// out16 = act(GroupNorm(x)), 32 groups, eps 1e-6.
    fn gnorm(&mut self, name: &str, x: &Buffer, v: &Vol, out: &Buffer, silu: bool) -> Result<()> {
        let g = self.v.gn[&format!("{P}{name}")]; let chunk = 2048u32; let nw = v.px().div_ceil(chunk);
        let mut err = Ok(());
        self.gpu += self.ctx.run(|cb| { err = (|| -> Result<()> {
            self.disp(cb, "k6red", &[x, &self.part, out, &self.stats], &[0, v.c, v.px(), chunk, nw, 0, 0, 0, 0, 0], nw)?;
            self.disp(cb, "k6red", &[x, &self.part, out, &self.stats], &[1, v.c, v.px(), chunk, nw, 1e-6f32.to_bits(), 0, 0, 0, 0], 1)?;
            self.ew(cb, [2, v.n(), v.c, 0, 0, 0, 0, 0, (!silu) as u32, 0, 0, g], x, &self.ych, x, out)
        })(); })?; err
    }
    /// x += conv2(act(gn2(conv1(act(gn1(x)))))) with optional 1x1 shortcut; returns the (possibly new) residual buffer index.
    fn resnet(&mut self, pre: &str, bufs: &[&Buffer; 4], xi: usize, v: &Vol) -> Result<(usize, Vol)> {
        let (x, act, h) = (bufs[xi], bufs[2], bufs[3]);
        self.gnorm(&format!("{pre}.norm1"), x, v, act, true)?;
        let hv = self.conv(&format!("{pre}.conv1"), act, v, false, false, h, false)?;
        self.gnorm(&format!("{pre}.norm2"), h, &hv, act, true)?;
        if self.v.conv.contains_key(&format!("{P}{pre}.conv_shortcut")) {
            let other = 1 - xi; let ov = self.conv(&format!("{pre}.conv_shortcut"), x, v, false, false, bufs[other], false)?;
            self.conv(&format!("{pre}.conv2"), act, &hv, false, false, bufs[other], true)?; Ok((other, ov))
        } else { let ov = self.conv(&format!("{pre}.conv2"), act, &hv, false, false, x, true)?; Ok((xi, ov)) }
    }
    fn attention(&mut self, x: &Buffer, act: &Buffer, v: &Vol, q16: &Buffer, k16: &Buffer, v16: &Buffer, vt: &Buffer, s: &Buffer, p16: &Buffer, o16: &Buffer) -> Result<()> {
        let pre = "mid_block.attentions.0"; self.gnorm(&format!("{pre}.group_norm"), x, v, act, false)?;
        let n = v.px(); let hw = v.h * v.w; let np = n.div_ceil(64) * 64; let c = 512u32;
        let lin = |nm: &str| self.v.lin[&format!("{P}{pre}.{nm}")];
        // q, k, v projections (chunked rows through ych)
        for (nm, dst) in [("to_q", q16), ("to_k", k16), ("to_v", v16)] { let l = lin(nm); let (wb, wr) = (&self.v.ws[l.w].0, self.v.ws[l.w].1);
            let ch = 16384u32; let mut r0 = 0; while r0 < n { let m = ch.min(n - r0); let mut err = Ok(());
                self.gpu += self.ctx.run(|cb| { err = (|| -> Result<()> { self.gemm(cb, wb, wr, act, c, r0, m, &self.ych, c)?;
                    self.ew(cb, [1, m * c, c, 0, 0, 0, r0, 0, 0, 0, 0, l.b], dst, &self.ych, act, act) })(); })?; err?; r0 += m; } }
        let mut err = Ok(());
        self.gpu += self.ctx.run(|cb| { err = self.ew(cb, [4, c * np, c, 0, n, np, 0, 0, 0, 0, 0, 0], x, &self.ych, v16, vt); })?; err?;
        let kw = WRef { arena: 0, off: 0, ty: GgmlType::F16, k: c, n, e: 1, bytes: (n * c * 2) as u64 };
        let vw = WRef { arena: 0, off: 0, ty: GgmlType::F16, k: np, n: c, e: 1, bytes: (c * np * 2) as u64 };
        let sc = 1.0f32 / (c as f32).sqrt();
        for f in 0..v.t { let r0 = f * hw; let nk = (f + 1) * hw; let mut err = Ok(());
            self.gpu += self.ctx.run(|cb| { err = (|| -> Result<()> {
                self.gemm(cb, k16, kw, q16, c, r0, hw, s, n)?;
                self.disp(cb, "k6red", &[x, s, p16, &self.stats], &[2, 0, 0, 0, 0, 0, nk, n, np, sc.to_bits()], hw)?;
                self.gemm(cb, vt, vw, p16, np, 0, hw, &self.ych, c)?;
                self.ew(cb, [1, hw * c, c, 0, 0, 0, r0, 0, 0, 0, 0, self.v.zero], o16, &self.ych, act, act)
            })(); })?; err?; }
        let l = lin("to_out.0"); let (wb, wr) = (&self.v.ws[l.w].0, self.v.ws[l.w].1);
        let ch = 16384u32; let mut r0 = 0; while r0 < n { let m = ch.min(n - r0); let mut err = Ok(());
            self.gpu += self.ctx.run(|cb| { err = (|| -> Result<()> { self.gemm(cb, wb, wr, o16, c, r0, m, &self.ych, c)?;
                self.ew(cb, [1, m * c, c, 0, 0, 0, r0, 0, 1, 0, 0, l.b], x, &self.ych, act, act) })(); })?; err?; r0 += m; }
        Ok(())
    }
}

impl Vae {
    /// lat [16][T][H][W] (pipeline latents, already divided by nothing) -> video [3][Tout][Hout][Wout] in [-1, 1] (approx).
    pub fn decode(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, lat: &[f32], t: u32, h: u32, w: u32, dump: Option<&str>) -> Result<(Vec<f32>, f64)> {
        let (tn, hn, wn) = (4 * (t - 1) + 1, 8 * h, 8 * w); let maxn = tn as u64 * hn as u64 * wn as u64 * 256;   // up block 2 emits 256 ch at full res
        let big = |n: u64| ctx.buffer(n * 2);
        let bufs = [big(maxn)?, big(maxn)?, big(maxn)?, big(maxn)?];
        let colb = (envu("K6_VAE_COLMB", 768) as u64) << 20;
        let npx_lat = (t * h * w) as u64; let np = npx_lat.div_ceil(64) * 64;
        let att = [big(npx_lat * 512)?, big(npx_lat * 512)?, big(npx_lat * 512)?, big(np * 512)?, ctx.buffer(h as u64 * w as u64 * npx_lat * 4)?, big(h as u64 * w as u64 * np)?, big(npx_lat * 512)?];
        let mut d = Dec { ctx, pf, kn, v: self, col: ctx.buffer(colb)?, ych: ctx.buffer(colb.max(16384 * 512 * 4) * 2)?, part: ctx.buffer(4 << 20)?, stats: ctx.buffer(4096)?, gpu: 0.0, colb };
        // z / scale -> post_quant_conv (CPU) -> channel-last f16
        { let n = (t * h * w) as usize; let o = unsafe { std::slice::from_raw_parts_mut(bufs[2].ptr as *mut f16, n * 16) };
          for p in 0..n { for co in 0..16 { let mut s = self.pq.1[co]; for ci in 0..16 { s += self.pq.0[co * 16 + ci] * lat[ci * n + p] / self.scale; } o[p * 16 + co] = f16::from_f32(s); } } }
        let br: [&Buffer; 4] = [&bufs[0], &bufs[1], &bufs[2], &bufs[3]];
        let iv = Vol { t, h, w, c: 16 };
        let mut v = d.conv("conv_in", br[2], &iv, false, false, br[0], false)?; let mut xi = 0;
        let dmp = |name: &str, b: &Buffer, v: &Vol| -> Result<()> { if let Some(dir) = dump { let n = v.n() as usize; eprintln!("dump {name}: {}x{}x{}x{} = {n} elems, buf {} bytes", v.t, v.h, v.w, v.c, b.size); anyhow::ensure!((n as u64) * 2 <= b.size, "dump {name} out of range");
            let s = unsafe { std::slice::from_raw_parts(b.ptr as *const f16, n) }; let f: Vec<f32> = s.iter().map(|x| x.to_f32()).collect();
            npy::write(&format!("{dir}/vae_{name}.npy"), &f, &[v.t as usize, v.h as usize, v.w as usize, v.c as usize])?; } Ok(()) };
        dmp("conv_in", br[xi], &v)?;
        let (r, nv) = d.resnet("mid_block.resnets.0", &br, xi, &v)?; xi = r; v = nv;
        d.attention(br[xi], br[2], &v, &att[0], &att[1], &att[2], &att[3], &att[4], &att[5], &att[6])?;
        let (r, nv) = d.resnet("mid_block.resnets.1", &br, xi, &v)?; xi = r; v = nv; dmp("mid", br[xi], &v)?;
        for b in 0..4 { for l in 0..3 { let (r, nv) = d.resnet(&format!("up_blocks.{b}.resnets.{l}"), &br, xi, &v)?; xi = r; v = nv; }
            if b < 3 { let ut = b == 1 || b == 2; let other = 1 - xi;
                v = d.conv(&format!("up_blocks.{b}.upsamplers.0.conv"), br[xi], &v, ut, true, br[other], false)?; xi = other; }
            dmp(&format!("up{b}"), br[xi], &v)?; }
        d.gnorm("conv_norm_out", br[xi], &v, br[2], true)?;
        let ov = d.conv("conv_out", br[2], &v, false, false, br[3], false)?;
        let n = ov.px() as usize; let s = unsafe { std::slice::from_raw_parts(br[3].ptr as *const f16, n * 3) };
        let mut out = vec![0f32; n * 3]; for p in 0..n { for c in 0..3 { out[c * n + p] = s[p * 3 + c].to_f32() } }
        let g = d.gpu; for b in bufs { ctx.free_buffer(b) } for b in att { ctx.free_buffer(b) }
        Ok((out, g))
    }
}

impl Vae {
    /// Width-tiled decode when the full-res 256-ch tensor would exceed the ~4 GB buffer range: latent tiles of `tw`
    /// columns overlapping by `ov`, linearly blended over the overlap (like diffusers' tiled decode; GroupNorm stats are per tile).
    pub fn decode_auto(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, lat: &[f32], t: u32, h: u32, w: u32, dump: Option<&str>) -> Result<(Vec<f32>, f64)> {
        let tn = 4 * (t - 1) + 1; let per_col = tn as u64 * 8 * h as u64 * 8 * 256 * 2;
        let maxw = ((3_900_000_000u64 / per_col) as u32).max(16);
        if w <= maxw { return self.decode(ctx, pf, kn, lat, t, h, w, dump); }
        let ov = 8u32; let tw = maxw.min(48); let hn = 8 * h as usize; let wn = 8 * w as usize; let tnn = tn as usize;
        let mut out = vec![0f32; 3 * tnn * hn * wn]; let mut wsum = vec![0f32; wn]; let mut gpu = 0.0;
        let mut x0 = 0u32;
        loop { let x1 = (x0 + tw).min(w); let tw2 = x1 - x0; let n = (t * h) as usize;
            let mut sub = vec![0f32; 16 * n * tw2 as usize];
            for c in 0..16 { for r in 0..n { for x in 0..tw2 as usize { sub[(c * n + r) * tw2 as usize + x] = lat[(c * n + r) * w as usize + x0 as usize + x]; } } }
            let (v, g) = self.decode(ctx, pf, kn, &sub, t, h, tw2, None)?; gpu += g; let tw8 = 8 * tw2 as usize;
            for xx in 0..tw8 { let gx = 8 * x0 as usize + xx;
                let lw = if x0 > 0 && xx < 8 * ov as usize { (xx as f32 + 0.5) / (8 * ov) as f32 } else { 1.0 };
                let rw = if x1 < w && xx >= tw8 - 8 * ov as usize { (tw8 - xx) as f32 / (8 * ov) as f32 } else { 1.0 };
                let wt = lw.min(rw); wsum[gx] += wt;
                for c in 0..3 { for f in 0..tnn { for y in 0..hn { out[((c * tnn + f) * hn + y) * wn + gx] += wt * v[((c * tnn + f) * hn + y) * tw8 + xx]; } } } }
            eprintln!("tile x {x0}..{x1} done");
            if x1 == w { break } x0 = x1 - ov; }
        for c in 0..3 * tnn * hn { for gx in 0..wn { out[c * wn + gx] /= wsum[gx]; } }
        Ok((out, gpu))
    }
}
