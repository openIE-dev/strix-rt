//! Qwen-Image-2.1 VAE decoder (AutoencoderKLQwenImage21, single image = first causal chunk, so every CausalConv3d is a
//! plain 2D conv and the upsample3d time_conv is skipped). Channel-last f32 activations [pixel][C]; 3x3 convs are
//! im2col (f16, K ordered (ky, kx, c), padded to 64) + prefill coop-matrix GEMM in pixel chunks; 1x1 convs GEMM the f16
//! activations directly. Weights are F32 on disk: mmapped and repacked once to f16 [Cout_pad][Kpad] in one GPU arena.
use crate::{st, Kern, Rec};
use anyhow::Result;
use gguf::GgmlType;
use half::f16;
use prefill::{GemmOp, Pf};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{Buffer, Ctx};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const KERNELS: &[(&str, &[u8], u32, u32)] = &[("vrow", spv!("qi_vrow"), 3, 32), ("vae", spv!("qi_vae"), 5, 48)];

#[derive(Clone, Copy)] pub struct Conv { w: WRef, b: u32, cin: u32, cout: u32, ks: u32, kpad: u32 }
pub struct Vae { arena: Vec<Buffer>, params: Buffer, conv: HashMap<String, Conv>, gamma: HashMap<String, u32>, zero: u32, pq: (Vec<f32>, Vec<f32>), mean: Vec<f32>, std: Vec<f32>, pub load_s: f64 }
const P: &str = "decoder.";

impl Vae {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let st = st::Model::open(ctx, dir, &|_| false)?;
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
        let v = |k: &str| -> Vec<f32> { cfg[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
        let mut names: Vec<String> = st.t.keys().filter(|k| k.starts_with(P) && k.ends_with(".weight")).cloned().collect(); names.sort();
        let mut wv: Vec<f16> = vec![]; let mut pv: Vec<f32> = vec![]; let mut conv = HashMap::new(); let mut gamma = HashMap::new();
        for n in &names { let t = st.get(n)?; if t.shape.len() != 4 { continue }
            let (co, ci, ks) = (t.shape[0], t.shape[1], t.shape[2]); let k = ci * ks * ks; let kpad = k.div_ceil(64) * 64; let npad = co.div_ceil(256) * 256;
            let w = st.f32(n)?; let base = wv.len(); wv.resize(base + npad * kpad, f16::ZERO);
            for o in 0..co { for c in 0..ci { for tap in 0..ks * ks { wv[base + o * kpad + tap * ci + c] = f16::from_f32(w[(o * ci + c) * ks * ks + tap]) } } }
            let b = pv.len() as u32; pv.extend(st.f32(&n.replace(".weight", ".bias"))?); while pv.len() % 64 != 0 { pv.push(0.0) }
            let key = n.trim_start_matches(P).trim_end_matches(".weight").to_string();
            conv.insert(key, Conv { w: WRef { arena: 0, off: base as u64 * 2, ty: GgmlType::F16, k: kpad as u32, n: co as u32, e: 1, bytes: (npad * kpad * 2) as u64 }, b, cin: ci as u32, cout: co as u32, ks: ks as u32, kpad: kpad as u32 }); }
        for n in st.t.keys().filter(|k| k.starts_with(P) && k.ends_with(".gamma")) { gamma.insert(n.trim_start_matches(P).trim_end_matches(".gamma").to_string(), pv.len() as u32); pv.extend(st.f32(n)?); while pv.len() % 64 != 0 { pv.push(0.0) } }
        let zero = pv.len() as u32; pv.extend(std::iter::repeat(0.0).take(4096));
        let a = ctx.buffer(wv.len() as u64 * 2)?; unsafe { std::ptr::copy_nonoverlapping(wv.as_ptr() as *const u8, a.ptr as *mut u8, wv.len() * 2) };
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        let pq = (st.f32("post_quant_conv.weight")?, st.f32("post_quant_conv.bias")?);
        Ok(Vae { arena: vec![a], params, conv, gamma, zero, pq, mean: v("latents_mean"), std: v("latents_std"), load_s: t0.elapsed().as_secs_f64() })
    }

    /// Packed DiT latents [h*w][64] -> image [4][16h][16w] clamped to [-1, 1] (channel-first f32), plus GPU seconds.
    pub fn decode(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, lat: &[f32], h: u32, w: u32) -> Result<(Vec<f32>, f64)> {
        let np = (h * w) as usize;
        let mut z = vec![0f32; np * 64];   // un-normalise + post_quant_conv (1x1, 64 -> 64) on the CPU
        for p in 0..np { let x: Vec<f32> = (0..64).map(|c| lat[p * 64 + c] * self.std[c] + self.mean[c]).collect();
            for o in 0..64 { let mut s = self.pq.1[o]; for c in 0..64 { s += self.pq.0[o * 64 + c] * x[c] } z[p * 64 + o] = s } }
        let (hf, wf) = (16 * h as u64, 16 * w as u64); let pmax = hf * wf; let ap = np as u64;
        let maxpc = [ap * 3456, ap * 16 * 1152, ap * 64 * 1152, pmax / 4 * 576, pmax * 288].into_iter().max().unwrap() + 256 * 1152;
        let f32b = |n: u64| ctx.buffer(n * 4); let f16b = |n: u64| ctx.buffer(n * 2);
        let (xa, xb, t) = (f32b(maxpc)?, f32b(maxpc)?, f32b(maxpc)?); let act = f16b(maxpc + 256 * 1152)?;
        let chunk_a: u64 = 192 << 20; let abuf = f16b(chunk_a / 2 + 256 * 10368)?; let ybuf = f32b(chunk_a / 2 + 256 * 1152)?;
        let attn = (f16b((ap + 256) * 1152)?, f16b((ap + 256) * 1152)?, f16b(1280 * ap + 64)?, f32b(ap * ap + 64)?, f16b((ap + 256) * ap)?);
        crate::upload_f16(&act, &z);
        let stop: i32 = std::env::var("QI_VAE_STOP").ok().and_then(|v| v.parse().ok()).unwrap_or(-1);
        let dbg = std::cell::Cell::new((0u8, 0u32, 0u32, 0u32));   // (buffer 1=xa 2=xb, C, hh, ww) of the stopping stage
        let mut err = Ok(());
        let g = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb }; let d = Dec { r: &r, v: self, abuf: &abuf, ybuf: &ybuf, chunk_a };
            let (mut hh, mut ww) = (h, w);
            d.conv("conv_in", &act, &xa, hh, ww, false)?;
            if stop == 0 { dbg.set((1, 1152, hh, ww)); return Ok(()) }
            d.resnet("mid_block.resnets.0", &xa, &xa, &t, &act, hh, ww)?;
            d.attention(&xa, &t, &act, &attn, hh, ww)?;
            if stop == 10 { dbg.set((1, 1152, hh, ww)); return Ok(()) }
            d.resnet("mid_block.resnets.1", &xa, &xa, &t, &act, hh, ww)?;
            if stop == 1 { dbg.set((1, 1152, hh, ww)); return Ok(()) }
            let (mut x, mut y) = (&xa, &xb);
            for i in 0..5 { let pre = format!("up_blocks.{i}");
                let cin = self.conv[&format!("{pre}.resnets.0.conv1")].cin;
                d.resnet(&format!("{pre}.resnets.0"), x, y, &t, &act, hh, ww)?;
                d.resnet(&format!("{pre}.resnets.1"), y, y, &t, &act, hh, ww)?;
                d.resnet(&format!("{pre}.resnets.2"), y, y, &t, &act, hh, ww)?;
                if i < 4 { let c = self.conv[&format!("{pre}.upsampler.resample.1")].cin; let n = hh * ww * c;
                    d.ew(3, n * 4, c, ww, hh, 0, 0, 0, 0, 0, y, &t, &act, &act)?;          // t = up2(y)
                    d.ew(2, n * 4, 0, 0, 0, 0, 0, 0, 0, 0, &t, &t, &act, &act)?;           // act = f16(t)
                    hh *= 2; ww *= 2; d.conv(&format!("{pre}.upsampler.resample.1"), &act, y, hh, ww, false)?;
                    d.ew(4, hh * ww * c, c, ww / 2, hh / 2, 0, 0, if i < 3 { 2 } else { 1 }, cin, 0, y, x, &act, &act)?; }  // y += DupUp(block input)
                if stop == 2 + i as i32 { dbg.set((if std::ptr::eq(y, &xa) { 1 } else { 2 }, self.conv[&format!("{pre}.resnets.2.conv2")].cout, hh, ww)); return Ok(()) }
                std::mem::swap(&mut x, &mut y); }
            d.norm(x, &act, hh * ww, 144, "norm_out", true)?;
            d.conv("conv_out", &act, &t, hh, ww, false)?;
            Ok(()) })(); })?;
        err?;
        let (bi, c, dh, dw) = dbg.get();
        if bi != 0 { let b = if bi == 1 { &xa } else { &xb }; let n = (c * dh * dw) as usize; let v = b.as_f32()[..n].to_vec();
            std::fs::write("/tmp/qi_vae_dbg.f32", v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?; eprintln!("vae dbg stage {stop}: [{dh}x{dw}][{c}]"); }
        let (hh, ww) = (hf as usize, wf as usize); let o = &t.as_f32()[..hh * ww * 4];
        let mut img = vec![0f32; 4 * hh * ww]; for p in 0..hh * ww { for c in 0..4 { img[c * hh * ww + p] = o[p * 4 + c].clamp(-1.0, 1.0) } }
        for b in [xa, xb, t, act, abuf, ybuf, attn.0, attn.1, attn.2, attn.3, attn.4] { ctx.free_buffer(b) }
        Ok((img, g))
    }
}

struct Dec<'a> { r: &'a Rec<'a>, v: &'a Vae, abuf: &'a Buffer, ybuf: &'a Buffer, chunk_a: u64 }
impl<'a> Dec<'a> {
    fn ew(&self, mode: u32, n: u32, c: u32, wd: u32, hd: u32, p0: u32, b_off: u32, flags: u32, c2: u32, kpad: u32, x: &Buffer, y: &Buffer, a: &Buffer, b: &Buffer) -> Result<()> {
        self.r.run("vae", &[x, y, a, b, &self.v.params], &[mode, n, c, wd, hd, p0, b_off, flags, c2, kpad, 0, 0], n.div_ceil(256).min(65535))
    }
    fn norm(&self, x: &Buffer, o: &Buffer, rows: u32, c: u32, g: &str, silu: bool) -> Result<()> {
        self.r.run("vrow", &[x, o, &self.v.params], &[0, c, c, c, self.v.gamma[g], silu as u32, 0, 0], rows)
    }
    fn gemm(&self, wb: &Buffer, w: WRef, a: &Buffer, lda: u32, a_row0: u32, m: u32, y: &Buffer, ldy: u32) -> Result<()> {
        self.r.pf.gemm(self.r.ctx, self.r.cb, wb, &self.v.params, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0, m, y, ldy, y_col: 0, acc: false })?; self.r.bar(); Ok(())
    }
    /// dst[p][cout] (= or +=) conv(src f16 [p][cin]) + bias over an hh x ww image, in pixel chunks.
    fn conv(&self, name: &str, src: &Buffer, dst: &Buffer, hh: u32, ww: u32, acc: bool) -> Result<()> {
        let c = self.v.conv[name]; let np = hh * ww; let wb = &self.v.arena[0];
        // GEMM over Cout rounded up to 64 (zero-padded weight rows): narrow / odd N (144, 4) mis-stores otherwise
        let ny = c.cout.div_ceil(64) * 64; let w = WRef { n: ny, ..c.w };
        let cp = ((self.chunk_a / (c.kpad as u64 * 2)).min(self.chunk_a / 2 / ny as u64) as u32 / 256 * 256).max(256);
        let mut p0 = 0; while p0 < np { let m = cp.min(np - p0);
            if c.ks == 3 || c.kpad != c.cin { self.ew(0, m * c.kpad, c.cin, ww, hh, p0, 0, (c.ks == 1) as u32, 0, c.kpad, self.ybuf, self.ybuf, src, self.abuf)?; self.gemm(wb, w, self.abuf, c.kpad, 0, m, self.ybuf, ny)?; }
            else { self.gemm(wb, w, src, c.kpad, p0, m, self.ybuf, ny)?; }
            self.ew(1, m * c.cout, c.cout, 0, 0, p0, c.b, acc as u32, ny, 0, dst, self.ybuf, src, self.abuf)?;
            p0 += m; }
        Ok(())
    }
    /// dst = shortcut(src) + conv2(silu(norm2(conv1(silu(norm1(src)))))); src may equal dst when cin == cout.
    fn resnet(&self, pre: &str, src: &Buffer, dst: &Buffer, t: &Buffer, act: &Buffer, hh: u32, ww: u32) -> Result<()> {
        let c1 = self.v.conv[&format!("{pre}.conv1")]; let np = hh * ww; let z = self.v.zero;
        self.norm(src, act, np, c1.cin, &format!("{pre}.norm1"), true)?;
        self.conv(&format!("{pre}.conv1"), act, t, hh, ww, false)?;
        self.norm(t, act, np, c1.cout, &format!("{pre}.norm2"), true)?;
        if c1.cin != c1.cout {
            self.conv(&format!("{pre}.conv2"), act, t, hh, ww, false)?;                     // t = residual branch
            self.ew(2, np * c1.cin, 0, 0, 0, 0, 0, 0, 0, 0, src, src, act, act)?;             // act = f16(src)
            self.conv(&format!("{pre}.conv_shortcut"), act, dst, hh, ww, false)?;
            self.ew(1, np * c1.cout, c1.cout, 0, 0, 0, z, 1, c1.cout, 0, dst, t, act, act)?;        // dst += t
        } else {
            if !std::ptr::eq(src, dst) { self.ew(1, np * c1.cout, c1.cout, 0, 0, 0, z, 0, c1.cout, 0, dst, src, act, act)?; }
            self.conv(&format!("{pre}.conv2"), act, dst, hh, ww, true)?;
        }
        Ok(())
    }
    /// Mid-block single-head attention over the hh*ww pixels (C = 1152), residual into x.
    fn attention(&self, x: &Buffer, t: &Buffer, act: &Buffer, at: &(Buffer, Buffer, Buffer, Buffer, Buffer), hh: u32, ww: u32) -> Result<()> {
        let np = hh * ww; let c = 1152u32; let pre = "mid_block.attentions.0";
        self.norm(x, act, np, c, &format!("{pre}.norm"), false)?;
        let qkv = self.v.conv[&format!("{pre}.to_qkv")];
        self.gemm(&self.v.arena[0], qkv.w, act, c, 0, np, t, 3 * c)?;
        for (i, (dst, tr)) in [(&at.0, 0u32), (&at.1, 0), (&at.2, 1)].into_iter().enumerate() {
            self.ew(5, np * c, c, np, 0, 0, i as u32 * c, tr, 3 * c, qkv.b, t, t, act, dst)?; }
        let kw = WRef { arena: 0, off: 0, ty: GgmlType::F16, k: c, n: np, e: 1, bytes: (np * c * 2) as u64 };
        self.gemm(&at.1, kw, &at.0, c, 0, np, &at.3, np)?;
        self.r.run("vrow", &[&at.3, &at.4, &self.v.params], &[1, np, np, np, 0, 0, (1.0f32 / (c as f32).sqrt()).to_bits(), 0], np)?;
        let vw = WRef { arena: 0, off: 0, ty: GgmlType::F16, k: np, n: c, e: 1, bytes: (np * c * 2) as u64 };
        self.gemm(&at.2, vw, &at.4, np, 0, np, t, c)?;
        self.ew(2, np * c, 0, 0, 0, 0, 0, 0, 0, 0, t, t, act, act)?;   // act = f16(O)
        self.conv(&format!("{pre}.proj"), act, x, hh, ww, true)
    }
}
