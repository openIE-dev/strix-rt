//! MiniCPM-V 4.7 vision tower (mmproj "minicpmv4_7", 16x merge) on the GPU:
//! patch conv (GEMM, K 588 -> 640) + bias + learned position embedding (70x70 buckets) -> 27 SigLIP layers
//! (LN, q/k/v/o with bias, 16 heads of 72 padded to 128 for qimage flash, LN, tanh-GELU MLP), with the ViT merger
//! after layer 6 (2x2-window attention, then 2x2 concat -> LN -> tanh-GELU MLP + mean residual, 4x fewer tokens),
//! post-LN, final DownsampleMLP (2x2 concat -> LN -> erf-GELU MLP -> 2048) and the separator rows (v.tok_embd_sep).
//! Every linear goes through prefill's GEMM straight from the BF16 weights (mmproj re-written as one aligned
//! safetensors file by ref/mmproj2st.py). Preprocessing follows llama.cpp mtmd (llava-uhd slicing, pillow bicubic).
use anyhow::{bail, Result};
use ash::vk;
use gguf::GgmlType;
use half::f16;
use prefill::{GemmOp, Pf};
use qimage::{st, Kern};
use rtcore::WRef;
use vkrt::{push_u32, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const D: u32 = 1152; pub const NH: u32 = 16; pub const HD: u32 = 72; pub const FF: u32 = 4304; pub const FFP: u32 = 4352;
pub const NL: usize = 27; pub const PS: usize = 14; pub const KP: u32 = 640; pub const DO: u32 = 2048;
const EPS: f32 = 1e-6;
fn rup(x: u32, m: u32) -> u32 { x.div_ceil(m) * m }

#[derive(Clone, Copy)] pub struct Lin { pub w: WRef, pub b: u32 }
#[derive(Clone, Copy)] pub struct Ln { pub w: u32, pub b: u32 }
pub struct Layer { pub ln1: Ln, pub ln2: Ln, pub q: WRef, pub k: WRef, pub v: WRef, pub qkv_b: u32, pub o: Lin, pub up: Lin, pub down: Lin }
struct PB { v: Vec<f32> }
impl PB { fn put(&mut self, x: Vec<f32>) -> u32 { let o = self.v.len() as u32; self.v.extend(x); while self.v.len() % 64 != 0 { self.v.push(0.0) } o } }

pub struct Vision {
    pub st: st::Model, pub p: Buffer, pub pw: Buffer, pub layers: Vec<Layer>, pub insert: usize,
    pub patch_b: u32, pub pos: u32, pub m_ln1: Ln, pub m_q: WRef, pub m_k: WRef, pub m_v: WRef, pub m_qkv_b: u32, pub m_o: Lin, pub m_dsln: Ln, pub m_up: Lin, pub m_down: Lin,
    pub post: Ln, pub f_ln: Ln, pub f_up: Lin, pub f_down: Lin, pub sep: Vec<f32>,
    pub pf: Pf, pub kn: Kern, ln: Kernel, ew: Kernel, pad: Kernel, win: Kernel, dsk: Kernel,
    bufs: std::cell::RefCell<Option<(u32, Bufs)>>,
}
struct Bufs { pin: Buffer, x: Buffer, a16: Buffer, h16: Buffer, hd16: Buffer, y: Buffer, q: Buffer, k: Buffer, v: Buffer, o: Buffer, cat: Buffer, x2: Buffer, out: Buffer, idx: Buffer }

/// One preprocessed tile: normalised pixels [3][h][w] (h, w multiples of 56) and its separator suffix rows.
pub struct Tile { pub w: usize, pub h: usize, pub px: Vec<f32>, pub suffix: Vec<usize> }

impl Vision {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|k: &str| k.ends_with(".weight") && !k.contains("ln") && !k.contains("norm") && !k.contains("patch_embd") && !k.contains("position_embd"))?;
        let mut pb = PB { v: vec![] };
        let lin = |pb: &mut PB, p: &str| -> Result<Lin> { Ok(Lin { w: st.wref(&format!("{p}.weight"))?, b: pb.put(st.f32(&format!("{p}.bias"))?) }) };
        let ln = |pb: &mut PB, p: &str| -> Result<Ln> { Ok(Ln { w: pb.put(st.f32(&format!("{p}.weight"))?), b: pb.put(st.f32(&format!("{p}.bias"))?) }) };
        let qkvb = |pb: &mut PB, p: &str| -> Result<u32> { let mut v = st.f32(&format!("{p}.attn_q.bias"))?; v.extend(st.f32(&format!("{p}.attn_k.bias"))?); v.extend(st.f32(&format!("{p}.attn_v.bias"))?); Ok(pb.put(v)) };
        let mut layers = vec![];
        for i in 0..NL { let p = format!("v.blk.{i}");
            layers.push(Layer { ln1: ln(&mut pb, &format!("{p}.ln1"))?, ln2: ln(&mut pb, &format!("{p}.ln2"))?, q: st.wref(&format!("{p}.attn_q.weight"))?, k: st.wref(&format!("{p}.attn_k.weight"))?,
                v: st.wref(&format!("{p}.attn_v.weight"))?, qkv_b: qkvb(&mut pb, &p)?, o: lin(&mut pb, &format!("{p}.attn_out"))?, up: lin(&mut pb, &format!("{p}.ffn_up"))?, down: lin(&mut pb, &format!("{p}.ffn_down"))? }); }
        if layers[0].down.w.k != FFP { bail!("ffn_down K {} (expected the K-padded {FFP}; re-run mmproj2st.py)", layers[0].down.w.k) }
        // patch conv F32 [1152][3][14][14] -> f16 [1152][640] (K order c, ky, kx; zero padded)
        let cw = st.f32("v.patch_embd.weight")?; let pw = ctx.buffer((D * KP * 2) as u64)?;
        { let o = unsafe { std::slice::from_raw_parts_mut(pw.ptr as *mut f16, (D * KP) as usize) };
          for j in 0..D as usize { for i in 0..KP as usize { o[j * KP as usize + i] = f16::from_f32(if i < 588 { cw[j * 588 + i] } else { 0.0 }) } } }
        let patch_b = pb.put(st.f32("v.patch_embd.bias")?);
        let pos = pb.put(st.f32("v.position_embd.weight")?);
        let m_ln1 = ln(&mut pb, "v.vit_merger.ln1")?;
        let (m_q, m_k, m_v) = (st.wref("v.vit_merger.attn_q.weight")?, st.wref("v.vit_merger.attn_k.weight")?, st.wref("v.vit_merger.attn_v.weight")?);
        let m_qkv_b = qkvb(&mut pb, "v.vit_merger")?; let m_o = lin(&mut pb, "v.vit_merger.attn_out")?;
        let m_dsln = ln(&mut pb, "v.vit_merger.ds_ln")?;
        let (m_up, m_down) = (lin(&mut pb, "v.vit_merger.ds_ffn_up")?, lin(&mut pb, "v.vit_merger.ds_ffn_down")?);
        let post = ln(&mut pb, "v.post_ln")?; let f_ln = ln(&mut pb, "mm.input_norm")?;
        let (f_up, f_down) = (lin(&mut pb, "mm.up")?, lin(&mut pb, "mm.down")?);
        let sep = st.f32("v.tok_embd_sep")?;
        let p = ctx.buffer(pb.v.len() as u64 * 4)?; p.as_f32_mut()[..pb.v.len()].copy_from_slice(&pb.v);
        let insert = std::env::var("MCV_INSERT").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        Ok(Vision { st, p, pw, layers, insert, patch_b, pos, m_ln1, m_q, m_k, m_v, m_qkv_b, m_o, m_dsln, m_up, m_down, post, f_ln, f_up, f_down, sep,
            pf: Pf::new(ctx)?, kn: Kern::new(ctx)?, ln: ctx.kernel(spv!("mcv_vln"), 4, 28, &[], Some(32))?, ew: ctx.kernel(spv!("mcv_vew"), 5, 32, &[], None)?,
            pad: ctx.kernel(spv!("mcv_vpad"), 4, 20, &[], None)?, win: ctx.kernel(spv!("mcv_vwin"), 2, 24, &[], None)?, dsk: ctx.kernel(spv!("mcv_vds"), 3, 16, &[], None)?,
            bufs: std::cell::RefCell::new(None) })
    }

    fn bufs(&self, ctx: &Ctx, n: u32) -> Result<()> {
        if let Some((m, _)) = &*self.bufs.borrow() { if *m >= n { return Ok(()) } }
        let r = rup(n, 256) + 256; let r2 = rup(n / 4, 256) + 256;
        let f = |e: u32| ctx.buffer(e as u64 * 4); let h = |e: u32| ctx.buffer(e as u64 * 2);
        let b = Bufs { pin: h(r * KP)?, x: f(r * D)?, a16: h((r * D).max(r2 * 4 * D))?, h16: h(r * FFP)?, hd16: h(r2 * m_up_n())?, y: f((r * 3 * D).max(r * FFP).max(r2 * m_up_n()))?,
            q: h(r * NH * 128)?, k: h(r * NH * 128)?, v: h(r * NH * 128)?, o: h(r * NH * 128)?, cat: f(r2 * 4 * D)?, x2: f(r2 * D)?, out: f(r2 * DO)?, idx: ctx.buffer(r as u64 * 4)? };
        *self.bufs.borrow_mut() = Some((n, b));
        Ok(())
    }

    fn run(&self, ctx: &Ctx, cb: vk::CommandBuffer, k: &Kernel, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<()> {
        let ds = self.pf.ds.get(ctx, k, bufs)?; ctx.dispatch(cb, k, ds, &push_u32(push), groups); ctx.barrier(cb); Ok(())
    }
    fn gemm(&self, ctx: &Ctx, cb: vk::CommandBuffer, w: WRef, a: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        let arena = if w.arena == usize::MAX { &self.pw } else { &self.st.bufs[w.arena] };
        self.pf.gemm(ctx, cb, arena, &self.p, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m, y, ldy, y_col, acc: false })?;
        ctx.barrier(cb); Ok(())
    }
    /// element-wise op (see mcv_vew.comp)
    fn ew(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, mode: u32, n: u32, d: u32, x: &Buffer, ldx: u32, y: &Buffer, ldy: u32, o16: &Buffer, ldo: u32, boff: u32, aux: u32) -> Result<()> {
        self.run(ctx, cb, &self.ew, &[x, y, o16, &self.p, &b.idx], &[mode, n, d, ldx, ldy, ldo, boff, aux], (n * d).div_ceil(256))
    }
    fn lnorm(&self, ctx: &Ctx, cb: vk::CommandBuffer, x: &Buffer, d: u32, n: u32, ln: Ln, o16: Option<&Buffer>, of: Option<&Buffer>, ldo: u32) -> Result<()> {
        let (a, bb, fl) = match (o16, of) { (Some(o), _) => (o, x, 1u32), (None, Some(o)) => (x, o, 0u32), _ => bail!("lnorm: no output") };
        self.run(ctx, cb, &self.ln, &[x, a, bb, &self.p], &[d, d, ldo, ln.w, ln.b, EPS.to_bits(), fl], n)
    }

    /// LN -> q|k|v (+bias) into y [n][3D]
    fn qkv(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, x: &Buffer, n: u32, ln: Ln, w: [WRef; 3], bias: u32) -> Result<()> {
        self.lnorm(ctx, cb, x, D, n, ln, Some(&b.a16), None, D)?;
        for (j, wj) in w.iter().enumerate() { self.gemm(ctx, cb, *wj, &b.a16, D, n, &b.y, 3 * D, j as u32 * D)?; }
        self.ew(ctx, cb, b, 4, n, 3 * D, &b.y, 3 * D, &b.y, 3 * D, &b.a16, 0, bias, 0)
    }
    /// full-attention SigLIP layer on x [n][D]
    fn layer(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, l: &Layer, x: &Buffer, n: u32) -> Result<()> {
        self.qkv(ctx, cb, b, x, n, l.ln1, [l.q, l.k, l.v], l.qkv_b)?;
        self.run(ctx, cb, &self.pad, &[&b.y, &b.q, &b.k, &b.v], &[0, n, NH, HD, 3 * D], (n * NH * 128).div_ceil(256))?;
        let fk = &self.kn.k["flash"]; let nt = n.div_ceil(128); let mut t0 = 0;
        while t0 < nt { let t1 = (t0 + 8).min(nt); let ds = self.pf.ds.get(ctx, fk, &[&b.q, &b.k, &b.v, &b.o])?;
            ctx.dispatch(cb, fk, ds, &push_u32(&[n, n, NH, NH * 128, NH * 128, NH * 128, NH * 128, (1.0f32 / (HD as f32).sqrt()).to_bits(), 1, t0]), (t1 - t0) * NH); t0 = t1; }
        ctx.barrier(cb);
        self.run(ctx, cb, &self.pad, &[&b.y, &b.o, &b.a16, &b.v], &[1, n, NH, HD, 3 * D], (n * D).div_ceil(256))?;
        self.gemm(ctx, cb, l.o.w, &b.a16, D, n, &b.y, D, 0)?;
        self.ew(ctx, cb, b, 1, n, D, x, D, &b.y, D, &b.a16, 0, l.o.b, 0)?;
        self.lnorm(ctx, cb, x, D, n, l.ln2, Some(&b.a16), None, D)?;
        self.gemm(ctx, cb, l.up.w, &b.a16, D, n, &b.y, FFP, 0)?;
        self.ew(ctx, cb, b, 2, n, FF, &b.x, 0, &b.y, FFP, &b.h16, FFP, l.up.b, 0)?;
        self.gemm(ctx, cb, l.down.w, &b.h16, FFP, n, &b.y, D, 0)?;
        self.ew(ctx, cb, b, 1, n, D, x, D, &b.y, D, &b.a16, 0, l.down.b, 0)
    }

    /// Encode one tile; returns [n_out + suffix rows][2048] f32 and the GPU wall time.
    pub fn encode(&self, ctx: &Ctx, t: &Tile) -> Result<(Vec<f32>, f64)> {
        let (hp, wp) = ((t.h / PS) as u32, (t.w / PS) as u32);
        if hp % 4 != 0 || wp % 4 != 0 { bail!("tile {}x{} not a multiple of 56", t.w, t.h) }
        let n = hp * wp; let (n2, n3) = (n / 4, n / 16);
        self.bufs(ctx, n)?;
        let bb = self.bufs.borrow(); let b = &bb.as_ref().unwrap().1;
        // im2col (c, ky, kx) -> f16 [n][640]; position buckets
        { let o = unsafe { std::slice::from_raw_parts_mut(b.pin.ptr as *mut f16, (n * KP) as usize) };
          let (w, h) = (t.w, t.h);
          for py in 0..hp as usize { for px in 0..wp as usize { let r = py * wp as usize + px;
            for c in 0..3 { for ky in 0..PS { for kx in 0..PS { o[r * KP as usize + c * 196 + ky * 14 + kx] = f16::from_f32(t.px[c * h * w + (py * PS + ky) * w + px * PS + kx]) } } }
            for i in 588..KP as usize { o[r * KP as usize + i] = f16::ZERO } } }
          let idx = unsafe { std::slice::from_raw_parts_mut(b.idx.ptr as *mut u32, n as usize) };
          for i in 0..hp as usize { for j in 0..wp as usize { idx[i * wp as usize + j] = ((70 * i / hp as usize) * 70 + 70 * j / wp as usize) as u32 } } }
        let tm = std::time::Instant::now();
        let mut err = Ok(());
        let cb = ctx.record_raw(|cb| { err = (|| -> Result<()> {
            let pwr = WRef { arena: usize::MAX, off: 0, ty: GgmlType::F16, k: KP, n: D, e: 1, bytes: (D * KP * 2) as u64 };
            self.gemm(ctx, cb, pwr, &b.pin, KP, n, &b.y, D, 0)?;
            self.ew(ctx, cb, b, 0, n, D, &b.x, D, &b.y, D, &b.a16, 0, self.patch_b, self.pos)?;
            for l in &self.layers[..=self.insert] { self.layer(ctx, cb, b, l, &b.x, n)?; }
            // ViT merger: window attention (residual) ...
            self.qkv(ctx, cb, b, &b.x, n, self.m_ln1, [self.m_q, self.m_k, self.m_v], self.m_qkv_b)?;
            self.run(ctx, cb, &self.win, &[&b.y, &b.a16], &[hp, wp, NH, HD, 3 * D, (1.0f32 / (HD as f32).sqrt()).to_bits()], (n * NH).div_ceil(64))?;
            self.gemm(ctx, cb, self.m_o.w, &b.a16, D, n, &b.y, D, 0)?;
            self.ew(ctx, cb, b, 1, n, D, &b.x, D, &b.y, D, &b.a16, 0, self.m_o.b, 0)?;
            // ... then 2x2 concat -> LN -> MLP (tanh GELU) + mean of the 4
            self.run(ctx, cb, &self.dsk, &[&b.x, &b.cat, &b.x2], &[hp, wp, D, 1], (n2 * 4 * D).div_ceil(256))?;
            self.lnorm(ctx, cb, &b.cat, 4 * D, n2, self.m_dsln, Some(&b.a16), None, 4 * D)?;
            let fm = self.m_up.w.n;
            self.gemm(ctx, cb, self.m_up.w, &b.a16, 4 * D, n2, &b.y, fm, 0)?;
            self.ew(ctx, cb, b, 2, n2, fm, &b.x, 0, &b.y, fm, &b.hd16, fm, self.m_up.b, 0)?;
            self.gemm(ctx, cb, self.m_down.w, &b.hd16, fm, n2, &b.y, D, 0)?;
            self.ew(ctx, cb, b, 1, n2, D, &b.x2, D, &b.y, D, &b.a16, 0, self.m_down.b, 0)?;
            for l in &self.layers[self.insert + 1..] { self.layer(ctx, cb, b, l, &b.x2, n2)?; }
            // post-LN, final 2x2 merge -> LN -> MLP (erf GELU) -> 2048
            self.lnorm(ctx, cb, &b.x2, D, n2, self.post, None, Some(&b.x), D)?;
            self.run(ctx, cb, &self.dsk, &[&b.x, &b.cat, &b.x2], &[hp / 2, wp / 2, D, 0], (n3 * 4 * D).div_ceil(256))?;
            self.lnorm(ctx, cb, &b.cat, 4 * D, n3, self.f_ln, Some(&b.a16), None, 4 * D)?;
            let fu = self.f_up.w.n;
            self.gemm(ctx, cb, self.f_up.w, &b.a16, 4 * D, n3, &b.y, fu, 0)?;
            self.ew(ctx, cb, b, 3, n3, fu, &b.x, 0, &b.y, fu, &b.hd16, fu, self.f_up.b, 0)?;
            self.gemm(ctx, cb, self.f_down.w, &b.hd16, fu, n3, &b.y, DO, 0)?;
            self.ew(ctx, cb, b, 4, n3, DO, &b.out, DO, &b.y, DO, &b.a16, 0, self.f_down.b, 0)?;
            Ok(()) })(); })?;
        err?;
        ctx.submit_repeated(cb, 1)?;
        let dt = tm.elapsed().as_secs_f64();
        let mut out = b.out.as_f32_mut()[..(n3 * DO) as usize].to_vec();
        for &r in &t.suffix { out.extend_from_slice(&self.sep[r * DO as usize..(r + 1) * DO as usize]); }
        Ok((out, dt))
    }
}

/// llama.cpp img_tool::resize_pillow (fixed-point separable resampling, PRECISION_BITS 22), bicubic a = -0.5.
pub fn resize_bicubic(src: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    const PB_: i32 = 22;
    let filt = |x: f64| -> f64 { let x = x.abs(); let a = -0.5; if x < 1.0 { ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 } else if x < 2.0 { (((x - 5.0) * x + 8.0) * x - 4.0) * a } else { 0.0 } };
    let pre = |ins: usize, outs: usize| -> (usize, Vec<(usize, usize)>, Vec<i32>) {
        let scale = ins as f64 / outs as f64; let fs = scale.max(1.0); let support = 2.0 * fs;
        let ks = support.ceil() as usize * 2 + 1; let mut b = vec![]; let mut w = vec![0i32; outs * ks];
        for xx in 0..outs { let center = (xx as f64 + 0.5) * scale; let ss = 1.0 / fs;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize; let xmax = ((center + support + 0.5) as i64).min(ins as i64) as usize - xmin;
            let mut pw = vec![0f64; ks]; let mut ww = 0.0;
            for x in 0..xmax { let v = filt((x as f64 + xmin as f64 - center + 0.5) * ss); pw[x] = v; ww += v; }
            for x in 0..xmax { if ww != 0.0 { pw[x] /= ww } }
            for x in 0..ks { let v = pw[x] * (1u64 << PB_) as f64; w[xx * ks + x] = (v + if pw[x] < 0.0 { -0.5 } else { 0.5 }) as i32; }
            b.push((xmin, xmax)); }
        (ks, b, w) };
    let clip8 = |v: i32| -> u8 { v.clamp(0, 255) as u8 };
    let mut cur = src.to_vec(); let (mut cw, ch) = (sw, sh);
    if tw != sw { let (ks, b, w) = pre(sw, tw); let mut o = vec![0u8; tw * ch * 3];
        for y in 0..ch { for xx in 0..tw { let (xmin, xc) = b[xx]; for c in 0..3 { let mut s: i32 = 1 << (PB_ - 1);
            for x in 0..xc { s += cur[(y * cw + xmin + x) * 3 + c] as i32 * w[xx * ks + x]; } o[(y * tw + xx) * 3 + c] = clip8(s >> PB_); } } }
        cur = o; cw = tw; }
    if th != ch { let (ks, b, w) = pre(ch, th); let mut o = vec![0u8; cw * th * 3];
        for yy in 0..th { let (ymin, yc) = b[yy]; for i in 0..cw * 3 { let mut s: i32 = 1 << (PB_ - 1);
            for y in 0..yc { s += cur[(ymin + y) * cw * 3 + i] as i32 * w[yy * ks + y]; } o[yy * cw * 3 + i] = clip8(s >> PB_); } }
        cur = o; }
    cur
}

fn round_even(v: f64) -> i64 { let f = v.floor(); let d = v - f; if d > 0.5 { f as i64 + 1 } else if d < 0.5 { f as i64 } else if (f as i64) % 2 == 0 { f as i64 } else { f as i64 + 1 } }
fn ensure_divide(len: usize, p: usize) -> usize { ((round_even(len as f64 / p as f64) as usize) * p).max(p) }
fn best_resize(w: usize, h: usize, scale_res: usize, p: usize, upscale: bool) -> (usize, usize) {
    let (mut w2, mut h2) = (w, h);
    if w * h > scale_res * scale_res || upscale { let r = w as f32 / h as f32; h2 = (scale_res as f32 / r.sqrt()) as usize; w2 = (h2 as f32 * r) as usize; }
    (ensure_divide(w2, p), ensure_divide(h2, p))
}

/// mtmd minicpmv preprocessing: overview (+ llava-uhd slices when the area exceeds 448^2), normalised (x/255-0.5)/0.5.
/// Separator rows per tile: 0 </image>, 1 <slice>, 2 </slice>, 3 \n. Returns tiles and the slice grid (gx, gy).
pub fn preprocess(rgb: &[u8], w: usize, h: usize) -> (Vec<Tile>, (usize, usize)) {
    let (ss, al) = (448usize, 56usize);
    let norm = |px: &[u8], tw: usize, th: usize| -> Vec<f32> { let mut o = vec![0f32; 3 * tw * th];
        for i in 0..tw * th { for c in 0..3 { o[c * tw * th + i] = (px[i * 3 + c] as f32 / 255.0 - 0.5) / 0.5 } } o };
    let ratio = (w * h) as f32 / (ss * ss) as f32;
    if ratio <= 1.0 {
        let (ow, oh) = best_resize(w, h, ss, al, true);
        return (vec![Tile { w: ow, h: oh, px: norm(&resize_bicubic(rgb, w, h, ow, oh), ow, oh), suffix: vec![0] }], (0, 0));
    }
    let (ow, oh) = best_resize(w, h, ss, al, false);
    let log_ratio = (w as f32 / h as f32).ln();
    let multiple = (ratio.ceil() as usize).min(9);
    let mut best = (1usize, 1usize); let mut me = f32::INFINITY;
    for nums in [multiple - 1, multiple, multiple + 1] { if nums == 1 || nums > 9 { continue }
        for m in 1..=nums { if nums % m == 0 { let g = (m, nums / m); let e = (log_ratio - (g.0 as f32 / g.1 as f32).ln()).abs(); if e < me { me = e; best = g } } } }
    let (gx, gy) = best;
    let (rw0, rh0) = (ensure_divide(w, gx), ensure_divide(h, gy));
    let (bw, bh) = best_resize(rw0 / gx, rh0 / gy, ss, al, true);
    let (rw, rh) = (bw * gx, bh * gy);
    let mut tiles = vec![Tile { w: ow, h: oh, px: norm(&resize_bicubic(rgb, w, h, ow, oh), ow, oh), suffix: vec![0, 1] }];
    let refined = resize_bicubic(rgb, w, h, rw, rh);
    for iy in 0..gy { for ix in 0..gx {
        let mut px = vec![0u8; bw * bh * 3];
        for y in 0..bh { let s = ((iy * bh + y) * rw + ix * bw) * 3; px[y * bw * 3..(y + 1) * bw * 3].copy_from_slice(&refined[s..s + bw * 3]); }
        let last = iy == gy - 1 && ix == gx - 1;
        let suffix = if last { vec![2] } else if ix == gx - 1 { vec![2, 3, 1] } else { vec![2, 1] };
        tiles.push(Tile { w: bw, h: bh, px: norm(&px, bw, bh), suffix }); } }
    (tiles, (gx, gy))
}

/// Minimal binary PPM (P6, maxval 255) reader.
pub fn read_ppm(path: &str) -> Result<(Vec<u8>, usize, usize)> {
    let d = std::fs::read(path)?; let mut i = 0; let mut f = vec![];
    while f.len() < 4 { while i < d.len() && (d[i] as char).is_whitespace() { i += 1 } if d[i] == b'#' { while d[i] != b'\n' { i += 1 } continue }
        let s = i; while i < d.len() && !(d[i] as char).is_whitespace() { i += 1 } f.push(String::from_utf8_lossy(&d[s..i]).to_string()); }
    if f[0] != "P6" || f[3] != "255" { bail!("only P6/255 PPM") }
    let (w, h): (usize, usize) = (f[1].parse()?, f[2].parse()?); i += 1;
    Ok((d[i..i + w * h * 3].to_vec(), w, h))
}
fn m_up_n() -> u32 { 17216 }
