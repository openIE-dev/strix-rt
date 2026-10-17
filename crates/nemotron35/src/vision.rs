//! Nemotron 3.5 Super VL vision tower (mmproj projector "nemotron_omni": C-RADIO ViT-H/16 + 2x2 unshuffle + RMSNorm/relu^2 MLP)
//! on the GPU, following llama.cpp tools/mtmd/models/nemotron-omni.cpp (still images):
//! patch conv (GEMM, K 768) + bilinear-resized 128x128 position embedding -> 10 register tokens prepended ->
//! 32 pre-LN ViT layers (fused qkv with bias, 16 heads of 80 padded to 128 for qimage flash, erf-GELU MLP 5120) -> post-LN ->
//! drop registers -> 2x2 pixel unshuffle (5120) -> RMSNorm(eps 1e-5) -> Linear 20480 -> relu^2 -> Linear 4096.
//! Linears use prefill's GEMM on the BF16 weights (mmproj re-written by crates/mcv/ref/mmproj2st.py). Patterned on crates/mcv/src/vision.rs.
use anyhow::{bail, Result};
use ash::vk;
use gguf::GgmlType;
use half::f16;
use prefill::{GemmOp, Pf};
use qimage::{st, Kern};
use rtcore::WRef;
use vkrt::{push_u32, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const D: u32 = 1280; pub const NH: u32 = 16; pub const HD: u32 = 80; pub const FF: u32 = 5120;
pub const NL: usize = 32; pub const PS: usize = 16; pub const KP: u32 = 768; pub const NR: u32 = 10; pub const DO: u32 = 4096; pub const G: usize = 128;
const EPS: f32 = 1e-6; const EPS_MM: f32 = 1e-5;
fn rup(x: u32, m: u32) -> u32 { x.div_ceil(m) * m }

#[derive(Clone, Copy)] pub struct Lin { pub w: WRef, pub b: u32 }
#[derive(Clone, Copy)] pub struct Ln { pub w: u32, pub b: u32 }
pub struct Layer { pub ln1: Ln, pub ln2: Ln, pub qkv: Lin, pub o: Lin, pub up: Lin, pub down: Lin }
struct PB { v: Vec<f32> }
impl PB { fn put(&mut self, x: Vec<f32>) -> u32 { let o = self.v.len() as u32; self.v.extend(x); while self.v.len() % 64 != 0 { self.v.push(0.0) } o } }

pub struct Vision {
    pub st: st::Model, pub p: Buffer, pub pw: Buffer, pub layers: Vec<Layer>, pub post: Ln, pub mm0: u32, pub mm1: WRef, pub mm3: WRef,
    pub cls: Vec<f32>, pub pos: Vec<f32>, pub pf: Pf, pub kn: Kern, nrm: Kernel, ew: Kernel, pad: Kernel, mrg: Kernel,
    bufs: std::cell::RefCell<Option<(u32, Bufs)>>,
}
struct Bufs { pin: Buffer, x: Buffer, a16: Buffer, h16: Buffer, y: Buffer, q: Buffer, k: Buffer, v: Buffer, o: Buffer, cat: Buffer, out: Buffer, posb: Buffer }

/// ggml_interpolate bilinear (half-pixel centres, edge clamp; GGML_SCALE_MODE_BILINEAR without align-corners) of the
/// [G][G][D] table to [h][w][D].
pub fn resize_pos(t: &[f32], h: usize, w: usize) -> Vec<f32> {
    if h == G && w == G { return t.to_vec() }
    let d = D as usize; let mut o = vec![0f32; h * w * d];
    let (sy, sx) = (G as f32 / h as f32, G as f32 / w as f32);
    for i in 0..h { for j in 0..w {
        let fy = ((i as f32 + 0.5) * sy - 0.5).max(0.0); let fx = ((j as f32 + 0.5) * sx - 0.5).max(0.0);
        let (y0, x0) = (fy.floor() as usize, fx.floor() as usize); let (y1, x1) = ((y0 + 1).min(G - 1), (x0 + 1).min(G - 1));
        let (dy, dx) = (fy - y0 as f32, fx - x0 as f32);
        for c in 0..d { let g = |y: usize, x: usize| t[(y * G + x) * d + c];
            o[(i * w + j) * d + c] = g(y0, x0) * (1.0 - dy) * (1.0 - dx) + g(y0, x1) * (1.0 - dy) * dx + g(y1, x0) * dy * (1.0 - dx) + g(y1, x1) * dy * dx; } } }
    o
}

impl Vision {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|k: &str| k.ends_with(".weight") && k.contains("blk") && !k.contains("ln") || k == "mm.model.mlp.1.weight" || k == "mm.model.mlp.3.weight")?;
        let mut pb = PB { v: vec![] };
        let lin = |pb: &mut PB, p: &str| -> Result<Lin> { Ok(Lin { w: st.wref(&format!("{p}.weight"))?, b: pb.put(st.f32(&format!("{p}.bias"))?) }) };
        let ln = |pb: &mut PB, p: &str| -> Result<Ln> { Ok(Ln { w: pb.put(st.f32(&format!("{p}.weight"))?), b: pb.put(st.f32(&format!("{p}.bias"))?) }) };
        let mut layers = vec![];
        for i in 0..NL { let p = format!("v.blk.{i}");
            layers.push(Layer { ln1: ln(&mut pb, &format!("{p}.ln1"))?, ln2: ln(&mut pb, &format!("{p}.ln2"))?, qkv: lin(&mut pb, &format!("{p}.attn_qkv"))?,
                o: lin(&mut pb, &format!("{p}.attn_out"))?, up: lin(&mut pb, &format!("{p}.ffn_up"))?, down: lin(&mut pb, &format!("{p}.ffn_down"))? }); }
        if layers[0].qkv.w.n != 3 * D || layers[0].up.w.n != FF { bail!("unexpected C-RADIO shapes") }
        // patch conv F32 [1280][3][16][16] -> f16 [1280][768] (K order c, ky, kx), no bias
        let cw = st.f32("v.patch_embd.weight")?; if cw.len() != (D * KP) as usize { bail!("patch_embd size") }
        let pw = ctx.buffer((D * KP * 2) as u64)?;
        { let o = unsafe { std::slice::from_raw_parts_mut(pw.ptr as *mut f16, (D * KP) as usize) }; for i in 0..o.len() { o[i] = f16::from_f32(cw[i]) } }
        let post = ln(&mut pb, "v.post_ln")?;
        let mm0 = pb.put(st.f32("mm.model.mlp.0.weight")?);
        let (mm1, mm3) = (st.wref("mm.model.mlp.1.weight")?, st.wref("mm.model.mlp.3.weight")?);
        let cls = st.f32("v.class_embd")?; let pos = st.f32("v.position_embd.weight")?;
        if cls.len() != (NR * D) as usize || pos.len() != G * G * D as usize { bail!("class/position embedding sizes") }
        pb.put(vec![0.0; 64]);
        let p = ctx.buffer(pb.v.len() as u64 * 4)?; p.as_f32_mut()[..pb.v.len()].copy_from_slice(&pb.v);
        Ok(Vision { st, p, pw, layers, post, mm0, mm1, mm3, cls, pos, pf: Pf::new(ctx)?, kn: Kern::new(ctx)?,
            nrm: ctx.kernel(spv!("n35_vnorm"), 4, 32, &[], Some(32))?, ew: ctx.kernel(spv!("n35_vew"), 4, 32, &[], None)?,
            pad: ctx.kernel(spv!("mcv_vpad"), 4, 20, &[], None)?, mrg: ctx.kernel(spv!("n35_vmerge"), 2, 16, &[], None)?, bufs: std::cell::RefCell::new(None) })
    }

    fn bufs(&self, ctx: &Ctx, n: u32) -> Result<()> {
        if let Some((m, _)) = &*self.bufs.borrow() { if *m >= n { return Ok(()) } }
        let r = rup(n + NR, 256) + 256; let r4 = rup(n / 4, 256) + 256; let fm = self.mm1.n;
        let f = |e: u32| ctx.buffer(e as u64 * 4); let h = |e: u32| ctx.buffer(e as u64 * 2);
        let b = Bufs { pin: h(r * KP)?, x: f(r * D)?, a16: h((r * D).max(r4 * 4 * D))?, h16: h((r * FF).max(r4 * fm))?, y: f((r * 3 * D).max(r * FF).max(r4 * fm))?,
            q: h(r * NH * 128)?, k: h(r * NH * 128)?, v: h(r * NH * 128)?, o: h(r * NH * 128)?, cat: f(r4 * 4 * D)?, out: f(r4 * DO)?, posb: f(n * D + 64)? };
        *self.bufs.borrow_mut() = Some((n, b));
        Ok(())
    }
    fn run(&self, ctx: &Ctx, cb: vk::CommandBuffer, k: &Kernel, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<()> {
        let ds = self.pf.ds.get(ctx, k, bufs)?; ctx.dispatch(cb, k, ds, &push_u32(push), groups); ctx.barrier(cb); Ok(())
    }
    fn gemm(&self, ctx: &Ctx, cb: vk::CommandBuffer, w: WRef, a: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32) -> Result<()> {
        let arena = if w.arena == usize::MAX { &self.pw } else { &self.st.bufs[w.arena] };
        self.pf.gemm(ctx, cb, arena, &self.p, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m, y, ldy, y_col: 0, acc: false })?;
        ctx.barrier(cb); Ok(())
    }
    fn ew(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, mode: u32, n: u32, d: u32, x: &Buffer, ldx: u32, y: &Buffer, ldy: u32, o16: &Buffer, ldo: u32, boff: u32, xoff: u32, p: Option<&Buffer>) -> Result<()> {
        let _ = b; self.run(ctx, cb, &self.ew, &[x, y, o16, p.unwrap_or(&self.p)], &[mode, n, d, ldx, ldy, ldo, boff, xoff], (n * d).div_ceil(256))
    }
    /// flags: 1 f16 out, 2 RMSNorm
    fn norm(&self, ctx: &Ctx, cb: vk::CommandBuffer, x: &Buffer, xoff: u32, d: u32, n: u32, w: u32, bb: u32, eps: f32, flags: u32, o16: &Buffer, of: &Buffer) -> Result<()> {
        self.run(ctx, cb, &self.nrm, &[x, o16, of, &self.p], &[d, d, d, w, bb, eps.to_bits(), flags, xoff], n)
    }
    fn layer(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, l: &Layer, n: u32) -> Result<()> {
        let x = &b.x;
        self.norm(ctx, cb, x, 0, D, n, l.ln1.w, l.ln1.b, EPS, 1, &b.a16, &b.y)?;
        self.gemm(ctx, cb, l.qkv.w, &b.a16, D, n, &b.y, 3 * D)?;
        self.ew(ctx, cb, b, 4, n, 3 * D, &b.y, 3 * D, &b.y, 3 * D, &b.a16, 0, l.qkv.b, 0, None)?;
        self.run(ctx, cb, &self.pad, &[&b.y, &b.q, &b.k, &b.v], &[0, n, NH, HD, 3 * D], (n * NH * 128).div_ceil(256))?;
        let fk = &self.kn.k["flash"]; let nt = n.div_ceil(128); let mut t0 = 0;
        while t0 < nt { let t1 = (t0 + 8).min(nt); let ds = self.pf.ds.get(ctx, fk, &[&b.q, &b.k, &b.v, &b.o])?;
            ctx.dispatch(cb, fk, ds, &push_u32(&[n, n, NH, NH * 128, NH * 128, NH * 128, NH * 128, (1.0f32 / (HD as f32).sqrt()).to_bits(), 1, t0]), (t1 - t0) * NH); t0 = t1; }
        ctx.barrier(cb);
        self.run(ctx, cb, &self.pad, &[&b.y, &b.o, &b.a16, &b.v], &[1, n, NH, HD, 3 * D], (n * D).div_ceil(256))?;
        self.gemm(ctx, cb, l.o.w, &b.a16, D, n, &b.y, D)?;
        self.ew(ctx, cb, b, 1, n, D, x, D, &b.y, D, &b.a16, 0, l.o.b, 0, None)?;
        self.norm(ctx, cb, x, 0, D, n, l.ln2.w, l.ln2.b, EPS, 1, &b.a16, &b.y)?;
        self.gemm(ctx, cb, l.up.w, &b.a16, D, n, &b.y, FF)?;
        self.ew(ctx, cb, b, 3, n, FF, &b.x, 0, &b.y, FF, &b.h16, FF, l.up.b, 0, None)?;
        self.gemm(ctx, cb, l.down.w, &b.h16, FF, n, &b.y, D)?;
        self.ew(ctx, cb, b, 1, n, D, x, D, &b.y, D, &b.a16, 0, l.down.b, 0, None)
    }

    /// Encode an image already in model input space: px [3][h][w] f32 (normalised), h and w multiples of 32.
    /// Returns [(h/32)(w/32)][4096] f32 and GPU wall seconds.
    pub fn encode(&self, ctx: &Ctx, px: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, f64)> {
        if h % (2 * PS) != 0 || w % (2 * PS) != 0 { bail!("image {w}x{h} not a multiple of 32") }
        let (hp, wp) = ((h / PS) as u32, (w / PS) as u32); let n = hp * wp; let nt = n + NR; let n4 = n / 4;
        self.bufs(ctx, n)?;
        let bb = self.bufs.borrow(); let b = &bb.as_ref().unwrap().1;
        { let o = unsafe { std::slice::from_raw_parts_mut(b.pin.ptr as *mut f16, (n * KP) as usize) };
          for py in 0..hp as usize { for pxi in 0..wp as usize { let r = py * wp as usize + pxi;
            for c in 0..3 { for ky in 0..PS { for kx in 0..PS { o[r * KP as usize + c * 256 + ky * 16 + kx] = f16::from_f32(px[c * h * w + (py * PS + ky) * w + pxi * PS + kx]) } } } } }
          let pos = resize_pos(&self.pos, hp as usize, wp as usize); b.posb.as_f32_mut()[..pos.len()].copy_from_slice(&pos);
          b.x.as_f32_mut()[..self.cls.len()].copy_from_slice(&self.cls); }
        let tm = std::time::Instant::now();
        // Large images (n patches > SPLIT_N) are submitted one stage per command buffer (embed, each ViT layer, tail): a single
        // multi-second submission trips the amdgpu job timeout (device lost at 13160 patches / 3290 tokens, ~8 s of attention).
        let split = n > std::env::var("N35V_SPLIT_N").ok().and_then(|v| v.parse().ok()).unwrap_or(2048u32);
        let steps = NL + 2; let groups: Vec<(usize, usize)> = if split { (0..steps).map(|i| (i, i + 1)).collect() } else { vec![(0, steps)] };
        for (s0, s1) in groups {
            let mut err = Ok(());
            let cb = ctx.record_raw(|cb| { err = (|| -> Result<()> {
                for st in s0..s1 { match st {
                    0 => { let pwr = WRef { arena: usize::MAX, off: 0, ty: GgmlType::F16, k: KP, n: D, e: 1, bytes: (D * KP * 2) as u64 };
                        self.gemm(ctx, cb, pwr, &b.pin, KP, n, &b.y, D)?;
                        self.ew(ctx, cb, b, 0, n, D, &b.x, D, &b.y, D, &b.a16, 0, 0, NR, Some(&b.posb))?; }
                    x if x <= NL => { self.layer(ctx, cb, b, &self.layers[x - 1], nt)?; }
                    _ => {
                        // post-LN (f32, in place is not possible: write to y), drop registers, unshuffle, RMSNorm -> MLP
                        self.norm(ctx, cb, &b.x, NR, D, n, self.post.w, self.post.b, EPS, 0, &b.a16, &b.y)?;
                        self.run(ctx, cb, &self.mrg, &[&b.y, &b.cat], &[hp, wp, D, 0], (n4 * 4 * D).div_ceil(256))?;
                        self.norm(ctx, cb, &b.cat, 0, 4 * D, n4, self.mm0, 0, EPS_MM, 3, &b.a16, &b.y)?;
                        let fm = self.mm1.n;
                        self.gemm(ctx, cb, self.mm1, &b.a16, 4 * D, n4, &b.y, fm)?;
                        self.ew(ctx, cb, b, 5, n4, fm, &b.x, 0, &b.y, fm, &b.h16, fm, 0, 0, None)?;
                        self.gemm(ctx, cb, self.mm3, &b.h16, fm, n4, &b.out, DO)?; }
                } }
                Ok(()) })(); })?;
            err?;
            ctx.submit_repeated(cb, 1)?;
        }
        let dt = tm.elapsed().as_secs_f64();
        Ok((b.out.as_f32_mut()[..(n4 * DO) as usize].to_vec(), dt))
    }
}

/// llama.cpp mtmd-debug "rainbow" test image (values in [0,1], fed to the encoder without normalisation), [3][s][s].
pub fn rainbow(s: usize) -> Vec<f32> {
    let mut o = vec![0f32; 3 * s * s]; let (cx, cy) = (s as f32 / 2.0, s as f32 / 2.0); let md = (cx * cx + cy * cy).sqrt();
    for y in 0..s { for x in 0..s {
        let (dx, dy) = (x as f32 - cx, y as f32 - cy);
        let mut hue = dy.atan2(dx) / (2.0 * 3.14159265f32); if hue < 0.0 { hue += 1.0 }
        let sat = ((dx * dx + dy * dy).sqrt() / md).min(1.0);
        let h6 = hue * 6.0; let i6 = h6 as i32; let f = h6 - i6 as f32;
        let (p, q, t) = (1.0 - sat, 1.0 - sat * f, 1.0 - sat * (1.0 - f));
        let (r, g, b) = match i6 % 6 { 0 => (1.0, t, p), 1 => (q, 1.0, p), 2 => (p, 1.0, t), 3 => (p, q, 1.0), 4 => (t, p, 1.0), _ => (1.0, p, q) };
        o[y * s + x] = r; o[s * s + y * s + x] = g; o[2 * s * s + y * s + x] = b; } }
    o
}
