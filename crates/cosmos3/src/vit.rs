//! Cosmos3 understanding path, vision side: the Qwen3-VL ViT in vision_encoder/ (27 SigLIP-style blocks, d 1152, 16 heads
//! of 72, tanh-GELU MLP 4304, axial 2D rope, learned 48x48 position table bilinearly resampled to the patch grid,
//! Conv3d patch embed 2x16x16), 2x2 patch merger -> 4096 and three DeepStack mergers (after blocks 8, 16, 24) whose
//! outputs are added to the und tower's hidden states after LLM layers 0, 1, 2 at the image-token rows.
//! Patches are in spatial-merge-block order (as the HF processor emits them), so each 2x2 merge is 4 consecutive rows.
//! Linears use prefill's GEMM straight from the BF16 weights; fc2 (K 4304, not a multiple of 64) and the patch conv are
//! repacked to f16 at load. Attention is qimage's flash with heads zero-padded to 128 (as in crates/mcv, not changed).
use anyhow::{bail, Result};
use gguf::GgmlType;
use half::f16;
use prefill::{GemmOp, Pf};
use qimage::{st, Kern};
use rtcore::WRef;
use vkrt::{push_u32, Buffer, Ctx, Kernel};
use ash::vk;

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const D: u32 = 1152; pub const NH: u32 = 16; pub const HD: u32 = 72; pub const FF: u32 = 4304; pub const FFP: u32 = 4352;
pub const NL: usize = 27; pub const KP: u32 = 1536; pub const DM: u32 = 4608; pub const DO: u32 = 4096; pub const DS: [usize; 3] = [8, 16, 24];
const EPS: f32 = 1e-6; const SIDE: usize = 48;

#[derive(Clone, Copy)] struct Lin { w: WRef, b: u32 }
#[derive(Clone, Copy)] struct Ln { w: u32, b: u32 }
struct Layer { n1: Ln, n2: Ln, qkv: Lin, proj: Lin, fc1: Lin, fc2: Lin }
struct Merger { norm: Ln, fc1: Lin, fc2: Lin }
struct Bufs { pin: Buffer, x: Buffer, a16: Buffer, h16: Buffer, y: Buffer, q: Buffer, k: Buffer, v: Buffer, o: Buffer, rope: Buffer, out: [Buffer; 4] }
pub struct Vit { st: st::Model, p: Buffer, pw: Buffer, layers: Vec<Layer>, merger: Merger, ds: Vec<Merger>, patch_b: u32, pos: Vec<f32>,
    pf: Pf, kn: Kern, ln: Kernel, ew: Kernel, pad: Kernel, bufs: std::cell::RefCell<Option<(u32, Bufs)>> }
/// ViT result for one image: merged embeddings [n/4][4096] and the 3 DeepStack features [n/4][4096].
pub struct VitOut { pub emb: Vec<f32>, pub ds: [Vec<f32>; 3], pub gpu_s: f64 }

/// (row, col) of patch `i` of an h x w patch grid in 2x2 merge-block order.
pub fn rc(i: usize, w: usize) -> (usize, usize) { let (bw, inn, blk) = (w / 2, i % 4, i / 4); ((blk / bw) * 2 + inn / 2, (blk % bw) * 2 + inn % 2) }

impl Vit {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|k: &str| k.ends_with(".weight") && (k.contains("qkv") || k.contains("proj.weight") && !k.contains("patch") || k.contains("linear_fc1") || k.contains("merger") && k.contains("linear_fc2")))?;
        let mut pv: Vec<f32> = vec![];
        let put = |pv: &mut Vec<f32>, v: Vec<f32>| -> u32 { let o = pv.len() as u32; pv.extend(v); while pv.len() % 64 != 0 { pv.push(0.0) } o };
        let ln = |pv: &mut Vec<f32>, p: &str| -> Result<Ln> { Ok(Ln { w: put(pv, st.f32(&format!("{p}.weight"))?), b: put(pv, st.f32(&format!("{p}.bias"))?) }) };
        // f16 side arena: 27 x fc2 [1152][4352] (K zero padded) then the patch conv [1152][1536]
        let fsz = (D * FFP) as usize; let pw = ctx.buffer(((NL * fsz) + (D * KP) as usize) as u64 * 2)?;
        let o16 = unsafe { std::slice::from_raw_parts_mut(pw.ptr as *mut f16, NL * fsz + (D * KP) as usize) };
        let mut layers = vec![];
        for i in 0..NL { let p = format!("blocks.{i}");
            let lin = |pv: &mut Vec<f32>, n: &str| -> Result<Lin> { Ok(Lin { w: st.wref(&format!("{p}.{n}.weight"))?, b: put(pv, st.f32(&format!("{p}.{n}.bias"))?) }) };
            let w2 = st.f32(&format!("{p}.mlp.linear_fc2.weight"))?;
            for r in 0..D as usize { for c in 0..FFP as usize { o16[i * fsz + r * FFP as usize + c] = f16::from_f32(if c < FF as usize { w2[r * FF as usize + c] } else { 0.0 }) } }
            let fc2 = Lin { w: WRef { arena: usize::MAX, off: (i * fsz * 2) as u64, ty: GgmlType::F16, k: FFP, n: D, e: 1, bytes: (fsz * 2) as u64 }, b: put(&mut pv, st.f32(&format!("{p}.mlp.linear_fc2.bias"))?) };
            layers.push(Layer { n1: ln(&mut pv, &format!("{p}.norm1"))?, n2: ln(&mut pv, &format!("{p}.norm2"))?, qkv: lin(&mut pv, "attn.qkv")?, proj: lin(&mut pv, "attn.proj")?, fc1: lin(&mut pv, "mlp.linear_fc1")?, fc2 }); }
        let cw = st.f32("patch_embed.proj.weight")?; let base = NL * fsz;
        for (j, v) in cw.iter().enumerate() { o16[base + j] = f16::from_f32(*v) }
        let mg = |pv: &mut Vec<f32>, p: &str| -> Result<Merger> { Ok(Merger { norm: ln(pv, &format!("{p}.norm"))?,
            fc1: Lin { w: st.wref(&format!("{p}.linear_fc1.weight"))?, b: put(pv, st.f32(&format!("{p}.linear_fc1.bias"))?) },
            fc2: Lin { w: st.wref(&format!("{p}.linear_fc2.weight"))?, b: put(pv, st.f32(&format!("{p}.linear_fc2.bias"))?) } }) };
        let merger = mg(&mut pv, "merger")?; let ds = (0..3).map(|i| mg(&mut pv, &format!("deepstack_merger_list.{i}"))).collect::<Result<Vec<_>>>()?;
        let patch_b = put(&mut pv, st.f32("patch_embed.proj.bias")?); let pos = st.f32("pos_embed.weight")?;
        if layers[0].qkv.w.k % 64 != 0 || merger.fc1.w.k != DM { bail!("vit: unexpected weight shapes") }
        let p = ctx.buffer(pv.len() as u64 * 4)?; p.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        Ok(Vit { st, p, pw, layers, merger, ds, patch_b, pos, pf: Pf::new(ctx)?, kn: Kern::new(ctx)?,
            ln: ctx.kernel(spv!("cos_vln"), 4, 28, &[], Some(32))?, ew: ctx.kernel(spv!("cos_vew"), 5, 32, &[], None)?, pad: ctx.kernel(spv!("cos_vpad"), 6, 20, &[], None)?,
            bufs: std::cell::RefCell::new(None) })
    }
    fn bufs(&self, ctx: &Ctx, n: u32) -> Result<()> {
        if let Some((m, _)) = &*self.bufs.borrow() { if *m >= n { return Ok(()) } }
        let r = n.div_ceil(256) * 256 + 256; let f = |e: u32| ctx.buffer(e as u64 * 4); let h = |e: u32| ctx.buffer(e as u64 * 2);
        let b = Bufs { pin: h(r * KP)?, x: f(r * D)?, a16: h(r * FFP)?, h16: h(r * FFP)?, y: f(r * FFP)?, q: h(r * NH * 128)?, k: h(r * NH * 128)?, v: h(r * NH * 128)?, o: h(r * NH * 128)?,
            rope: f(r * HD)?, out: [f(r / 4 * DO)?, f(r / 4 * DO)?, f(r / 4 * DO)?, f(r / 4 * DO)?] };
        unsafe { std::ptr::write_bytes(b.h16.ptr as *mut u8, 0, (r * FFP * 2) as usize); } // fc2 reads K-pad columns 4304..4352
        *self.bufs.borrow_mut() = Some((n, b)); Ok(())
    }
    fn run(&self, ctx: &Ctx, cb: vk::CommandBuffer, k: &Kernel, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<()> {
        let ds = self.pf.ds.get(ctx, k, bufs)?; ctx.dispatch(cb, k, ds, &push_u32(push), groups); ctx.barrier(cb); Ok(())
    }
    fn gemm(&self, ctx: &Ctx, cb: vk::CommandBuffer, w: WRef, a: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32) -> Result<()> {
        let arena = if w.arena == usize::MAX { &self.pw } else { &self.st.bufs[w.arena] };
        self.pf.gemm(ctx, cb, arena, &self.p, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m, y, ldy, y_col: 0, acc: false })?; ctx.barrier(cb); Ok(())
    }
    /// cos_vew: 1 X += Y + b, 2 O16 = gelu_tanh(Y + b), 3 O16 = gelu_erf(Y + b), 4 X = Y + b
    fn ew(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, mode: u32, n: u32, d: u32, x: &Buffer, ldx: u32, ldy: u32, ldo: u32, boff: u32) -> Result<()> { self.ewo(ctx, cb, b, mode, n, d, x, ldx, ldy, &b.h16, ldo, boff) }
    fn ewo(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, mode: u32, n: u32, d: u32, x: &Buffer, ldx: u32, ldy: u32, o16: &Buffer, ldo: u32, boff: u32) -> Result<()> {
        self.run(ctx, cb, &self.ew, &[x, &b.y, o16, &self.p, &b.pin], &[mode, n, d, ldx, ldy, ldo, boff, 0], (n * d).div_ceil(256))
    }
    fn lnorm(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, x: &Buffer, d: u32, n: u32, ln: Ln) -> Result<()> {
        self.run(ctx, cb, &self.ln, &[x, &b.a16, &b.y, &self.p], &[d, d, d, ln.w, ln.b, EPS.to_bits(), 1], n)
    }
    fn layer(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, l: &Layer, n: u32) -> Result<()> {
        self.lnorm(ctx, cb, b, &b.x, D, n, l.n1)?;
        self.gemm(ctx, cb, l.qkv.w, &b.a16, D, n, &b.y, 3 * D)?;
        self.run(ctx, cb, &self.pad, &[&b.y, &b.q, &b.k, &b.v, &self.p, &b.rope], &[0, n, NH, HD, l.qkv.b], (n * NH * 128).div_ceil(256))?;
        let fk = &self.kn.k["flash"]; let nt = n.div_ceil(128); let mut t0 = 0;
        while t0 < nt { let t1 = (t0 + 8).min(nt); let ds = self.pf.ds.get(ctx, fk, &[&b.q, &b.k, &b.v, &b.o])?;
            ctx.dispatch(cb, fk, ds, &push_u32(&[n, n, NH, NH * 128, NH * 128, NH * 128, NH * 128, (1.0f32 / (HD as f32).sqrt()).to_bits(), 1, t0]), (t1 - t0) * NH); t0 = t1; }
        ctx.barrier(cb);
        self.run(ctx, cb, &self.pad, &[&b.y, &b.o, &b.a16, &b.v, &self.p, &b.rope], &[1, n, NH, HD, 0], (n * D).div_ceil(256))?;
        self.gemm(ctx, cb, l.proj.w, &b.a16, D, n, &b.y, D)?;
        self.ew(ctx, cb, b, 1, n, D, &b.x, D, D, 0, l.proj.b)?;
        self.lnorm(ctx, cb, b, &b.x, D, n, l.n2)?;
        self.gemm(ctx, cb, l.fc1.w, &b.a16, D, n, &b.y, FFP)?;
        self.ew(ctx, cb, b, 2, n, FF, &b.x, 0, FFP, FFP, l.fc1.b)?;
        self.gemm(ctx, cb, l.fc2.w, &b.h16, FFP, n, &b.y, D)?;
        self.ew(ctx, cb, b, 1, n, D, &b.x, D, D, 0, l.fc2.b)
    }
    /// merger: (post-shuffle ? LN(4608) on the 2x2 concat : LN(1152) per patch) -> fc1 -> erf GELU -> fc2 -> out [n/4][4096]
    fn merge(&self, ctx: &Ctx, cb: vk::CommandBuffer, b: &Bufs, m: &Merger, post: bool, n: u32, out: &Buffer) -> Result<()> {
        if post { self.lnorm(ctx, cb, b, &b.x, DM, n / 4, m.norm)? } else { self.lnorm(ctx, cb, b, &b.x, D, n, m.norm)? }
        self.gemm(ctx, cb, m.fc1.w, &b.a16, DM, n / 4, &b.y, DM)?;
        self.ewo(ctx, cb, b, 3, n / 4, DM, &b.x, 0, DM, &b.a16, DM, m.fc1.b)?; // into a16: h16's fc2 K-pad columns must stay zero
        self.gemm(ctx, cb, m.fc2.w, &b.a16, DM, n / 4, &b.y, DO)?;
        self.ew(ctx, cb, b, 4, n / 4, DO, out, DO, DO, 0, m.fc2.b)
    }
    /// Encode one image: `patches` [n][1536] in merge-block order (HF pixel_values layout), patch grid gh x gw (even).
    pub fn encode(&self, ctx: &Ctx, patches: &[f32], gh: usize, gw: usize) -> Result<VitOut> {
        let n = gh * gw; if gh % 2 != 0 || gw % 2 != 0 || patches.len() != n * KP as usize { bail!("vit: bad grid {gh}x{gw} / {} values", patches.len()) }
        self.bufs(ctx, n as u32)?; let bb = self.bufs.borrow(); let b = &bb.as_ref().unwrap().1; let nu = n as u32;
        qimage::upload_f16(&b.pin, patches);
        // learned position table, bilinear align_corners resample (4 taps, clamped), straight into x; axial rope angles
        let x = b.x.as_f32_mut(); let rope = b.rope.as_f32_mut(); let du = D as usize;
        let inv: Vec<f64> = (0..18).map(|j| 10000f64.powf(-(2.0 * j as f64) / 36.0)).collect();
        let tap = |i: usize, len: usize| -> (usize, usize, f64) { let s = if len > 1 { i as f64 * (SIDE - 1) as f64 / (len - 1) as f64 } else { 0.0 };
            let f = s.floor() as usize; let c = (f + 1).min(SIDE - 1); (f, c, s - f as f64) };
        for i in 0..n { let (r, c) = rc(i, gw); let (r0, r1, fr) = tap(r, gh); let (c0, c1, fc) = tap(c, gw);
            let wts = [((r0, c0), (1.0 - fr) * (1.0 - fc)), ((r0, c1), (1.0 - fr) * fc), ((r1, c0), fr * (1.0 - fc)), ((r1, c1), fr * fc)];
            for j in 0..du { let mut s = 0f64; for ((a, bq), wt) in wts { s += wt * self.pos[(a * SIDE + bq) * du + j] as f64 } x[i * du + j] = s as f32; }
            for j in 0..36 { let a = if j < 18 { r as f64 * inv[j] } else { c as f64 * inv[j - 18] }; rope[i * 72 + 2 * j] = a.cos() as f32; rope[i * 72 + 2 * j + 1] = a.sin() as f32; } }
        let pwr = WRef { arena: usize::MAX, off: (NL * (D * FFP) as usize * 2) as u64, ty: GgmlType::F16, k: KP, n: D, e: 1, bytes: (D * KP * 2) as u64 };
        let mut err = Ok(());
        let gpu = ctx.run(|cb| { err = (|| -> Result<()> {
            self.gemm(ctx, cb, pwr, &b.pin, KP, nu, &b.y, D)?;
            self.ew(ctx, cb, b, 1, nu, D, &b.x, D, D, 0, self.patch_b)?;
            for (li, l) in self.layers.iter().enumerate() { self.layer(ctx, cb, b, l, nu)?;
                if let Some(k) = DS.iter().position(|&d| d == li) { self.merge(ctx, cb, b, &self.ds[k], true, nu, &b.out[1 + k])?; } }
            self.merge(ctx, cb, b, &self.merger, false, nu, &b.out[0]) })(); })?; err?;
        let g = |i: usize| b.out[i].as_f32()[..n / 4 * DO as usize].to_vec();
        Ok(VitOut { emb: g(0), ds: [g(1), g(2), g(3)], gpu_s: gpu })
    }
}

/// Qwen2-VL smart_resize (factor 32 = patch 16 x merge 2): target (h, w) for an image of h x w.
pub fn smart_resize(h: usize, w: usize, min_px: usize, max_px: usize) -> (usize, usize) {
    let f = 32.0; let (hf, wf) = (h as f64, w as f64);
    let (mut hb, mut wb) = (((hf / f).round() * f).max(f), ((wf / f).round() * f).max(f));
    if hb * wb > max_px as f64 { let beta = (hf * wf / max_px as f64).sqrt(); hb = ((hf / beta / f).floor() * f).max(f); wb = ((wf / beta / f).floor() * f).max(f); }
    else if hb * wb < min_px as f64 { let beta = (min_px as f64 / (hf * wf)).sqrt(); hb = (hf * beta / f).ceil() * f; wb = (wf * beta / f).ceil() * f; }
    (hb as usize, wb as usize)
}
/// RGB8 h x w (multiples of 32) -> normalised patches [n][(c, t, ky, kx)] in merge-block order, frame duplicated for t = 2.
pub fn patchify_image(rgb: &[u8], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (gh, gw) = (h / 16, w / 16); let n = gh * gw; let mut o = vec![0f32; n * KP as usize];
    for i in 0..n { let (r, c) = rc(i, gw); for ch in 0..3 { for t in 0..2 { for ky in 0..16 { for kx in 0..16 {
        let v = rgb[((r * 16 + ky) * w + c * 16 + kx) * 3 + ch] as f32 / 255.0;
        o[i * KP as usize + ((ch * 2 + t) * 16 + ky) * 16 + kx] = (v - 0.5) / 0.5; } } } } }
    (o, gh, gw)
}
