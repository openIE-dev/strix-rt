//! Wan2.2 video VAE decoder (AutoencoderKLWan, z 48, decoder_base_dim 256, is_residual, patch 2) for Cosmos3-Nano.
//! Whole-clip (non-chunked) formulation of diffusers' frame-by-frame decode with feat_cache: every WanCausalConv3d is a
//! causal conv over the full frame sequence (zero front padding of 2 frames), except
//!  - upsample3d: frame 0 bypasses time_conv (the "Rep" cache) and frames 1.. go through time_conv as their own causal
//!    sequence, each giving 2 output frames;
//!  - DupUp3D shortcut on frame 0 keeps only the last of its factor_t frames (first_chunk).
//! Activations are channel-last f32 [frame][pixel][C]; 3x3x3 / 3x1x1 / 3x3 convs are im2col (new causal conv3d im2col in
//! cos_vae.comp, K ordered (kt, ky, kx, c), padded to 64) + prefill coop-matrix GEMM on f16 weights in pixel chunks.
use anyhow::Result;
use half::f16;
use prefill::{GemmOp, Pf};
use qimage::{st, Kern, Rec};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{Buffer, Ctx};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub fn add_kernels(ctx: &Ctx, kn: &mut Kern) -> Result<()> {
    kn.k.insert("cvae", ctx.kernel(spv!("cos_vae"), 5, 48, &[], None)?);
    kn.k.insert("cvattn", ctx.kernel(spv!("cos_vattn"), 3, 16, &[], None)?);
    kn.k.insert("cvattn_t", ctx.kernel(spv!("cos_vattn_t"), 3, 16, &[], None)?);
    kn.k.insert("cconv", ctx.kernel(spv!("cos_conv"), 3, 52, &[], Some(32))?);
    kn.k.insert("cgemv", ctx.kernel(spv!("cos_gemv"), 3, 24, &[], None)?);
    Ok(())
}
#[derive(Clone, Copy)] pub struct Conv { w: WRef, b: u32, cin: u32, cout: u32, kt: u32, ks: u32, kpad: u32 }
pub struct Vae { arena: Vec<Buffer>, params: Buffer, conv: HashMap<String, Conv>, gamma: HashMap<String, u32>, zero: u32, pq: (Vec<f32>, Vec<f32>), pub mean: Vec<f32>, pub std: Vec<f32>, pub load_s: f64 }
const P: &str = "decoder.";

impl Vae {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let st = st::Model::open(ctx, dir, &|_| false)?;
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
        let v = |k: &str| -> Vec<f32> { cfg[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
        let mut names: Vec<String> = st.t.keys().filter(|k| k.starts_with(P) && k.ends_with(".weight")).cloned().collect(); names.sort();
        let mut wv: Vec<f16> = vec![]; let mut pv: Vec<f32> = vec![]; let mut conv = HashMap::new(); let mut gamma = HashMap::new();
        for n in &names { let t = st.get(n)?; if t.shape.len() < 4 { continue }
            let (co, ci) = (t.shape[0], t.shape[1]); let (kt, ks) = if t.shape.len() == 5 { (t.shape[2], t.shape[3]) } else { (1, t.shape[2]) };
            let taps = kt * ks * ks; let k = ci * taps; let kpad = k.div_ceil(64) * 64; let npad = co.div_ceil(256) * 256;
            let w = st.f32(n)?; let base = wv.len(); wv.resize(base + npad * kpad, f16::ZERO);
            for o in 0..co { for c in 0..ci { for tap in 0..taps { wv[base + o * kpad + tap * ci + c] = f16::from_f32(w[(o * ci + c) * taps + tap]) } } }
            let b = pv.len() as u32; pv.extend(st.f32(&n.replace(".weight", ".bias"))?); while pv.len() % 64 != 0 { pv.push(0.0) }
            let key = n.trim_start_matches(P).trim_end_matches(".weight").to_string();
            conv.insert(key, Conv { w: WRef { arena: 0, off: base as u64 * 2, ty: gguf_f16(), k: kpad as u32, n: co as u32, e: 1, bytes: (npad * kpad * 2) as u64 }, b, cin: ci as u32, cout: co as u32, kt: kt as u32, ks: ks as u32, kpad: kpad as u32 }); }
        for n in st.t.keys().filter(|k| k.starts_with(P) && k.ends_with(".gamma")) { gamma.insert(n.trim_start_matches(P).trim_end_matches(".gamma").to_string(), pv.len() as u32); pv.extend(st.f32(n)?); while pv.len() % 64 != 0 { pv.push(0.0) } }
        let zero = pv.len() as u32; pv.extend(std::iter::repeat(0.0).take(4096));
        let a = ctx.buffer(wv.len() as u64 * 2)?; unsafe { std::ptr::copy_nonoverlapping(wv.as_ptr() as *const u8, a.ptr as *mut u8, wv.len() * 2) };
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        let pq = (st.f32("post_quant_conv.weight")?, st.f32("post_quant_conv.bias")?);
        Ok(Vae { arena: vec![a], params, conv, gamma, zero, pq, mean: v("latents_mean"), std: v("latents_std"), load_s: t0.elapsed().as_secs_f64() })
    }

    /// Normalized latents [48][t][h][w] (the transformer's space) -> video [3][F][16h][16w] in [-1, 1], plus GPU seconds.
    /// Optional dumps of the stage outputs (channel-first [C][F][H][W]) for parity checks.
    /// COS_VAE_CHUNK=n decodes latent frame 0 alone and then n latent frames at a time with per-conv causal caches (the last 2
    /// input frames of every 3-tap temporal conv), which is exact; 0 = whole clip. Default: whole clip unless an activation
    /// buffer would pass 3.5 GB, then 2 (1 faults the GPU).
    pub fn decode(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, lat: &[f32], t: u32, h: u32, w: u32, dumps: &mut Vec<(String, Vec<f32>)>) -> Result<(Vec<f32>, f64)> {
        let np0 = (t * h * w) as usize; let hw0 = (h * w) as usize;
        let mut z = vec![0f32; np0 * 48];
        for p in 0..np0 { let (f, q) = (p / hw0, p % hw0); let x: Vec<f32> = (0..48).map(|c| lat[(c * t as usize + f) * hw0 + q] * self.std[c] + self.mean[c]).collect();
            for o in 0..48 { let mut s = self.pq.1[o]; for c in 0..48 { s += self.pq.0[o * 48 + c] * x[c] } z[p * 48 + o] = s } }
        let whole_bytes = (1 + 4 * (t as u64 - 1)) * 64 * hw0 as u64 * 512 * 4;
        let cl: u32 = std::env::var("COS_VAE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(if whole_bytes > (3584u64 << 20) { 2 } else { 0 });
        if cl == 0 || t == 1 { return self.decode_part(ctx, pf, kn, &z, t, h, w, None, false, None, dumps) }
        let cache = std::cell::RefCell::new(HashMap::new());
        let cat = ctx.buffer((4 * cl as u64 + 2) * 64 * hw0 as u64 * 512 * 2)?;
        let maxe = [cl as u64 * hw0 as u64 * 1024 * 2, 2 * cl as u64 * 4 * hw0 as u64 * 1024 * 2, 4 * cl as u64 * 16 * hw0 as u64 * 1024, 4 * cl as u64 * 64 * hw0 as u64 * 512].into_iter().max().unwrap() + 4096;
        let chunk_a: u64 = 192 << 20; let (f32b, f16b) = (|n: u64| ctx.buffer(n * 4), |n: u64| ctx.buffer(n * 2));
        let pool = DecBufs { xa: f32b(maxe)?, xb: f32b(maxe)?, tb: f32b(maxe)?, ub: f32b(maxe)?, act: f16b(maxe)?, abuf: f16b(chunk_a / 2 + 256 * 27648)?, ybuf: f32b(chunk_a / 2 + 256 * 2048)? };
        let (ho, wo) = (16 * h as usize, 16 * w as usize); let nf = 1 + 4 * (t as usize - 1);
        let mut vid = vec![0f32; 3 * nf * ho * wo]; let mut gpu = 0.0; let mut f_out = 0usize;
        let mut parts = vec![(0u32, 1u32)]; let mut s0 = 1; while s0 < t { parts.push((s0, cl.min(t - s0))); s0 += cl; }
        for (s, n) in parts { let zs = &z[s as usize * hw0 * 48..(s + n) as usize * hw0 * 48];
            let (v, g) = self.decode_part(ctx, pf, kn, zs, n, h, w, Some((&cache, &cat)), s > 0, Some(&pool), &mut vec![])?; gpu += g;
            let fp = v.len() / (3 * ho * wo);
            for ch in 0..3 { for f in 0..fp { let src = &v[(ch * fp + f) * ho * wo..(ch * fp + f + 1) * ho * wo];
                vid[(ch * nf + f_out + f) * ho * wo..(ch * nf + f_out + f + 1) * ho * wo].copy_from_slice(src); } }
            f_out += fp; }
        for (_, b) in cache.into_inner() { ctx.free_buffer(b) } ctx.free_buffer(cat); for x in [pool.xa, pool.xb, pool.tb, pool.ub, pool.act, pool.abuf, pool.ybuf] { ctx.free_buffer(x) }
        Ok((vid, gpu))
    }
    fn decode_part(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, z: &[f32], t: u32, h: u32, w: u32, cache: Option<(&std::cell::RefCell<HashMap<String, Buffer>>, &Buffer)>, later: bool, pool: Option<&DecBufs>, dumps: &mut Vec<(String, Vec<f32>)>) -> Result<(Vec<f32>, f64)> {
        let np0 = (t * h * w) as usize; let hw0 = (h * w) as usize;
        let nf = if later { 4 * t as u64 } else { 1 + 4 * (t as u64 - 1) }; let (hf, wf) = (8 * h as u64, 8 * w as u64);
        let maxe = [np0 as u64 * 1024 * 2, (if later { 2 * t as u64 } else { 1 + 2 * (t as u64 - 1) }) * 4 * hw0 as u64 * 1024 * 2, nf * 16 * hw0 as u64 * 1024, nf * hf * wf * 512].into_iter().max().unwrap() + 4096;
        let f32b = |n: u64| ctx.buffer(n * 4); let f16b = |n: u64| ctx.buffer(n * 2);
        let chunk_a: u64 = 192 << 20;
        // temporal chunks reuse one buffer set (sized by the caller): freeing and re-allocating per part could hand the
        // descriptor-set cache a recycled VkBuffer handle
        let own = if pool.is_none() { Some(DecBufs { xa: f32b(maxe)?, xb: f32b(maxe)?, tb: f32b(maxe)?, ub: f32b(maxe)?, act: f16b(maxe)?, abuf: f16b(chunk_a / 2 + 256 * 27648)?, ybuf: f32b(chunk_a / 2 + 256 * 2048)? }) } else { None };
        let bf = pool.unwrap_or_else(|| own.as_ref().unwrap());
        qimage::upload_f16(&bf.act, z);
        let d = Dec { v: self, abuf: &bf.abuf, ybuf: &bf.ybuf, chunk_a, cache, chunk: later };
        let mut gpu = 0.0;
        let trace = std::env::var("COS_VAE_TRACE").is_ok(); let mut nsub = 0;
        let mut sub = |f: &dyn Fn(&Rec) -> Result<()>| -> Result<()> { nsub += 1; let mut err = Ok(()); let g = ctx.run(|cb| { let r = Rec { ctx, pf, kn, cb }; err = f(&r); })?; gpu += g;
            if trace { eprintln!("vae part t{t} later {later} sub {nsub}: {:.1} ms", g * 1e3); } err };
        let dump = std::env::var("COS_VAE_DUMP").is_ok();
        let grab = |b: &Buffer, c: u32, f: u32, hh: u32, ww: u32| -> Vec<f32> { let s = b.as_f32(); let n = (f * hh * ww) as usize; let mut o = vec![0f32; n * c as usize];
            for p in 0..n { for k in 0..c as usize { o[(k * (f as usize)) * (hh * ww) as usize + (p / (hh * ww) as usize) * (hh * ww) as usize + p % (hh * ww) as usize] = s[p * c as usize + k] } } o };
        let (mut hh, mut ww, mut nfr) = (h, w, t);
        sub(&|r| d.conv(r, "conv_in", &bf.act, 0, &bf.xa, nfr, hh, ww, false))?;
        sub(&|r| d.resnet(r, "mid_block.resnets.0", &bf.xa, &bf.xa, &bf.tb, &bf.act, nfr, hh, ww))?;
        sub(&|r| d.attention(r, &bf.xa, &bf.tb, &bf.act, nfr, hh, ww))?;
        sub(&|r| d.resnet(r, "mid_block.resnets.1", &bf.xa, &bf.xa, &bf.tb, &bf.act, nfr, hh, ww))?;
        if dump { dumps.push(("mid".into(), grab(&bf.xa, 1024, nfr, hh, ww))); }
        let (mut x, mut y) = (&bf.xa, &bf.xb);
        for i in 0..4 { let pre = format!("up_blocks.{i}"); let cin = self.conv[&format!("{pre}.resnets.0.conv1")].cin; let cout = self.conv[&format!("{pre}.resnets.0.conv1")].cout;
            let (f0, h0, w0) = (nfr, hh, ww);
            sub(&|r| { d.resnet(r, &format!("{pre}.resnets.0"), x, y, &bf.tb, &bf.act, f0, h0, w0)?; Ok(()) })?;
            sub(&|r| d.resnet(r, &format!("{pre}.resnets.1"), y, y, &bf.tb, &bf.act, f0, h0, w0))?;
            sub(&|r| d.resnet(r, &format!("{pre}.resnets.2"), y, y, &bf.tb, &bf.act, f0, h0, w0))?;
            if i < 3 { let temporal = self.conv.contains_key(&format!("{pre}.upsampler.time_conv")); let c = cout; let hw = h0 * w0;
                let f1 = if temporal { if later { 2 * f0 } else { 1 + 2 * (f0 - 1) } } else { f0 };
                sub(&|r| {
                    // ub = frames after (optional) time_conv, at h0 x w0
                    if temporal && later {   // later chunk: every frame goes through time_conv (causal cache carries the history)
                        d.ew(r, 2, f0 * hw * c, 0, 0, 0, 0, 0, 0, 0, 0, 0, y, y, &bf.act, &bf.act)?;
                        d.conv(r, &format!("{pre}.upsampler.time_conv"), &bf.act, 0, &bf.tb, f0, h0, w0, false)?;
                        d.ew(r, 5, f0 * 2 * hw * c, c, w0, h0, 0, 0, 0, 0, 0, 0, &bf.ub, &bf.tb, &bf.act, &bf.act)?;
                    } else if temporal {
                        d.ew(r, 6, hw * c, 0, 0, 0, 0, 0, 0, 0, 0, 0, &bf.ub, y, &bf.act, &bf.act)?;                       // frame 0 passthrough
                        if f0 > 1 { d.ew(r, 2, (f0 - 1) * hw * c, 0, 0, 0, hw * c, 0, 0, 0, 0, 0, y, y, &bf.act, &bf.act)?;   // act = f16(frames 1..)
                            d.conv(r, &format!("{pre}.upsampler.time_conv"), &bf.act, 0, &bf.tb, f0 - 1, h0, w0, false)?;     // tb [f0-1][hw][2c]
                            d.ew(r, 5, (f0 - 1) * 2 * hw * c, c, w0, h0, 0, 0, 0, 0, 0, 0, &bf.ub, &bf.tb, &bf.act, &bf.act)?; }
                    } else { d.ew(r, 6, f0 * hw * c, 0, 0, 0, 0, 0, 0, 0, 0, 0, &bf.ub, y, &bf.act, &bf.act)?; }
                    d.ew(r, 3, f1 * hw * 4 * c, c, w0, h0 * f1, 0, 0, 0, 0, 0, 0, &bf.ub, &bf.tb, &bf.act, &bf.act)?;            // tb = up2(ub)
                    d.ew(r, 2, f1 * hw * 4 * c, 0, 0, 0, 0, 0, 0, 0, 0, 0, &bf.tb, &bf.tb, &bf.act, &bf.act)?;                    // act = f16(tb)
                    d.conv(r, &format!("{pre}.upsampler.resample.1"), &bf.act, 0, y, f1, 2 * h0, 2 * w0, false)?;
                    d.ew(r, 4, f1 * hw * 4 * c, c, w0, h0, 0, 0, if temporal { 2 } else { 1 }, cin, 0, 0, y, x, &bf.act, &bf.act)?;  // y += DupUp(block input)
                    Ok(()) })?;
                nfr = f1; hh *= 2; ww *= 2; }
            if dump && i < 3 { dumps.push((format!("up{i}"), grab(y, cout, nfr, hh, ww))); }
            std::mem::swap(&mut x, &mut y); }
        let (f0, h0, w0) = (nfr, hh, ww);
        sub(&|r| { r.run("vrow", &[x, &bf.act, &self.params], &[0, 256, 256, 256, self.gamma["norm_out"], 1, 0, 0], f0 * h0 * w0)?;
            d.conv(r, "conv_out", &bf.act, 0, &bf.tb, f0, h0, w0, false) })?;
        if dump { dumps.push(("dec".into(), grab(&bf.tb, 12, f0, h0, w0))); }
        // unpatchify (channel = ch*4 + pa*2 + pb -> row 2y + pb, col 2x + pa) and clamp
        let (nfu, ho, wo) = (f0 as usize, 2 * h0 as usize, 2 * w0 as usize); let o = bf.tb.as_f32(); let hw = (h0 * w0) as usize;
        let mut vid = vec![0f32; 3 * nfu * ho * wo];
        for f in 0..nfu { for yy in 0..h0 as usize { for xx in 0..w0 as usize { let p = f * hw + yy * w0 as usize + xx; for ch in 0..3 { for pa in 0..2 { for pb in 0..2 {
            vid[((ch * nfu + f) * ho + 2 * yy + pb) * wo + 2 * xx + pa] = o[p * 12 + ch * 4 + pa * 2 + pb].clamp(-1.0, 1.0); } } } } } }
        let _ = self.zero;
        if let Some(b) = own { for x in [b.xa, b.xb, b.tb, b.ub, b.act, b.abuf, b.ybuf] { ctx.free_buffer(x) } }
        Ok((vid, gpu))
    }
}
pub struct DecBufs { xa: Buffer, xb: Buffer, tb: Buffer, ub: Buffer, act: Buffer, abuf: Buffer, ybuf: Buffer }
fn gguf_f16() -> gguf::GgmlType { gguf::GgmlType::F16 }

struct Dec<'a> { v: &'a Vae, abuf: &'a Buffer, ybuf: &'a Buffer, chunk_a: u64, cache: Option<(&'a std::cell::RefCell<HashMap<String, Buffer>>, &'a Buffer)>, chunk: bool }
impl<'a> Dec<'a> {
    fn ew(&self, r: &Rec, mode: u32, n: u32, c: u32, wd: u32, hd: u32, p0: u32, b_off: u32, flags: u32, c2: u32, kpad: u32, aoff: u32, x: &Buffer, y: &Buffer, a: &Buffer, b: &Buffer) -> Result<()> {
        let _ = b_off; r.run("cvae", &[x, y, a, b, &self.v.params], &[mode, n, c, wd, hd, p0, b_off, flags, c2, kpad, aoff, self.chunk as u32], n.div_ceil(256).min(65535))
    }
    fn gemm(&self, r: &Rec, w: WRef, a: &Buffer, lda: u32, a_row0: u32, m: u32, y: &Buffer, ldy: u32) -> Result<()> {
        r.pf.gemm(r.ctx, r.cb, &self.v.arena[0], &self.v.params, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0, m, y, ldy, y_col: 0, acc: false })?; r.bar(); Ok(())
    }
    /// Implicit-GEMM conv chunk (no im2col): ybuf[m][cout] = conv rows p0..p0+m of `src` (f16 frame stack, element offset aoff).
    fn conv_impl_ok(&self, c: &Conv) -> bool { c.cin % 32 == 0 && c.cout % 128 == 0 && !(c.kt == 1 && c.ks == 1) && std::env::var("COS_IMPL").map_or(true, |v| v != "0") }
    fn conv_impl(&self, r: &Rec, c: &Conv, src: &Buffer, aoff: u32, p0: u32, m: u32, hh: u32, ww: u32) -> Result<()> {
        r.run("cconv", &[src, &self.v.arena[0], self.ybuf], &[m, p0, 0, hh * ww, ww, hh, c.cin, c.kt | c.ks << 2, aoff, c.kpad, c.cout, (c.w.off / 16) as u32, c.cout], m.div_ceil(128) * (c.cout / 128))
    }
    /// dst[p][cout] (= or +=) causal conv(src f16 [aoff + p][cin]) + bias over nf frames of hh x ww, in pixel chunks.
    fn conv(&self, r: &Rec, name: &str, src: &Buffer, aoff: u32, dst: &Buffer, nf: u32, hh: u32, ww: u32, acc: bool) -> Result<()> {
        let c = self.v.conv[name]; let np = nf * hh * ww;
        let ny = c.cout.div_ceil(64) * 64; let w = WRef { n: ny, ..c.w };
        let cp = ((self.chunk_a / (c.kpad as u64 * 2)).min(self.chunk_a / 2 / ny as u64) as u32 / 256 * 256).max(256);
        let direct = c.kt == 1 && c.ks == 1 && c.kpad == c.cin;
        if let (Some((cache, cat)), 3) = (self.cache, c.kt) {   // temporal chunk: [2 cached input frames | this chunk] in cat
            let hw = hh * ww; let mut cm = cache.borrow_mut();
            let cb = cm.entry(name.to_string()).or_insert_with(|| { let b = r.ctx.buffer((2 * hw * c.cin) as u64 * 2).unwrap(); unsafe { std::ptr::write_bytes(b.ptr, 0, b.size as usize) }; b });
            self.ew(r, 7, 2 * hw * c.cin, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.ybuf, self.ybuf, cb, cat)?;
            self.ew(r, 7, np * c.cin, 0, 0, 0, 2 * hw * c.cin, 0, 0, 0, 0, aoff * c.cin, self.ybuf, self.ybuf, src, cat)?;
            let mut p0 = 2 * hw; while p0 < np + 2 * hw { let m = cp.min(np + 2 * hw - p0);
                if self.conv_impl_ok(&c) { self.conv_impl(r, &c, cat, 0, p0, m, hh, ww)?; } else { self.ew(r, 0, m * c.kpad, c.cin, ww, hh, p0, 0, c.kt | c.ks << 2, 0, c.kpad, 0, self.ybuf, self.ybuf, cat, self.abuf)?; self.gemm(r, w, self.abuf, c.kpad, 0, m, self.ybuf, ny)?; }
                self.ew(r, 1, m * c.cout, c.cout, 0, 0, p0 - 2 * hw, c.b, acc as u32, ny, 0, 0, dst, self.ybuf, src, self.abuf)?;
                p0 += m; }
            self.ew(r, 7, 2 * hw * c.cin, 0, 0, 0, 0, 0, 0, 0, 0, np * c.cin, self.ybuf, self.ybuf, cat, cb)?;
            return Ok(());
        }
        let mut p0 = 0; while p0 < np { let m = cp.min(np - p0);
            if !direct && self.conv_impl_ok(&c) { self.conv_impl(r, &c, src, aoff, p0, m, hh, ww)?; }
            else if !direct { self.ew(r, 0, m * c.kpad, c.cin, ww, hh, p0, 0, c.kt | c.ks << 2, 0, c.kpad, aoff, self.ybuf, self.ybuf, src, self.abuf)?; self.gemm(r, w, self.abuf, c.kpad, 0, m, self.ybuf, ny)?; }
            else { self.gemm(r, w, src, c.kpad, aoff + p0, m, self.ybuf, ny)?; }
            self.ew(r, 1, m * c.cout, c.cout, 0, 0, p0, c.b, acc as u32, ny, 0, 0, dst, self.ybuf, src, self.abuf)?;
            p0 += m; }
        Ok(())
    }
    fn norm(&self, r: &Rec, x: &Buffer, o: &Buffer, rows: u32, c: u32, g: &str, silu: bool) -> Result<()> {
        r.run("vrow", &[x, o, &self.v.params], &[0, c, c, c, self.v.gamma[g], silu as u32, 0, 0], rows)
    }
    fn resnet(&self, r: &Rec, pre: &str, src: &Buffer, dst: &Buffer, t: &Buffer, act: &Buffer, nf: u32, hh: u32, ww: u32) -> Result<()> {
        let c1 = self.v.conv[&format!("{pre}.conv1")]; let np = nf * hh * ww; let z = self.v.zero;
        self.norm(r, src, act, np, c1.cin, &format!("{pre}.norm1"), true)?;
        self.conv(r, &format!("{pre}.conv1"), act, 0, t, nf, hh, ww, false)?;
        self.norm(r, t, act, np, c1.cout, &format!("{pre}.norm2"), true)?;
        if c1.cin != c1.cout {
            self.conv(r, &format!("{pre}.conv2"), act, 0, t, nf, hh, ww, false)?;
            self.ew(r, 2, np * c1.cin, 0, 0, 0, 0, 0, 0, 0, 0, 0, src, src, act, act)?;
            self.conv(r, &format!("{pre}.conv_shortcut"), act, 0, dst, nf, hh, ww, false)?;
            self.ew(r, 1, np * c1.cout, c1.cout, 0, 0, 0, z, 1, c1.cout, 0, 0, dst, t, act, act)?;
        } else {
            if !std::ptr::eq(src, dst) { self.ew(r, 1, np * c1.cout, c1.cout, 0, 0, 0, z, 0, c1.cout, 0, 0, dst, src, act, act)?; }
            self.conv(r, &format!("{pre}.conv2"), act, 0, dst, nf, hh, ww, true)?;
        }
        Ok(())
    }
    /// Mid-block attention (per frame, single head, C 1024), residual into x.
    fn attention(&self, r: &Rec, x: &Buffer, t: &Buffer, act: &Buffer, nf: u32, hh: u32, ww: u32) -> Result<()> {
        let npf = hh * ww; let np = nf * npf; let c = 1024u32; let pre = "mid_block.attentions.0";
        self.norm(r, x, act, np, c, &format!("{pre}.norm"), false)?;
        let qkv = self.v.conv[&format!("{pre}.to_qkv")];
        self.gemm(r, qkv.w, act, c, 0, np, t, 3 * c)?;
        // Naive kernel keeps all scores in shared memory (<= 2048 px/frame); tiled online-softmax kernel for 720p etc.
        let tiled = npf > 2048 || std::env::var("COS_VATTN_T").map_or(false, |v| v == "1");
        let pc = [npf, c, qkv.b, (1.0f32 / (c as f32).sqrt()).to_bits()];
        if tiled { r.run("cvattn_t", &[t, act, &self.v.params], &pc, nf * npf.div_ceil(4))?; }
        else { r.run("cvattn", &[t, act, &self.v.params], &pc, np)?; }
        self.conv(r, &format!("{pre}.proj"), act, 0, x, nf, hh, ww, true)
    }
}
