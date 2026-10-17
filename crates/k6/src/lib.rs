//! Kandinsky 6.0 Pro-distill (text -> video + audio) on strix-rt.
//!  - DiT: 4+4 text blocks and 60 fused video/audio blocks (d 4096 / 2048, head 128), every linear through prefill's
//!    coop-matrix GEMM straight from the bf16 safetensors pages (qimage::st memory tier, zero copy; the checkpoint is
//!    re-headered once so every tensor is 16-byte aligned), qimage flash attention for all attention (self, text cross,
//!    video<->audio cross), qimage qk kernel (per-head RMSNorm + 3-D / 1-D RoPE), AdaLN modulation as m=1 GEMMs;
//!  - one GPU submission per block (K6_BPS), so no single long submission;
//!  - PiFlow DX-policy scheduler on the CPU (10 grid predictions per channel, 128 substeps per unit raw time).
pub mod npy;
pub mod sched;
pub mod vae;
pub mod audio;
pub mod te;
use anyhow::Result;
use ash::vk;
use gguf::GgmlType;
use half::{bf16, f16};
use prefill::{GemmOp, Pf};
use qimage::{st, Kern};
use rtcore::WRef;
use vkrt::{push_u32, Buffer, Ctx};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const D: u32 = 4096; pub const DA: u32 = 2048; pub const FF: u32 = 16384; pub const FFA: u32 = 7168; pub const TD: usize = 1024;
pub const NBLK: usize = 60; pub const NTB: usize = 4; pub const NG: usize = 10; pub const CV: usize = 16; pub const CA: usize = 40;
const FQ: u32 = 128;
fn envu(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn qk_eps() -> f32 { std::env::var("K6_QKEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0078125) }   // torch.finfo(bf16).eps (nn.RMSNorm eps=None)
const LN_EPS: f32 = 1e-5;
pub fn rup(x: u32, m: u32) -> u32 { x.div_ceil(m) * m }

pub fn kern(ctx: &Ctx) -> Result<Kern> {
    let mut kn = Kern::new(ctx)?;
    kn.k.insert("k6norm", ctx.kernel(spv!("k6_norm"), 4, 32, &[], None)?);
    kn.k.insert("k6ew", ctx.kernel(spv!("k6_ew"), 4, 32, &[], None)?);
    kn.k.insert("k6vae", ctx.kernel(spv!("k6_vae"), 6, 48, &[], None)?);
    kn.k.insert("k6red", ctx.kernel(spv!("k6_red"), 4, 40, &[], None)?);
    kn.k.insert("k6a1d", ctx.kernel(spv!("k6_a1d"), 5, 48, &[], None)?);
    Ok(kn)
}

#[derive(Clone, Copy)] pub struct Lin { pub w: WRef, pub b: u32 }
#[derive(Clone, Copy)] pub struct Attn { pub q: Lin, pub k: Lin, pub v: Lin, pub o: Lin, pub qn: u32, pub kn: u32 }
#[derive(Clone, Copy)] pub struct Dec { pub sa: Attn, pub ca: Attn, pub f1: WRef, pub f2: WRef, pub m: Lin, pub mo: u32 }
pub struct Blk { pub v: Dec, pub a: Dec, pub va: Attn, pub av: Attn, pub vam: Lin, pub avm: Lin, pub vamo: u32, pub avmo: u32 }
pub struct TBlk { pub at: Attn, pub f1: WRef, pub f2: WRef, pub m: Lin, pub mo: u32 }
pub struct CpuLin { pub w: Vec<f32>, pub b: Vec<f32>, pub k: usize, pub n: usize }
impl CpuLin {
    fn load(s: &st::Model, p: &str) -> Result<Self> { let t = s.get(&format!("{p}.weight"))?; let (n, k) = (t.shape[0], t.shape[1]);
        Ok(CpuLin { w: s.f32(&format!("{p}.weight"))?, b: s.f32(&format!("{p}.bias"))?, k, n }) }
    pub fn fwd(&self, x: &[f32]) -> Vec<f32> { let rows = x.len() / self.k; let mut y = vec![0f32; rows * self.n];
        std::thread::scope(|sc| { let nt = 16usize; let chunk = self.n.div_ceil(nt);
            let mut parts: Vec<&mut [f32]> = vec![]; let _ = &mut parts;
            let hs: Vec<_> = (0..nt).map(|ti| { let (j0, j1) = (ti * chunk, ((ti + 1) * chunk).min(self.n)); let w = &self.w; let b = &self.b; let k = self.k;
                sc.spawn(move || { let mut o = vec![0f32; rows * (j1.saturating_sub(j0))];
                    for r in 0..rows { let xr = &x[r * k..(r + 1) * k]; for j in j0..j1 { let wr = &w[j * k..(j + 1) * k]; let mut s = 0f64; for i in 0..k { s += (xr[i] * wr[i]) as f64 } o[r * (j1 - j0) + j - j0] = s as f32 + b[j]; } }
                    (j0, j1, o) }) }).collect();
            for h in hs { let (j0, j1, o) = h.join().unwrap(); for r in 0..rows { y[r * self.n + j0..r * self.n + j1].copy_from_slice(&o[r * (j1 - j0)..(r + 1) * (j1 - j0)]); } } });
        y }
}
pub fn layernorm(x: &mut [f32], d: usize, g: Option<(&[f32], &[f32])>) {
    for r in x.chunks_mut(d) { let m = r.iter().map(|&v| v as f64).sum::<f64>() / d as f64; let var = r.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + LN_EPS as f64).sqrt(); for (i, v) in r.iter_mut().enumerate() { let y = ((*v as f64 - m) * inv) as f32; *v = match g { Some((w, b)) => y * w[i] + b[i], None => y } } }
}
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
pub fn bf(x: f32) -> f32 { bf16::from_f32(x).to_f32() }

pub struct Model {
    pub st: st::Model, pub p: Buffer, pub blocks: Vec<Blk>, pub tv: Vec<TBlk>, pub ta: Vec<TBlk>,
    pub outm: Lin, pub out: Lin, pub aoutm: Lin, pub aout: Lin, pub outmo: u32, pub aoutmo: u32, pub mlen: u32,
    pub vin: Buffer, pub vin_w: WRef, pub vin_b: u32, pub ain: CpuLin, pub ain_n: (Vec<f32>, Vec<f32>),
    pub tin: [CpuLin; 2], pub tin_n: [(Vec<f32>, Vec<f32>); 2], pub pool: [CpuLin; 2], pub pool_n: [(Vec<f32>, Vec<f32>); 2],
    pub time1: [CpuLin; 2], pub time2: [CpuLin; 2],
}
struct PB { v: Vec<f32> }
impl PB { fn put(&mut self, x: Vec<f32>) -> u32 { let o = self.v.len() as u32; self.v.extend(x); while self.v.len() % 64 != 0 { self.v.push(0.0) } o } }

impl Model {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|k: &str| k.ends_with(".weight") && !k.contains("_norm.") && !k.contains("embeddings") && !k.contains(".norm."))?;
        let mut pb = PB { v: vec![] }; let mut mlen = 0u32;
        let lin = |pb: &mut PB, p: &str| -> Result<Lin> { Ok(Lin { w: st.wref(&format!("{p}.weight"))?, b: pb.put(st.f32(&format!("{p}.bias"))?) }) };
        let attn = |pb: &mut PB, p: &str| -> Result<Attn> { Ok(Attn { q: lin(pb, &format!("{p}.to_query"))?, k: lin(pb, &format!("{p}.to_key"))?, v: lin(pb, &format!("{p}.to_value"))?,
            o: lin(pb, &format!("{p}.out_layer"))?, qn: pb.put(st.f32(&format!("{p}.query_norm.weight"))?), kn: pb.put(st.f32(&format!("{p}.key_norm.weight"))?) }) };
        let mut alloc = |n: u32| { let o = mlen; mlen += rup(n, 64); o };
        let mut blocks = vec![];
        for b in 0..NBLK { let p = format!("visual_transformer_blocks.{b}");
            let dec = |pb: &mut PB, q: &str, alloc: &mut dyn FnMut(u32) -> u32| -> Result<Dec> { let m = lin(pb, &format!("{p}.{q}.visual_modulation.out_layer"))?; let mo = alloc(m.w.n);
                Ok(Dec { sa: attn(pb, &format!("{p}.{q}.self_attention"))?, ca: attn(pb, &format!("{p}.{q}.cross_attention"))?,
                    f1: st.wref(&format!("{p}.{q}.feed_forward.net.0.proj.weight"))?, f2: st.wref(&format!("{p}.{q}.feed_forward.net.2.weight"))?, m, mo }) };
            let v = dec(&mut pb, "video_dec_block", &mut alloc)?; let a = dec(&mut pb, "audio_dec_block", &mut alloc)?;
            let vam = lin(&mut pb, &format!("{p}.va_modulation.out_layer"))?; let vamo = alloc(vam.w.n);
            let avm = lin(&mut pb, &format!("{p}.av_modulation.out_layer"))?; let avmo = alloc(avm.w.n);
            blocks.push(Blk { v, a, va: attn(&mut pb, &format!("{p}.va_cross_attention"))?, av: attn(&mut pb, &format!("{p}.av_cross_attention"))?, vam, avm, vamo, avmo });
        }
        let tb = |pb: &mut PB, pre: &str, alloc: &mut dyn FnMut(u32) -> u32| -> Result<Vec<TBlk>> { (0..NTB).map(|i| { let p = format!("{pre}.{i}");
            let m = lin(pb, &format!("{p}.text_modulation.out_layer"))?; let mo = alloc(m.w.n);
            Ok(TBlk { at: attn(pb, &format!("{p}.attn"))?, f1: st.wref(&format!("{p}.feed_forward.net.0.proj.weight"))?, f2: st.wref(&format!("{p}.feed_forward.net.2.weight"))?, m, mo }) }).collect() };
        let tv = tb(&mut pb, "video_text_transformer_blocks", &mut alloc)?; let ta = tb(&mut pb, "audio_text_transformer_blocks", &mut alloc)?;
        let outm = lin(&mut pb, "out_layer.modulation.out_layer")?; let outmo = alloc(outm.w.n);
        let aoutm = lin(&mut pb, "audio_out_layer.modulation.out_layer")?; let aoutmo = alloc(aoutm.w.n);
        let out = lin(&mut pb, "out_layer.out_layer")?; let aout = lin(&mut pb, "audio_out_layer.out_layer")?;
        // visual patch embedding: F32 [4096][132] -> f16 [4096][192] (K padded to the GEMM's 64 multiple)
        let vw = st.f32("visual_embeddings.in_layer.weight")?; let vin = ctx.buffer(4096 * 192 * 2)?;
        { let o = unsafe { std::slice::from_raw_parts_mut(vin.ptr as *mut f16, 4096 * 192) }; for j in 0..4096 { for i in 0..192 { o[j * 192 + i] = f16::from_f32(if i < 132 { vw[j * 132 + i] } else { 0.0 }) } } }
        let vin_w = WRef { arena: usize::MAX, off: 0, ty: GgmlType::F16, k: 192, n: 4096, e: 1, bytes: 4096 * 192 * 2 };
        let vin_b = pb.put(st.f32("visual_embeddings.in_layer.bias")?);
        let nrm = |p: &str| -> Result<(Vec<f32>, Vec<f32>)> { Ok((st.f32(&format!("{p}.norm.weight"))?, st.f32(&format!("{p}.norm.bias"))?)) };
        let ain = CpuLin::load(&st, "audio_embeddings.in_layer")?; let ain_n = nrm("audio_embeddings")?;
        let tin = [CpuLin::load(&st, "video_text_embeddings.in_layer")?, CpuLin::load(&st, "audio_text_embeddings.in_layer")?];
        let tin_n = [nrm("video_text_embeddings")?, nrm("audio_text_embeddings")?];
        let pool = [CpuLin::load(&st, "video_pooled_text_embeddings.in_layer")?, CpuLin::load(&st, "audio_pooled_text_embeddings.in_layer")?];
        let pool_n = [nrm("video_pooled_text_embeddings")?, nrm("audio_pooled_text_embeddings")?];
        let time1 = [CpuLin::load(&st, "video_time_embeddings.timestep_embedder.linear_1")?, CpuLin::load(&st, "audio_time_embeddings.timestep_embedder.linear_1")?];
        let time2 = [CpuLin::load(&st, "video_time_embeddings.timestep_embedder.linear_2")?, CpuLin::load(&st, "audio_time_embeddings.timestep_embedder.linear_2")?];
        let p = ctx.buffer(pb.v.len() as u64 * 4)?; p.as_f32_mut()[..pb.v.len()].copy_from_slice(&pb.v);
        Ok(Model { st, p, blocks, tv, ta, outm, out, aoutm, aout, outmo, aoutmo, mlen, vin, vin_w, vin_b, ain, ain_n, tin, tin_n, pool, pool_n, time1, time2 })
    }
    /// temb (video, audio) = TimestepEmbedding(sinusoid(t)) + pooled-text embedding, f32 on the CPU.
    pub fn temb(&self, t: f32, pooled: &[f32]) -> [Vec<f32>; 2] {
        let t = if std::env::var("K6_TBF").is_ok() { bf(t) } else { t };
        let mut out = [vec![], vec![]];
        for m in 0..2 { let dim = if m == 0 { D as usize } else { DA as usize }; let half = dim / 2;
            let mut e = vec![0f32; dim];
            for i in 0..half { let f = (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32; let a = t * f; e[i] = a.cos(); e[half + i] = a.sin(); }   // flip_sin_to_cos
            let h: Vec<f32> = self.time1[m].fwd(&e).into_iter().map(silu).collect(); let te = self.time2[m].fwd(&h);
            let mut pe = self.pool[m].fwd(pooled); layernorm(&mut pe, TD, Some((&self.pool_n[m].0, &self.pool_n[m].1)));
            out[m] = te.iter().zip(&pe).map(|(a, b)| a + b).collect(); }
        out
    }
    /// Text embeddings (Linear + affine LayerNorm) for the video (d 4096) and audio (d 2048) text branches.
    pub fn text_embed(&self, emb: &[f32]) -> [Vec<f32>; 2] {
        let mut o = [self.tin[0].fwd(emb), self.tin[1].fwd(emb)];
        layernorm(&mut o[0], D as usize, Some((&self.tin_n[0].0, &self.tin_n[0].1))); layernorm(&mut o[1], DA as usize, Some((&self.tin_n[1].0, &self.tin_n[1].1))); o
    }
}

/// Rope table rows (cos, sin) x 64 pairs for qimage's qk kernel (style 0, adjacent pairs).
fn rope1d(n: usize, scaling: f32) -> Vec<f32> { let mut r = vec![0f32; n * 128];
    for p in 0..n { for i in 0..64 { let f = (-(10000f64.ln()) * i as f64 / 64.0).exp() as f32 * scaling; let a = p as f32 * f; r[p * 128 + 2 * i] = a.cos(); r[p * 128 + 2 * i + 1] = a.sin(); } } r }
fn rope3d(t: usize, h: usize, w: usize) -> Vec<f32> { let mut r = vec![0f32; t * h * w * 128];
    let fr = |d: usize| -> Vec<f32> { (0..d / 2).map(|i| (-(10000f64.ln()) * i as f64 / (d / 2) as f64).exp() as f32).collect() };
    let (ft, fh, fw) = (fr(32), fr(48), fr(48));
    for a in 0..t { for b in 0..h { for c in 0..w { let o = ((a * h + b) * w + c) * 128; let mut k = 0;
        for (pos, fs, sc) in [(a, &ft, 1.0f32), (b, &fh, 2.0), (c, &fw, 2.0)] { for f in fs.iter() { let ang = pos as f32 * f / sc; r[o + 2 * k] = ang.cos(); r[o + 2 * k + 1] = ang.sin(); k += 1; } } } } } r }

pub struct State {
    pub n: u32, pub na: u32, pub lt: u32, pub t: u32, pub h: u32, pub w: u32,
    pub xv: Buffer, pub xa: Buffer, pub a16: Buffer, pub y: Buffer, pub yf: Buffer, pub y2: Buffer, pub q: Buffer, pub k: Buffer, pub v: Buffer, pub o: Buffer,
    pub vpre: Buffer, pub apre: Buffer, pub vout: Buffer, pub vfa: Buffer, pub afv: Buffer,
    pub tx: [Buffer; 2], pub tx0: [Buffer; 2], pub t16: [Buffer; 2], pub rv: Buffer, pub ra: Buffer, pub rt: Buffer, pub md: Buffer, pub tm: Buffer, pub pin: Buffer,
    pub ov: Buffer, pub oa: Buffer,
}
impl State {
    /// t, h, w: latent grid (h, w at full latent resolution; tokens are 2x2 patches). na audio tokens, lt text tokens.
    pub fn new(ctx: &Ctx, m: &Model, t: u32, h: u32, w: u32, na: u32, lt: u32, text: &[Vec<f32>; 2]) -> Result<Self> {
        let n = t * (h / 2) * (w / 2); let np = rup(n, 256) + 256; let nap = rup(na, 256) + 256; let ltp = rup(lt, 256) + 256;
        let f32b = |r: u32, c: u32| ctx.buffer(r as u64 * c as u64 * 4); let f16b = |r: u32, c: u32| ctx.buffer(r as u64 * c as u64 * 2);
        let rows = np.max(nap).max(ltp);
        let s = State { n, na, lt, t, h, w,
            xv: f32b(np, D)?, xa: f32b(nap, DA)?, a16: f16b(rows, FF)?, y: f32b(rows, 3 * D)?, yf: f32b(rows, FF)?, y2: f32b(rows, D)?,
            q: f16b(rows, D)?, k: f16b(rows, D)?, v: f16b(rows, D)?, o: f16b(rows, D)?,
            vpre: f16b(np, D)?, apre: f16b(nap, DA)?, vout: f32b(np, D)?, vfa: f32b(np, D)?, afv: f32b(nap, DA)?,
            tx: [f32b(ltp, D)?, f32b(ltp, DA)?], tx0: [f32b(ltp, D)?, f32b(ltp, DA)?], t16: [f16b(ltp, D)?, f16b(ltp, DA)?],
            rv: f32b(np, 128)?, ra: f32b(nap, 128)?, rt: f32b(ltp, 128)?, md: ctx.buffer(m.mlen as u64 * 4 + 4096)?, tm: f16b(256, TD as u32)?, pin: f16b(np, 192)?,
            ov: f32b(np, 640)?, oa: f32b(nap, 400)? };
        let rv = rope3d(t as usize, (h / 2) as usize, (w / 2) as usize); s.rv.as_f32_mut()[..rv.len()].copy_from_slice(&rv);
        let ra = rope1d(na as usize, 0.144); s.ra.as_f32_mut()[..ra.len()].copy_from_slice(&ra);
        let rt = rope1d(lt as usize, 1.0); s.rt.as_f32_mut()[..rt.len()].copy_from_slice(&rt);
        for i in 0..2 { s.tx0[i].as_f32_mut()[..text[i].len()].copy_from_slice(&text[i]); }
        Ok(s)
    }
}

pub struct R<'a> { pub ctx: &'a Ctx, pub pf: &'a Pf, pub kn: &'a Kern, pub cb: vk::CommandBuffer, pub m: &'a Model }
impl<'a> R<'a> {
    fn run(&self, name: &str, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<()> {
        let k = &self.kn.k[name]; let ds = self.pf.ds.get(self.ctx, k, bufs)?; self.ctx.dispatch(self.cb, k, ds, &push_u32(push), groups); self.ctx.barrier(self.cb); Ok(())
    }
    /// y[m][ldy] (cols y_col..) = a[a_row0..][lda] . W^T + bias; barrier after.
    fn lin(&self, l: Lin, a: &Buffer, lda: u32, a_row0: u32, m: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        self.gemm(l.w, a, lda, a_row0, m, y, ldy, y_col)?;
        self.ew(1, m, l.w.n, y, y, ldy, y_col, y, 0, &self.m.p, l.b)
    }
    fn gemm(&self, w: WRef, a: &Buffer, lda: u32, a_row0: u32, m: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        let arena = if w.arena == usize::MAX { &self.m.vin } else { &self.m.st.bufs[w.arena] };
        self.pf.gemm(self.ctx, self.cb, arena, &self.m.p, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0, m, y, ldy, y_col, acc: false })?;
        self.ctx.barrier(self.cb); Ok(())
    }
    fn ew(&self, mode: u32, rows: u32, d: u32, x: &Buffer, y: &Buffer, ldy: u32, ycol: u32, a: &Buffer, lda: u32, md: &Buffer, g_off: u32) -> Result<()> {
        self.run("k6ew", &[x, y, a, md], &[mode, rows * d, d, ldy, ycol, lda, g_off, 0], (rows * d).div_ceil(256))
    }
    /// AdaLN: out16 = f16(LN(x) * (1 + md[s]) + md[h])  (mode 2: followed by a plain LN)
    fn adaln(&self, mode: u32, x: &Buffer, d: u32, rows: u32, out: &Buffer, s_off: u32, h_off: u32, md: &Buffer) -> Result<()> {
        self.run("k6norm", &[x, out, x, md], &[d, d, d, s_off, h_off, mode, LN_EPS.to_bits(), 0], rows)
    }
    /// qk: f16 copy with optional per-head RMSNorm (weight w_off) and rope (table).
    fn qk(&self, y: &Buffer, ldy: u32, col0: u32, nh: u32, rows: u32, dst: &Buffer, w_off: Option<u32>, rope: Option<&Buffer>) -> Result<()> {
        let flags = w_off.map(|_| 1).unwrap_or(0) | rope.map(|_| 2).unwrap_or(0);
        self.run("qk", &[y, dst, &self.m.p, rope.unwrap_or(y)], &[nh, ldy, col0, w_off.unwrap_or(0), qk_eps().to_bits(), nh * 128, 0, 0, flags, 0], rows * nh)
    }
    fn flash(&self, q: &Buffer, k: &Buffer, v: &Buffer, o: &Buffer, sq: u32, nk: u32, nh: u32) -> Result<()> {
        let ld = nh * 128; let nt = sq.div_ceil(FQ); let qt = envu("K6_FLASH_QT", 8) as u32; let mut t0 = 0;
        let kk = &self.kn.k["flash"];
        while t0 < nt { let t1 = (t0 + qt).min(nt); let ds = self.pf.ds.get(self.ctx, kk, &[q, k, v, o])?;
            self.ctx.dispatch(self.cb, kk, ds, &push_u32(&[sq, nk, nh, ld, ld, ld, ld, (1.0f32 / 128f32.sqrt()).to_bits(), 1, t0]), (t1 - t0) * nh); t0 = t1; }
        self.ctx.barrier(self.cb); Ok(())
    }
    /// q from `aq` (f16 [sq][dq]); k, v from `akv` (f16 [nk][dkv]); out (+bias) into `out` f32 [sq][d]. q rope `rq`, k rope `rk`.
    fn attn(&self, s: &State, at: &Attn, aq: &Buffer, dq: u32, sq: u32, akv: &Buffer, dkv: u32, nk: u32, d: u32, rq: Option<&Buffer>, rk: Option<&Buffer>, out: &Buffer) -> Result<()> {
        let nh = d / 128;
        self.lin(at.q, aq, dq, 0, sq, &s.y, 3 * D, 0)?;
        self.qk(&s.y, 3 * D, 0, nh, sq, &s.q, Some(at.qn), rq)?;
        self.lin(at.k, akv, dkv, 0, nk, &s.y, 3 * D, D)?;
        self.lin(at.v, akv, dkv, 0, nk, &s.y, 3 * D, 2 * D)?;
        self.qk(&s.y, 3 * D, D, nh, nk, &s.k, Some(at.kn), rk)?;
        self.qk(&s.y, 3 * D, 2 * D, nh, nk, &s.v, None, None)?;
        self.flash(&s.q, &s.k, &s.v, &s.o, sq, nk, nh)?;
        self.lin(at.o, &s.o, d, 0, sq, out, d, 0)
    }
    fn ff(&self, s: &State, x: &Buffer, d: u32, ffd: u32, rows: u32, f1: WRef, f2: WRef, mo: u32) -> Result<()> {
        self.adaln(0, x, d, rows, &s.a16, mo + d, mo, &s.md)?;
        self.gemm(f1, &s.a16, d, 0, rows, &s.yf, ffd, 0)?;
        self.ew(2, rows, ffd, &s.yf, &s.yf, ffd, 0, &s.a16, ffd, &s.md, 0)?;
        self.gemm(f2, &s.a16, ffd, 0, rows, &s.y2, d, 0)?;
        self.ew(0, rows, d, x, &s.y2, d, 0, &s.a16, 0, &s.md, mo + 2 * d)
    }
    fn text_block(&self, s: &State, b: &TBlk, mi: usize) -> Result<()> {
        let (d, ffd) = if mi == 0 { (D, FF) } else { (DA, FFA) }; let x = &s.tx[mi]; let lt = s.lt;
        self.adaln(0, x, d, lt, &s.a16, b.mo + d, b.mo, &s.md)?;
        self.attn(s, &b.at, &s.a16, d, lt, &s.a16, d, lt, d, Some(&s.rt), Some(&s.rt), &s.y2)?;
        self.ew(0, lt, d, x, &s.y2, d, 0, &s.a16, 0, &s.md, b.mo + 2 * d)?;
        self.ff(s, x, d, ffd, lt, b.f1, b.f2, b.mo + 3 * d)
    }
    pub fn block(&self, s: &State, b: &Blk) -> Result<()> {
        let (n, na, lt) = (s.n, s.na, s.lt); let md = &s.md; let (vm, am) = (b.v.mo, b.a.mo);
        // video self-attention
        self.adaln(0, &s.xv, D, n, &s.a16, vm + D, vm, md)?;
        self.attn(s, &b.v.sa, &s.a16, D, n, &s.a16, D, n, D, Some(&s.rv), Some(&s.rv), &s.y2)?;
        self.ew(0, n, D, &s.xv, &s.y2, D, 0, &s.a16, 0, md, vm + 2 * D)?;
        // video text cross-attention (kept in vout, applied after the audio path like the reference)
        self.adaln(0, &s.xv, D, n, &s.vpre, vm + 4 * D, vm + 3 * D, md)?;
        self.attn(s, &b.v.ca, &s.vpre, D, n, &s.t16[0], D, lt, D, None, None, &s.vout)?;
        // audio self-attention + text cross-attention
        self.adaln(0, &s.xa, DA, na, &s.a16, am + DA, am, md)?;
        self.attn(s, &b.a.sa, &s.a16, DA, na, &s.a16, DA, na, DA, Some(&s.ra), Some(&s.ra), &s.afv)?;
        self.ew(0, na, DA, &s.xa, &s.afv, DA, 0, &s.a16, 0, md, am + 2 * DA)?;
        self.adaln(0, &s.xa, DA, na, &s.apre, am + 4 * DA, am + 3 * DA, md)?;
        self.attn(s, &b.a.ca, &s.apre, DA, na, &s.t16[1], DA, lt, DA, None, None, &s.afv)?;
        self.ew(0, na, DA, &s.xa, &s.afv, DA, 0, &s.a16, 0, md, am + 5 * DA)?;
        self.ew(0, n, D, &s.xv, &s.vout, D, 0, &s.a16, 0, md, vm + 5 * D)?;
        // video <-> audio cross-attention (cross gates: va_mod = [shift D | scale D | gate DA], av_mod = [shift DA | scale DA | gate D])
        let (vo, ao) = (b.vamo, b.avmo);
        self.adaln(0, &s.xv, D, n, &s.a16, vo + D, vo, md)?;
        self.attn(s, &b.va, &s.a16, D, n, &s.apre, DA, na, D, Some(&s.rv), Some(&s.ra), &s.vfa)?;
        self.adaln(0, &s.xa, DA, na, &s.a16, ao + DA, ao, md)?;
        self.attn(s, &b.av, &s.a16, DA, na, &s.vpre, D, n, DA, Some(&s.ra), Some(&s.rv), &s.afv)?;
        self.ew(0, n, D, &s.xv, &s.vfa, D, 0, &s.a16, 0, md, ao + 2 * DA)?;
        self.ew(0, na, DA, &s.xa, &s.afv, DA, 0, &s.a16, 0, md, vo + 2 * D)?;
        // feed-forwards
        self.ff(s, &s.xv, D, FF, n, b.v.f1, b.v.f2, vm + 6 * D)?;
        self.ff(s, &s.xa, DA, FFA, na, b.a.f1, b.a.f2, am + 6 * DA)
    }
}

/// Visual input [T][H][W][c] -> f16 patch rows [n][192] (feature (ph*2+pw)*c + ch, zero pad).
pub fn patchify(inp: &[f32], t: usize, h: usize, w: usize, c: usize, out: &mut [f16]) {
    let (h2, w2) = (h / 2, w / 2);
    for a in 0..t { for b in 0..h2 { for d in 0..w2 { let tok = (a * h2 + b) * w2 + d; let o = &mut out[tok * 192..(tok + 1) * 192];
        for z in o.iter_mut() { *z = f16::ZERO }
        for ph in 0..2 { for pw in 0..2 { let src = ((a * h + 2 * b + ph) * w + 2 * d + pw) * c; for ch in 0..c { o[(ph * 2 + pw) * c + ch] = f16::from_f32(inp[src + ch]); } } } } } }
}
/// Out-layer rows [n][640] (feature c160*4 + ph*2 + pw) -> [T][H][W][160].
pub fn unpatchify(ov: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (h / 2, w / 2); let mut o = vec![0f32; t * h * w * 160];
    for a in 0..t { for b in 0..h2 { for d in 0..w2 { let tok = (a * h2 + b) * w2 + d;
        for ph in 0..2 { for pw in 0..2 { let dst = ((a * h + 2 * b + ph) * w + 2 * d + pw) * 160; for c in 0..160 { o[dst + c] = ov[tok * 640 + c * 4 + ph * 2 + pw]; } } } } } }
    o
}

pub struct StepOut { pub vis: Vec<f32>, pub aud: Vec<f32>, pub gpu: f64, pub wall: f64 }
/// One denoiser evaluation. vis_in [T][H][W][33], aud_in [na][40], t = sigma * 1000. Returns the PiFlow grids
/// vis [T][H][W][160], aud [na][400]. K6_NB limits the block count (debug); `hook` sees the state after each submission.
pub fn forward(ctx: &Ctx, pf: &Pf, kn: &Kern, m: &Model, s: &State, vis_in: &[f32], aud_in: &[f32], t: f32, pooled: &[f32],
               mut hook: impl FnMut(&str, &State)) -> Result<StepOut> {
    let t0 = std::time::Instant::now();
    let te = m.temb(t, pooled);
    { let p = unsafe { std::slice::from_raw_parts_mut(s.tm.ptr as *mut f16, 2 * TD) }; for i in 0..TD { p[i] = f16::from_f32(silu(te[0][i])); p[TD + i] = f16::from_f32(silu(te[1][i])); } }
    { let p = unsafe { std::slice::from_raw_parts_mut(s.pin.ptr as *mut f16, s.n as usize * 192) }; patchify(vis_in, s.t as usize, s.h as usize, s.w as usize, 33, p); }
    { let mut xa = m.ain.fwd(aud_in); layernorm(&mut xa, DA as usize, Some((&m.ain_n.0, &m.ain_n.1))); s.xa.as_f32_mut()[..xa.len()].copy_from_slice(&xa); }
    for i in 0..2 { let n = s.lt as usize * if i == 0 { D } else { DA } as usize; s.tx[i].as_f32_mut()[..n].copy_from_slice(&s.tx0[i].as_f32()[..n]); }
    let nb = envu("K6_NB", NBLK).min(NBLK); let bps = envu("K6_BPS", 1).max(1);
    let mut gpu = 0.0; let mut err = Ok(());
    // submission 0: all modulations, embeddings, text blocks
    gpu += ctx.run(|cb| { err = (|| -> Result<()> { let r = R { ctx, pf, kn, cb, m };
        let ml = |l: Lin, row: u32, off: u32| r.lin(l, &s.tm, TD as u32, row, 1, &s.md, 0, off);
        for b in &m.blocks { ml(b.v.m, 0, b.v.mo)?; ml(b.a.m, 1, b.a.mo)?; ml(b.vam, 0, b.vamo)?; ml(b.avm, 1, b.avmo)?; }
        for b in &m.tv { ml(b.m, 0, b.mo)?; } for b in &m.ta { ml(b.m, 1, b.mo)?; }
        ml(m.outm, 0, m.outmo)?; ml(m.aoutm, 1, m.aoutmo)?;
        r.gemm(m.vin_w, &s.pin, 192, 0, s.n, &s.xv, D, 0)?; r.ew(1, s.n, D, &s.xv, &s.xv, D, 0, &s.xv, 0, &m.p, m.vin_b)?;
        Ok(()) })(); })?; err?;
    let mut err = Ok(());
    gpu += ctx.run(|cb| { err = (|| -> Result<()> { let r = R { ctx, pf, kn, cb, m };
        for b in &m.tv { r.text_block(s, b, 0)?; } for b in &m.ta { r.text_block(s, b, 1)?; }
        r.ew(3, s.lt, D, &s.tx[0], &s.tx[0], D, 0, &s.t16[0], D, &s.md, 0)?; r.ew(3, s.lt, DA, &s.tx[1], &s.tx[1], DA, 0, &s.t16[1], DA, &s.md, 0)?;
        Ok(()) })(); })?; err?;
    hook("pre", s);
    for b0 in (0..nb).step_by(bps) { let mut err = Ok(());
        gpu += ctx.run(|cb| { err = (|| -> Result<()> { let r = R { ctx, pf, kn, cb, m };
            for b in b0..(b0 + bps).min(nb) { r.block(s, &m.blocks[b])?; } Ok(()) })(); })?; err?;
        for b in b0..(b0 + bps).min(nb) { hook(&format!("blk{b}"), s); } }
    let mut err = Ok(());
    gpu += ctx.run(|cb| { err = (|| -> Result<()> { let r = R { ctx, pf, kn, cb, m };
        r.adaln(0, &s.xv, D, s.n, &s.a16, m.outmo + D, m.outmo, &s.md)?; r.lin(m.out, &s.a16, D, 0, s.n, &s.ov, 640, 0)?;
        r.adaln(2, &s.xa, DA, s.na, &s.a16, m.aoutmo + DA, m.aoutmo, &s.md)?; r.lin(m.aout, &s.a16, DA, 0, s.na, &s.oa, 400, 0)?;
        Ok(()) })(); })?; err?;
    let vis = unpatchify(&s.ov.as_f32()[..s.n as usize * 640], s.t as usize, s.h as usize, s.w as usize);
    let aud = s.oa.as_f32()[..s.na as usize * 400].to_vec();
    Ok(StepOut { vis, aud, gpu, wall: t0.elapsed().as_secs_f64() })
}
pub fn err_cmp(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut d, mut n, mut ab, mut aa, mut bb, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) { let (x, y) = (x as f64, y as f64); d += (x - y).powi(2); n += y * y; ab += x * y; aa += x * x; bb += y * y; mx = mx.max((x - y).abs()); }
    (ab / (aa.sqrt() * bb.sqrt() + 1e-30), (d / (n + 1e-30)).sqrt(), mx)
}
