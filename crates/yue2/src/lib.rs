//! YuE2-3B (m-a-p) music generation on strix-rt.
//!  - one Qwen3-style AR-NAR Mixture-of-Transformers backbone (28 layers, d 2048, 16 q / 8 kv heads of 128, SwiGLU 6144):
//!    the AR path writes the ABC score and the 25 Hz semantic codec tokens (sampled on the host from GPU logits of the
//!    allowed vocabulary slice only); the NAR path (separate attention / MLP / norms per layer) is a flow-matching velocity
//!    field over the 64-d VAE latents that attends to the AR K/V of the chunk (computed once) plus its own tokens;
//!  - every linear runs through prefill's coop-matrix GEMM straight from the bf16 safetensors pages (qimage::st, zero copy);
//!    attention is a causal / bidirectional variant of qimage's coop-matrix flash kernel; norms, q/k-norm + RoPE from qimage;
//!  - 32-step midpoint ODE exactly as the reference schedule; Oobleck VAE decoder (vae.rs) with im2col + coop-matrix GEMM.
pub mod vae;
pub mod npy;
use anyhow::{bail, Result};
use half::{bf16, f16};
use prefill::{GemmOp, Pf};
use qimage::{st, Kern, Rec, NOG};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const D: u32 = 2048; pub const NH: u32 = 16; pub const NKV: u32 = 8; pub const HD: u32 = 128; pub const FFD: u32 = 6144; pub const NL: usize = 28;
pub const QKV: u32 = D + 2 * NKV * HD; // 4096
pub const LAT: u32 = 64; const PAD: u64 = 320; const EPS: f32 = 1e-6;
pub const EOD: u32 = 151643; pub const ABC_START: u32 = 151847; pub const ABC_END: u32 = 151848;
pub const MUSIC_START: u32 = 151851; pub const MUSIC_END: u32 = 151852; pub const CODEC_OFFSET: u32 = 151853; pub const CODEC_SIZE: u32 = 32768;
pub const CONTEXT: usize = 24576;

pub struct YK { pub flash: Kernel, pub misc: Kernel }
impl YK { pub fn new(ctx: &Ctx) -> Result<Self> { Ok(YK { flash: ctx.kernel(spv!("ye_flash"), 4, 44, &[], Some(32))?, misc: ctx.kernel(spv!("ye_misc"), 4, 48, &[], None)? }) } }

pub struct Env<'a> { pub ctx: &'a Ctx, pub pf: &'a Pf, pub kn: &'a Kern, pub yk: &'a YK }
impl<'a> Env<'a> {
    pub fn misc(&self, cb: ash::vk::CommandBuffer, bufs: &[&Buffer], push: &[u32; 12]) -> Result<()> {
        let ds = self.pf.ds.get(self.ctx, &self.yk.misc, bufs)?; self.ctx.dispatch(cb, &self.yk.misc, ds, &vkrt::push_u32(push), push[1].div_ceil(256)); self.ctx.barrier(cb); Ok(())
    }
    pub fn flash(&self, cb: ash::vk::CommandBuffer, q: &Buffer, k: &Buffer, v: &Buffer, o: &Buffer, sq: u32, nk: u32, causal: bool, koff: u32) -> Result<()> {
        let ds = self.pf.ds.get(self.ctx, &self.yk.flash, &[q, k, v, o])?;
        let p = [sq, nk, NH, D, NKV * HD, NKV * HD, D, (1.0f32 / (HD as f32).sqrt()).to_bits(), NH / NKV, causal as u32, koff];
        self.ctx.dispatch(cb, &self.yk.flash, ds, &vkrt::push_u32(&p), sq.div_ceil(64) * NH); self.ctx.barrier(cb); Ok(())
    }
}

struct Path { w: HashMap<&'static str, WRef>, p: HashMap<&'static str, u32> }
/// Copy the GEMM weights out of the safetensors pages into <= 3.5 GiB arenas at 256-byte aligned offsets (safetensors
/// data offsets are only 2-byte aligned; the GEMM loads 16-byte vectors). Rows that are a multiple of 4 KiB get 256 bytes
/// of padding (rtcore PAD_BF16) to avoid memory-channel conflicts between the rows of a GEMM tile.
fn arenas(ctx: &Ctx, st: &st::Model, names: &[String]) -> Result<(Vec<Buffer>, HashMap<String, WRef>)> {
    const MAXC: u64 = 3584 << 20; let pad_ok = std::env::var("YE_PADBF16").map(|v| v != "0").unwrap_or(true);
    let mut plan: Vec<(String, u64, u64, bool)> = vec![]; // name, row bytes (dst), rows, padded
    for n in names { let t = st.get(n)?; if t.dtype != "BF16" { bail!("{n}: {}", t.dtype) } let k = t.shape[1..].iter().product::<usize>() as u64; let rb = k * 2;
        let pad = pad_ok && rb % 4096 == 0; plan.push((n.clone(), if pad { rb + 256 } else { rb }, t.shape[0] as u64, pad)); }
    let (mut bufs, mut w) = (vec![], HashMap::new()); let mut cur: Vec<(usize, u64)> = vec![]; let mut size = 0u64;
    let mut flush = |cur: &mut Vec<(usize, u64)>, size: &mut u64, bufs: &mut Vec<Buffer>, w: &mut HashMap<String, WRef>| -> Result<()> {
        if cur.is_empty() { return Ok(()) } let b = ctx.buffer(*size)?; let dst = b.bytes_mut();
        for &(i, off) in cur.iter() { let (n, rb, rows, pad) = &plan[i]; let src = st.bytes(n)?; let srb = if *pad { rb - 256 } else { *rb };
            for r in 0..*rows as usize { dst[off as usize + r * *rb as usize..off as usize + r * *rb as usize + srb as usize].copy_from_slice(&src[r * srb as usize..(r + 1) * srb as usize]) }
            w.insert(n.clone(), WRef { arena: bufs.len(), off, ty: if *pad { gguf::GgmlType::Other(rtcore::PAD_BF16) } else { gguf::GgmlType::Other(30) }, k: (srb / 2) as u32, n: *rows as u32, e: 1, bytes: rb * rows }); }
        bufs.push(b); cur.clear(); *size = 0; Ok(()) };
    for (i, (_, rb, rows, _)) in plan.iter().enumerate() { let bytes = rb * rows; if size + bytes > MAXC { flush(&mut cur, &mut size, &mut bufs, &mut w)?; } cur.push((i, size)); size = (size + bytes).next_multiple_of(256); }
    flush(&mut cur, &mut size, &mut bufs, &mut w)?;
    Ok((bufs, w))
}

pub struct Model { pub st: st::Model, pub bufs: Vec<Buffer>, params: Buffer, ar: Vec<Path>, nar: Vec<Path>, norm: u32, head: WRef, llm2vae: WRef, vae2llm: WRef, pub l2v_b: Vec<f32>, v2l_b: Vec<f32>,
    t1: (Vec<f32>, Vec<f32>), t2: (Vec<f32>, Vec<f32>), fused: bool, emb: (usize, usize) }

fn bf(b: &[u8], i: usize) -> f32 { bf16::from_le_bytes([b[2 * i], b[2 * i + 1]]).to_f32() }

impl Model {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|_| false)?;
        let mut names: Vec<String> = st.t.keys().filter(|k| k.ends_with("proj.weight") || *k == "lm_head.weight" || *k == "llm2vae.weight" || *k == "vae2llm.weight").cloned().collect(); names.sort();
        let (bufs, wm) = arenas(ctx, &st, &names)?; let wr = |n: &str| -> Result<WRef> { wm.get(n).copied().ok_or_else(|| anyhow::anyhow!("missing {n}")) };
        let mut pv: Vec<f32> = vec![];
        let push = |pv: &mut Vec<f32>, v: Vec<f32>| -> u32 { let o = pv.len() as u32; pv.extend(v); while pv.len() % 64 != 0 { pv.push(0.0) } o };
        let (mut ar, mut nar) = (vec![], vec![]);
        for l in 0..NL { for (np, list) in [("", &mut ar), ("nar_", &mut nar)] {
            let p = format!("model.layers.{l}.");
            let (attn, mlp, n_in, n_post) = if np.is_empty() { ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm") } else { ("nar_self_attn", "nar_mlp", "nar_input_layernorm", "nar_pre_mlp_layernorm") };
            let mut w = HashMap::new();
            for (k, n) in [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")] { w.insert(k, wr(&format!("{p}{attn}.{n}.weight"))?); }
            for (k, n) in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")] { w.insert(k, wr(&format!("{p}{mlp}.{n}.weight"))?); }
            let mut o = HashMap::new();
            o.insert("in", push(&mut pv, st.f32(&format!("{p}{n_in}.weight"))?)); o.insert("post", push(&mut pv, st.f32(&format!("{p}{n_post}.weight"))?));
            o.insert("qn", push(&mut pv, st.f32(&format!("{p}{attn}.q_norm.weight"))?)); o.insert("kn", push(&mut pv, st.f32(&format!("{p}{attn}.k_norm.weight"))?));
            list.push(Path { w, p: o }); } }
        let norm = push(&mut pv, st.f32("model.norm.weight")?);
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        let fused = ar.iter().chain(nar.iter()).all(|p| p.w["gate"].arena == p.w["up"].arena) && std::env::var("YE_SWIGLU").map(|v| v != "0").unwrap_or(true);
        let e = st.get("model.embed_tokens.weight")?; let emb = (e.shard, e.start);
        Ok(Model { head: wr("lm_head.weight")?, llm2vae: wr("llm2vae.weight")?, vae2llm: wr("vae2llm.weight")?, bufs, l2v_b: st.f32("llm2vae.bias")?, v2l_b: st.f32("vae2llm.bias")?,
            t1: (st.f32("time_embedder.mlp.0.weight")?, st.f32("time_embedder.mlp.0.bias")?), t2: (st.f32("time_embedder.mlp.2.weight")?, st.f32("time_embedder.mlp.2.bias")?),
            st, params, ar, nar, norm, fused, emb })
    }
    /// bf16 embedding rows of `toks` -> x rows [row0..)
    pub fn embed(&self, x: &Buffer, row0: usize, toks: &[u32]) {
        let mm = &self.st.shards[self.emb.0].mm[self.emb.1..]; let xs = x.as_f32_mut(); let d = D as usize;
        for (i, &t) in toks.iter().enumerate() { let r = &mm[t as usize * d * 2..(t as usize + 1) * d * 2]; for j in 0..d { xs[(row0 + i) * d + j] = bf(r, j) } }
    }
    /// Time embedding (CPU, f32 math on the bf16 weights; t is rounded to bf16 as the reference's sigmoid output) + vae2llm bias.
    pub fn cvec(&self, t: f32) -> Vec<f32> {
        let t = bf16::from_f32(t).to_f32(); let r16 = |v: f32| bf16::from_f32(v).to_f32();
        let mut e = [0f32; 256];
        for i in 0..128 { let f = (-(10000f32.ln()) * i as f32 / 128.0).exp(); let a = t * f; e[i] = r16(a.cos()); e[128 + i] = r16(a.sin()); }
        let d = D as usize;
        let h: Vec<f32> = (0..d).map(|o| { let w = &self.t1.0[o * 256..(o + 1) * 256]; let s: f32 = w.iter().zip(&e).map(|(a, b)| a * b).sum::<f32>() + self.t1.1[o]; let s = r16(s); r16(s / (1.0 + (-s).exp())) }).collect();
        (0..d).map(|o| { let w = &self.t2.0[o * d..(o + 1) * d]; r16(w.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>() + self.t2.1[o]) + self.v2l_b[o] }).collect()
    }
}

/// Device state: activations for up to `rmax` rows, K/V caches of `kmax` rows per layer (AR then NAR rows), RoPE table.
pub struct State { pub rmax: u32, pub kmax: u32, pub x: Buffer, pub a: Buffer, pub y: Buffer, y2: Buffer, h: Buffer, q: Buffer, kc: Vec<Buffer>, vc: Vec<Buffer>, rope: Buffer,
    pub logits: Buffer, pub lat: Buffer, pub vel: Buffer, aux: Buffer }
impl State {
    pub fn new(ctx: &Ctx, rmax: u32, kmax: u32) -> Result<Self> {
        let r = rmax as u64 + PAD; let k = kmax as u64 + 64;
        let mut kc = vec![]; let mut vc = vec![];
        for _ in 0..NL { kc.push(ctx.buffer(k * (NKV * HD) as u64 * 2)?); vc.push(ctx.buffer(k * (NKV * HD) as u64 * 2)?); }
        let rope = ctx.buffer(k * 128 * 4)?;
        { let rt = rope.as_f32_mut(); for p in 0..k as usize { for i in 0..64 { let inv = 1.0 / 1_000_000f64.powf(2.0 * i as f64 / 128.0);
            let (s, c) = ((p as f64) * inv).sin_cos(); rt[p * 128 + 2 * i] = c as f32; rt[p * 128 + 2 * i + 1] = s as f32; } } }
        let fy = if std::env::var("YE_SWIGLU").map(|v| v == "0").unwrap_or(false) { 2 * FFD } else { QKV };
        Ok(State { rmax, kmax, x: ctx.buffer(r * D as u64 * 4)?, a: ctx.buffer(r * FFD as u64 * 2)?, y: ctx.buffer(r * fy as u64 * 4)?, y2: ctx.buffer(r * D as u64 * 4)?,
            h: ctx.buffer(r * FFD as u64 * 2)?, q: ctx.buffer((r + 64) * D as u64 * 2)?, kc, vc, rope, logits: ctx.buffer(64 * 192000 * 4)?,
            lat: ctx.buffer(r * LAT as u64 * 2)?, vel: ctx.buffer(r * LAT as u64 * 4)?, aux: ctx.buffer((r + 1) * D as u64 * 4)? })
    }
}

impl Model {
    /// One layer over m rows at positions pos0.. (rope rows) whose K/V go to cache rows kv0..; attention over keys [0, nk).
    fn layer(&self, e: &Env, r: &Rec, s: &State, l: usize, nar: bool, m: u32, pos0: u32, kv0: u32, nk: u32, causal: bool) -> Result<()> {
        let pth = if nar { &self.nar[l] } else { &self.ar[l] }; let w = &pth.w; let p = &pth.p; let pr = &self.params; let ar = &self.bufs; let eps = EPS.to_bits();
        r.run("norm", &[&s.x, &s.a, pr, pr], &[D, D, D, p["in"], 0, 0, eps, 0], m)?;
        if std::env::var("YE_STOP").as_deref() == Ok("4") { return Ok(()) }
        r.gemm(ar, pr, w["q"], &s.a, D, m, &s.y, QKV, 0, false)?; r.gemm(ar, pr, w["k"], &s.a, D, m, &s.y, QKV, D, false)?; r.gemm(ar, pr, w["v"], &s.a, D, m, &s.y, QKV, D + 1024, true)?;
        let stop = std::env::var("YE_STOP").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0); if stop == 3 { return Ok(()) }
        r.run("qk", &[&s.y, &s.q, pr, &s.rope], &[NH, QKV, 0, p["qn"], eps, D, 0, 1, 3, pos0], m * NH)?;
        r.run("qk", &[&s.y, &s.kc[l], pr, &s.rope], &[NKV, QKV, D, p["kn"], eps, 1024, kv0, 1, 3, pos0], m * NKV)?;
        r.run("qk", &[&s.y, &s.vc[l], pr, &s.rope], &[NKV, QKV, D + 1024, 0, 0, 1024, kv0, 1, 0, pos0], m * NKV)?;
        e.flash(r.cb, &s.q, &s.kc[l], &s.vc[l], &s.a, m, nk, causal, pos0)?;
        r.gemm(ar, pr, w["o"], &s.a, D, m, &s.y2, D, 0, true)?;
        r.ew(0, m, D, &s.x, &s.y2, D, 0, &s.a, 0, pr, NOG)?; if stop == 1 { return Ok(()) }
        r.run("norm", &[&s.x, &s.a, pr, pr], &[D, D, D, p["post"], 0, 0, eps, 0], m)?;
        if self.fused { r.pf.gemm_swiglu(r.ctx, r.cb, &ar[w["gate"].arena], pr, w["gate"], w["up"], 0, FFD, &s.a, D, m, &s.h, FFD, 0)?; r.bar(); }
        else { r.gemm(ar, pr, w["gate"], &s.a, D, m, &s.y, 2 * FFD, 0, false)?; r.gemm(ar, pr, w["up"], &s.a, D, m, &s.y, 2 * FFD, FFD, true)?; r.ew(2, m, FFD, &s.x, &s.y, 2 * FFD, 0, &s.h, FFD, pr, 0)?; }
        r.gemm(ar, pr, w["down"], &s.h, FFD, m, &s.y2, D, 0, true)?;
        r.ew(0, m, D, &s.x, &s.y2, D, 0, &s.a, 0, pr, NOG)?;
        Ok(())
    }
    /// Final norm of row `row` + LM head rows [row0, row0 + n) -> logits[0..n] (f32).
    fn head(&self, r: &Rec, s: &State, rows: u32, ranges: &[(u32, u32)]) -> Result<()> {
        let pr = &self.params;
        r.run("norm", &[&s.x, &s.a, pr, pr], &[D, D, D, self.norm, 0, 0, EPS.to_bits(), 0], rows)?;
        let mut col = 0; let ld: u32 = ranges.iter().map(|x| x.1).sum();
        for &(row0, n) in ranges { r.pf.gemm(r.ctx, r.cb, &self.bufs[self.head.arena], pr, &GemmOp { w: self.head, row0, rows: n, a: &s.a, lda: D, a_row0: 0, m: rows, y: &s.logits, ldy: ld, y_col: col, acc: false })?; col += n; }
        r.bar(); Ok(())
    }
    /// AR forward of `toks` at positions pos0.. (K/V cache rows = positions). Chunks of s.rmax rows, several submits.
    /// If `ranges` is given, the logits of the last token over those vocab row ranges are left in s.logits.
    pub fn ar_forward(&self, e: &Env, s: &State, toks: &[u32], pos0: u32, ranges: Option<&[(u32, u32)]>) -> Result<f64> {
        let mut g = 0.0; let mut off = 0usize;
        while off < toks.len() {
            let n = (toks.len() - off).min(s.rmax as usize); let p0 = pos0 + off as u32; let last = off + n == toks.len();
            self.embed(&s.x, 0, &toks[off..off + n]);
            let lps = if n == 1 { NL } else { 7 }; let nlim = std::env::var("YE_NL").ok().and_then(|v| v.parse().ok()).unwrap_or(NL);
            for l0 in (0..nlim).step_by(lps) {
                let mut err = Ok(());
                g += e.ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx: e.ctx, pf: e.pf, kn: e.kn, cb };
                    for l in l0..(l0 + lps).min(nlim) { self.layer(e, &r, s, l, false, n as u32, p0, p0, p0 + n as u32, true)?; }
                    if last && l0 + lps >= NL { if let Some(rg) = ranges { 
                        if n > 1 { r.ew(5, 1, D, &s.x, &s.x, D, (n as u32 - 1) * D, &s.a, 0, &self.params, 0)?; } // last row -> row 0
                        self.head(&r, s, 1, rg)?; } }
                    Ok(()) })(); })?;
                err?;
            }
            off += n;
        }
        Ok(g)
    }
    /// Teacher-forced full-vocab logits at the given rows of a single forward (rows must fit s.rmax).
    pub fn ar_logits_rows(&self, e: &Env, s: &State, toks: &[u32], rows: &[u32]) -> Result<Vec<f32>> {
        let n = toks.len() as u32; if n > s.rmax { bail!("ar_logits_rows: {} > rmax", n) }
        self.ar_forward(e, s, toks, 0, None)?;
        let xs = s.x.as_f32().to_vec(); let d = D as usize; let vocab = self.head.n;
        let mut out = vec![]; let per = (64 * 192000 / vocab.max(1)).min(64).max(1) as usize;
        for ch in rows.chunks(per) {
            for (i, &rw) in ch.iter().enumerate() { s.x.as_f32_mut()[i * d..(i + 1) * d].copy_from_slice(&xs[rw as usize * d..(rw as usize + 1) * d]); }
            let mut err = Ok(());
            e.ctx.run(|cb| { err = self.head(&Rec { ctx: e.ctx, pf: e.pf, kn: e.kn, cb }, s, ch.len() as u32, &[(0, vocab)]); })?; err?;
            out.extend_from_slice(&s.logits.as_f32()[..ch.len() * vocab as usize]);
        }
        Ok(out)
    }
}

/// NAR engine for one acoustic chunk: AR K/V of the chunk tokens computed once; velocity(state, t) per ODE evaluation.
pub struct Nar { pub ar_len: u32, pub t: u32 }
impl Model {
    pub fn nar_prefill(&self, e: &Env, s: &State, ar_tokens: &[u32], t_lat: u32) -> Result<(Nar, f64)> {
        let ar_len = ar_tokens.len() as u32; let nn = t_lat + 2;
        if ar_len + nn > s.kmax || nn > s.rmax { bail!("nar chunk {} + {} exceeds state", ar_len, nn) }
        let g = self.ar_forward(e, s, ar_tokens, 0, None)?;
        // positional table rows 0..nn (bf16 checkpoint buffer)
        let pe = self.st.bytes("latent_pos_embed.pe")?; let d = D as usize; let aux = s.aux.as_f32_mut();
        for i in 0..nn as usize * d { aux[i] = bf(pe, i) }
        Ok((Nar { ar_len, t: t_lat }, g))
    }
    /// v(x_t, t) for state [T][64] (host f32), t = the (sigmoid-shifted) flow time. Returns [T][64].
    pub fn velocity(&self, e: &Env, s: &State, nar: &Nar, state: &[f32], t: f32) -> Result<(Vec<f32>, f64)> {
        let nn = nar.t + 2; let d = D as usize;
        { let lat = unsafe { std::slice::from_raw_parts_mut(s.lat.ptr as *mut f16, (nn as usize) * 64) };
          lat[..64].fill(f16::ZERO); lat[(nn as usize - 1) * 64..].fill(f16::ZERO);
          for (o, v) in lat[64..(nn as usize - 1) * 64].iter_mut().zip(state) { *o = f16::from_f32(*v) } }
        let cv = self.cvec(t); s.aux.as_f32_mut()[nn as usize * d..(nn as usize + 1) * d].copy_from_slice(&cv);
        let pr = &self.params; let ar = &self.bufs;
        let mut err = Ok(());
        let g = e.ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx: e.ctx, pf: e.pf, kn: e.kn, cb };
            r.gemm(ar, pr, self.vae2llm, &s.lat, LAT, nn, &s.y2, D, 0, true)?;
            e.misc(cb, &[&s.x, &s.y2, &s.a, &s.aux], &[0, nn * D, D, 0, nn * D, 0, 0, 0, 0, 0, 0, 0])?;
            for l in 0..NL { self.layer(e, &r, s, l, true, nn, nar.ar_len, nar.ar_len, nar.ar_len + nn, false)?; }
            r.run("norm", &[&s.x, &s.a, pr, pr], &[D, D, D, self.norm, 0, 0, EPS.to_bits(), 0], nn)?;
            r.gemm(ar, pr, self.llm2vae, &s.a, D, nn, &s.vel, LAT, 0, true)?;
            Ok(()) })(); })?;
        err?;
        let v = s.vel.as_f32(); let mut out = vec![0f32; nar.t as usize * 64];
        for i in 0..nar.t as usize { for c in 0..64 { out[i * 64 + c] = v[(i + 1) * 64 + c] + self.l2v_b[c] } }
        Ok((out, g))
    }
    /// 32-step (default) midpoint solve from `noise` [T][64]; returns latents [T][64] and GPU seconds.
    pub fn solve(&self, e: &Env, s: &State, nar: &Nar, noise: &[f32], steps: usize) -> Result<(Vec<f32>, f64)> {
        let mut st = noise.to_vec(); let dt = 1.0 / steps as f64; let mut g = 0.0;
        let tval = |t: f64| -> f32 { let raw = bf16::from_f64((t / (1.0 - t)).ln().clamp(-20.0, 20.0)).to_f32(); bf16::from_f32(1.0 / (1.0 + (-raw).exp())).to_f32() };
        for k in 0..steps {
            let t = 1.0 - k as f64 * dt;
            let (v1, g1) = self.velocity(e, s, nar, &st, tval(t))?;
            let mid: Vec<f32> = st.iter().zip(&v1).map(|(a, b)| a - b * (dt / 2.0) as f32).collect();
            let (v2, g2) = self.velocity(e, s, nar, &mid, tval(t - dt / 2.0))?;
            for (a, b) in st.iter_mut().zip(&v2) { *a -= b * dt as f32 }
            g += g1 + g2;
        }
        Ok((st, g))
    }
}

/// Song chunks exactly as yue2.protocol.chunk_ranges: [(a, b)] codec frame ranges.
pub fn chunk_ranges(frames: usize, prefix: usize) -> Vec<(usize, usize)> { let size = ((CONTEXT - prefix - 3) / 2).min(CONTEXT); (0..frames).step_by(size).map(|a| (a, (a + size).min(frames))).collect() }

/// Host sampler matching yue2.sampling.distribution (+ multinomial with our own RNG).
pub struct Sampling { pub temperature: f32, pub top_p: f32, pub top_k: usize, pub rep: f32, pub window: usize, pub min_tokens: usize, pub max_tokens: usize }
impl Sampling {
    pub fn abc() -> Self { Sampling { temperature: 0.7, top_p: 0.9, top_k: 30, rep: 1.005, window: 100, min_tokens: 32, max_tokens: 4096 } }
    pub fn semantic() -> Self { Sampling { temperature: 1.0, top_p: 0.95, top_k: 100, rep: 1.2, window: 50, min_tokens: 200, max_tokens: 9000 } }
}
pub struct Rng(pub u64);
impl Rng { pub fn next_f64(&mut self) -> f64 { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^= z >> 31; (z >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn normal(&mut self) -> f32 { let u1 = self.next_f64().max(1e-300); let u2 = self.next_f64(); ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32 } }
/// `cand`: (token id, logit) of the allowed vocabulary; `end` the end token; returns the sampled id.
pub fn sample(cand: &mut Vec<(u32, f32)>, sp: &Sampling, hist: &[u32], step: usize, end: u32, rng: &mut Rng) -> u32 {
    if step < sp.min_tokens { cand.retain(|c| c.0 != end) }
    if sp.rep != 1.0 { let w = &hist[hist.len().saturating_sub(sp.window)..]; let mut cnt: HashMap<u32, i32> = HashMap::new(); for &t in w { *cnt.entry(t).or_default() += 1 }
        for c in cand.iter_mut() { if let Some(&f) = cnt.get(&c.0) { let a = sp.rep.powi(f); c.1 = if c.1 < 0.0 { c.1 * a } else { c.1 / a } } } }
    if sp.temperature == 0.0 { return cand.iter().fold((0u32, f32::MIN), |b, c| if c.1 > b.1 { *c } else { b }).0 }
    for c in cand.iter_mut() { c.1 /= sp.temperature }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let thr = cand[(sp.top_k.min(cand.len())) - 1].1; cand.retain(|c| c.1 >= thr);
    let mx = cand[0].1; let ps: Vec<f64> = cand.iter().map(|c| ((c.1 - mx) as f64).exp()).collect(); let z: f64 = ps.iter().sum();
    let mut keep = cand.len(); let mut cum = 0.0; for (i, p) in ps.iter().enumerate() { if i > 0 && cum > sp.top_p as f64 { keep = i; break } cum += p / z; }
    let z2: f64 = ps[..keep].iter().sum(); let mut u = rng.next_f64() * z2;
    for i in 0..keep { u -= ps[i]; if u <= 0.0 { return cand[i].0 } } cand[keep - 1].0
}

impl Model {
    /// Autoregressive generation after `prefix` (phase abc or semantic). Returns (generated ids without end, eos, gpu s, wall s).
    pub fn generate(&self, e: &Env, s: &State, prefix: &[u32], semantic: bool, sp: &Sampling, rng: &mut Rng, mut on_tok: impl FnMut(u32)) -> Result<(Vec<u32>, bool, f64, f64)> {
        let t0 = std::time::Instant::now();
        let (ranges, end): (Vec<(u32, u32)>, u32) = if semantic { (vec![(MUSIC_END, CODEC_SIZE + 1)], MUSIC_END) } else { (vec![(0, EOD), (ABC_END, 1)], ABC_END) };
        let ids = |i: usize| -> u32 { if semantic { MUSIC_END + i as u32 } else if i < EOD as usize { i as u32 } else { ABC_END } };
        let nl: usize = ranges.iter().map(|r| r.1 as usize).sum();
        if prefix.len() + sp.max_tokens + 1 > s.kmax as usize { bail!("generate: context {} + {} > kmax", prefix.len(), sp.max_tokens) }
        let mut g = self.ar_forward(e, s, prefix, 0, Some(&ranges))?;
        let mut hist: Vec<u32> = vec![]; let mut eos = false;
        for step in 0..sp.max_tokens {
            let lg = &s.logits.as_f32()[..nl];
            let mut cand: Vec<(u32, f32)> = lg.iter().enumerate().map(|(i, &v)| (ids(i), v)).collect();
            let tok = sample(&mut cand, sp, &hist, step, end, rng);
            on_tok(tok);
            if tok == end { eos = true; break }
            hist.push(tok);
            if step + 1 < sp.max_tokens { g += self.ar_forward(e, s, &[tok], (prefix.len() + hist.len() - 1) as u32, Some(&ranges))?; }
        }
        Ok((hist, eos, g, t0.elapsed().as_secs_f64()))
    }
}
