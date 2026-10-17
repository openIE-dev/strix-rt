//! NVIDIA Cosmos3-Nano (Cosmos3OmniTransformer, Mixture-of-Transformers) on strix-rt.
//! One 36-layer Qwen3-VL-style stack with two weight sets per layer:
//!  - understanding ("und", text) tower: causal attention over the text tokens only; its per-layer K/V are computed once
//!    per prompt (they never see the generation tokens) and cached;
//!  - generation ("gen", diffusion) tower: full attention of the latent tokens over [und K/V, gen K/V]; GQA 32/8, q/k RMS
//!    norm, interleaved 3D mRoPE (sections 24/20/20, theta 5e6). Timestep conditioning is additive (no adaLN):
//!    x = proj_in(patchify(latent)) + time_embedder(time_proj(sigma)).
//! Kernels and the zero-copy safetensors loader come from crates/qimage (qi_norm, qi_qk, qi_small, qi_flash, qi_ew,
//! prefill coop-matrix GEMMs); nothing in qimage is changed.
pub mod sched;
pub mod vae;
pub mod prompt;
pub mod vit;
use anyhow::Result;
use half::{bf16, f16};
use qimage::{st, Kern, Rec, NOG};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{push_u32, Buffer, Ctx};
use prefill::Pf;

pub const D: u32 = 4096; pub const FFD: u32 = 12288; pub const NH: u32 = 32; pub const NKV: u32 = 8; pub const KVW: u32 = NKV * 128;
pub const PD: u32 = 192; pub const LC: usize = 48; const PAD: u64 = 320;
fn ald() -> u32 { D + 64 } fn hld() -> u32 { FFD + 64 }
fn lps() -> usize { std::env::var("COS_LPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2).max(1) }
fn flash_qt() -> u32 { std::env::var("COS_FLASH_QT").ok().and_then(|v| v.parse().ok()).unwrap_or(8).max(1) }
const EPS: f32 = 1e-6;

pub struct Model { pub st: st::Model, params: Buffer, lw: Vec<HashMap<&'static str, WRef>>, lp: Vec<HashMap<&'static str, u32>>, pub nl: usize,
    fin_u: u32, fin_g: u32, proj_in: WRef, proj_out: WRef, proj_in_b: Vec<f32>, proj_out_b: Vec<f32>, t1: (Vec<f32>, Vec<f32>), t2: (Vec<f32>, Vec<f32>), fused: bool }
/// Per-prompt cache (und K/V in rows [0, lt) of every layer) plus scratch for n gen tokens.
pub struct State { pub lt: u32, pub n: u32, kc: Vec<Buffer>, vc: Vec<Buffer>, x: Buffer, a: Buffer, y: Buffer, y2: Buffer, h: Buffer, q: Buffer,
    rope_u: Buffer, rope_g: Buffer, tb: Buffer, lat: Buffer, pub v: Buffer, pub dumps: Vec<(String, Vec<f32>)>,
    /// understanding path: DeepStack features (first image row, 3 x [nv][4096]) added after und layers 0, 1, 2
    pub ds: Option<(usize, Vec<Vec<f32>>)>,
    /// und KV-cache decode: > 0 = one new token at row `dec` attending to the cached rows [0, dec] (flash), rope from rope_g row 0
    pub dec: u32 }

/// Interleaved 3D mRoPE angles for one token: freq p in 0..64 uses axis H if p%3==1 && p<60, W if p%3==2 && p<60, else T.
pub fn mrope_row(t: f64, h: f64, w: f64, out: &mut [f32]) {
    for p in 0..64usize { let pos = if p < 60 && p % 3 == 1 { h } else if p < 60 && p % 3 == 2 { w } else { t };
        let a = pos * 5_000_000f64.powf(-(2.0 * p as f64) / 128.0); let (s, c) = a.sin_cos(); out[2 * p] = c as f32; out[2 * p + 1] = s as f32; }
}
fn matvec(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> { let k = x.len(); (0..b.len()).map(|i| b[i] + w[i * k..(i + 1) * k].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect() }
fn silu(v: f32) -> f32 { v / (1.0 + (-v).exp()) }

impl Model {
    pub fn load(ctx: &Ctx, dir: &str, lm_head: bool) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|k| (k.starts_with("layers.") && k.ends_with(".weight") && !k.contains("norm")) || k == "proj_in.weight" || k == "proj_out.weight" || (lm_head && k == "lm_head.weight"))?;
        let nl = (0..).take_while(|i| st.t.contains_key(&format!("layers.{i}.input_layernorm.weight"))).count();
        let mut pv: Vec<f32> = vec![]; let mut lw = vec![]; let mut lp = vec![];
        let mut push = |pv: &mut Vec<f32>, v: Vec<f32>| -> u32 { let o = pv.len() as u32; pv.extend(v); while pv.len() % 64 != 0 { pv.push(0.0) } o };
        for l in 0..nl { let p = format!("layers.{l}.");
            let mut w = HashMap::new();
            for (k, n) in [("uq", "self_attn.to_q"), ("uk", "self_attn.to_k"), ("uv", "self_attn.to_v"), ("uo", "self_attn.to_out"), ("ugate", "mlp.gate_proj"), ("uup", "mlp.up_proj"), ("udown", "mlp.down_proj"),
                ("gq", "self_attn.add_q_proj"), ("gk", "self_attn.add_k_proj"), ("gv", "self_attn.add_v_proj"), ("go", "self_attn.to_add_out"), ("ggate", "mlp_moe_gen.gate_proj"), ("gup", "mlp_moe_gen.up_proj"), ("gdown", "mlp_moe_gen.down_proj")] {
                w.insert(k, st.wref(&format!("{p}{n}.weight"))?); }
            let mut o = HashMap::new();
            for (k, n) in [("uin", "input_layernorm"), ("upost", "post_attention_layernorm"), ("uqn", "self_attn.norm_q"), ("ukn", "self_attn.norm_k"),
                ("gin", "input_layernorm_moe_gen"), ("gpost", "post_attention_layernorm_moe_gen"), ("gqn", "self_attn.norm_added_q"), ("gkn", "self_attn.norm_added_k")] {
                o.insert(k, push(&mut pv, st.f32(&format!("{p}{n}.weight"))?)); }
            lw.push(w); lp.push(o); }
        let fin_u = push(&mut pv, st.f32("norm.weight")?); let fin_g = push(&mut pv, st.f32("norm_moe_gen.weight")?);
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        let fused = lw.iter().all(|w| w["ggate"].arena == w["gup"].arena) && std::env::var("COS_SWIGLU").map_or(true, |v| v != "0");
        Ok(Model { proj_in: st.wref("proj_in.weight")?, proj_out: st.wref("proj_out.weight")?, proj_in_b: st.f32("proj_in.bias")?, proj_out_b: st.f32("proj_out.bias")?,
            t1: (st.f32("time_embedder.linear_1.weight")?, st.f32("time_embedder.linear_1.bias")?), t2: (st.f32("time_embedder.linear_2.weight")?, st.f32("time_embedder.linear_2.bias")?),
            st, params, lw, lp, nl, fin_u, fin_g, fused })
    }
    /// time_embedder(time_proj(t * timestep_scale)) in f32 (diffusers keeps time_embedder in fp32), Timesteps(256, flip_sin_to_cos, shift 0).
    pub fn temb(&self, timestep: f64) -> Vec<f32> {
        let t = timestep * 0.001; let mut e = vec![0f32; 256];
        for i in 0..128 { let f = (-(10000f64.ln()) * i as f64 / 128.0).exp(); let a = t * f; e[i] = a.cos() as f32; e[128 + i] = a.sin() as f32; }
        let h: Vec<f32> = matvec(&self.t1.0, &self.t1.1, &e).into_iter().map(silu).collect(); matvec(&self.t2.0, &self.t2.1, &h)
    }
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> { let e = self.st.bytes("embed_tokens.weight")?; let mut o = vec![0f32; ids.len() * D as usize];
        for (i, &t) in ids.iter().enumerate() { let r = &e[t as usize * 8192..(t as usize + 1) * 8192]; for j in 0..4096 { o[i * 4096 + j] = bf16::from_le_bytes([r[2 * j], r[2 * j + 1]]).to_f32() } } Ok(o) }

    pub fn state(&self, ctx: &Ctx, lt: u32, n: u32) -> Result<State> {
        let nk = (lt + n) as u64 + 64; let rows = n.max(lt) as u64 + PAD; let d = D as u64;
        let mut kc = vec![]; let mut vc = vec![];
        for _ in 0..self.nl { kc.push(ctx.buffer(nk * KVW as u64 * 2)?); vc.push(ctx.buffer(nk * KVW as u64 * 2)?); }
        let s = State { lt, n, kc, vc, x: ctx.buffer(rows * d * 4)?, a: ctx.buffer(rows * hld() as u64 * 2)?, y: ctx.buffer(rows * 2 * FFD as u64 * 4)?, y2: ctx.buffer(rows * d * 4)?,
            h: ctx.buffer(rows * hld() as u64 * 2)?, q: ctx.buffer((rows + 128) * d * 2)?, rope_u: ctx.buffer(lt.max(1) as u64 * 512)?, rope_g: ctx.buffer(n as u64 * 512)?,
            tb: ctx.buffer(d * 4)?, lat: ctx.buffer(rows * PD as u64 * 2)?, v: ctx.buffer(rows * PD as u64 * 4)?, dumps: vec![], ds: None, dec: 0 };
        let ru = s.rope_u.as_f32_mut(); for t in 0..lt as usize { mrope_row(t as f64, t as f64, t as f64, &mut ru[t * 128..(t + 1) * 128]) }
        Ok(s)
    }
    /// Und-token 3D mRoPE positions (understanding path: image tokens get (t, h, w) grids, see Qwen3-VL get_rope_index).
    pub fn set_rope_u(s: &State, pos: &[[f64; 3]]) { let ru = s.rope_u.as_f32_mut(); for (t, p) in pos.iter().enumerate() { mrope_row(p[0], p[1], p[2], &mut ru[t * 128..(t + 1) * 128]) } }
    /// Gen-token positions: frame f, patch row y, col x -> (f * fps_scale + lt + margin, y, x) (Cosmos unified 3D mRoPE, reset spatial ids).
    pub fn set_grid(s: &State, t: u32, h: u32, w: u32) {
        let rg = s.rope_g.as_f32_mut(); let off = s.lt as f64 + 15000.0;
        for f in 0..t { for y in 0..h { for x in 0..w { let r = ((f * h + y) * w + x) as usize; mrope_row(f as f64 + off, y as f64, x as f64, &mut rg[r * 128..(r + 1) * 128]) } } }
    }
    /// y[y_col..] = A[0][..k] . W^T for one row (BF16 GEMV; falls back to the coop GEMM if the weight is not 16 B aligned). COS_GEMV=0 disables.
    fn gm(&self, r: &Rec, w: WRef, a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32, bar: bool) -> Result<()> {
        let ar = &self.st.bufs; let _ = lda;
        if w.off % 16 == 0 && w.k % 2048 == 0 && w.bytes == w.n as u64 * w.k as u64 * 2 && std::env::var("COS_GEMV").map_or(true, |v| v != "0") {
            r.run("cgemv", &[&ar[w.arena], a, y], &[w.k, w.n, (w.off / 16) as u32, 0, ldy, y_col], w.n)?; if bar { } return Ok(()) }
        r.gemm(ar, &self.params, w, a, lda, 1, y, ldy, y_col, bar)
    }
    fn swiglu(&self, r: &Rec, s: &State, w: &HashMap<&'static str, WRef>, g: bool, m: u32) -> Result<()> {
        let (gk, uk) = if g { ("ggate", "gup") } else { ("ugate", "uup") }; let ar = &self.st.bufs; let p = &self.params;
        if s.dec > 0 && !g { self.gm(r, w[gk], &s.a, ald(), &s.y, 2 * FFD, 0, true)?; self.gm(r, w[uk], &s.a, ald(), &s.y, 2 * FFD, FFD, true)?; r.ew(2, m, FFD, &s.x, &s.y, 2 * FFD, 0, &s.h, hld(), p, 0)?; }
        else if self.fused && w[gk].arena == w[uk].arena { r.pf.gemm_swiglu(r.ctx, r.cb, &ar[w[gk].arena], p, w[gk], w[uk], 0, FFD, &s.a, ald(), m, &s.h, hld(), 0)?; r.bar(); }
        else { r.gemm(ar, p, w[gk], &s.a, ald(), m, &s.y, 2 * FFD, 0, false)?; r.gemm(ar, p, w[uk], &s.a, ald(), m, &s.y, 2 * FFD, FFD, true)?; r.ew(2, m, FFD, &s.x, &s.y, 2 * FFD, 0, &s.h, hld(), p, 0)?; }
        Ok(())
    }
    pub const OPS: [&'static str; 10] = ["norm1", "qkv", "qk", "attn", "o", "res1", "norm2", "swiglu", "down", "res2"];
    /// One layer of one tower. und: m = lt causal (qi_small), K/V rows [0, lt). gen: m = n, K/V rows [lt, lt+n), flash over lt+n keys.
    fn layer(&self, r: &Rec, s: &State, l: usize, g: bool, full: bool) -> Result<()> {
        for op in Self::OPS { if !g && !full && l == self.nl - 1 && op == "attn" { return Ok(()) } self.layer_op(r, s, l, g, op)?; } Ok(())
    }
    fn layer_op(&self, r: &Rec, s: &State, l: usize, g: bool, op: &str) -> Result<()> {
        let w = &self.lw[l]; let o = &self.lp[l]; let p = &self.params; let ar = &self.st.bufs; let eps = EPS.to_bits();
        let k = |n: &str| -> &'static str { let key = format!("{}{}", if g { "g" } else { "u" }, n); *w.keys().chain(o.keys()).find(|x| **x == key).unwrap() };
        let (m, row0, rope) = if g { (s.n, s.lt, &s.rope_g) } else if s.dec > 0 { (1, s.dec, &s.rope_g) } else { (s.lt, 0, &s.rope_u) };
        let sc = (1.0f32 / 128f32.sqrt()).to_bits();
        match op {
            "norm1" => r.run("norm", &[&s.x, &s.a, p, p], &[D, D, ald(), o[k("in")], 0, 0, eps, 0], m)?,
            "qkv" if s.dec > 0 && !g => { self.gm(r, w[k("q")], &s.a, ald(), &s.y, 6144, 0, true)?; self.gm(r, w[k("k")], &s.a, ald(), &s.y, 6144, 4096, true)?; self.gm(r, w[k("v")], &s.a, ald(), &s.y, 6144, 5120, true)?; }
            "qkv" => { r.gemm(ar, p, w[k("q")], &s.a, ald(), m, &s.y, 6144, 0, false)?; r.gemm(ar, p, w[k("k")], &s.a, ald(), m, &s.y, 6144, 4096, false)?; r.gemm(ar, p, w[k("v")], &s.a, ald(), m, &s.y, 6144, 5120, true)?; }
            "qk" => { r.run("qk", &[&s.y, &s.q, p, rope], &[NH, 6144, 0, o[k("qn")], eps, D, 0, 1, 3, 0], m * NH)?;
                r.run("qk", &[&s.y, &s.kc[l], p, rope], &[NKV, 6144, 4096, o[k("kn")], eps, KVW, row0, 1, 3, 0], m * NKV)?;
                r.run("qk", &[&s.y, &s.vc[l], p, rope], &[NKV, 6144, 5120, 0, 0, KVW, row0, 1, 0, 0], m * NKV)?; }
            "attn" => if !g && s.dec == 0 { r.run("small", &[&s.q, &s.kc[l], &s.vc[l], &s.a], &[m, NH, NH / NKV, D, KVW, KVW, ald(), sc], m * NH)?; }
                else { let nt = m.div_ceil(128); let mut t0 = 0; while t0 < nt { let t1 = (t0 + flash_qt()).min(nt);
                    r.bar(); let kk = &r.kn.k["flash"]; let ds = r.pf.ds.get(r.ctx, kk, &[&s.q, &s.kc[l], &s.vc[l], &s.a])?;
                    r.ctx.dispatch(r.cb, kk, ds, &push_u32(&[m, if g { s.lt + s.n } else { s.dec + 1 }, NH, D, KVW, KVW, ald(), sc, NH / NKV, t0]), (t1 - t0) * NH); t0 = t1; } r.bar(); },
            "o" if s.dec > 0 && !g => self.gm(r, w[k("o")], &s.a, ald(), &s.y2, D, 0, true)?,
            "o" => r.gemm(ar, p, w[k("o")], &s.a, ald(), m, &s.y2, D, 0, true)?,
            "res1" | "res2" => r.ew(0, m, D, &s.x, &s.y2, D, 0, &s.a, 0, p, NOG)?,
            "norm2" => r.run("norm", &[&s.x, &s.a, p, p], &[D, D, ald(), o[k("post")], 0, 0, eps, 0], m)?,
            "swiglu" => self.swiglu(r, s, w, g, m)?,
            _ if s.dec > 0 && !g => self.gm(r, w[k("down")], &s.h, hld(), &s.y2, D, 0, true)?,
            _ => r.gemm(ar, p, w[k("down")], &s.h, hld(), m, &s.y2, D, 0, true)?,
        }
        Ok(())
    }
    /// Per-op GPU time of gen layer 0 (each op in its own submission, median of `reps`).
    pub fn prof(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &State, reps: usize) -> Result<Vec<(&'static str, f64)>> {
        let mut out = vec![];
        for op in Self::OPS { let mut t = vec![]; for _ in 0..reps { let mut err = Ok(()); t.push(ctx.run(|cb| { let r = Rec { ctx, pf, kn, cb }; err = self.layer_op(&r, s, 0, true, op); })?); err?; }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap()); out.push((op, t[t.len() / 2])); }
        Ok(out)
    }
    fn dump_set() -> Vec<usize> { std::env::var("COS_DUMP").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default() }
    fn stack(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &mut State, g: bool, full: bool) -> Result<f64> {
        let dl = Self::dump_set(); let rows = if g { s.n } else { s.lt } as usize; let lt_env = std::env::var("COS_LTIME").is_ok() && g; let mut lts = vec![];
        let mut gpu = 0.0; let step = if dl.is_empty() && (g || s.ds.is_none()) && !lt_env { lps() } else { 1 };
        for l0 in (0..self.nl).step_by(step) { let mut err = Ok(());
            let gl = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb }; let same: Option<usize> = std::env::var("COS_SAMEL").ok().map(|v| v.parse().unwrap_or(0)); for l in l0..(l0 + step).min(self.nl) { self.layer(&r, s, match same { Some(x) if g => x, _ => l }, g, full)?; } Ok(()) })(); })?; err?; gpu += gl; lts.push(gl);
            if !g { if let Some((r0, ds)) = &s.ds { if l0 < ds.len() { let x = s.x.as_f32_mut(); let d = D as usize;
                for (i, v) in ds[l0].iter().enumerate() { x[r0 * d + i] += *v; } } } }
            if dl.contains(&l0) && (g || full || l0 < self.nl - 1) { s.dumps.push((format!("L{l0}_{}", if g { "gen" } else { "und" }), s.x.as_f32()[..rows * D as usize].to_vec())); } }
        if lt_env { eprintln!("layer ms: {}", lts.iter().map(|t| format!("{:.1}", t * 1e3)).collect::<Vec<_>>().join(" ")); }
        Ok(gpu)
    }
    /// Und tower over the prompt (once per prompt): fills the K/V caches rows [0, lt). `full` also finishes the last
    /// layer and returns logits of the last token (AR text path).
    pub fn prefix(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &mut State, ids: &[u32], full: bool) -> Result<(f64, Option<Vec<f32>>)> {
        let e = self.embed(ids)?; self.prefix_emb(ctx, pf, kn, s, &e, full)
    }
    /// `prefix` from input embeddings [lt][4096] (understanding path: image rows replaced by ViT embeddings).
    pub fn prefix_emb(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &mut State, e: &[f32], full: bool) -> Result<(f64, Option<Vec<f32>>)> {
        s.x.as_f32_mut()[..e.len()].copy_from_slice(e);
        let g = self.stack(ctx, pf, kn, s, false, full)?;
        if !full { return Ok((g, None)) }
        let lt = s.lt as usize; let p = &self.params; let lm = self.st.wref("lm_head.weight")?; let mut err = Ok(());
        let g2 = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            r.run("norm", &[&s.x, &s.a, p, p], &[D, D, ald(), self.fin_u, 0, 0, EPS.to_bits(), 0], s.lt)?;
            // lm_head on the last row only (y holds (lt + PAD) x 24576 floats, not lt x 151936)
            r.pf.gemm(r.ctx, r.cb, &self.st.bufs[lm.arena], p, &prefill::GemmOp { w: lm, row0: 0, rows: lm.n, a: &s.a, lda: ald(), a_row0: s.lt - 1, m: 1, y: &s.y, ldy: lm.n, y_col: 0, acc: false })?; r.bar(); Ok(()) })(); })?; err?;
        let _ = lt; let n = lm.n as usize; Ok((g + g2, Some(s.y.as_f32()[..n].to_vec())))
    }
    /// Und KV-cache decode step: token embedding `e` [4096] at 3D position `pos`, appended at row s.dec (= tokens cached so
    /// far; the caller advances it). Returns last-token logits.
    pub fn decode_step(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &mut State, e: &[f32], pos: [f64; 3]) -> Result<(f64, Vec<f32>)> {
        anyhow::ensure!(s.dec > 0 && s.ds.is_none(), "decode_step: prefill first and clear deepstack");
        s.x.as_f32_mut()[..D as usize].copy_from_slice(e); mrope_row(pos[0], pos[1], pos[2], &mut s.rope_g.as_f32_mut()[..128]);
        let g = self.stack(ctx, pf, kn, s, false, true)?;
        let p = &self.params; let lm = self.st.wref("lm_head.weight")?; let mut err = Ok(());
        let g2 = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            r.run("norm", &[&s.x, &s.a, p, p], &[D, D, ald(), self.fin_u, 0, 0, EPS.to_bits(), 0], 1)?;
            self.gm(&r, lm, &s.a, ald(), &s.y, lm.n, 0, true)?; Ok(()) })(); })?; err?;
        if std::env::var("COS_DEC_TRACE").is_ok() { eprintln!("decode_step: tower {:.1} ms, norm+lm_head {:.1} ms", g * 1e3, g2 * 1e3); }
        Ok((g + g2, s.y.as_f32()[..lm.n as usize].to_vec()))
    }
    /// One gen-tower evaluation: packed latent tokens [n][192] at `timestep` -> velocity tokens [n][192] (returned).
    pub fn step(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &mut State, tok: &[f32], timestep: f64) -> Result<(Vec<f32>, f64)> {
        let n = s.n; let p = &self.params; let ar = &self.st.bufs;
        qimage::upload_f16(&s.lat, tok);
        let te = self.temb(timestep); let tb = s.tb.as_f32_mut();
        for i in 0..D as usize { tb[i] = bf16::from_f32(te[i]).to_f32() + self.proj_in_b[i] }
        let mut err = Ok(());
        let mut g = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            r.gemm(ar, p, self.proj_in, &s.lat, PD, n, &s.x, D, 0, true)?;
            r.ew(0, n, D, &s.x, &s.tb, 0, 0, &s.a, 0, p, NOG)?; Ok(()) })(); })?; err?;
        if !Self::dump_set().is_empty() { s.dumps.push(("in_gen".into(), s.x.as_f32()[..(n * D) as usize].to_vec())); }
        g += self.stack(ctx, pf, kn, s, true, true)?;
        let mut err = Ok(());
        g += ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            r.run("norm", &[&s.x, &s.a, p, p], &[D, D, ald(), self.fin_g, 0, 0, EPS.to_bits(), 0], n)?;
            r.gemm(ar, p, self.proj_out, &s.a, ald(), n, &s.v, PD, 0, true)?; Ok(()) })(); })?; err?;
        let v = s.v.as_f32()[..(n * PD) as usize].iter().enumerate().map(|(i, x)| x + self.proj_out_b[i % PD as usize]).collect();
        Ok((v, g))
    }
}
/// latent [48][t][h][w] -> tokens [(t, h/2, w/2)][(p, q, c)] (p, q = row / col inside the 2x2 patch)
pub fn patchify(lat: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
    let (hp, wp) = (h / 2, w / 2); let mut o = vec![0f32; lat.len()];
    for f in 0..t { for y in 0..hp { for x in 0..wp { let tk = (f * hp + y) * wp + x; for pp in 0..2 { for qq in 0..2 { for c in 0..LC {
        o[tk * 192 + (pp * 2 + qq) * LC + c] = lat[((c * t + f) * h + y * 2 + pp) * w + x * 2 + qq]; } } } } } }
    o
}
pub fn unpatchify(tok: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
    let (hp, wp) = (h / 2, w / 2); let mut o = vec![0f32; tok.len()];
    for f in 0..t { for y in 0..hp { for x in 0..wp { let tk = (f * hp + y) * wp + x; for pp in 0..2 { for qq in 0..2 { for c in 0..LC {
        o[((c * t + f) * h + y * 2 + pp) * w + x * 2 + qq] = tok[tk * 192 + (pp * 2 + qq) * LC + c]; } } } } } }
    o
}
pub fn f16s(v: &[f32]) -> Vec<f32> { v.iter().map(|x| f16::from_f32(*x).to_f32()).collect() }
