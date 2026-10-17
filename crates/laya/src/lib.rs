//! Laya (convaiinnovations/laya, Apache-2.0) on strix-rt: a typed decision model = ModernBERT-large encoder (28 layers,
//! d 1024, 16 heads x 64, GeGLU 2624, bias-free LayerNorms, RoPE theta 160k on every 3rd (global) layer and 10k with a
//! +-64 sliding window elsewhere, bidirectional) + decision head: type_emb[qtype] added, 2 torch TransformerEncoderLayers
//! (norm_first, ReLU FFN 4096, biases), a scorer LayerNorm -> Linear -> GELU -> Linear(1) read at each option's [MASK]
//! marker (softmax over a question's options / per-type temperature), and an act/escalate head on [CLS] (CPU).
//! Weights: ref/to_gguf.py (2-D F16 in GEMM arenas, 1-D F32). GEMMs use prefill's coop-matrix F16 GEMM; LayerNorm,
//! attention and activations are shaders/ly_*.comp. One question row (<= 512 tokens) per forward.
use anyhow::{anyhow, bail, Result};
use prefill::{GemmOp, Pf};
use rtcore::Weights;
use vkrt::{Buffer, Ctx};

const LN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ly_ln.spv"));
const ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ly_attn.spv"));
const ROPE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ly_rope.spv"));
const ACT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ly_act.spv"));
const NONE: u32 = u32::MAX;

pub struct Laya { pub ctx: Ctx, pub wt: Weights, pub pf: Pf, pub d: u32, pub nl: u32, pub nh: u32, pub ffn: u32, pub win: u32, pub every: u32, pub hl: u32,
    pub th_g: f32, pub th_l: f32, pub eps: f32, pub lmax: u32, x: Buffer, a: Buffer, h: Buffer, pub temperature: [f32; 3] }

pub struct Out { pub logits: Vec<f32>, pub act: [f32; 2], pub gpu_ms: f64 }

impl Laya {
    pub fn load(path: &str, lmax: u32) -> Result<Self> {
        let ctx = Ctx::new()?; let wt = Weights::load(&ctx, path)?; let mut pf = Pf::new(&ctx)?;
        pf.k.insert("ly_ln", ctx.kernel(LN, 4, 28, &[], None)?); pf.k.insert("ly_attn", ctx.kernel(ATTN, 3, 28, &[], Some(64))?); pf.k.insert("ly_rope", ctx.kernel(ROPE, 1, 16, &[], None)?); pf.k.insert("ly_act", ctx.kernel(ACT, 3, 24, &[], None)?);
        let u = |k: &str| wt.g.u64(&format!("laya.{k}")).map(|v| v as u32); let fl = |k: &str| wt.g.f32(&format!("laya.{k}"));
        let (d, ffn) = (u("hidden")?, u("ffn")?); if d % 64 != 0 || lmax > 1024 { bail!("d {d} / lmax {lmax}") }
        let rows = (lmax + 256) as u64; let wmax = (2 * ffn).max(4 * d).max(3 * d) as u64;
        let t = wt.f("temperature")? as usize; let tf = &wt.f32b.as_f32()[t..t + 3];
        Ok(Self { d, nl: u("layers")?, nh: u("heads")?, ffn, win: u("window")? / 2, every: u("global_every")?, hl: u("head_layers")?, th_g: fl("theta_global")?, th_l: fl("theta_local")?,
            eps: fl("norm_eps")?, lmax, x: ctx.buffer(rows * d as u64 * 4)?, a: ctx.buffer(rows * wmax * 2)?, h: ctx.buffer(rows * wmax * 4)?, temperature: [tf[0], tf[1], tf[2]], ctx, wt, pf })
    }
    fn f(&self, n: &str) -> Result<u32> { self.wt.f(n) }
    fn ln(&self, cb: ash_cb::Cb, l: u32, w: &str, b: Option<&str>, mode: u32, add: u32, a_ld: u32) -> Result<()> {
        let b = match b { Some(n) => self.f(n)?, None => NONE };
        self.pf.run(&self.ctx, "ly_ln", &[&self.x, &self.a, &self.wt.f32b, &self.x], &[self.d, a_ld, self.f(w)?, b, self.eps.to_bits(), mode, add], l, cb)
    }
    fn gemm(&self, cb: ash_cb::Cb, name: &str, l: u32, lda: u32, y: &Buffer, ldy: u32, acc: bool) -> Result<()> {
        let w = self.wt.wref(name)?; if w.k != lda && !(lda >= w.k) { bail!("{name}: k {} vs lda {lda}", w.k) }
        self.pf.gemm(&self.ctx, cb, &self.wt.arenas[w.arena], &self.wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &self.a, lda, a_row0: 0, m: l, y, ldy, y_col: 0, acc })
    }
    fn act(&self, cb: ash_cb::Cb, l: u32, n: u32, ldh: u32, lda: u32, b: Option<&str>, mode: u32, h: &Buffer) -> Result<()> {
        let b = match b { Some(nm) => self.f(nm)?, None => NONE };
        self.pf.run(&self.ctx, "ly_act", &[h, &self.a, &self.wt.f32b], &[l * n, n, ldh, lda, b, mode], (l * n).div_ceil(256), cb)
    }
    fn attn(&self, cb: ash_cb::Cb, l: u32, b: Option<&str>, theta: f32, win: u32) -> Result<()> {
        let b = match b { Some(nm) => self.f(nm)?, None => NONE };
        if theta > 0.0 { self.pf.run(&self.ctx, "ly_rope", &[&self.h], &[l, self.d, self.nh, theta.to_bits()], (l * self.nh * 32).div_ceil(256), cb)?; self.ctx.barrier(cb); }
        self.pf.run(&self.ctx, "ly_attn", &[&self.h, &self.a, &self.wt.f32b], &[l, self.d, self.d, b, theta.to_bits(), win, self.nh], l * self.nh, cb)
    }
    /// Encoder + head for one row. Returns raw option logits at `markers` (before temperature) and the act logits.
    /// stop: 1 = after the encoder (x = encoder last hidden), 2 = after the head layers (x = head output), 0 = full.
    pub fn forward(&self, ids: &[u32], qtype: u32, markers: &[usize], stop: u32) -> Result<Out> {
        let l = ids.len() as u32; if l == 0 || l > self.lmax { bail!("row length {l}") }
        let d = self.d as usize;
        let t = self.wt.g.tensor("encoder.embeddings.tok_embeddings.weight").ok_or_else(|| anyhow!("no tok_embeddings"))?;
        let data = self.wt.g.data(t); let rb = data.len() / t.dims[1] as usize; let x = self.x.as_f32_mut();
        for (i, &tk) in ids.iter().enumerate() { gguf::dequant::dequant_row(t.ty, &data[tk as usize * rb..(tk as usize + 1) * rb], &mut x[i * d..(i + 1) * d]); }
        let (dd, ff) = (self.d, self.ffn); let mut err = None;
        let gpu = self.ctx.run(|cb| { let r = (|| -> Result<()> {
            let bar = || self.ctx.barrier(cb);
            self.ln(cb, l, "encoder.embeddings.norm.weight", None, 2, NONE, dd)?; bar();
            for i in 0..self.nl {
                let p = |s: &str| format!("encoder.layers.{i}.{s}");
                let glob = i % self.every == 0;
                if i == 0 { self.ln_identity(cb, l)?; } else { self.ln(cb, l, &p("attn_norm.weight"), None, 1, NONE, dd)?; } bar();
                self.gemm(cb, &p("attn.Wqkv.weight"), l, dd, &self.h, 3 * dd, false)?; bar();
                self.attn(cb, l, None, if glob { self.th_g } else { self.th_l }, if glob { 0 } else { self.win })?; bar();
                self.gemm(cb, &p("attn.Wo.weight"), l, dd, &self.x, dd, true)?; bar();
                self.ln(cb, l, &p("mlp_norm.weight"), None, 1, NONE, dd)?; bar();
                self.gemm(cb, &p("mlp.Wi.weight"), l, dd, &self.h, 2 * ff, false)?; bar();
                self.act(cb, l, ff, 2 * ff, ff, None, 0, &self.h)?; bar();
                self.gemm(cb, &p("mlp.Wo.weight"), l, ff, &self.x, dd, true)?; bar();
            }
            let te = self.f("type_emb.weight")? + qtype * dd;
            self.ln(cb, l, "encoder.final_norm.weight", None, 2, if stop == 1 { NONE } else { te }, dd)?; bar();
            if stop == 1 { return Ok(()) }
            for i in 0..self.hl {
                let p = |s: &str| format!("head.layers.{i}.{s}");
                self.ln(cb, l, &p("norm1.weight"), Some(&p("norm1.bias")), 1, NONE, dd)?; bar();
                self.gemm(cb, &p("self_attn.in_proj_weight"), l, dd, &self.h, 3 * dd, false)?; bar();
                self.attn(cb, l, Some(&p("self_attn.in_proj_bias")), 0.0, 0)?; bar();
                self.gemm(cb, &p("self_attn.out_proj.weight"), l, dd, &self.x, dd, true)?; bar();
                self.act(cb, l, dd, dd, dd, Some(&p("self_attn.out_proj.bias")), 2, &self.x)?; bar();
                self.ln(cb, l, &p("norm2.weight"), Some(&p("norm2.bias")), 1, NONE, dd)?; bar();
                self.gemm(cb, &p("linear1.weight"), l, dd, &self.h, 4 * dd, false)?; bar();
                self.act(cb, l, 4 * dd, 4 * dd, 4 * dd, Some(&p("linear1.bias")), 1, &self.h)?; bar();
                self.gemm(cb, &p("linear2.weight"), l, 4 * dd, &self.x, dd, true)?; bar();
                self.act(cb, l, dd, dd, dd, Some(&p("linear2.bias")), 2, &self.x)?; bar();
            }
            if stop == 2 { return Ok(()) }
            self.ln(cb, l, "scorer.0.weight", Some("scorer.0.bias"), 1, NONE, dd)?; bar();
            self.gemm(cb, "scorer.1.weight", l, dd, &self.h, dd, false)?; bar();
            self.act(cb, l, dd, dd, dd, Some("scorer.1.bias"), 5, &self.h)?; bar();
            Ok(()) })(); if let Err(e) = r { err = Some(e) } })?;
        if let Some(e) = err { return Err(e) }
        if stop != 0 { return Ok(Out { logits: vec![], act: [0.0; 2], gpu_ms: gpu * 1e3 }) }
        // scorer.3 (1024 -> 1) at the markers and the act head on [CLS]: tiny, on the CPU from the GPU's outputs
        let f32b = self.wt.f32b.as_f32();         let g = self.h.as_f32(); let w3 = self.w16("scorer.3.weight")?; let b3 = f32b[self.f("scorer.3.bias")? as usize];
        let logits: Vec<f32> = markers.iter().map(|&m| g[m * d..(m + 1) * d].iter().zip(&w3).map(|(a, b)| a * b).sum::<f32>() + b3).collect();
        let p = softmax(&logits); let k = markers.len() as f32;
        let mut s = p.clone(); s.sort_by(|a, b| b.total_cmp(a)); let (t1, t2) = (s[0], if s.len() > 1 { s[1] } else { 0.0 });
        let ent = -p.iter().map(|v| v * v.max(1e-9).ln()).sum::<f32>() / k.max(2.0).ln();
        let x = self.x.as_f32(); let mut inp: Vec<f32> = x[..d].to_vec(); inp.extend([t1, t1 - t2, ent, k.max(2.0) / 255.0]);
        let (w0, b0, w2, b2) = (self.f32m("act_head.0.weight")?, self.f32m("act_head.0.bias")?, self.w16("act_head.2.weight").or_else(|_| self.f32m("act_head.2.weight"))?, self.f32m("act_head.2.bias")?);
        let hid: Vec<f32> = (0..b0.len()).map(|j| { let v = b0[j] + w0[j * inp.len()..(j + 1) * inp.len()].iter().zip(&inp).map(|(a, b)| a * b).sum::<f32>(); 0.5 * v * (1.0 + erf(v / 2f32.sqrt())) }).collect();
        let act = [0, 1].map(|o| b2[o] + w2[o * hid.len()..(o + 1) * hid.len()].iter().zip(&hid).map(|(a, b)| a * b).sum::<f32>());
        Ok(Out { logits, act, gpu_ms: gpu * 1e3 })
    }
    fn ln_identity(&self, cb: ash_cb::Cb, l: u32) -> Result<()> {   // layer 0 has no attn_norm: a = f16(x)
        self.act(cb, l, self.d, self.d, self.d, None, 4, &self.x)
    }
    fn f32m(&self, n: &str) -> Result<Vec<f32>> { let t = self.wt.g.tensor(n).ok_or_else(|| anyhow!("no {n}"))?; let mut v = vec![0f32; t.n_elems() as usize];
        gguf::dequant::dequant_row(t.ty, self.wt.g.data(t), &mut v); Ok(v) }
    fn w16(&self, n: &str) -> Result<Vec<f32>> { self.f32m(n) }
    /// encoder last hidden (stop=1) or head output (stop=2) rows, from x
    pub fn x_rows(&self, l: usize) -> Vec<f32> { self.x.as_f32()[..l * self.d as usize].to_vec() }
}

pub fn softmax(z: &[f32]) -> Vec<f32> { let m = z.iter().cloned().fold(f32::MIN, f32::max); let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect(); let s: f32 = e.iter().sum(); e.iter().map(|v| v / s).collect() }
fn erf(x: f32) -> f32 { let z = x.abs() as f64; let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    (if x >= 0.0 { 1.0 - r } else { r - 1.0 }) as f32 }

mod ash_cb { pub type Cb = ash::vk::CommandBuffer; }
