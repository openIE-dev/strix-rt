//! Gemma 4 vision encoder (gemma4v mmproj): bicubic resize to a 48-multiple grid, patch embed (CPU, F32 conv 16x16)
//! + learned x/y position tables, 16 ViT blocks on the GPU (RMS sandwich norms, Gemma4ClippableLinear clamps around
//! every Q8_0 linear, q/k RMSNorm + 2D NeoX RoPE theta 100, weightless v RMSNorm, non-causal attention scale 1,
//! GeGLU FFN), 3x3 average pool * sqrt(768), weightless RMSNorm, mm.input_projection -> LM-width embeddings.
//! Unverified against llama.cpp mtmd yet (needs a GPU session); see logs/g4/vision_*.
use crate::load_weights;
use anyhow::{bail, Result};
use ash::vk;
use prefill::{GemmOp, Pf};
use rtcore::{DsCache, WRef, Weights};
use vkrt::{push_u32, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
const NONE: u32 = 0xFFFF_FFFF;
const P: usize = 16;
const MERGE: usize = 3;

pub struct Vision<'a> {
    ctx: &'a Ctx, pub wt: Weights, pf: Pf, ds: DsCache,
    norm: Kernel, h: Kernel, qkr: Kernel, attn: Kernel, comb: Kernel, clamp: Kernel,
    pub e: u32, pub n_head: u32, pub ff: u32, pub n_layer: u32, pub out_dim: u32, eps: f32, theta: f32,
}

/// Bicubic (a = -0.5) resize of interleaved RGB u8 -> planar f32 in [-1, 1].
pub fn resize(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    let cub = |x: f32| { let a = -0.5f32; let x = x.abs(); if x <= 1.0 { (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0 } else if x < 2.0 { a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a } else { 0.0 } };
    let mut out = vec![0f32; 3 * ow * oh];
    for oy in 0..oh { for ox in 0..ow {
        let sx = (ox as f32 + 0.5) * w as f32 / ow as f32 - 0.5; let sy = (oy as f32 + 0.5) * h as f32 / oh as f32 - 0.5;
        let (x0, y0) = (sx.floor() as i64, sy.floor() as i64);
        for c in 0..3 {
            let mut acc = 0f32;
            for dy in -1..=2i64 { for dx in -1..=2i64 {
                let (xx, yy) = ((x0 + dx).clamp(0, w as i64 - 1) as usize, (y0 + dy).clamp(0, h as i64 - 1) as usize);
                acc += src[(yy * w + xx) * 3 + c] as f32 * cub(sx - (x0 + dx) as f32) * cub(sy - (y0 + dy) as f32);
            } }
            out[(c * oh + oy) * ow + ox] = (acc.clamp(0.0, 255.0) / 255.0) * 2.0 - 1.0;
        }
    } }
    out
}

/// Target size as llama.cpp mtmd calc_size_preserved_ratio: align 48 (patch 16 x merge 3), pixel budget [70, max_tok] tokens.
pub fn target(w: usize, h: usize, max_tok: usize) -> (usize, usize) {
    let f = (P * MERGE) as f32; let (wf, hf) = (w as f32, h as f32);
    let (minp, maxp) = (70.0 * f * f, max_tok as f32 * f * f);
    let rnd = |x: f32| ((x / f).round() * f) as usize; let ceil = |x: f32| ((x / f).ceil() * f) as usize; let floor = |x: f32| ((x / f).floor() * f) as usize;
    let (mut wb, mut hb) = (rnd(wf).max(f as usize), rnd(hf).max(f as usize));
    if (wb * hb) as f32 > maxp { let beta = (hf * wf / maxp).sqrt(); hb = floor(hf / beta).max(f as usize); wb = floor(wf / beta).max(f as usize); }
    else if ((wb * hb) as f32) < minp { let beta = (minp / (hf * wf)).sqrt(); hb = ceil(hf * beta); wb = ceil(wf * beta); }
    (wb, hb)
}

/// Read a binary PPM (P6, maxval 255).
pub fn read_ppm(path: &str) -> Result<(Vec<u8>, usize, usize)> {
    let b = std::fs::read(path)?;
    let mut f = Vec::new(); let mut i = 0;
    while f.len() < 4 {
        while i < b.len() && b[i].is_ascii_whitespace() { i += 1 }
        if b[i] == b'#' { while b[i] != b'\n' { i += 1 } continue }
        let s = i; while !b[i].is_ascii_whitespace() { i += 1 }
        f.push(String::from_utf8_lossy(&b[s..i]).to_string());
    }
    if f[0] != "P6" || f[3] != "255" { bail!("need P6 maxval 255 PPM") }
    let (w, h): (usize, usize) = (f[1].parse()?, f[2].parse()?);
    Ok((b[i + 1..i + 1 + w * h * 3].to_vec(), w, h))
}

impl<'a> Vision<'a> {
    pub fn load(ctx: &'a Ctx, path: &str) -> Result<Self> {
        let wt = load_weights(ctx, path)?;
        let u = |k: &str, d: u32| wt.g.u64(k).map(|v| v as u32).unwrap_or(d);
        let e = u("clip.vision.embedding_length", 768); let n_head = u("clip.vision.attention.head_count", 12);
        if e / n_head != 64 { bail!("vision head dim {} != 64", e / n_head) }
        let out_dim = wt.wref("mm.input_projection.weight")?.n;
        Ok(Vision { pf: Pf::new(ctx)?, ds: DsCache::default(),
            norm: ctx.kernel(spv!("g4_norm"), 4, 32, &[1024], None)?, h: ctx.kernel(spv!("g4_h"), 3, 36, &[], None)?,
            qkr: ctx.kernel(spv!("g4_vqkr"), 4, 28, &[], None)?, attn: ctx.kernel(spv!("g4_attn"), 5, 48, &[64], None)?,
            comb: ctx.kernel(spv!("g4_comb"), 2, 16, &[64], None)?, clamp: ctx.kernel(spv!("g4_clamp"), 1, 20, &[], None)?,
            e, n_head, ff: u("clip.vision.feed_forward_length", 3072), n_layer: u("clip.vision.block_count", 16), out_dim,
            eps: wt.g.f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6), theta: 100.0, ctx, wt })
    }
    fn sc(&self, name: &str, d: f32) -> f32 { self.wt.f(name).map(|o| self.wt.f32b.as_f32()[o as usize]).unwrap_or(d) }
    fn lim(&self, w: &str, io: &str) -> (f32, f32) { (self.sc(&format!("{w}.{io}_min"), -65504.0), self.sc(&format!("{w}.{io}_max"), 65504.0)) }

    /// CPU patch embed + position tables: img planar [3][H][W] in [-1,1] -> x [N][E] (N = (H/16)*(W/16), row-major).
    fn embed(&self, img: &[f32], iw: usize, ih: usize) -> Result<Vec<f32>> {
        let (nc, nr, e) = (iw / P, ih / P, self.e as usize);
        let f = self.wt.f32b.as_f32();
        let pw = &f[self.wt.f("v.patch_embd.weight")? as usize..][..e * 3 * P * P];
        let pe = &f[self.wt.f("v.position_embd.weight")? as usize..];
        let npos = pe.len().min(2 * 10240 * e) / (2 * e);
        let mut x = vec![0f32; nc * nr * e];
        std::thread::scope(|s| {
            for (ci, ch) in x.chunks_mut(e * nc).enumerate() {
                s.spawn(move || {
                    let mut pv = vec![0f32; 3 * P * P];
                    for px in 0..nc {
                        for c in 0..3 { for ky in 0..P { for kx in 0..P { pv[(c * P + ky) * P + kx] = img[(c * ih + ci * P + ky) * iw + px * P + kx]; } } }
                        let o = &mut ch[px * e..(px + 1) * e];
                        for (co, v) in o.iter_mut().enumerate() {
                            let wr = &pw[co * 3 * P * P..(co + 1) * 3 * P * P];
                            *v = wr.iter().zip(&pv).map(|(a, b)| a * b).sum::<f32>() + pe[px.min(npos - 1) * e + co] + pe[(npos + ci.min(npos - 1)) * e + co];
                        }
                    }
                });
            }
        });
        Ok(x)
    }

    fn set(&self, k: &Kernel, b: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(self.ctx, k, b) }

    /// Encode an RGB u8 image; returns (embeddings [T][out_dim], pooled cols, pooled rows).
    pub fn encode(&self, rgb: &[u8], w: usize, h: usize, max_tok: usize) -> Result<(Vec<f32>, usize, usize)> {
        let (tw, th) = target(w, h, max_tok);
        let img = resize(rgb, w, h, tw, th);
        let x0 = self.embed(&img, tw, th)?;
        let (nc, nr) = (tw / P, th / P); let n = (nc * nr) as u32;
        let (c, e, ff) = (self.ctx, self.e, self.ff);
        let al = ff.max(e).max(self.out_dim).next_multiple_of(64);
        let ldp = 3 * e; let ldy = (2 * ff).max(ldp);
        let fb = |k: u64| c.buffer(k.max(64) * 4);
        let (x, t, y, ao) = (fb(n as u64 * e as u64)?, fb(n as u64 * e as u64)?, fb(n as u64 * ldy as u64)?, fb(n as u64 * e as u64)?);
        let a = c.buffer((n as u64 + 256) * al as u64 * 2)?;
        let (kc, vc) = (c.buffer(n as u64 * e as u64 * 2)?, c.buffer(n as u64 * e as u64 * 2)?);
        let part = fb(n as u64 * self.n_head as u64 * 66)?;
        let aux = c.buffer(4096)?;
        x.as_f32_mut()[..x0.len()].copy_from_slice(&x0);
        let wt = &self.wt; let f32b = &wt.f32b;
        let wr = |nm: &str| -> Result<WRef> { wt.wref(nm) };
        // linear: clamp(in) -> f16 A -> GEMM -> clamp(out) at y[:, col..col+rows]
        let lin = |cb: vk::CommandBuffer, nm: &str, inp: &Buffer, k: u32, xcs: u32, mode: u32, out: &Buffer, ldo: u32, col: u32, m: u32| -> Result<()> {
            let (lo, hi) = self.lim(nm, "input");
            let ds = self.set(&self.h, &[inp, inp, &a])?;
            c.dispatch3(cb, &self.h, ds, &push_u32(&[k, al, xcs, mode, 1f32.to_bits(), 0, 0, lo.to_bits(), hi.to_bits()]), k.div_ceil(256), m, 1); c.barrier(cb);
            let w = wr(&format!("{nm}.weight"))?;
            self.pf.gemm(c, cb, &wt.arenas[w.arena], f32b, &GemmOp { w, row0: 0, rows: w.n, a: &a, lda: al, a_row0: 0, m, y: out, ldy: ldo, y_col: col, acc: false })?; c.barrier(cb);
            let (lo, hi) = self.lim(nm, "output");
            if lo > -65504.0 || hi < 65504.0 {
                let ds = self.set(&self.clamp, &[out])?;
                c.dispatch3(cb, &self.clamp, ds, &push_u32(&[w.n, ldo, col, lo.to_bits(), hi.to_bits()]), w.n.div_ceil(256), m, 1); c.barrier(cb);
            }
            Ok(())
        };
        let norm = |cb: vk::CommandBuffer, inp: &Buffer, out: &Buffer, w: u32, mode: u32| -> Result<()> {
            let ds = self.set(&self.norm, &[inp, out, &aux, f32b])?;
            c.dispatch3(cb, &self.norm, ds, &push_u32(&[e, w, self.eps.to_bits(), mode, NONE, 0, e, e]), 1, n, 1); c.barrier(cb); Ok(())
        };
        for l in 0..self.n_layer {
            let b = |s: &str| format!("v.blk.{l}.{s}");
            let fo = |s: &str| wt.f(&b(s));
            let mut err = Ok(());
            let rec = c.record(|cb| { err = (|| -> Result<()> {
                c.barrier(cb);
                norm(cb, &x, &t, fo("ln1.weight")?, 0)?;
                for (j, p) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() { lin(cb, &b(p), &t, e, e, 0, &y, ldp, j as u32 * e, n)?; }
                let ds = self.set(&self.qkr, &[&y, &kc, &vc, f32b])?;
                c.dispatch3(cb, &self.qkr, ds, &push_u32(&[self.n_head, ldp, fo("attn_q_norm.weight")?, fo("attn_k_norm.weight")?, self.theta.to_bits(), self.eps.to_bits(), nc as u32]), self.n_head, n, 1); c.barrier(cb);
                let ds = self.set(&self.attn, &[&y, &kc, &vc, &part, &aux])?;
                c.dispatch3(cb, &self.attn, ds, &push_u32(&[self.n_head, self.n_head, 0, 1, 64, 0, 1f32.to_bits(), 0, n, NONE, n, ldp]), self.n_head, 1, n); c.barrier(cb);
                let ds = self.set(&self.comb, &[&part, &ao])?;
                c.dispatch3(cb, &self.comb, ds, &push_u32(&[1, 0, e, self.n_head]), self.n_head, n, 1); c.barrier(cb);
                lin(cb, &b("attn_out"), &ao, e, e, 0, &t, e, 0, n)?;
                norm(cb, &t, &x, fo("attn_post_norm.weight")?, 1)?;
                norm(cb, &x, &t, fo("ln2.weight")?, 0)?;
                lin(cb, &b("ffn_gate"), &t, e, e, 0, &y, 2 * ff, 0, n)?;
                lin(cb, &b("ffn_up"), &t, e, e, 0, &y, 2 * ff, ff, n)?;
                lin(cb, &b("ffn_down"), &y, ff, 2 * ff, 1, &t, e, 0, n)?;
                norm(cb, &t, &x, fo("ffn_post_norm.weight")?, 1)?;
                Ok(()) })(); })?; err?;
            c.submit(&rec)?;
        }
        // pool 3x3 (CPU) * sqrt(E), then weightless RMSNorm
        let (pc, pr) = (nc / MERGE, nr / MERGE); let eu = e as usize;
        let xv = &x.as_f32()[..n as usize * eu];
        let s = (e as f32).sqrt() / (MERGE * MERGE) as f32;
        let mut pooled = vec![0f32; pc * pr * eu];
        for py in 0..pr { for px in 0..pc { let o = &mut pooled[(py * pc + px) * eu..][..eu];
            for dy in 0..MERGE { for dx in 0..MERGE { let r = &xv[((py * MERGE + dy) * nc + px * MERGE + dx) * eu..][..eu]; for i in 0..eu { o[i] += r[i] * s } } }
            if let (Ok(bo), Ok(so)) = (wt.f("v.std_bias"), wt.f("v.std_scale")) { for i in 0..eu { o[i] = (o[i] - f32b.as_f32()[bo as usize + i]) * f32b.as_f32()[so as usize + i] } }
            let r = 1.0 / (o.iter().map(|v| v * v).sum::<f32>() / eu as f32 + self.eps).sqrt(); for v in o.iter_mut() { *v *= r }
        } }
        let m = (pc * pr) as u32;
        let tt = fb(m as u64 * self.out_dim as u64)?;
        x.as_f32_mut()[..pooled.len()].copy_from_slice(&pooled);
        let mut err = Ok(());
        let rec = c.record(|cb| { c.barrier(cb); err = lin(cb, "mm.input_projection", &x, e, e, 0, &tt, self.out_dim, 0, m); })?; err?;
        c.submit(&rec)?;
        Ok((tt.as_f32()[..(m * self.out_dim) as usize].to_vec(), pc, pr))
    }
}
