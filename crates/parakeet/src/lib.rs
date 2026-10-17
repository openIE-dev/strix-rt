//! Parakeet Redux (FastConformer encoder + TDT decoder, thrush-ternary-v2 weights) on strix-rt.
//! Encoder: Vulkan; every linear goes through prefill's coop-matrix GEMM (ternary layers repacked to PQ2_0 at load).
//! Front end (log-mel) and the TDT greedy decoder (2-layer LSTM + joint) run on the CPU.
pub mod audio;
pub mod pool;
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use gguf::GgmlType;
use half::f16;
use prefill::{GemmOp, Pf};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{push_u32, Buffer, Ctx, Kernel};

const D: u32 = 1024; const FF: u32 = 4096; const H: u32 = 8; const KW: u32 = 9; const C_SUB: u32 = 256;
const DP: usize = 640; const VOCAB: usize = 8193; const BLANK: usize = 8192; const DURS: [usize; 5] = [0, 1, 2, 3, 4];
const PQ2: GgmlType = GgmlType::Other(142);

struct St { mm: memmap2::Mmap, base: usize, t: HashMap<String, (String, Vec<usize>, usize, usize)> }
impl St {
    fn open(p: &str) -> Result<Self> {
        let f = std::fs::File::open(p).with_context(|| p.to_string())?; let mm = unsafe { memmap2::Mmap::map(&f)? };
        let n = u64::from_le_bytes(mm[0..8].try_into()?) as usize; let h: serde_json::Value = serde_json::from_slice(&mm[8..8 + n])?;
        let mut t = HashMap::new();
        for (k, v) in h.as_object().ok_or(anyhow!("st header"))? { if k == "__metadata__" { continue }
            let o = &v["data_offsets"]; t.insert(k.clone(), (v["dtype"].as_str().unwrap().to_string(), v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(), o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize)); }
        Ok(St { mm, base: 8 + n, t })
    }
    fn raw(&self, k: &str) -> Result<(&str, &[usize], &[u8])> { let (d, s, a, b) = self.t.get(k).ok_or_else(|| anyhow!("missing tensor {k}"))?; Ok((d, s, &self.mm[self.base + a..self.base + b])) }
    fn f32(&self, k: &str) -> Result<Vec<f32>> {
        let (d, _, b) = self.raw(k)?;
        Ok(match d { "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(), _ => bail!("{k}: dtype {d}") })
    }
}

/// Repack one thrush-ternary-v2 matrix (qweight u8 [n][ceil(k/5)], base-3 LSD-first digits; scales f16 [n][k/128])
/// into PQ2_0 rows: per 128 weights one f16 scale + 32 bytes of 2-bit codes (code 0/1/2 = -1/0/+1).
fn repack_ternary(q: &[u8], sc: &[u8], n: usize, k: usize) -> Vec<u8> {
    let nb = q.len() / n; let g = k / 128; let rb = g * 34;
    let mut lut = [[0u8; 5]; 256]; for b in 0..243usize { let mut v = b; for d in 0..5 { lut[b][d] = (v % 3) as u8; v /= 3; } }
    let mut out = vec![0u8; n * rb];
    let nth = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8);
    let rows_per = n.div_ceil(nth);
    std::thread::scope(|s| { for (ci, ch) in out.chunks_mut(rows_per * rb).enumerate() { let lut = &lut; s.spawn(move || {
        let mut codes = vec![0u8; nb * 5];
        for (ri, orow) in ch.chunks_mut(rb).enumerate() { let r = ci * rows_per + ri;
            for (j, &b) in q[r * nb..(r + 1) * nb].iter().enumerate() { codes[j * 5..j * 5 + 5].copy_from_slice(&lut[b as usize]); }
            for bi in 0..g { let o = &mut orow[bi * 34..bi * 34 + 34]; o[0] = sc[(r * g + bi) * 2]; o[1] = sc[(r * g + bi) * 2 + 1];
                for j in 0..32 { let e = bi * 128 + j * 4; o[2 + j] = codes[e] | codes[e + 1] << 2 | codes[e + 2] << 4 | codes[e + 3] << 6; } }
        } }); } });
    out
}

struct Layer { ff1: [WRef; 2], ff2: [WRef; 2], q: WRef, k: WRef, v: WRef, o: WRef, pos: WRef, pw1: WRef, pw2: WRef, p: HashMap<&'static str, u32> }

pub struct Scratch { cap: u32, qu: Buffer, qv: Buffer, kh: Buffer, vt: Buffer, ph: Buffer, ac: Buffer, bd: Buffer, pa: Buffer, ao: Buffer, mel: Buffer, s1: Buffer, sy: Buffer, sa: Buffer, x: Buffer, a: Buffer, pos: Buffer, y: Buffer, y2: Buffer, pp: Buffer, ep: Buffer }

pub struct Model {
    pub arena: Buffer, pub params: Buffer, layers: Vec<Layer>, sub: HashMap<&'static str, u32>, sub_pw: [WRef; 2], sub_lin: WRef, proj: WRef,
    k: HashMap<&'static str, Kernel>, scratch: Option<Scratch>,
    // decoder (CPU, f32)
    emb: Vec<f32>, lstm: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>, dproj: (Vec<f32>, Vec<f32>), head: (Vec<f32>, Vec<f32>),
    pub vocab: Vec<String>, pub load_s: f64,
}

pub struct Timing { pub mel_ms: f64, pub enc_ms: f64, pub enc_gpu_ms: f64, pub dec_ms: f64 }

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }

impl Model {
    pub fn load(ctx: &Ctx, dir: &str, vocab_path: &str) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let st = St::open(&format!("{dir}/model.safetensors"))?;
        let nl = (0..).take_while(|i| st.t.contains_key(&format!("encoder.layers.{i}.norm_out.weight"))).count();
        // --- weight arena: ternary -> PQ2_0, dense f16 kept as f16 rows
        let mut blobs: Vec<(String, Vec<u8>, GgmlType, u32, u32)> = vec![];
        let mut tern = |name: &str| -> Result<()> {
            let (_, sh, q) = st.raw(&format!("{name}.qweight"))?; let (_, ss, s) = st.raw(&format!("{name}.scales"))?;
            let (n, k) = (sh[0], ss[1] * 128); blobs.push((name.to_string(), repack_ternary(q, s, n, k), PQ2, k as u32, n as u32)); Ok(()) };
        let lin = ["feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2", "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj", "self_attn.relative_k_proj", "conv.pointwise_conv1", "conv.pointwise_conv2"];
        for l in 0..nl { for m in lin { tern(&format!("encoder.layers.{l}.{m}"))?; } }
        for name in ["encoder.subsampling.layers.3", "encoder.subsampling.layers.6", "encoder.subsampling.linear", "encoder_projector"] {
            let (d, sh, b) = st.raw(&format!("{name}.weight"))?; if d != "F16" { bail!("{name}: {d}") }
            blobs.push((name.to_string(), b.to_vec(), GgmlType::F16, sh[1] as u32, sh[0] as u32)); }
        let total: u64 = blobs.iter().map(|b| (b.1.len() as u64 + 255) & !255).sum();
        let arena = ctx.buffer(total)?; let mut w: HashMap<String, WRef> = HashMap::new(); let mut off = 0u64;
        for (name, bytes, ty, k, n) in blobs { arena.bytes_mut()[off as usize..off as usize + bytes.len()].copy_from_slice(&bytes);
            w.insert(name, WRef { arena: 0, off, ty, k, n, e: 1, bytes: bytes.len() as u64 }); off += (bytes.len() as u64 + 255) & !255; }
        // --- small f32 params
        let mut pv: Vec<f32> = vec![];
        let put = |pv: &mut Vec<f32>, k: &str| -> Result<u32> { let o = pv.len() as u32; pv.extend(st.f32(k)?); while pv.len() % 64 != 0 { pv.push(0.0) } Ok(o) };
        let mut layers = vec![];
        for l in 0..nl { let pre = format!("encoder.layers.{l}."); let mut p = HashMap::new();
            for (key, t) in [("ff1_g", "norm_feed_forward1.weight"), ("ff1_b", "norm_feed_forward1.bias"), ("ff2_g", "norm_feed_forward2.weight"), ("ff2_b", "norm_feed_forward2.bias"),
                ("att_g", "norm_self_att.weight"), ("att_b", "norm_self_att.bias"), ("conv_g", "norm_conv.weight"), ("conv_b", "norm_conv.bias"), ("out_g", "norm_out.weight"), ("out_b", "norm_out.bias"),
                ("u", "self_attn.bias_u"), ("v", "self_attn.bias_v"), ("dw", "conv.depthwise_conv.weight"), ("bn_m", "conv.norm.running_mean"), ("bn_v", "conv.norm.running_var"),
                ("bn_g", "conv.norm.weight"), ("bn_b", "conv.norm.bias")] { p.insert(key, put(&mut pv, &format!("{pre}{t}"))?); }
            let g = |m: &str| w[&format!("{pre}{m}")];
            layers.push(Layer { ff1: [g(lin[0]), g(lin[1])], ff2: [g(lin[2]), g(lin[3])], q: g(lin[4]), k: g(lin[5]), v: g(lin[6]), o: g(lin[7]), pos: g(lin[8]), pw1: g(lin[9]), pw2: g(lin[10]), p }); }
        let mut sub = HashMap::new();
        for (key, t) in [("w0", "layers.0.weight"), ("b0", "layers.0.bias"), ("w2", "layers.2.weight"), ("b2", "layers.2.bias"), ("b3", "layers.3.bias"), ("w5", "layers.5.weight"),
            ("b5", "layers.5.bias"), ("b6", "layers.6.bias"), ("blin", "linear.bias")] { sub.insert(key, put(&mut pv, &format!("encoder.subsampling.{t}"))?); }
        sub.insert("bproj", put(&mut pv, "encoder_projector.bias")?);
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        // --- kernels
        let mut k = HashMap::new();
        k.insert("ln", ctx.kernel(spv!("pk_ln"), 3, 24, &[], None)?);
        k.insert("ew", ctx.kernel(spv!("pk_ew"), 4, 32, &[], None)?);
        k.insert("dwc", ctx.kernel(spv!("pk_dwc"), 3, 40, &[], None)?);
        k.insert("attn", ctx.kernel(spv!("pk_attn"), 4, 28, &[], None)?);
        k.insert("aprep", ctx.kernel(spv!("pk_aprep"), 8, 28, &[], None)?);
        k.insert("smax", ctx.kernel(spv!("pk_smax"), 3, 16, &[], None)?);
        k.insert("sub", ctx.kernel(spv!("pk_sub"), 4, 28, &[], None)?);
        // --- decoder
        let mut lstm = vec![];
        for l in 0..2 { let (bi, bh) = (st.f32(&format!("decoder.lstm.bias_ih_l{l}"))?, st.f32(&format!("decoder.lstm.bias_hh_l{l}"))?);
            lstm.push((st.f32(&format!("decoder.lstm.weight_ih_l{l}"))?, st.f32(&format!("decoder.lstm.weight_hh_l{l}"))?, bi.iter().zip(&bh).map(|(a, b)| a + b).collect())); }
        let vocab: Vec<String> = serde_json::from_str(&std::fs::read_to_string(vocab_path)?)?;
        Ok(Model { arena, params, layers, sub, sub_pw: [w["encoder.subsampling.layers.3"], w["encoder.subsampling.layers.6"]], sub_lin: w["encoder.subsampling.linear"],
            proj: w["encoder_projector"], k, scratch: None, emb: st.f32("decoder.embedding.weight")?, lstm,
            dproj: (st.f32("decoder.decoder_projector.weight")?, st.f32("decoder.decoder_projector.bias")?), head: (st.f32("joint.head.weight")?, st.f32("joint.head.bias")?),
            vocab, load_s: t0.elapsed().as_secs_f64() })
    }

    fn scratch(&mut self, ctx: &Ctx, t0: u32) -> Result<()> {
        if self.scratch.as_ref().map(|s| s.cap >= t0).unwrap_or(false) { return Ok(()) }
        let cap = t0.next_power_of_two().max(1024); let (l1, l2, t) = subs(cap);
        let f = |n: u64| ctx.buffer(n * 4); let h = |n: u64| ctx.buffer(n * 2);
        let pad = 256u64; let t = t as u64; let rows = (2 * t + pad).max(l2 as u64 * 32 + pad);
        let (hh, tk) = (H as u64, (t + 63) / 64 * 64);
        self.scratch = Some(Scratch { cap, qu: h((hh * t + pad) * 128)?, qv: h((hh * t + pad) * 128)?, kh: h(hh * t * 128)?, vt: h(hh * 128 * tk)?, ph: h(hh * 2 * t * 128)?,
            ac: f(t * hh * t)?, bd: f(t * hh * 2 * t)?, pa: h((hh * t + pad) * tk)?, ao: f(t * D as u64)?, mel: f(cap as u64 * 128)?, s1: f(l1 as u64 * 64 * 256)?, sy: f(l2 as u64 * 32 * 256)?, sa: h((l2 as u64 * 32 + pad) * 256)?,
            x: f(t * D as u64)?, a: h(rows * FF as u64)?, pos: h((2 * t + pad) * D as u64)?, y: f(t * FF as u64)?, y2: f(t * D as u64)?, pp: f(2 * t * D as u64)?, ep: f(t * DP as u64)? });
        Ok(())
    }

    /// Encoder on the GPU: features [t0][128] -> projected encoder output [T][640] (and the encoder hidden state [T][1024] if `hidden`).
    pub fn encode(&mut self, ctx: &Ctx, pf: &Pf, mel: &[f32], hidden: bool) -> Result<(Vec<f32>, Option<Vec<f32>>, u32, f64)> {
        let t0 = (mel.len() / 128) as u32; self.scratch(ctx, t0)?;
        let (l1, l2, t) = subs(t0); if t > 4096 { bail!("audio too long for one pass ({t} frames > 4096)") }
        let s = self.scratch.as_ref().unwrap(); s.mel.as_f32_mut()[..mel.len()].copy_from_slice(mel);
        // relative positional embedding, rows k <-> position T-1-k, interleaved (sin, cos)
        { let pb = unsafe { std::slice::from_raw_parts_mut(s.pos.ptr as *mut f16, (2 * t as usize - 1) * D as usize) };
          let inv: Vec<f64> = (0..D as usize / 2).map(|m| 1.0 / 10000f64.powf(2.0 * m as f64 / D as f64)).collect();
          let nth = 16usize; let rows = 2 * t as usize - 1; let per = rows.div_ceil(nth);
          std::thread::scope(|sc| { for (ci, ch) in pb.chunks_mut(per * D as usize).enumerate() { let inv = &inv; sc.spawn(move || {
            for (ri, row) in ch.chunks_mut(D as usize).enumerate() { let ps = (t as i64 - 1 - (ci * per + ri) as i64) as f64;
              for m in 0..D as usize / 2 { let (sn, cs) = (ps * inv[m]).sin_cos(); row[2 * m] = f16::from_f32(sn as f32); row[2 * m + 1] = f16::from_f32(cs as f32); } } }); } }); }
        let (k, p, ar) = (&self.k, &self.params, &self.arena);
        let mut err: Result<()> = Ok(());
        let gpu = ctx.run(|cb| { err = (|| -> Result<()> {
            // PK_DUP (profiling only, results invalid): issue a kernel class twice; bit 1 gemm, 2 attention, 4 ln/ew/dwc, 8 subsampling
            let dup = prefill::env("PK_DUP", 0); let attn_gemm = prefill::env("PK_ATTN_GEMM", 1) != 0;
            let cls = |name: &str| match name { "attn" | "aprep" | "smax" => 2, "sub" => 8, _ => 4 };
            let run = |name: &str, bufs: &[&Buffer], push: &[u32], groups: u32| -> Result<()> { let kk = &k[name]; let ds = pf.ds.get(ctx, kk, bufs)?;
                for _ in 0..1 + (dup & cls(name) != 0) as u32 { ctx.dispatch(cb, kk, ds, &push_u32(push), groups); ctx.barrier(cb); } Ok(()) };
            let gemm = |w: WRef, a: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32, y_col: u32| -> Result<()> {
                for _ in 0..1 + (dup & 1) { pk_gemm(pf, ctx, cb, ar, p, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m, y, ldy, y_col, acc: false })?; } Ok(()) };
            let g = |n: u32| n.div_ceil(256);
            let sb = &self.sub;
            // subsampling
            run("sub", &[&s.mel, &s.s1, &s.sa, p], &[0, t0, 128, l1, 64, sb["w0"], sb["b0"]], g(l1 * 64 * C_SUB))?;
            run("sub", &[&s.s1, &s.sy, &s.sa, p], &[1, l1, 64, l2, 32, sb["w2"], sb["b2"]], g(l2 * 32 * C_SUB))?;
            gemm(self.sub_pw[0], &s.sa, C_SUB, l2 * 32, &s.sy, C_SUB, 0)?; ctx.barrier(cb);
            run("sub", &[&s.s1, &s.sy, &s.sa, p], &[2, 0, 0, l2, 32, 0, sb["b3"]], g(l2 * 32 * C_SUB))?;
            run("sub", &[&s.sy, &s.s1, &s.sa, p], &[1, l2, 32, t, 16, sb["w5"], sb["b5"]], g(t * 16 * C_SUB))?;
            gemm(self.sub_pw[1], &s.sa, C_SUB, t * 16, &s.sy, C_SUB, 0)?; ctx.barrier(cb);
            run("sub", &[&s.sy, &s.s1, &s.a, p], &[3, 0, 0, t, 16, 0, sb["b6"]], g(t * 16 * C_SUB))?;
            gemm(self.sub_lin, &s.a, 4096, t, &s.x, D, 0)?; ctx.barrier(cb);
            let ew = |mode: u32, n: u32, dd: u32, y: &Buffer, ldy: u32, ycol: u32, sc: f32, boff: u32, lda: u32| run("ew", &[&s.x, y, &s.a, p], &[mode, n, dd, ldy, ycol, sc.to_bits(), boff, lda], g(n));
            ew(2, t * D, D, &s.y, 0, 0, 0.0, sb["blin"], 0)?;
            let ln = |gk: u32, bk: u32, mode: u32| run("ln", &[&s.x, &s.a, p], &[D, D, gk, bk, mode, 1e-5f32.to_bits()], t);
            for l in &self.layers { let lp = &l.p;
                for (ff, gn, bn) in [(&l.ff1, "ff1_g", "ff1_b"), (&l.ff2, "ff2_g", "ff2_b")] {
                    if gn == "ff2_g" { // conv module and attention come between the two half-step FFNs
                        ln(lp["att_g"], lp["att_b"], 0)?;
                        gemm(l.q, &s.a, D, t, &s.y, 3 * D, 0)?; gemm(l.k, &s.a, D, t, &s.y, 3 * D, D)?; gemm(l.v, &s.a, D, t, &s.y, 3 * D, 2 * D)?;
                        gemm(l.pos, &s.pos, D, 2 * t - 1, &s.pp, D, 0)?; ctx.barrier(cb);
                        if attn_gemm { // scores and P.V through the coop-matrix GEMM (per head), softmax with the relative shift in between
                            let tk = t.div_ceil(64) * 64; let r = 2 * t - 1;
                            let ab: [&Buffer; 8] = [&s.y, &s.pp, &s.qu, &s.qv, &s.kh, &s.vt, &s.ph, p];
                            run("aprep", &ab, &[0, t, H, 3 * D, tk, lp["u"], lp["v"]], g(t * D))?;
                            run("aprep", &ab, &[1, t, H, 3 * D, tk, 0, 0], g(r * D))?;
                            if tk > t { run("aprep", &ab, &[2, t, H, 3 * D, tk, 0, 0], g(D * (tk - t)))?; }
                            for h in 0..H {
                                let f16w = |off: u64, n: u32, kk: u32| WRef { arena: 0, off, ty: GgmlType::F16, k: kk, n, e: 1, bytes: n as u64 * kk as u64 * 2 };
                                pk_gemm(pf, ctx, cb, &s.kh, p, &GemmOp { w: f16w((h * t) as u64 * 256, t, 128), row0: 0, rows: t, a: &s.qu, lda: 128, a_row0: h * t, m: t, y: &s.ac, ldy: H * t, y_col: h * t, acc: false })?;
                                pk_gemm(pf, ctx, cb, &s.ph, p, &GemmOp { w: f16w((h * r) as u64 * 256, r, 128), row0: 0, rows: r, a: &s.qv, lda: 128, a_row0: h * t, m: t, y: &s.bd, ldy: H * r, y_col: h * r, acc: false })?;
                            }
                            ctx.barrier(cb);
                            run("smax", &[&s.ac, &s.bd, &s.pa], &[t, H, tk, (1.0f32 / 128f32.sqrt()).to_bits()], t * H)?;
                            for h in 0..H {
                                let w = WRef { arena: 0, off: (h * 128 * tk) as u64 * 2, ty: GgmlType::F16, k: tk, n: 128, e: 1, bytes: 128 * tk as u64 * 2 };
                                pk_gemm(pf, ctx, cb, &s.vt, p, &GemmOp { w, row0: 0, rows: 128, a: &s.pa, lda: tk, a_row0: h * t, m: t, y: &s.ao, ldy: D, y_col: h * 128, acc: false })?;
                            }
                            ctx.barrier(cb);
                            run("ew", &[&s.x, &s.ao, &s.a, p], &[3, t * D, D, D, 0, 0, 0, D], g(t * D))?;
                        } else {
                            run("attn", &[&s.y, &s.pp, &s.a, p], &[t, H, 3 * D, lp["u"], lp["v"], (1.0f32 / 128f32.sqrt()).to_bits(), D], t * H)?;
                        }
                        gemm(l.o, &s.a, D, t, &s.y2, D, 0)?; ctx.barrier(cb); ew(1, t * D, D, &s.y2, D, 0, 1.0, 0, 0)?;
                        ln(lp["conv_g"], lp["conv_b"], 0)?;
                        gemm(l.pw1, &s.a, D, t, &s.y, 2 * D, 0)?; ctx.barrier(cb);
                        run("dwc", &[&s.y, &s.a, p], &[t, D, KW, lp["dw"], lp["bn_m"], lp["bn_v"], lp["bn_g"], lp["bn_b"], 1e-5f32.to_bits(), D], g(t * D))?;
                        gemm(l.pw2, &s.a, D, t, &s.y2, D, 0)?; ctx.barrier(cb); ew(1, t * D, D, &s.y2, D, 0, 1.0, 0, 0)?;
                    }
                    ln(lp[gn], lp[bn], 0)?;
                    gemm(ff[0], &s.a, D, t, &s.y, FF, 0)?; ctx.barrier(cb);
                    ew(0, t * FF, FF, &s.y, FF, 0, 0.0, 0, FF)?;
                    gemm(ff[1], &s.a, FF, t, &s.y2, D, 0)?; ctx.barrier(cb); ew(1, t * D, D, &s.y2, D, 0, 0.5, 0, 0)?;
                }
                ln(lp["out_g"], lp["out_b"], 1)?;
            }
            ew(3, t * D, D, &s.x, D, 0, 0.0, 0, D)?;
            gemm(self.proj, &s.a, D, t, &s.ep, DP as u32, 0)?; ctx.barrier(cb);
            run("ew", &[&s.ep, &s.y, &s.a, p], &[2, t * DP as u32, DP as u32, 0, 0, 0, sb["bproj"], 0], g(t * DP as u32))?;
            Ok(()) })(); })?;
        err?;
        let ep = s.ep.as_f32()[..(t as usize * DP)].to_vec();
        let hid = if hidden { Some(s.x.as_f32()[..(t * D) as usize].to_vec()) } else { None };
        Ok((ep, hid, t, gpu))
    }

    fn lstm_step(&self, pool: &pool::Pool, tok: usize, st: &mut [(Vec<f32>, Vec<f32>); 2], gates: &mut [f32], dec: &mut [f32]) {
        let mut x = self.emb[tok * DP..(tok + 1) * DP].to_vec();
        for (l, (wi, wh, b)) in self.lstm.iter().enumerate() {
            let (h, c) = &mut st[l]; let (wi, wh, b) = (&wi[..], &wh[..], &b[..]); let g = pool::Out(gates.as_mut_ptr());
            { let (x, h) = (&x, &*h); pool.run(&|w| pool::gemv_rows(w, pool.n, 4 * DP, DP, wi, x, Some((wh, h)), b, g)); }
            let sg = |v: f32| 1.0 / (1.0 + (-v).exp());
            for j in 0..DP { let (i, f, gg, o) = (sg(gates[j]), sg(gates[DP + j]), gates[2 * DP + j].tanh(), sg(gates[3 * DP + j]));
                c[j] = f * c[j] + i * gg; h[j] = o * c[j].tanh(); }
            x.copy_from_slice(h);
        }
        let d = pool::Out(dec.as_mut_ptr()); let x = &x; let dp = &self.dproj;
        pool.run(&|w| pool::gemv_rows(w, pool.n, DP, DP, &dp.0, x, None, &dp.1, d));
    }

    /// TDT greedy decoding over projected encoder frames [t][640]; returns token ids (no blanks).
    pub fn decode(&self, ep: &[f32], t: usize) -> Vec<usize> {
        let nth = pool::env("PK_DEC_THREADS", 16);
        pool::Pool::scope(nth, |pool| {
            let mut st = [(vec![0f32; DP], vec![0f32; DP]), (vec![0f32; DP], vec![0f32; DP])];
            let (mut gates, mut dec) = (vec![0f32; 4 * DP], vec![0f32; DP]);
            self.lstm_step(pool, BLANK, &mut st, &mut gates, &mut dec);
            let nh = self.head.1.len(); let mut logits = vec![0f32; nh]; let mut z = vec![0f32; DP];
            let (mut out, mut f, mut syms_here) = (vec![], 0usize, 0);
            while f < t {
                for j in 0..DP { z[j] = (ep[f * DP + j] + dec[j]).max(0.0); }
                { let lo = pool::Out(logits.as_mut_ptr()); let z = &z; let hd = &self.head; pool.run(&|w| pool::gemv_rows(w, pool.n, nh, DP, &hd.0, z, None, &hd.1, lo)); }
                let tok = argmax(&logits[..VOCAB]); let mut dur = DURS[argmax(&logits[VOCAB..])];
                if tok != BLANK { out.push(tok); self.lstm_step(pool, tok, &mut st, &mut gates, &mut dec); syms_here += 1; if syms_here >= 10 && dur == 0 { dur = 1 } }
                else if dur == 0 { dur = 1 }
                if dur > 0 { syms_here = 0 }
                f += dur;
            }
            out
        })
    }

    pub fn detok(&self, ids: &[usize]) -> String {
        let mut s = String::new();
        for &i in ids { let p = &self.vocab[i]; if p.starts_with('<') && p.ends_with('>') { continue } s.push_str(&p.replace('\u{2581}', " ")); }
        s.trim().to_string()
    }

    /// Full pipeline on 16 kHz mono audio.
    pub fn transcribe(&mut self, ctx: &Ctx, pf: &Pf, fe: &audio::MelFrontend, audio: &[f32]) -> Result<(String, Vec<usize>, Timing)> {
        let t0 = std::time::Instant::now(); let mel = fe.features(audio); let t1 = std::time::Instant::now();
        let (ep, _, t, gpu) = self.encode(ctx, pf, &mel, false)?; let t2 = std::time::Instant::now();
        let ids = self.decode(&ep, t as usize); let t3 = std::time::Instant::now();
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        Ok((self.detok(&ids), ids, Timing { mel_ms: ms(t0, t1), enc_ms: ms(t1, t2), enc_gpu_ms: gpu * 1e3, dec_ms: ms(t2, t3) }))
    }
}

/// Frame counts after each stride-2 stage: (l1, l2, T).
pub fn subs(t0: u32) -> (u32, u32, u32) { let s = |l: u32| (l + 1) / 2; (s(t0), s(s(t0)), s(s(s(t0)))) }
fn argmax(v: &[f32]) -> usize { let mut b = 0; for i in 1..v.len() { if v[i] > v[b] { b = i } } b }

/// Tile choice for the encoder's small-M GEMMs (M = frames, a few 10s to a few 100s): keep enough workgroups in flight.
/// PK_TILE overrides (variant id, see Pf::gemm_tile).
fn pk_gemm(pf: &Pf, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, op: &GemmOp) -> Result<()> {
    let forced = prefill::env("PK_TILE", u32::MAX);
    // measured on Strix Halo (8 clips, 61..368 frames): 32 x 128 tiles win up to ~250 rows, 128 x 128 beyond
    let v = if forced != u32::MAX { forced } else if op.m <= 16 { 0x200 } else if op.m <= prefill::env("PK_M32", 256) { 0x400 } else { 0x2000 };
    pf.gemm_tile(ctx, cb, arena, f32b, op, Some(v))
}
