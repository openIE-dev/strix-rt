//! Qwen-Image-2.1 text-to-image on strix-rt.
//!  - text encoder: Qwen3-VL-8B language model (text path), last hidden state before the final norm;
//!  - denoiser: QwenImage21 single-stream DiT (32 blocks, d 4096, 32 heads, SwiGLU 12288), all linears through prefill's
//!    coop-matrix GEMM straight from the bf16 safetensors pages (memory tier, zero copy); flash-style coop-matrix attention
//!    for image tokens; text prefix K/V computed once (causal_condition) and reused by every step;
//!  - FlowMatch Euler scheduler (exponential dynamic shift + terminal stretch); VAE decoder in vae.rs.
pub mod st;
pub mod vae;
pub mod tok;
pub mod png;
use anyhow::Result;
use ash::vk;
use half::f16;
use prefill::{GemmOp, Pf};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{push_u32, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const D: u32 = 4096; const NH: u32 = 32; const FFD: u32 = 12288; const PAD: u64 = 320;

/// Blocks per GPU submission (QI_BPS, default 2): keeps each submit well under the amdgpu ring watchdog.
fn bps() -> usize { std::env::var("QI_BPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2).max(1) }
/// Query tiles (FLASH_Q rows) per flash-attention dispatch (QI_FLASH_QT, default 8), bounding single-dispatch runtime.
fn flash_qt() -> u32 { std::env::var("QI_FLASH_QT").ok().and_then(|v| v.parse().ok()).unwrap_or(8).max(1) }
/// QI_FLASH_NW=4: 4-wave flash workgroups (64 query rows), else 8 waves (128 rows).
fn flash_nw4() -> bool { std::env::var("QI_FLASH_NW").map_or(false, |v| v == "4") }
fn flash_q() -> u32 { if flash_nw4() { 64 } else { 128 } }
/// Row stride of the SwiGLU output h (QI_HPAD, default 64 halves of padding): a 24 KiB stride makes every A row-block
/// of the down GEMM hit the same DRAM channels.
/// Row stride of the f16 GEMM operand s.a (normed activations / attention output), padded like h (QI_APAD, default 64).
fn ald() -> u32 { D + std::env::var("QI_APAD").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(64) / 8 * 8 }
fn hld() -> u32 { FFD + std::env::var("QI_HPAD").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(64) / 8 * 8 }
pub struct Kern { pub k: HashMap<&'static str, Kernel> }
impl Kern {
    pub fn new(ctx: &Ctx) -> Result<Self> {
        let mut k = HashMap::new();
        k.insert("gemm", ctx.kernel(spv!("qi_gemm"), 3, 32, &[], Some(32))?);
        k.insert("flash", ctx.kernel(if std::env::var("QI_FLASH_PF").map_or(true, |v| v != "0") { if flash_nw4() { spv!("qi_flash_pf4") } else if std::env::var("QI_FLASH_SW").map_or(true, |v| v != "0") { spv!("qi_flash_sw") } else { spv!("qi_flash_pf") } } else { spv!("qi_flash") }, 4, 40, &[], Some(32))?); // QI_FLASH_PF=0: no K/V register prefetch
        k.insert("small", ctx.kernel(spv!("qi_small"), 4, 32, &[], None)?);
        k.insert("qk", ctx.kernel(spv!("qi_qk"), 4, 40, &[], None)?);
        k.insert("norm", ctx.kernel(spv!("qi_norm"), 4, 32, &[], None)?);
        k.insert("ew", ctx.kernel(spv!("qi_ew"), 4, 32, &[], None)?);
        for (n, s, b, p) in vae::KERNELS { k.insert(n, ctx.kernel(s, *b, *p, &[], None)?); }
        Ok(Kern { k })
    }
}

/// qimage's own BF16 GEMM (qi_gemm, 128 x 128 tiles) applies to the big image-token GEMMs; opt-in with QI_GEMM=1 (slower than prefill at 1024 so far: qkv 12.7 vs 13.5 TFLOPS)
pub fn own_gemm(w: &WRef, m: u32) -> bool {
    matches!(w.ty, gguf::GgmlType::Other(30)) && m % 128 == 0 && w.n % 128 == 0 && w.k % 32 == 0 && w.off % 8 == 0 && w.off / 8 <= u32::MAX as u64
        && std::env::var("QI_GEMM").map(|v| v == "1").unwrap_or(false)
}
/// Recording helper: kernels + GEMMs into one command buffer, a barrier after each dispatch.
pub struct Rec<'a> { pub ctx: &'a Ctx, pub pf: &'a Pf, pub kn: &'a Kern, pub cb: vk::CommandBuffer }
impl<'a> Rec<'a> {
    pub fn run(&self, name: &str, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<()> {
        let k = &self.kn.k[name]; let ds = self.pf.ds.get(self.ctx, k, bufs)?; self.ctx.dispatch(self.cb, k, ds, &push_u32(push), groups); self.ctx.barrier(self.cb); Ok(())
    }
    /// y[:, y_col..] = a[m][lda] . W^T (no barrier unless `bar`)
    pub fn gemm(&self, arenas: &[Buffer], f32b: &Buffer, w: WRef, a: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32, y_col: u32, bar: bool) -> Result<()> {
        if own_gemm(&w, m) {
            let k = &self.kn.k["gemm"]; let ds = self.pf.ds.get(self.ctx, k, &[a, &arenas[w.arena], y])?;
            self.ctx.dispatch(self.cb, k, ds, &push_u32(&[m, w.n, w.k, (w.off / 8) as u32, lda, ldy, y_col, 0]), (m / 128) * (w.n / 128));
        } else { self.pf.gemm(self.ctx, self.cb, &arenas[w.arena], f32b, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m, y, ldy, y_col, acc: false })?; }
        if bar { self.ctx.barrier(self.cb) } Ok(())
    }
    pub fn bar(&self) { self.ctx.barrier(self.cb) }
    pub fn ew(&self, mode: u32, rows: u32, d: u32, x: &Buffer, y: &Buffer, ldy: u32, ycol: u32, a: &Buffer, lda: u32, m: &Buffer, g_off: u32) -> Result<()> {
        self.run("ew", &[x, y, a, m], &[mode, rows * d, d, ldy, ycol, lda, g_off, 0], (rows * d).div_ceil(256))
    }
}
pub const NOG: u32 = 0xffff_ffff;

fn f16buf(ctx: &Ctx, n: u64) -> Result<Buffer> { ctx.buffer(n * 2) }
fn f32buf(ctx: &Ctx, n: u64) -> Result<Buffer> { ctx.buffer(n * 4) }
pub fn upload_f16(b: &Buffer, v: &[f32]) { let p = unsafe { std::slice::from_raw_parts_mut(b.ptr as *mut f16, v.len()) }; for (o, x) in p.iter_mut().zip(v) { *o = f16::from_f32(*x) } }

// ------------------------------------------------------------------------------------------------------------- text encoder
pub struct TextEncoder { pub st: st::Model, params: Buffer, lw: Vec<HashMap<&'static str, WRef>>, lp: Vec<HashMap<&'static str, u32>>, nl: usize, embed: String }
impl TextEncoder {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|k| k.starts_with("model.language_model.layers.") && k.ends_with("proj.weight"))?;
        let nl = (0..).take_while(|i| st.t.contains_key(&format!("model.language_model.layers.{i}.input_layernorm.weight"))).count();
        let mut pv: Vec<f32> = vec![]; let mut lw = vec![]; let mut lp = vec![];
        for l in 0..nl { let p = format!("model.language_model.layers.{l}.");
            let mut w = HashMap::new();
            for (k, n) in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")] {
                w.insert(k, st.wref(&format!("{p}{n}.weight"))?); }
            let mut o = HashMap::new();
            for (k, n) in [("in", "input_layernorm"), ("post", "post_attention_layernorm"), ("qn", "self_attn.q_norm"), ("kn", "self_attn.k_norm")] {
                o.insert(k, pv.len() as u32); pv.extend(st.f32(&format!("{p}{n}.weight"))?); while pv.len() % 64 != 0 { pv.push(0.0) } }
            lw.push(w); lp.push(o); }
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        Ok(TextEncoder { st, params, lw, lp, nl, embed: "model.language_model.embed_tokens.weight".into() })
    }
    /// Hidden states of the last layer (pre final norm) for `ids`, rows [drop..) -> [L - drop][4096] f32.
    pub fn encode(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, ids: &[u32], drop: usize) -> Result<(Vec<f32>, f64)> {
        let l = ids.len() as u64; let d = D as u64; let rows = l + PAD;
        let x = f32buf(ctx, rows * d)?; let a = f16buf(ctx, rows * FFD as u64)?; let y = f32buf(ctx, rows * 2 * FFD as u64)?; let y2 = f32buf(ctx, rows * d)?;
        let qh = f16buf(ctx, rows * d)?; let kh = f16buf(ctx, rows * 1024)?; let vh = f16buf(ctx, rows * 1024)?;
        let rope = f32buf(ctx, l * 128)?;
        { let e = self.st.bytes(&self.embed)?; let xs = x.as_f32_mut();
          for (i, &t) in ids.iter().enumerate() { let r = &e[t as usize * 8192..(t as usize + 1) * 8192];
              for j in 0..4096 { xs[i * 4096 + j] = half::bf16::from_le_bytes([r[2 * j], r[2 * j + 1]]).to_f32() } }
          let rt = rope.as_f32_mut();
          for p in 0..l as usize { for i in 0..64 { let inv = 1.0 / 5_000_000f64.powf(2.0 * i as f64 / 128.0); let (s, c) = (p as f64 * inv).sin_cos(); rt[p * 128 + 2 * i] = c as f32; rt[p * 128 + 2 * i + 1] = s as f32; } } }
        let ar = &self.st.bufs; let pr = &self.params; let lu = l as u32;
        let mut err = Ok(());
        let gpu = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            let dbg = std::env::var("QI_TE_DBG").is_ok();
            let nl = std::env::var("QI_TE_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(self.nl);
            for li in 0..nl { let w = &self.lw[li]; let p = &self.lp[li];
                r.run("norm", &[&x, &a, pr, pr], &[D, D, D, p["in"], 0f32.to_bits(), 0, 1e-6f32.to_bits(), 0], lu)?;
                r.gemm(ar, pr, w["q"], &a, D, lu, &y, 6144, 0, false)?; r.gemm(ar, pr, w["k"], &a, D, lu, &y, 6144, 4096, false)?; r.gemm(ar, pr, w["v"], &a, D, lu, &y, 6144, 5120, true)?;
                if dbg { return Ok(()) }
                r.run("qk", &[&y, &qh, pr, &rope], &[32, 6144, 0, p["qn"], 1e-6f32.to_bits(), D, 0, 1, 3, 0], lu * 32)?;
                r.run("qk", &[&y, &kh, pr, &rope], &[8, 6144, 4096, p["kn"], 1e-6f32.to_bits(), 1024, 0, 1, 3, 0], lu * 8)?;
                r.run("qk", &[&y, &vh, pr, &rope], &[8, 6144, 5120, 0, 0, 1024, 0, 1, 0, 0], lu * 8)?;
                r.run("small", &[&qh, &kh, &vh, &a], &[lu, 32, 4, D, 1024, 1024, D, (1.0f32 / 128f32.sqrt()).to_bits()], lu * 32)?;
                r.gemm(ar, pr, w["o"], &a, D, lu, &y2, D, 0, true)?;
                r.ew(0, lu, D, &x, &y2, D, 0, &a, 0, pr, NOG)?;
                r.run("norm", &[&x, &a, pr, pr], &[D, D, D, p["post"], 0f32.to_bits(), 0, 1e-6f32.to_bits(), 0], lu)?;
                r.gemm(ar, pr, w["gate"], &a, D, lu, &y, 2 * FFD, 0, false)?; r.gemm(ar, pr, w["up"], &a, D, lu, &y, 2 * FFD, FFD, true)?;
                r.ew(2, lu, FFD, &x, &y, 2 * FFD, 0, &a, FFD, pr, 0)?;
                r.gemm(ar, pr, w["down"], &a, FFD, lu, &y2, D, 0, true)?;
                r.ew(0, lu, D, &x, &y2, D, 0, &a, 0, pr, NOG)?;
            }
            Ok(()) })(); })?;
        err?;
        if std::env::var("QI_TE_DBG").is_ok() { let mut o = vec![]; for t in 0..l as usize { o.extend((unsafe { std::slice::from_raw_parts(a.ptr as *const f16, (t + 1) * 4096) })[t * 4096..].iter().map(|v| v.to_f32())); }
            for t in 0..l as usize { o.extend_from_slice(&y.as_f32()[t * 6144..(t + 1) * 6144]); } return Ok((o, gpu)) }
        let out = x.as_f32()[drop * 4096..(l as usize) * 4096].to_vec();
        for b in [x, a, y, y2, qh, kh, vh, rope] { ctx.free_buffer(b) }
        Ok((out, gpu))
    }
}

// --------------------------------------------------------------------------------------------------------------------- DiT
pub struct Dit { pub st: st::Model, params: Buffer, blocks: Vec<HashMap<&'static str, WRef>>, bp: Vec<HashMap<&'static str, u32>>, g: HashMap<&'static str, WRef>, txt_norm: u32, pub fused: bool }
pub struct DitState { pub lt: u32, pub n: u32, kc: Vec<Buffer>, vc: Vec<Buffer>, x: Buffer, a: Buffer, y: Buffer, y2: Buffer, h: Buffer, q: Buffer, rope: Buffer, md: Buffer, ta: Buffer, ty: Buffer, pub v: Buffer, lat: Buffer }

/// FlowMatchEulerDiscreteScheduler sigmas as Qwen-Image-2.1 configures them (exponential dynamic shift, terminal 0.02).
pub fn sigmas(steps: usize, image_seq_len: usize) -> Vec<f32> {
    let (base, maxl, bs, ms) = (256.0f64, 8192.0f64, 0.5f64, 0.9f64);
    let mu = image_seq_len as f64 * (ms - bs) / (maxl - base) + (bs - (ms - bs) / (maxl - base) * base);
    let lin: Vec<f64> = (0..steps).map(|i| 1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps as f64 - 1.0).max(1.0)).collect();
    let sh: Vec<f64> = lin.iter().map(|&s| mu.exp() / (mu.exp() + (1.0 / s - 1.0))).collect();
    let sf = (1.0 - sh[steps - 1]) / (1.0 - 0.02);
    let mut o: Vec<f32> = sh.iter().map(|s| (1.0 - (1.0 - s) / sf) as f32).collect(); o.push(0.0); o
}

impl Dit {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let st = st::Model::open(ctx, dir, &|k| k.ends_with(".weight") && !k.contains("norm_q") && !k.contains("norm_k") && !k.contains("text_norm"))?;
        let nb = (0..).take_while(|i| st.t.contains_key(&format!("transformer_blocks.{i}.attn.to_q.weight"))).count();
        let mut pv: Vec<f32> = vec![]; let mut blocks = vec![]; let mut bp = vec![];
        for b in 0..nb { let p = format!("transformer_blocks.{b}.");
            let mut w = HashMap::new();
            for (k, n) in [("q", "attn.to_q"), ("k", "attn.to_k"), ("v", "attn.to_v"), ("o", "attn.to_out.0"), ("gate", "img_mlp.gate_layer"), ("up", "img_mlp.proj"), ("down", "img_mlp.out")] {
                w.insert(k, st.wref(&format!("{p}{n}.weight"))?); }
            let mut o = HashMap::new();
            for (k, n) in [("qn", "attn.norm_q"), ("kn", "attn.norm_k")] { o.insert(k, pv.len() as u32); pv.extend(st.f32(&format!("{p}{n}.weight"))?); while pv.len() % 64 != 0 { pv.push(0.0) } }
            blocks.push(w); bp.push(o); }
        let txt_norm = pv.len() as u32; pv.extend(st.f32("txt_in.text_norm.weight")?);
        let mut g = HashMap::new();
        for (k, n) in [("img_in", "img_in"), ("mod", "modulation.1"), ("norm_out", "norm_out.linear"), ("proj_out", "proj_out"), ("t1", "time_text_embed.timestep_embedder.linear_1"),
            ("t2", "time_text_embed.timestep_embedder.linear_2"), ("txt_in1", "txt_in.in_layer"), ("txt_in2", "txt_in.out_layer")] { g.insert(k, st.wref(&format!("{n}.weight"))?); }
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        let fused = blocks.iter().all(|w| w["gate"].arena == w["up"].arena) && std::env::var("QI_SWIGLU").map(|v| v != "0").unwrap_or(true);
        Ok(Dit { st, params, blocks, bp, g, txt_norm, fused })
    }

    pub fn state(&self, ctx: &Ctx, lt: u32, n: u32) -> Result<DitState> {
        let nk = (lt + n) as u64 + 64; let d = D as u64; let rows = n.max(lt) as u64 + PAD;
        let mut kc = vec![]; let mut vc = vec![];
        for _ in 0..self.blocks.len() { kc.push(f16buf(ctx, nk * d)?); vc.push(f16buf(ctx, nk * d)?); }
        Ok(DitState { lt, n, kc, vc, x: f32buf(ctx, rows * d)?, a: f16buf(ctx, rows * FFD as u64)?, y: f32buf(ctx, rows * 2 * FFD as u64)?, y2: f32buf(ctx, rows * d)?, h: f16buf(ctx, rows * (FFD as u64 + 256))?,
            q: f16buf(ctx, (rows + 64) * d)?, rope: f32buf(ctx, (lt + n) as u64 * 128)?, md: f32buf(ctx, 2 * 4 * d + 2 * d)?, ta: f16buf(ctx, PAD * d)?, ty: f32buf(ctx, PAD * d)?,
            v: f32buf(ctx, n as u64 * 64)?, lat: f16buf(ctx, rows * 64)? })
    }

    /// RoPE table rows: text positions 0..lt (all three axes), then the h x w image grid (frame = lt, centred h/w).
    pub fn rope_table(st: &DitState, h: u32, w: u32) {
        let rt = st.rope.as_f32_mut(); let lt = st.lt as usize;
        let ang = |row: usize, f: f64, hh: f64, ww: f64, rt: &mut [f32]| {
            for p in 0..64usize { let (pos, i, dim) = if p < 8 { (f, p, 16.0) } else if p < 36 { (hh, p - 8, 56.0) } else { (ww, p - 36, 56.0) };
                let a = pos / 10000f64.powf(2.0 * i as f64 / dim); let (s, c) = a.sin_cos(); rt[row * 128 + 2 * p] = c as f32; rt[row * 128 + 2 * p + 1] = s as f32; } };
        for t in 0..lt { ang(t, t as f64, t as f64, t as f64, rt) }
        for y in 0..h as i64 { for x in 0..w as i64 {
            let hy = y - (h as i64 - h as i64 / 2); let wx = x - (w as i64 - w as i64 / 2);
            ang(lt + (y * w as i64 + x) as usize, lt as f64, hy as f64, wx as f64, rt) } }
    }

    /// temb + modulation for (sigma, 0) -> md: rows 0/1 = [scale1|gate1|scale2|gate2] at 0 / 4D, norm_out scale at 8D / 9D.
    fn modulation(&self, r: &Rec, s: &DitState, sigma: f32) -> Result<()> {
        let p = &self.params; let ar = &self.st.bufs;
        let mut tp = vec![0f32; 2 * 256];
        for (row, t) in [sigma, 0.0].iter().enumerate() { for i in 0..128 { let f = (-(10000f64.ln()) * i as f64 / 128.0).exp(); let a = 1000.0 * *t as f64 * f;
            tp[row * 256 + i] = a.cos() as f32; tp[row * 256 + 128 + i] = a.sin() as f32; } }
        upload_f16(&s.ta, &tp);
        r.gemm(ar, p, self.g["t1"], &s.ta, 256, 2, &s.ty, D, 0, true)?;
        r.ew(4, 2, D, &s.x, &s.ty, D, 0, &s.ta, D, p, 0)?;
        r.gemm(ar, p, self.g["t2"], &s.ta, D, 2, &s.y2, D, 0, true)?;
        r.ew(4, 2, D, &s.x, &s.y2, D, 0, &s.ta, D, p, 0)?;
        r.gemm(ar, p, self.g["mod"], &s.ta, D, 2, &s.md, 4 * D, 0, false)?;
        // norm_out.linear scale rows (sigma, 0) -> ty (read by the final norm)
        r.gemm(ar, p, self.g["norm_out"], &s.ta, D, 2, &s.ty, D, 0, true)?;
        Ok(())
    }

    /// Per-op GPU time of one image block (each op in its own submission, `reps` medians).
    pub fn prof_block(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &DitState, reps: usize) -> Result<Vec<(&'static str, f64)>> {
        let mut out = vec![];
        for op in ["norm1", "qkv", "qk", "flash", "o", "res1", "norm2", "swiglu", "down", "res2"] {
            let mut t = vec![];
            for _ in 0..reps { let mut err = Ok(()); let g = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb }; self.block_op(&r, s, 0, op) })(); })?; err?; t.push(g); }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap()); out.push((op, t[t.len() / 2])); }
        Ok(out)
    }
    fn block_op(&self, r: &Rec, s: &DitState, b: usize, op: &str) -> Result<()> {
        let w = &self.blocks[b]; let bp = &self.bp[b]; let p = &self.params; let ar = &self.st.bufs;
        let (m, row0) = (s.n, s.lt); let mo = 0; let eps = 1e-6f32.to_bits(); let sc = (1.0f32 / 128f32.sqrt()).to_bits();
        match op {
            "norm1" => r.run("norm", &[&s.x, &s.a, p, &s.md], &[D, D, ald(), 0, 0, 1, eps, mo], m)?,
            "qkv" => { r.gemm(ar, p, w["q"], &s.a, ald(), m, &s.y, 3 * D, 0, false)?; r.gemm(ar, p, w["k"], &s.a, ald(), m, &s.y, 3 * D, D, false)?; r.gemm(ar, p, w["v"], &s.a, ald(), m, &s.y, 3 * D, 2 * D, true)?; }
            "qk" => { r.run("qk", &[&s.y, &s.q, p, &s.rope], &[NH, 3 * D, 0, bp["qn"], eps, D, 0, 0, 3, row0], m * NH)?;
                r.run("qk", &[&s.y, &s.kc[b], p, &s.rope], &[NH, 3 * D, D, bp["kn"], eps, D, row0, 0, 3, row0], m * NH)?;
                r.run("qk", &[&s.y, &s.vc[b], p, &s.rope], &[NH, 3 * D, 2 * D, 0, 0, D, row0, 0, 0, row0], m * NH)?; }
            "flash" => { let nt = m.div_ceil(flash_q()); let k = &r.kn.k["flash"]; let ds = r.pf.ds.get(r.ctx, k, &[&s.q, &s.kc[b], &s.vc[b], &s.a])?;
                r.ctx.dispatch(r.cb, k, ds, &push_u32(&[m, s.lt + s.n, NH, D, D, D, ald(), sc, 1, 0]), nt * NH); }
            "o" => r.gemm(ar, p, w["o"], &s.a, ald(), m, &s.y2, D, 0, true)?,
            "res1" => r.ew(0, m, D, &s.x, &s.y2, D, 0, &s.a, 0, &s.md, mo + D)?,
            "norm2" => r.run("norm", &[&s.x, &s.a, p, &s.md], &[D, D, ald(), 0, 0, 1, eps, mo + 2 * D], m)?,
            "swiglu" => { if own_gemm(&w["gate"], m) { r.gemm(ar, p, w["gate"], &s.a, ald(), m, &s.y, 2 * FFD, 0, false)?; r.gemm(ar, p, w["up"], &s.a, ald(), m, &s.y, 2 * FFD, FFD, true)?; r.ew(2, m, FFD, &s.x, &s.y, 2 * FFD, 0, &s.h, hld(), p, 0)?; }
                else { r.pf.gemm_swiglu(r.ctx, r.cb, &ar[w["gate"].arena], p, w["gate"], w["up"], 0, FFD, &s.a, ald(), m, &s.h, hld(), 0)?; } }
            "down" => r.gemm(ar, p, w["down"], &s.h, hld(), m, &s.y2, D, 0, true)?,
            _ => r.ew(0, m, D, &s.x, &s.y2, D, 0, &s.a, 0, &s.md, mo + 3 * D)?,
        }
        Ok(())
    }

    fn block(&self, r: &Rec, s: &DitState, b: usize, text: bool) -> Result<()> {
        let w = &self.blocks[b]; let bp = &self.bp[b]; let p = &self.params; let ar = &self.st.bufs;
        let (m, row0, mrow) = if text { (s.lt, 0u32, 1u32) } else { (s.n, s.lt, 0u32) };
        let mo = mrow * 4 * D; let eps = 1e-6f32.to_bits();
        r.run("norm", &[&s.x, &s.a, p, &s.md], &[D, D, ald(), 0, 0, 1, eps, mo], m)?;
        r.gemm(ar, p, w["q"], &s.a, ald(), m, &s.y, 3 * D, 0, false)?; r.gemm(ar, p, w["k"], &s.a, ald(), m, &s.y, 3 * D, D, false)?; r.gemm(ar, p, w["v"], &s.a, ald(), m, &s.y, 3 * D, 2 * D, true)?;
        r.run("qk", &[&s.y, &s.q, p, &s.rope], &[NH, 3 * D, 0, bp["qn"], eps, D, 0, 0, 3, row0], m * NH)?;
        r.run("qk", &[&s.y, &s.kc[b], p, &s.rope], &[NH, 3 * D, D, bp["kn"], eps, D, row0, 0, 3, row0], m * NH)?;
        r.run("qk", &[&s.y, &s.vc[b], p, &s.rope], &[NH, 3 * D, 2 * D, 0, 0, D, row0, 0, 0, row0], m * NH)?;
        let sc = (1.0f32 / 128f32.sqrt()).to_bits();
        if text { r.run("small", &[&s.q, &s.kc[b], &s.vc[b], &s.a], &[m, NH, 1, D, D, D, ald(), sc], m * NH)?; }
        else { let nt = m.div_ceil(flash_q()); let mut t0 = 0; while t0 < nt { let t1 = (t0 + flash_qt() * (128 / flash_q())).min(nt);
            r.ctx.barrier(r.cb); let k = &r.kn.k["flash"]; let ds = r.pf.ds.get(r.ctx, k, &[&s.q, &s.kc[b], &s.vc[b], &s.a])?;
            r.ctx.dispatch(r.cb, k, ds, &push_u32(&[m, s.lt + s.n, NH, D, D, D, ald(), sc, 1, t0]), (t1 - t0) * NH); t0 = t1; } r.bar(); }
        r.gemm(ar, p, w["o"], &s.a, ald(), m, &s.y2, D, 0, true)?;
        r.ew(0, m, D, &s.x, &s.y2, D, 0, &s.a, 0, &s.md, mo + D)?;
        r.run("norm", &[&s.x, &s.a, p, &s.md], &[D, D, ald(), 0, 0, 1, eps, mo + 2 * D], m)?;
        if self.fused && !own_gemm(&w["gate"], m) { r.pf.gemm_swiglu(r.ctx, r.cb, &ar[w["gate"].arena], p, w["gate"], w["up"], 0, FFD, &s.a, ald(), m, &s.h, hld(), 0)?; r.bar(); }
        else { r.gemm(ar, p, w["gate"], &s.a, ald(), m, &s.y, 2 * FFD, 0, false)?; r.gemm(ar, p, w["up"], &s.a, ald(), m, &s.y, 2 * FFD, FFD, true)?; r.ew(2, m, FFD, &s.x, &s.y, 2 * FFD, 0, &s.h, hld(), p, 0)?; }
        r.gemm(ar, p, w["down"], &s.h, hld(), m, &s.y2, D, 0, true)?;
        r.ew(0, m, D, &s.x, &s.y2, D, 0, &s.a, 0, &s.md, mo + 3 * D)?;
        Ok(())
    }

    /// Text prefix (once): txt_in + all blocks over the text tokens with t = 0 modulation; fills rows [0, lt) of the K/V caches.
    pub fn prefix(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &DitState, emb: &[f32]) -> Result<f64> {
        let lt = s.lt; let p = &self.params; let ar = &self.st.bufs;
        s.y.as_f32_mut()[..emb.len()].copy_from_slice(emb);
        let mut err = Ok(());
        let g = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            self.modulation(&r, s, 0.0)?;
            r.run("norm", &[&s.y, &s.a, p, p], &[D, D, D, self.txt_norm, 1f32.to_bits(), 0, 1e-6f32.to_bits(), 0], lt)?;
            r.gemm(ar, p, self.g["txt_in1"], &s.a, D, lt, &s.y, D, 0, true)?;
            r.ew(3, lt, D, &s.x, &s.y, D, 0, &s.a, D, p, 0)?;
            r.gemm(ar, p, self.g["txt_in2"], &s.a, D, lt, &s.x, D, 0, true)?;
            Ok(()) })(); })?;
        err?; let mut g = g;
        for b0 in (0..self.blocks.len()).step_by(bps()) { let mut err = Ok(());
            g += ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
                for b in b0..(b0 + bps()).min(self.blocks.len()) { self.block(&r, s, b, true)?; } Ok(()) })(); })?; err?; }
        Ok(g)
    }

    /// One denoiser evaluation of the image tokens at `sigma`: latents [n][64] -> velocity in s.v [n][64].
    pub fn step(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, s: &DitState, lat: &[f32], sigma: f32) -> Result<f64> {
        let n = s.n; let p = &self.params; let ar = &self.st.bufs;
        upload_f16(&s.lat, lat);
        let mut err = Ok(());
        let g = ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            self.modulation(&r, s, sigma)?;
            r.gemm(ar, p, self.g["img_in"], &s.lat, 64, n, &s.x, D, 0, true)?;
            Ok(()) })(); })?;
        err?; let mut g = g;
        for b0 in (0..self.blocks.len()).step_by(bps()) { let mut err = Ok(());
            g += ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
                for b in b0..(b0 + bps()).min(self.blocks.len()) { self.block(&r, s, b, false)?; } Ok(()) })(); })?; err?; }
        let mut err = Ok(());
        g += ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
            r.run("norm", &[&s.x, &s.a, p, &s.ty], &[D, D, ald(), 0, 0, 1, 1e-6f32.to_bits(), 0], n)?;
            r.gemm(ar, p, self.g["proj_out"], &s.a, ald(), n, &s.v, 64, 0, true)?;
            Ok(()) })(); })?;
        err?; Ok(g)
    }
}
