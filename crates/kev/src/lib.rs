//! Kev decision models (jaredpalmer/kev, Apache-2.0) on strix-rt: a Qwen3.5 / Qwen3.8 hybrid (Gated DeltaNet + full
//! attention) backbone with the LoRA folded in (ref/merge_lora.py -> convert_hf_to_gguf), run by the qwen35 batched prefill
//! kernels, plus Kev's pointer head on the GPU (shaders/kev_head.comp):
//!   logits_k = (W_k h_opt_k + b_k) . (W_q h_decide + b_q) / sqrt(256) / T, softmax over a question's options.
//! A question is one causal row [<state> state][<q> instr (<opt> o </opt>)* <decide>] (kev.model.encode / rows_of).
//! Serving runs the state once, snapshots the DeltaNet conv/recurrent state (the state's KV stays in place at positions
//! 0..Ls), and runs each question branch from that snapshot at positions Ls.. (exact: the branches never see each other).
use anyhow::{bail, Result};
use prefill::{qwen35, Pf};

pub struct PointerHead { pub temperature: f32, pub dp: usize, pub d: usize, pub qw: Vec<f32>, pub qb: Vec<f32>, pub kw: Vec<f32>, pub kb: Vec<f32> }

fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }

impl PointerHead {
    /// head.bin from ref/pack.py: f32 T, u32 dp, u32 d, q.w, q.b, k.w, k.b (f32 LE)
    pub fn load(path: &str) -> Result<Self> {
        let b = std::fs::read(path)?;
        let t = f32::from_le_bytes(b[0..4].try_into()?); let dp = u32::from_le_bytes(b[4..8].try_into()?) as usize; let d = u32::from_le_bytes(b[8..12].try_into()?) as usize;
        let v = f32s(&b[12..]); if v.len() != 2 * (dp * d + dp) || dp != 256 { bail!("head.bin size / dp") }
        let (qw, r) = v.split_at(dp * d); let (qb, r) = r.split_at(dp); let (kw, kb) = r.split_at(dp * d);
        Ok(Self { temperature: t, dp, d, qw: qw.to_vec(), qb: qb.to_vec(), kw: kw.to_vec(), kb: kb.to_vec() })
    }
    fn proj(&self, w: &[f32], b: &[f32], h: &[f32]) -> Vec<f32> {
        (0..self.dp).map(|i| b[i] as f64 + w[i * self.d..(i + 1) * self.d].iter().zip(h).map(|(a, x)| *a as f64 * *x as f64).sum::<f64>()).map(|v| v as f32).collect()
    }
    /// CPU reference of the head (calibrated logits), used by the parity test
    pub fn logits(&self, h_decide: &[f32], h_opts: &[Vec<f32>]) -> Vec<f32> {
        let q = self.proj(&self.qw, &self.qb, h_decide); let sc = 1.0 / (self.dp as f32).sqrt();
        h_opts.iter().map(|h| { let k = self.proj(&self.kw, &self.kb, h); k.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * sc / self.temperature }).collect()
    }
    /// [qT, qb, kT, kb] with W transposed to [d][256]
    fn packed(&self) -> Vec<f32> {
        let tr = |w: &[f32]| { let mut o = vec![0f32; w.len()]; for i in 0..self.dp { for j in 0..self.d { o[j * self.dp + i] = w[i * self.d + j] } } o };
        let mut v = tr(&self.qw); v.extend_from_slice(&self.qb); v.extend(tr(&self.kw)); v.extend_from_slice(&self.kb); v
    }
}

pub fn softmax(z: &[f32]) -> Vec<f32> { let m = z.iter().cloned().fold(f32::MIN, f32::max); let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect(); let s: f32 = e.iter().sum(); e.iter().map(|v| v / s).collect() }

const KEV_HEAD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kev_head.spv"));
const MAX_OPTS: usize = 256;

/// One question branch: its tokens (after the state), and the offsets within it of <decide> and each </opt>.
#[derive(Clone, Debug)]
pub struct Branch { pub ids: Vec<u32>, pub decide: usize, pub opts: Vec<usize> }

pub struct Kev { pub m: qwen38::Model, pub pf: Pf, pub s: qwen35::Q35, pub head: PointerHead, hp: vkrt::Buffer, pb: vkrt::Buffer, rb: vkrt::Buffer, snap: (Vec<u8>, Vec<u8>), multi: Option<Multi>,
    /// when Some, each branch pushes its final-normed <decide> hidden row here (probe routers)
    pub hsink: std::cell::RefCell<Option<Vec<Vec<f32>>>>,
    /// Base Labs (joey/elephant) mode: block-bidirectional full attention inside each forwarded block, and slot_lm readout
    /// (LM-head logits of these token ids at a branch's last row) instead of the pointer head
    pub bidir: bool, pub slots: Option<Vec<u32>> }

impl Kev {
    /// gguf: merged backbone; head: head.bin; max_ctx: longest state + branch; chunk: tokens per prefill chunk (>= longest branch)
    pub fn load(gguf: &str, head: &str, max_ctx: u32, chunk: u32) -> Result<Self> {
        // Kev only runs prefill GEMMs, so BF16 matrices with 4 KiB-multiple rows get padded rows (rtcore PAD_BF16): 9B bf16 ~9.5x faster
        if std::env::var("RT_PAD16").is_err() { std::env::set_var("RT_PAD16", "1") }
        let m = qwen38::Model::load(gguf, max_ctx + 8)?; let mut pf = Pf::new(&m.ctx)?;
        pf.k.insert("kev_head", m.ctx.kernel(KEV_HEAD, 5, 24, &[], None)?);
        let s = qwen35::Q35::new(&m, chunk.next_multiple_of(64))?;
        let head = if head.is_empty() { let d = m.cfg.dim as usize; PointerHead { temperature: 1.0, dp: 256, d, qw: vec![0.0; 256 * d], qb: vec![0.0; 256], kw: vec![0.0; 256 * d], kb: vec![0.0; 256] } } else { PointerHead::load(head)? }; if head.d != m.cfg.dim as usize || head.d > 8192 { bail!("head d {} vs model dim {}", head.d, m.cfg.dim) }
        let pk = head.packed(); let hp = m.ctx.buffer(pk.len() as u64 * 4)?; hp.as_f32_mut()[..pk.len()].copy_from_slice(&pk);
        let pb = m.ctx.buffer(((MAX_OPTS + 1) * 256 + MAX_OPTS) as u64 * 4)?; let rb = m.ctx.buffer((MAX_OPTS + 1) as u64 * 4)?;
        let mut k = Self { m, pf, s, head, hp, pb, rb, snap: (vec![], vec![]), multi: None, hsink: std::cell::RefCell::new(None), bidir: false, slots: None }; if prefill::env("KEV_MULTI", 1) > 0 { k.init_multi()?; } Ok(k)
    }
    fn embed(&self, toks: &[u32]) -> Result<()> {
        let t = self.m.wt.g.tensor("token_embd.weight").ok_or_else(|| anyhow::anyhow!("no token_embd"))?;
        let data = self.m.wt.g.data(t); let d = self.m.cfg.dim as usize; let rb = data.len() / self.m.cfg.vocab as usize;
        let x = &mut self.s.x.as_f32_mut()[..toks.len() * d];
        for (i, &tk) in toks.iter().enumerate() { gguf::dequant::dequant_row(t.ty, &data[tk as usize * rb..(tk as usize + 1) * rb], &mut x[i * d..(i + 1) * d]); }
        Ok(())
    }
    /// Causal forward of `toks` at positions pos0.. continuing the current recurrent state / KV cache. s.x holds the final
    /// residual of the last chunk's rows on return. Returns GPU seconds.
    pub fn forward_at(&self, toks: &[u32], pos0: usize) -> Result<f64> {
        let (m, s, pf) = (&self.m, &self.s, &self.pf);
        if pos0 + toks.len() + 1 > m.cfg.max_ctx as usize { bail!("context: {} + {} > {}", pos0, toks.len(), m.cfg.max_ctx) }
        if self.bidir && toks.len() > s.mmax as usize { bail!("bidirectional block of {} tokens > chunk {}", toks.len(), s.mmax) }
        let nl = m.cfg.n_layer as usize; let lps = prefill::env("PF_LPS", 8) as usize; let mut gpu = 0.0;
        for c0 in (0..toks.len()).step_by(s.mmax as usize) {
            let part = &toks[c0..(c0 + s.mmax as usize).min(toks.len())]; self.embed(part)?;
            for l0 in (0..nl).step_by(lps) {
                let mut err = None;
                gpu += m.ctx.run(|cb| if let Err(e) = qwen35::record(pf, m, s, cb, part.len() as u32, (pos0 + c0) as u32, l0..(l0 + lps).min(nl)) { err = Some(e) })?;
                if let Some(e) = err { return Err(e) }
            }
        }
        Ok(gpu)
    }
    /// Run the state from position 0 and snapshot the recurrent state.
    pub fn set_state(&mut self, state: &[u32]) -> Result<f64> {
        self.m.cst.bytes_mut().fill(0); self.m.sst.bytes_mut().fill(0);
        let t = self.forward_at(state, 0)?;
        self.snap = (self.m.cst.bytes_mut().to_vec(), self.m.sst.bytes_mut().to_vec()); Ok(t)
    }
    /// GPU head over rows of the last chunk: rows[0] = <decide>, rows[1..] = options. Returns calibrated logits.
    pub fn head_gpu(&self, rows: &[usize]) -> Result<Vec<f32>> {
        let k = rows.len() - 1; if k == 0 || k > MAX_OPTS { bail!("options: {k}") }
        let rv: Vec<u32> = rows.iter().map(|&r| r as u32).collect(); unsafe { std::slice::from_raw_parts_mut(self.rb.ptr as *mut u32, rv.len()) }.copy_from_slice(&rv);
        let (c, ctx) = (&self.m.cfg, &self.m.ctx); let off = self.m.wt.f("output_norm.weight")?;
        let sc = 1.0 / 16.0 / self.head.temperature; let mut err = None;
        ctx.run(|cb| {
            let bufs = [&self.s.x, &self.m.wt.f32b, &self.hp, &self.pb, &self.rb];
            let r = self.pf.run(ctx, "kev_head", &bufs, &[c.dim, off, rows.len() as u32, c.eps.to_bits(), sc.to_bits(), 0], rows.len() as u32, cb)
                .and_then(|_| { ctx.barrier(cb); self.pf.run(ctx, "kev_head", &bufs, &[c.dim, off, rows.len() as u32, c.eps.to_bits(), sc.to_bits(), 1], k as u32, cb) });
            if let Err(e) = r { err = Some(e) } })?;
        if let Some(e) = err { return Err(e) }
        let o = rows.len() * 256; Ok(self.pb.as_f32()[o..o + k].to_vec())
    }
    /// Calibrated logits of one branch continuing the snapshotted state of `state_len` tokens.
    pub fn branch(&self, state_len: usize, b: &Branch) -> Result<Vec<f32>> {
        if b.ids.len() > self.s.mmax as usize { bail!("branch of {} tokens > chunk {}", b.ids.len(), self.s.mmax) }
        self.m.cst.bytes_mut().copy_from_slice(&self.snap.0); self.m.sst.bytes_mut().copy_from_slice(&self.snap.1);
        self.forward_at(&b.ids, state_len)?;
        let mut rows = vec![b.decide]; rows.extend_from_slice(&b.opts);
        if self.hsink.borrow().is_some() { let h = self.hidden_rows(&[b.decide])?.remove(0); self.hsink.borrow_mut().as_mut().unwrap().push(h) }
        if self.slots.is_some() { return self.slot_logits(b.decide) }
        self.head_gpu(&rows)
    }
    /// Probabilities for every question about one state (state once, then each branch).
    pub fn decide_all(&mut self, state: &[u32], branches: &[Branch]) -> Result<Vec<Vec<f32>>> {
        self.set_state(state)?;
        Ok(self.branches(state.len(), branches)?.iter().map(|z| softmax(z)).collect())
    }
    /// Switch to Base Labs mode: block-bidirectional full attention (prefix block and each suffix block) + slot_lm readout.
    pub fn base_labs(&mut self, slots: Vec<u32>) -> Result<()> {
        let k = self.m.ctx.kernel(KB_ATTN_BIDIR, 4, 36, &[], None)?; self.pf.k.insert("attn", k); let k2 = self.m.ctx.kernel(KB_ATTN_BIDIR, 4, 36, &[], None)?; self.pf.k.insert("attn_nofa", k2); self.bidir = true; self.slots = Some(slots); Ok(())
    }
    /// slot_lm readout: LM-head logits (output.weight, or tied token_embd) of the slot tokens at one final-normed row.
    pub fn slot_logits(&self, row: usize) -> Result<Vec<f32>> {
        let h = self.hidden_rows(&[row])?.remove(0); let g = &self.m.wt.g;
        let t = g.tensor("output.weight").or_else(|| g.tensor("token_embd.weight")).ok_or_else(|| anyhow::anyhow!("no lm head"))?;
        let data = g.data(t); let d = self.m.cfg.dim as usize; let rb = data.len() / self.m.cfg.vocab as usize; let mut w = vec![0f32; d];
        Ok(self.slots.as_ref().unwrap().iter().map(|&tk| { gguf::dequant::dequant_row(t.ty, &data[tk as usize * rb..(tk as usize + 1) * rb], &mut w);
            w.iter().zip(&h).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32 }).collect())
    }
    /// Final-normed hidden rows of the last chunk (CPU; parity diagnostics)
    pub fn hidden_rows(&self, rows: &[usize]) -> Result<Vec<Vec<f32>>> {
        let d = self.m.cfg.dim as usize; let x = self.s.x.as_f32();
        let off = self.m.wt.f("output_norm.weight")? as usize; let w = &self.m.wt.f32b.as_f32()[off..off + d];
        Ok(rows.iter().map(|&r| { let h = &x[r * d..(r + 1) * d];
            let ms = (h.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() / d as f64 + self.m.cfg.eps as f64).sqrt() as f32;
            h.iter().zip(w).map(|(v, g)| v / ms * g).collect() }).collect())
    }
}

pub mod json;
pub mod systemone;
pub mod decisions;
pub mod tok;
pub mod ucat;
pub mod hftok;
pub mod laya_s1;

/// Kev's delimiters (kev.model.SPECIAL): <state>, <q>, <opt>, </opt>, <decide>
pub const SPECIAL: [&str; 5] = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"];

/// kev.model.user_tokens: `<|name|>` is rewritten to `<¦name¦>` so caller text can never produce a delimiter
pub fn user_tokens(t: &tok::Tokenizer, text: &str) -> Vec<u32> {
    let mut s = String::with_capacity(text.len()); let mut rest = text;
    while let Some(p) = rest.find("<|") {
        s.push_str(&rest[..p]); let after = &rest[p + 2..];
        let n = after.bytes().take_while(|b| b.is_ascii_alphanumeric() || *b == b'_').count();
        if n > 0 && after[n..].starts_with("|>") { s.push_str("<¦"); s.push_str(&after[..n]); s.push_str("¦>"); rest = &after[n + 2..]; } else { s.push_str("<|"); rest = after; }
    }
    s.push_str(rest); t.encode_plain(&s)
}

/// kev.model.encode (no option isolation) split into the state row and one branch per question
pub fn encode(t: &tok::Tokenizer, state: &str, qs: &[(String, Vec<String>)]) -> Result<(Vec<u32>, Vec<Branch>)> {
    let id = |s: &str| t.id(s).ok_or_else(|| anyhow::anyhow!("tokenizer has no {s}"));
    let mut st = vec![id(SPECIAL[0])?]; st.extend(user_tokens(t, state));
    let (q, o, c, d) = (id(SPECIAL[1])?, id(SPECIAL[2])?, id(SPECIAL[3])?, id(SPECIAL[4])?);
    let brs = qs.iter().map(|(instr, opts)| {
        let mut ids = vec![q]; ids.extend(user_tokens(t, instr)); let mut ends = vec![];
        for op in opts { ids.push(o); ids.extend(user_tokens(t, op)); ids.push(c); ends.push(ids.len() - 1); }
        ids.push(d); Branch { decide: ids.len() - 1, ids, opts: ends } }).collect();
    Ok((st, brs))
}

// ---- all questions in one pass: every branch's rows packed into one chunk, GEMMs/norms over all rows at once, the
// sequence kernels (DeltaNet conv + scan, RoPE + attention) once per branch with a row offset (shaders/kb_*.comp).
// Branches read the document's conv/DeltaNet state without writing it back and keep their own K/V in a scratch cache,
// so nothing needs restoring between requests on the same document.
const KB_CONV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kb_conv.spv"));
const KB_SCAN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kb_scan.spv"));
const KB_ROPE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kb_rope.spv"));
const KB_ATTN_BIDIR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kb_attn_bidir.spv"));
const KB_ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kb_attn.spv"));

pub struct Multi { kcb: vkrt::Buffer, vcb: vkrt::Buffer }

impl Kev {
    pub fn init_multi(&mut self) -> Result<()> {
        let ctx = &self.m.ctx;
        self.pf.k.insert("kb_conv", ctx.kernel(KB_CONV, 4, 36, &[], None)?); self.pf.k.insert("kb_scan", ctx.kernel(KB_SCAN, 5, 44, &[], None)?);
        self.pf.k.insert("kb_rope", ctx.kernel(KB_ROPE, 4, 56, &[256], None)?); self.pf.k.insert("kb_attn", ctx.kernel(KB_ATTN, 6, 48, &[], None)?);
        let c = &self.m.cfg; let na = c.recr.iter().filter(|r| !**r).count() as u64;
        let sz = na * self.s.mmax as u64 * (c.n_kv * c.hd) as u64 * 2;
        self.multi = Some(Multi { kcb: ctx.buffer(sz.max(256))?, vcb: ctx.buffer(sz.max(256))? }); Ok(())
    }
    fn record_multi(&self, cb: ash::vk::CommandBuffer, spans: &[(u32, u32)], pos0: u32, layers: std::ops::Range<usize>) -> Result<()> {
        let (m, s, pf) = (&self.m, &self.s, &self.pf); let (c, wt, ctx) = (&m.cfg, &m.wt, &m.ctx); let mu = self.multi.as_ref().unwrap();
        let n: u32 = spans.iter().map(|x| x.1).sum(); let ds_ = c.d_state; let conv_ch = 2 * c.nk * ds_ + c.nv * ds_; let al = s.al;
        let bar = || ctx.barrier(cb);
        let gemm = |name: &str, y: &vkrt::Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> { let w = wt.wref(name)?;
            pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &prefill::GemmOp { w, row0: 0, rows: w.n, a: &s.a, lda: al, a_row0: 0, m: n, y, ldy, y_col, acc }) };
        let norm_to = |wn: &str, dst: &vkrt::Buffer| -> Result<()> { pf.run(ctx, "rmsnorm", &[&s.x, dst, &wt.f32b], &[c.dim, al, wt.f(wn)?, c.eps.to_bits(), 0], n, cb)?; bar(); Ok(()) };
        for l in layers {
            let nm = |x: &str| format!("blk.{l}.{x}"); let li = s.lidx[l];
            norm_to(&nm("attn_norm.weight"), &s.a)?;
            if c.recr[l] {
                let ldp = conv_ch + c.nv * ds_ + 2 * c.nv; let (zo, ao, bo) = (conv_ch, conv_ch + c.nv * ds_, conv_ch + c.nv * ds_ + c.nv);
                gemm(&nm("attn_qkv.weight"), &s.p, ldp, 0, false)?; gemm(&nm("attn_gate.weight"), &s.p, ldp, zo, false)?;
                gemm(&nm("ssm_alpha.weight"), &s.p, ldp, ao, false)?; gemm(&nm("ssm_beta.weight"), &s.p, ldp, bo, false)?; bar();
                let st_off = li * conv_ch * (c.d_conv - 1);
                for &(r0, len) in spans { pf.run(ctx, "kb_conv", &[&s.p, &m.cst, &s.cv, &wt.f32b], &[c.nk, st_off, wt.f(&nm("ssm_conv1d.weight"))?, c.d_conv, c.eps.to_bits(), ldp, conv_ch, conv_ch / 128, r0], len * (conv_ch / 128), cb)?; } bar();
                for &(r0, len) in spans { pf.run(ctx, "kb_scan", &[&s.cv, &s.p, &m.sst, &s.o, &wt.f32b], &[c.nk, c.nv, li * c.nv * ds_ * ds_ / 4, ao, bo, wt.f(&nm("ssm_dt.bias"))?, wt.f(&nm("ssm_a"))?, ldp, conv_ch, len, r0], c.nv, cb)?; } bar();
                pf.run(ctx, "gnorm", &[&s.o, &s.p, &s.a, &wt.f32b], &[c.nv, ldp, zo, wt.f(&nm("ssm_norm.weight"))?, c.eps.to_bits(), al], n * c.nv, cb)?; bar();
                gemm(&nm("ssm_out.weight"), &s.x, c.dim, 0, true)?; bar();
            } else {
                let (qn, kn) = (c.n_head * c.hd * 2, c.n_kv * c.hd); let ldp = qn + 2 * kn;
                let cache_off = li * c.max_ctx * c.n_kv * c.hd;
                gemm(&nm("attn_q.weight"), &s.p, ldp, 0, false)?; gemm(&nm("attn_k.weight"), &s.p, ldp, qn, false)?; gemm(&nm("attn_v.weight"), &s.p, ldp, qn + kn, false)?; bar();
                for &(r0, len) in spans { let boff = (li * s.mmax + r0) * c.n_kv * c.hd;
                    pf.run(ctx, "kb_rope", &[&s.p, &mu.kcb, &mu.vcb, &wt.f32b], &[c.n_head, c.n_kv, boff, c.theta.to_bits(), wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?,
                        c.eps.to_bits(), 2 * c.hd, qn, qn + kn, c.n_rot, pos0, ldp, r0], len * (c.n_head + c.n_kv), cb)?; } bar();
                for &(r0, len) in spans { let boff = (li * s.mmax + r0) * c.n_kv * c.hd;
                    pf.run(ctx, "kb_attn", &[&s.p, &m.kc, &m.vc, &s.a, &mu.kcb, &mu.vcb], &[c.n_head, c.n_kv, cache_off, 2 * c.hd, pos0, (1.0f32 / (c.hd as f32).sqrt()).to_bits(), ldp, al, len, r0, boff, self.bidir as u32], len.div_ceil(16) * c.n_head, cb)?; } bar();
                gemm(&nm("attn_output.weight"), &s.x, c.dim, 0, true)?; bar();
            }
            let (wg, wu) = (wt.wref(&nm("ffn_gate.weight"))?, wt.wref(&nm("ffn_up.weight"))?);
            if wg.arena == wu.arena && prefill::env("PF_SWI", 1) > 0 {
                norm_to(&nm("post_attention_norm.weight"), &s.a2)?;
                pf.gemm_swiglu(ctx, cb, &wt.arenas[wg.arena], &wt.f32b, wg, wu, 0, c.ffn, &s.a2, al, n, &s.a, al, 0)?; bar();
            } else {
                norm_to(&nm("post_attention_norm.weight"), &s.a)?;
                gemm(&nm("ffn_gate.weight"), &s.p, 2 * c.ffn, 0, false)?; gemm(&nm("ffn_up.weight"), &s.p, 2 * c.ffn, c.ffn, false)?; bar();
                pf.run(ctx, "swiglu", &[&s.p, &s.a], &[c.ffn, n, al], (n * c.ffn).div_ceil(256), cb)?; bar();
            }
            gemm(&nm("ffn_down.weight"), &s.x, c.dim, 0, true)?; bar();
        }
        Ok(())
    }
    /// Calibrated logits of every branch in one pass over the weights (groups of up to one chunk of rows).
    pub fn branches(&self, state_len: usize, bs: &[Branch]) -> Result<Vec<Vec<f32>>> {
        if self.multi.is_none() { return bs.iter().map(|b| self.branch(state_len, b)).collect() }
        let mut out = vec![]; let mut i = 0; let cap = self.s.mmax as usize;
        while i < bs.len() {
            let mut j = i; let mut tot = 0; while j < bs.len() && tot + bs[j].ids.len() <= cap { tot += bs[j].ids.len(); j += 1 }
            if j == i { bail!("branch of {} tokens > chunk {}", bs[i].ids.len(), cap) }
            let mut toks = vec![]; let mut spans = vec![];
            for b in &bs[i..j] { spans.push((toks.len() as u32, b.ids.len() as u32)); toks.extend_from_slice(&b.ids); }
            self.embed(&toks)?;
            let nl = self.m.cfg.n_layer as usize; let lps = prefill::env("PF_LPS", 8) as usize;
            for l0 in (0..nl).step_by(lps) { let mut err = None;
                self.m.ctx.run(|cb| if let Err(e) = self.record_multi(cb, &spans, state_len as u32, l0..(l0 + lps).min(nl)) { err = Some(e) })?;
                if let Some(e) = err { return Err(e) } }
            for (b, &(r0, _)) in bs[i..j].iter().zip(&spans) { let mut rows = vec![r0 as usize + b.decide]; rows.extend(b.opts.iter().map(|o| r0 as usize + o)); 
                if self.hsink.borrow().is_some() { let h = self.hidden_rows(&rows[..1])?.remove(0); self.hsink.borrow_mut().as_mut().unwrap().push(h) }
                out.push(if self.slots.is_some() { self.slot_logits(rows[0])? } else { self.head_gpu(&rows)? }); }
            i = j;
        }
        Ok(out)
    }
}
