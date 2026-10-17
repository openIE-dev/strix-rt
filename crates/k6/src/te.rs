//! Kandinsky 6 text encoders.
//!  - Qwen2.5-VL-7B text path (28 layers, d 3584, 28 q / 4 kv heads, qkv bias, RoPE theta 1e6, SwiGLU 18944) on the GPU:
//!    zero-copy BF16 weights through prefill's GEMM, qimage's norm / qk (RoPE) / small (GQA causal attention) / ew kernels,
//!    one submission per layer. Output = last hidden state after the final RMSNorm, template rows [129..) dropped.
//!    Text-only M-RoPE (all three sections share the position) is plain 1D RoPE.
//!  - CLIP-L text model on the CPU (~20 tokens, 123M params): pooled output = final LayerNorm at the first EOS.
use crate::Kern;
use anyhow::{anyhow, Result};
use half::bf16;
use prefill::Pf;
use qimage::{st, Rec, NOG};
use rtcore::WRef;
use std::collections::HashMap;
use vkrt::{Buffer, Ctx};

pub const TEMPLATE: &str = include_str!("qwen_template.txt");
pub const CROP: usize = 129;
const D: u32 = 3584; const KV: u32 = 512; const FFD: u32 = 18944; const QKV: u32 = D + 2 * KV;
const P: &str = "model.layers.";

pub struct Qwen { pub st: st::Model, params: Buffer, lw: Vec<HashMap<&'static str, WRef>>, lp: Vec<HashMap<&'static str, u32>>, nl: usize, fnorm: Vec<f32>, pub load_s: f64 }
impl Qwen {
    pub fn load(ctx: &Ctx, dir: &str) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let st = st::Model::open(ctx, dir, &|k| k.starts_with(P) && k.ends_with("proj.weight"))?;
        let nl = (0..).take_while(|i| st.t.contains_key(&format!("{P}{i}.input_layernorm.weight"))).count();
        let mut pv: Vec<f32> = vec![]; let mut lw = vec![]; let mut lp = vec![];
        let pad = |pv: &mut Vec<f32>| while pv.len() % 64 != 0 { pv.push(0.0) };
        for l in 0..nl { let p = format!("{P}{l}.");
            let mut w = HashMap::new();
            for (k, n) in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")] {
                let wr = st.wref(&format!("{p}{n}.weight"))?; anyhow::ensure!(wr.off % 16 == 0, "{p}{n} not 16-byte aligned (use text_encoder_al)"); w.insert(k, wr); }
            let mut o = HashMap::new();
            for (k, n) in [("in", "input_layernorm"), ("post", "post_attention_layernorm")] { o.insert(k, pv.len() as u32); pv.extend(st.f32(&format!("{p}{n}.weight"))?); pad(&mut pv); }
            o.insert("bias", pv.len() as u32); for n in ["q", "k", "v"] { pv.extend(st.f32(&format!("{p}self_attn.{n}_proj.bias"))?); } pad(&mut pv);
            lw.push(w); lp.push(o); }
        let params = ctx.buffer(pv.len() as u64 * 4)?; params.as_f32_mut()[..pv.len()].copy_from_slice(&pv);
        let fnorm = st.f32("model.norm.weight")?;
        Ok(Qwen { st, params, lw, lp, nl, fnorm, load_s: t0.elapsed().as_secs_f64() })
    }
    /// -> ([L - drop][3584] f32 final-normed hidden states, gpu seconds)
    pub fn encode(&self, ctx: &Ctx, pf: &Pf, kn: &Kern, ids: &[u32], drop: usize) -> Result<(Vec<f32>, f64)> {
        let l = ids.len(); let lu = l as u32; let rows = (l as u64 + 64) & !63;
        let x = ctx.buffer(rows * D as u64 * 4)?; let a = ctx.buffer(rows * FFD as u64 * 2)?; let y = ctx.buffer(rows * 2 * FFD as u64 * 4)?;
        let y2 = ctx.buffer(rows * D as u64 * 4)?; let qh = ctx.buffer(rows * D as u64 * 2)?; let kh = ctx.buffer(rows * KV as u64 * 2)?; let vh = ctx.buffer(rows * KV as u64 * 2)?;
        let rope = ctx.buffer(rows * 128 * 4)?;
        { let e = self.st.bytes("model.embed_tokens.weight")?; let xs = x.as_f32_mut(); let rb = D as usize * 2;
          for (i, &t) in ids.iter().enumerate() { let r = &e[t as usize * rb..(t as usize + 1) * rb];
              for j in 0..D as usize { xs[i * D as usize + j] = bf16::from_le_bytes([r[2 * j], r[2 * j + 1]]).to_f32() } }
          let rt = rope.as_f32_mut();
          for p in 0..l { for i in 0..64 { let inv = 1.0 / 1_000_000f64.powf(2.0 * i as f64 / 128.0); let (s, c) = (p as f64 * inv).sin_cos(); rt[p * 128 + 2 * i] = c as f32; rt[p * 128 + 2 * i + 1] = s as f32; } } }
        let ar = &self.st.bufs; let pr = &self.params; let mut gpu = 0.0;
        for li in 0..self.nl { let w = &self.lw[li]; let p = &self.lp[li]; let mut err = Ok(());
            gpu += ctx.run(|cb| { err = (|| -> Result<()> { let r = Rec { ctx, pf, kn, cb };
                r.run("norm", &[&x, &a, pr, pr], &[D, D, D, p["in"], 0f32.to_bits(), 0, 1e-6f32.to_bits(), 0], lu)?;
                r.gemm(ar, pr, w["q"], &a, D, lu, &y, QKV, 0, false)?; r.gemm(ar, pr, w["k"], &a, D, lu, &y, QKV, D, false)?; r.gemm(ar, pr, w["v"], &a, D, lu, &y, QKV, D + KV, true)?;
                r.run("k6ew", &[&x, &y, &a, pr], &[1, lu * QKV, QKV, QKV, 0, 0, p["bias"], 0], (lu * QKV).div_ceil(256))?;
                r.run("qk", &[&y, &qh, pr, &rope], &[28, QKV, 0, 0, 0, D, 0, 1, 2, 0], lu * 28)?;
                r.run("qk", &[&y, &kh, pr, &rope], &[4, QKV, D, 0, 0, KV, 0, 1, 2, 0], lu * 4)?;
                r.run("qk", &[&y, &vh, pr, &rope], &[4, QKV, D + KV, 0, 0, KV, 0, 1, 0, 0], lu * 4)?;
                r.run("small", &[&qh, &kh, &vh, &a], &[lu, 28, 7, D, KV, KV, D, (1.0f32 / 128f32.sqrt()).to_bits()], lu * 28)?;
                r.gemm(ar, pr, w["o"], &a, D, lu, &y2, D, 0, true)?;
                r.ew(0, lu, D, &x, &y2, D, 0, &a, 0, pr, NOG)?;
                r.run("norm", &[&x, &a, pr, pr], &[D, D, D, p["post"], 0f32.to_bits(), 0, 1e-6f32.to_bits(), 0], lu)?;
                r.gemm(ar, pr, w["gate"], &a, D, lu, &y, 2 * FFD, 0, false)?; r.gemm(ar, pr, w["up"], &a, D, lu, &y, 2 * FFD, FFD, true)?;
                r.ew(2, lu, FFD, &x, &y, 2 * FFD, 0, &a, FFD, pr, 0)?;
                r.gemm(ar, pr, w["down"], &a, FFD, lu, &y2, D, 0, true)?;
                r.ew(0, lu, D, &x, &y2, D, 0, &a, 0, pr, NOG)
            })(); })?; err?; }
        let xs = x.as_f32(); let d = D as usize; let mut out = Vec::with_capacity((l - drop) * d);
        for t in drop..l { let row = &xs[t * d..(t + 1) * d]; let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
            let r = 1.0 / (ms + 1e-6).sqrt(); for j in 0..d { out.push((row[j] as f64 * r) as f32 * self.fnorm[j]); } }
        for b in [x, a, y, y2, qh, kh, vh, rope] { ctx.free_buffer(b) }
        Ok((out, gpu))
    }
}

// ------------------------------------------------------------------------------------------------ CLIP-L (CPU)
pub struct Clip { w: HashMap<String, Vec<f32>>, nl: usize, ids: HashMap<String, u32>, ranks: HashMap<(String, String), usize>, b2u: [char; 256], pub load_s: f64 }
fn b2u() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect(); let mut cs = bs.clone(); let mut n = 0;
    for b in 0..256u32 { if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; } }
    let mut m = [' '; 256]; for (b, c) in bs.iter().zip(&cs) { m[*b as usize] = char::from_u32(*c).unwrap(); } m
}
impl Clip {
    pub fn load(ctx: &Ctx, dir: &str, tok_json: &str) -> Result<Self> {
        let t0 = std::time::Instant::now(); let st = st::Model::open(ctx, dir, &|_| false)?;
        let mut w = HashMap::new(); for k in st.t.keys() { w.insert(k.trim_start_matches("text_model.").to_string(), st.f32(k)?); }
        let nl = (0..).take_while(|i| w.contains_key(&format!("encoder.layers.{i}.layer_norm1.weight"))).count();
        use kev::json::{self, J};
        let j = json::parse(&std::fs::read_to_string(tok_json)?)?; let m = j.get("model").ok_or_else(|| anyhow!("no model"))?;
        let num = |x: &J| match x { J::Num(t) => t.parse().unwrap_or(0), _ => 0 };
        let mut ids = HashMap::new(); if let Some(J::Obj(v)) = m.get("vocab") { for (k, x) in v { ids.insert(k.clone(), num(x)); } }
        let mut ranks = HashMap::new();
        if let Some(J::Arr(v)) = m.get("merges") { for (r, x) in v.iter().enumerate() { let p = match x {
            J::Arr(ab) if ab.len() == 2 => (ab[0].as_str().unwrap_or("").to_string(), ab[1].as_str().unwrap_or("").to_string()),
            J::Str(s) => match s.split_once(' ') { Some((a, b)) => (a.to_string(), b.to_string()), None => continue }, _ => continue };
            ranks.entry(p).or_insert(r); } }
        Ok(Clip { w, nl, ids, ranks, b2u: b2u(), load_s: t0.elapsed().as_secs_f64() })
    }
    fn bpe(&self, word: &str) -> Vec<u32> {
        let chars: Vec<String> = word.bytes().map(|b| self.b2u[b as usize].to_string()).collect();
        let mut parts = chars; if let Some(last) = parts.last_mut() { last.push_str("</w>"); }
        loop { let mut best: Option<(usize, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) { if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) { if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); } } }
            match best { None => break, Some((_, i)) => { let m = format!("{}{}", parts[i], parts[i + 1]); parts.splice(i..i + 2, [m]); } } }
        parts.iter().map(|p| *self.ids.get(p).unwrap_or(&0)).collect()
    }
    /// CLIP tokenizer: lowercase, collapse whitespace, split into contractions / letter runs / single digits / other runs, BPE with </w>.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let t = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase(); let c: Vec<char> = t.chars().collect();
        let mut out = vec![49406u32]; let mut i = 0;
        while i < c.len() { let ch = c[i];
            if ch.is_whitespace() { i += 1; continue }
            if ch == '\'' { let rest: String = c[i..].iter().take(3).collect(); let mut hit = false;
                for s in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"] { if rest.starts_with(s) { out.extend(self.bpe(s)); i += s.chars().count(); hit = true; break } } if hit { continue } }
            let j = if ch.is_alphabetic() { let mut j = i; while j < c.len() && c[j].is_alphabetic() { j += 1 } j }
                else if ch.is_numeric() { i + 1 }
                else { let mut j = i; while j < c.len() && !c[j].is_whitespace() && !c[j].is_alphabetic() && !c[j].is_numeric() { j += 1 } j };
            let word: String = c[i..j].iter().collect(); out.extend(self.bpe(&word)); i = j; }
        out.truncate(76); out.push(49407); out
    }
    fn ln(x: &mut [f32], w: &[f32], b: &[f32]) { let n = x.len() as f32; let m = x.iter().sum::<f32>() / n; let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / n; let r = 1.0 / (v + 1e-5).sqrt(); for i in 0..x.len() { x[i] = (x[i] - m) * r * w[i] + b[i]; } }
    fn lin(&self, x: &[f32], l: usize, name: &str) -> Vec<f32> {   // y[l][n] = x[l][k] W^T + b, threaded over output columns
        let w = &self.w[&format!("{name}.weight")]; let b = &self.w[&format!("{name}.bias")]; let n = b.len(); let k = w.len() / n;
        let mut y = vec![0f32; l * n]; let nt = 16usize; let cols = n.div_ceil(nt);
        let parts: Vec<Vec<f32>> = std::thread::scope(|s| { let hs: Vec<_> = (0..nt).map(|ti| { s.spawn(move || {
            let (c0, c1) = (ti * cols, ((ti + 1) * cols).min(n)); let mut o = vec![0f32; l * (c1.saturating_sub(c0))];
            for r in 0..l { for c in c0..c1 { let mut acc = b[c]; let wr = &w[c * k..(c + 1) * k]; let xr = &x[r * k..(r + 1) * k];
                for q in 0..k { acc += xr[q] * wr[q] } o[r * (c1 - c0) + c - c0] = acc; } } o }) }).collect(); hs.into_iter().map(|h| h.join().unwrap()).collect() });
        for ti in 0..nt { let (c0, c1) = (ti * cols, ((ti + 1) * cols).min(n)); if c1 <= c0 { continue } for r in 0..l { y[r * n + c0..r * n + c1].copy_from_slice(&parts[ti][r * (c1 - c0)..(r + 1) * (c1 - c0)]); } }
        y
    }
    /// pooled [768] for ids (through the first EOS; causal, so padding beyond it is irrelevant)
    pub fn pooled(&self, ids: &[u32]) -> Vec<f32> {
        let eos = ids.iter().position(|&t| t == 49407).unwrap_or(ids.len() - 1); let l = eos + 1; let d = 768;
        let te = &self.w["embeddings.token_embedding.weight"]; let pe = &self.w["embeddings.position_embedding.weight"];
        let mut x: Vec<f32> = (0..l).flat_map(|p| (0..d).map(move |j| (p, j))).map(|(p, j)| te[ids[p] as usize * d + j] + pe[p * d + j]).collect();
        for li in 0..self.nl { let p = format!("encoder.layers.{li}.");
            let mut h = x.clone(); for r in 0..l { Self::ln(&mut h[r * d..(r + 1) * d], &self.w[&format!("{p}layer_norm1.weight")], &self.w[&format!("{p}layer_norm1.bias")]); }
            let q = self.lin(&h, l, &format!("{p}self_attn.q_proj")); let k = self.lin(&h, l, &format!("{p}self_attn.k_proj")); let v = self.lin(&h, l, &format!("{p}self_attn.v_proj"));
            let mut o = vec![0f32; l * d]; let sc = 1.0 / 8.0f32;
            for hd in 0..12 { for i in 0..l { let s: Vec<f32> = (0..=i).map(|j| (0..64).map(|e| q[i * d + hd * 64 + e] * k[j * d + hd * 64 + e]).sum::<f32>() * sc).collect();
                let m = s.iter().cloned().fold(f32::MIN, f32::max); let ex: Vec<f32> = s.iter().map(|a| (a - m).exp()).collect(); let z: f32 = ex.iter().sum();
                for e in 0..64 { o[i * d + hd * 64 + e] = (0..=i).map(|j| ex[j] * v[j * d + hd * 64 + e]).sum::<f32>() / z; } } }
            let ao = self.lin(&o, l, &format!("{p}self_attn.out_proj")); for i in 0..l * d { x[i] += ao[i]; }
            let mut h = x.clone(); for r in 0..l { Self::ln(&mut h[r * d..(r + 1) * d], &self.w[&format!("{p}layer_norm2.weight")], &self.w[&format!("{p}layer_norm2.bias")]); }
            let mut f = self.lin(&h, l, &format!("{p}mlp.fc1")); for a in f.iter_mut() { *a = *a / (1.0 + (-1.702 * *a).exp()); }
            let f2 = self.lin(&f, l, &format!("{p}mlp.fc2")); for i in 0..l * d { x[i] += f2[i]; } }
        let mut last = x[eos * d..(eos + 1) * d].to_vec(); Self::ln(&mut last, &self.w["final_layer_norm.weight"], &self.w["final_layer_norm.bias"]); last
    }
}
