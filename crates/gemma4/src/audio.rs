//! Gemma 4 audio encoder (gemma4a mmproj, "a.*" + "mm.a.*" tensors), CPU f32 reference port of llama.cpp
//! tools/mtmd/models/gemma4a.cpp and mtmd_audio_preprocessor_gemma4a:
//! - log-mel: 16 kHz, 320-sample periodic Hann zero-padded to n_fft 512, hop 160, semicausal left pad 160, magnitude
//!   spectrum, 128 HTK mel bins (no area norm), ln(max(mel, 1e-3)), trimmed to the PyTorch unfold frame count;
//! - SSCP: 2x (conv2d 3x3 stride 2 pad 1 over (freq, time) -> LayerNorm over channels -> ReLU), flatten (c fastest),
//!   a.input_projection;
//! - 12 conformer blocks: FFN half-step, chunked local causal attention (window 12 past incl. self, Transformer-XL
//!   sinusoidal RPE through attn_k_rel, q scale 1/sqrt(d)/ln2 * per_dim_scale, k scale ln(1+e)/ln2, softcap 50),
//!   light conv (pw1 -> GLU -> causal depthwise k=5 -> RMS -> SiLU -> pw2), FFN half-step, ln2;
//! - a.pre_encode.out (+bias) -> weightless RMSNorm -> mm.a.input_projection (LM width).
//! Every linear that has input/output min/max tensors is clamped like Gemma4ClippableLinear.
//! The attention below is the direct form of the blocked graph: for query i, keys j in [i-11, i], score
//! q_i . (k_j + P[i-j]) where P[d] = attn_k_rel . sinusoid(d).
use anyhow::{anyhow, bail, Result};
use gguf::{GgmlType, Gguf};

const EPS: f32 = 1e-6;

struct Lin { w: Vec<f32>, k: usize, n: usize, b: Option<Vec<f32>>, cl: Option<[f32; 4]> }

fn tf32(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
    let d = g.data(t); let n = t.n_elems() as usize; let k = t.dims[0] as usize; let rb = t.row_bytes();
    let mut y = vec![0f32; n];
    if t.ty == GgmlType::F32 { for i in 0..n { y[i] = f32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) } }
    else { for r in 0..n / k { gguf::dequant::dequant_row(t.ty, &d[r * rb..(r + 1) * rb], &mut y[r * k..(r + 1) * k]) } }
    Ok(y)
}
fn lin(g: &Gguf, base: &str) -> Result<Lin> {
    let wn = format!("{base}.weight");
    let t = g.tensor(&wn).ok_or_else(|| anyhow!("missing tensor {wn}"))?;
    let (k, n) = (t.dims[0] as usize, t.dims.get(1).copied().unwrap_or(1) as usize);
    let sc = |s: &str| g.tensor(&format!("{base}.{s}")).map(|_| tf32(g, &format!("{base}.{s}")).map(|v| v[0]));
    let cl = match (sc("input_min"), sc("input_max"), sc("output_min"), sc("output_max")) {
        (Some(a), Some(b), Some(c), Some(d)) => Some([a?, b?, c?, d?]), _ => None };
    let b = if g.tensor(&format!("{base}.bias")).is_some() { Some(tf32(g, &format!("{base}.bias"))?) } else { None };
    Ok(Lin { w: tf32(g, &wn)?, k, n, b, cl })
}

/// y[t][n] = W[n] . x[t] over t rows (threaded over output rows)
fn mm(l: &Lin, x: &[f32], t: usize) -> Vec<f32> {
    let (k, n) = (l.k, l.n);
    let xc: Vec<f32> = match l.cl { Some(c) => x.iter().map(|v| v.clamp(c[0], c[1])).collect(), None => x.to_vec() };
    let mut y = vec![0f32; t * n];
    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(32);
    let per = n.div_ceil(nt);
    let mut cols: Vec<Vec<f32>> = vec![vec![]; nt];
    std::thread::scope(|s| {
        for (ti, out) in cols.iter_mut().enumerate() {
            let (xc, w) = (&xc, &l.w);
            s.spawn(move || {
                let (r0, r1) = (ti * per, ((ti + 1) * per).min(n));
                if r0 >= r1 { return }
                *out = vec![0f32; (r1 - r0) * t];
                for r in r0..r1 { let wr = &w[r * k..(r + 1) * k];
                    for tt in 0..t { let xr = &xc[tt * k..(tt + 1) * k];
                        let mut a = [0f32; 8];
                        for (wc, xc) in wr.chunks_exact(8).zip(xr.chunks_exact(8)) { for i in 0..8 { a[i] += wc[i] * xc[i] } }
                        out[(r - r0) * t + tt] = a.iter().sum();
                    } }
            });
        }
    });
    for (ti, out) in cols.iter().enumerate() { let r0 = ti * per;
        for (j, v) in out.iter().enumerate() { let (r, tt) = (r0 + j / t, j % t); y[tt * n + r] = *v } }
    for tt in 0..t { for r in 0..n {
        let mut v = y[tt * n + r] + l.b.as_ref().map(|b| b[r]).unwrap_or(0.0);
        if let Some(c) = l.cl { v = v.clamp(c[2], c[3]) }
        y[tt * n + r] = v;
    } }
    y
}
fn rms(x: &[f32], d: usize, w: Option<&[f32]>) -> Vec<f32> {
    let mut y = x.to_vec();
    for r in y.chunks_exact_mut(d) {
        let s = (r.iter().map(|v| v * v).sum::<f32>() / d as f32 + EPS).sqrt().recip();
        for (i, v) in r.iter_mut().enumerate() { *v *= s * w.map(|w| w[i]).unwrap_or(1.0) }
    }
    y
}
fn silu(v: f32) -> f32 { v / (1.0 + (-v).exp()) }

struct Ffn { norm: Vec<f32>, up: Lin, down: Lin, post: Vec<f32> }
struct Blk { f1: Ffn, f2: Ffn, pre: Vec<f32>, q: Lin, k: Lin, v: Lin, o: Lin, pds: Vec<f32>, pdk: Option<Vec<f32>>, krel: Lin, post: Vec<f32>,
    cnorm: Vec<f32>, pw1: Lin, dw: Vec<f32>, nconv: Vec<f32>, pw2: Lin, ln2: Vec<f32> }

pub struct Audio { conv: [(Vec<f32>, Vec<f32>, usize, usize); 2], inp: Lin, blk: Vec<Blk>, outp: Lin, mmp: Lin, soft: Option<Vec<f32>>,
    pub e: usize, pub n_head: usize, pub n_mel: usize, pub out_dim: usize, filt: Vec<f32> }

const NFFT: usize = 512; const WIN: usize = 320; const HOP: usize = 160; const SR: usize = 16000;

impl Audio {
    pub fn load(path: &str) -> Result<Self> {
        let g = Gguf::open(path)?;
        let e = g.u64("clip.audio.embedding_length")? as usize; let n_head = g.u64("clip.audio.attention.head_count")? as usize;
        let nl = g.u64("clip.audio.block_count")? as usize; let n_mel = g.u64("clip.audio.num_mel_bins")? as usize;
        let f = |n: &str| tf32(&g, n);
        let opt = |n: &str| if g.tensor(n).is_some() { tf32(&g, n).map(Some) } else { Ok(None) };
        let cv = |i: usize| -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
            let t = g.tensor(&format!("a.conv1d.{i}.weight")).ok_or_else(|| anyhow!("no conv {i}"))?;
            if t.dims[0] != 3 || t.dims[1] != 3 { bail!("conv {i} kernel {:?}", t.dims) }
            if g.tensor(&format!("a.conv1d.{i}.bias")).is_some() { bail!("conv bias unsupported") }
            Ok((f(&format!("a.conv1d.{i}.weight"))?, f(&format!("a.conv1d.{i}.norm.weight"))?, t.dims[2] as usize, t.dims[3] as usize))
        };
        let conv = [cv(0)?, cv(1)?];
        let mut blk = vec![];
        for l in 0..nl {
            let p = |s: &str| format!("a.blk.{l}.{s}");
            let ffn = |sfx: &str| -> Result<Ffn> { Ok(Ffn { norm: f(&p(&format!("ffn_norm{sfx}.weight")))?, up: lin(&g, &p(&format!("ffn_up{sfx}")))?,
                down: lin(&g, &p(&format!("ffn_down{sfx}")))?, post: f(&p(&format!("ffn_post_norm{sfx}.weight")))? }) };
            if g.tensor(&p("conv_dw.bias")).is_some() || g.tensor(&p("attn_out.bias")).is_some() { bail!("unexpected bias in block {l}") }
            blk.push(Blk { f1: ffn("")?, f2: ffn("_1")?, pre: f(&p("attn_pre_norm.weight"))?, q: lin(&g, &p("attn_q"))?, k: lin(&g, &p("attn_k"))?,
                v: lin(&g, &p("attn_v"))?, o: lin(&g, &p("attn_out"))?, pds: f(&p("per_dim_scale.weight"))?, pdk: opt(&p("per_dim_k_scale.weight"))?,
                krel: lin(&g, &p("attn_k_rel"))?, post: f(&p("attn_post_norm.weight"))?,
                // gguf names swapped upstream: "conv_norm" is the pre-pw1 norm, "norm_conv" the post-depthwise one (clip.cpp)
                cnorm: f(&p("conv_norm.weight"))?, pw1: lin(&g, &p("conv_pw1"))?, dw: f(&p("conv_dw.weight"))?, nconv: f(&p("norm_conv.weight"))?,
                pw2: lin(&g, &p("conv_pw2"))?, ln2: f(&p("ln2.weight"))? });
        }
        let outp = lin(&g, "a.pre_encode.out")?; let mmp = lin(&g, "mm.a.input_projection")?;
        let out_dim = mmp.n;
        Ok(Audio { conv, inp: lin(&g, "a.input_projection")?, blk, outp, mmp, soft: opt("mm.a.soft_emb_norm.weight")?, e, n_head, n_mel, out_dim, filt: filterbank(n_mel) })
    }

    /// log-mel [frames][n_mel] for one <= 30 s chunk of 16 kHz mono samples
    pub fn mel(&self, x: &[f32]) -> Result<(Vec<f32>, usize)> {
        if x.is_empty() || x.len() > 30 * SR { bail!("audio must be 0..30 s (got {} samples)", x.len()) }
        let pad_left = WIN / 2; let n_with_left = x.len() + pad_left;
        if n_with_left < WIN + 1 { bail!("audio too short") }
        let pt = (n_with_left - (WIN + 1)) / HOP + 1;
        let need = (pt - 1) * HOP + NFFT;
        let total_pad = (need as i64 - x.len() as i64).max(pad_left as i64) as usize;
        let mut s = vec![0f32; total_pad + x.len()]; s[pad_left..pad_left + x.len()].copy_from_slice(x);
        let n_len = ((s.len() - NFFT) / HOP + 1).min(pt);
        let hann: Vec<f32> = (0..WIN).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / WIN as f32).cos()).collect();
        let nb = NFFT / 2 + 1;
        let (cs, sn): (Vec<f64>, Vec<f64>) = (0..NFFT).map(|i| { let th = 2.0 * std::f64::consts::PI * i as f64 / NFFT as f64; (th.cos(), th.sin()) }).unzip();
        let mut out = vec![0f32; n_len * self.n_mel];
        for i in 0..n_len {
            let o = i * HOP; let fr: Vec<f64> = (0..WIN).map(|j| (hann[j] * s.get(o + j).copied().unwrap_or(0.0)) as f64).collect();
            let mag: Vec<f64> = (0..nb).map(|k| { let (mut re, mut im) = (0f64, 0f64);
                for (j, v) in fr.iter().enumerate() { let p = (j * k) % NFFT; re += v * cs[p]; im -= v * sn[p]; } (re * re + im * im).sqrt() }).collect();
            for m in 0..self.n_mel {
                let sum: f64 = (0..nb).map(|k| mag[k] * self.filt[m * nb + k] as f64).sum();
                out[i * self.n_mel + m] = sum.max(1e-3).ln() as f32;
            }
        }
        Ok((out, n_len))
    }

    /// embeddings [tokens][out_dim] for a log-mel [frames][n_mel]
    pub fn encode_mel(&self, mel: &[f32], nf: usize) -> Result<(Vec<f32>, usize)> {
        // SSCP: layout [t][f][c]
        let (mut x, mut tl, mut fl, mut cl) = (mel.to_vec(), nf, self.n_mel, 1usize);
        for (w, nw, ci, co) in &self.conv {
            if *ci != cl { bail!("conv cin {ci} != {cl}") }
            let (t2, f2) = ((tl - 1) / 2 + 1, (fl - 1) / 2 + 1);
            let mut y = vec![0f32; t2 * f2 * co];
            let xr = &x;
            let nt = 32usize; let per = t2.div_ceil(nt);
            std::thread::scope(|s| { for (ti, ych) in y.chunks_mut(per * f2 * co).enumerate() { s.spawn(move || {
                for (tt, yrow) in ych.chunks_mut(f2 * co).enumerate() { let t = ti * per + tt;
                    for f in 0..f2 { for o in 0..*co {
                        let mut a = 0f32;
                        for kh in 0..3 { let it = (2 * t + kh) as i64 - 1; if it < 0 || it >= tl as i64 { continue }
                            for kw in 0..3 { let iff = (2 * f + kw) as i64 - 1; if iff < 0 || iff >= fl as i64 { continue }
                                let xb = (it as usize * fl + iff as usize) * cl;
                                for c in 0..cl { a += w[kw + 3 * (kh + 3 * (c + cl * o))] * xr[xb + c] } } }
                        yrow[f * co + o] = a;
                    } }
                } }); } });
            for r in y.chunks_exact_mut(*co) {   // LayerNorm over channels, weight only, then ReLU
                let m = r.iter().sum::<f32>() / *co as f32; let v = r.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / *co as f32;
                let s = (v + EPS).sqrt().recip();
                for (i, a) in r.iter_mut().enumerate() { *a = ((*a - m) * s * nw[i]).max(0.0) }
            }
            x = y; tl = t2; fl = f2; cl = *co;
        }
        let n = tl; let e = self.e; let (h, dh) = (self.n_head, self.e / self.n_head);
        if fl * cl != self.inp.k { bail!("sscp flatten {} != {}", fl * cl, self.inp.k) }
        // [t][f][c] with c fastest == ggml permute(1,2,0) + reshape (feature c + C*f)
        let mut r = mm(&self.inp, &x, n);
        // sinusoid RPE for distances 0..12 (pos_emb row p = distance 12-p; direct-form index = distance)
        let nts = e / 2; let inc = (10000f32).ln() / (nts.max(2) - 1) as f32;
        let mut pe = vec![0f32; 13 * e];
        for d in 0..13 { for i in 0..nts { let a = d as f32 * (-(i as f32) * inc).exp(); pe[d * e + i] = a.sin(); pe[d * e + i + nts] = a.cos(); } }
        let qs = (1.0 / (dh as f32).sqrt()) / 2f32.ln(); let ks = (1.0 + 1f32.exp()).ln() / 2f32.ln();
        let ffn = |f: &Ffn, r: &mut Vec<f32>| {
            let u = mm(&f.up, &rms(r, e, Some(&f.norm)), n).into_iter().map(silu).collect::<Vec<_>>();
            let d = rms(&mm(&f.down, &u, n), e, Some(&f.post));
            for (a, b) in r.iter_mut().zip(d) { *a += 0.5 * b }
        };
        for b in &self.blk {
            ffn(&b.f1, &mut r);
            let hn = rms(&r, e, Some(&b.pre));
            let (mut q, mut k, v) = (mm(&b.q, &hn, n), mm(&b.k, &hn, n), mm(&b.v, &hn, n));
            for t in 0..n { for hh in 0..h { for d in 0..dh { let i = t * e + hh * dh + d;
                q[i] *= qs * b.pds[d]; k[i] *= ks * b.pdk.as_ref().map(|w| w[d]).unwrap_or(1.0); } } }
            let p = mm(&b.krel, &pe, 13);
            let mut o = vec![0f32; n * e];
            for t in 0..n { for hh in 0..h {
                let qv = &q[t * e + hh * dh..t * e + (hh + 1) * dh];
                let j0 = t.saturating_sub(11);
                let mut sc: Vec<f32> = (j0..=t).map(|j| {
                    let (kv, pv) = (&k[j * e + hh * dh..j * e + (hh + 1) * dh], &p[(t - j) * e + hh * dh..(t - j) * e + (hh + 1) * dh]);
                    let s: f32 = (0..dh).map(|d| qv[d] * (kv[d] + pv[d])).sum(); 50.0 * (s / 50.0).tanh() }).collect();
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max); let mut z = 0f32;
                for s in sc.iter_mut() { *s = (*s - mx).exp(); z += *s }
                for (jj, j) in (j0..=t).enumerate() { let wgt = sc[jj] / z;
                    for d in 0..dh { o[t * e + hh * dh + d] += wgt * v[j * e + hh * dh + d] } }
            } }
            let a = rms(&mm(&b.o, &o, n), e, Some(&b.post));
            for (x, y) in r.iter_mut().zip(a) { *x += y }
            // light conv
            let g2 = mm(&b.pw1, &rms(&r, e, Some(&b.cnorm)), n);
            let gl: Vec<f32> = (0..n * e).map(|i| { let (t, c) = (i / e, i % e); g2[t * 2 * e + c] / (1.0 + (-g2[t * 2 * e + e + c]).exp()) }).collect();
            let mut dw = vec![0f32; n * e];
            for t in 0..n { for c in 0..e { let mut a = 0f32;
                for kk in 0..5 { let ti = t as i64 - 4 + kk as i64; if ti >= 0 { a += b.dw[kk + 5 * c] * gl[ti as usize * e + c] } }
                dw[t * e + c] = a; } }
            let dn: Vec<f32> = rms(&dw, e, Some(&b.nconv)).into_iter().map(silu).collect();
            let c2 = mm(&b.pw2, &dn, n);
            for (x, y) in r.iter_mut().zip(c2) { *x += y }
            ffn(&b.f2, &mut r);
            r = rms(&r, e, Some(&b.ln2));
        }
        let y = rms(&mm(&self.outp, &r, n), self.outp.n, self.soft.as_deref());
        Ok((mm(&self.mmp, &y, n), n))
    }

    pub fn encode(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> { let (m, nf) = self.mel(samples)?; self.encode_mel(&m, nf) }
}

/// HTK mel filterbank [n_mel][257], 0..8 kHz, no area normalization (mtmd_audio_cache::fill_mel_filterbank_matrix)
fn filterbank(n_mel: usize) -> Vec<f32> {
    let hz2m = |f: f64| 2595.0 * (1.0 + f / 700.0).log10(); let m2hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (lo, hi) = (hz2m(0.0), hz2m(SR as f64 / 2.0));
    let hz: Vec<f64> = (0..n_mel + 2).map(|i| m2hz(lo + (hi - lo) * (i as f64 / (n_mel + 1) as f64))).collect();
    let nb = NFFT / 2 + 1; let step = SR as f64 / NFFT as f64;
    let mut out = vec![0f32; n_mel * nb];
    for m in 0..n_mel { let (l, c, r) = (hz[m], hz[m + 1], hz[m + 2]);
        for k in 0..nb { let f = k as f64 * step;
            let w = if f >= l && f <= c { (f - l) / (c - l).max(1e-30) } else if f > c && f <= r { (r - f) / (r - c).max(1e-30) } else { 0.0 };
            out[m * nb + k] = w as f32; } }
    out
}

/// 16 kHz mono WAV (PCM16 or float32; multi-channel is averaged)
pub fn read_wav(path: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    if &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" { bail!("not a WAV file") }
    let (mut p, mut fmt, mut ch, mut sr, mut bits) = (12usize, 0u16, 1u16, 0u32, 0u16);
    let u16a = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]); let u32a = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    while p + 8 <= b.len() {
        let (id, sz) = (&b[p..p + 4], u32a(p + 4) as usize);
        if id == b"fmt " { fmt = u16a(p + 8); ch = u16a(p + 10); sr = u32a(p + 12); bits = u16a(p + 22); if fmt == 0xFFFE { fmt = u16a(p + 32) } }
        if id == b"data" {
            if sr != SR as u32 { bail!("WAV is {sr} Hz; need 16 kHz") }
            let d = &b[p + 8..(p + 8 + sz).min(b.len())]; let ch = ch.max(1) as usize;
            let s: Vec<f32> = match (fmt, bits) {
                (1, 16) => d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
                (3, 32) => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                _ => bail!("WAV format {fmt}/{bits} unsupported") };
            return Ok(s.chunks_exact(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect());
        }
        p += 8 + sz + (sz & 1);
    }
    bail!("no data chunk")
}
