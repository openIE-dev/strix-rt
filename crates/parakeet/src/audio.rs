//! WAV reading and the Parakeet (NeMo / HF ParakeetFeatureExtractor) log-mel front end, on the CPU.
use anyhow::{bail, Result};

/// Read a RIFF WAV (PCM16/32 or float32, any channel count, averaged to mono). Returns (samples, sample_rate).
pub fn read_wav(path: &str) -> Result<(Vec<f32>, u32)> {
    let b = std::fs::read(path)?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" { bail!("{path}: not a WAV file") }
    let (mut p, mut fmt, mut ch, mut sr, mut bits) = (12usize, 0u16, 1u16, 0u32, 16u16);
    while p + 8 <= b.len() {
        let id = &b[p..p + 4]; let n = u32::from_le_bytes(b[p + 4..p + 8].try_into()?) as usize; let d = &b[p + 8..(p + 8 + n).min(b.len())];
        if id == b"fmt " {
            fmt = u16::from_le_bytes([d[0], d[1]]); ch = u16::from_le_bytes([d[2], d[3]]); sr = u32::from_le_bytes(d[4..8].try_into()?); bits = u16::from_le_bytes([d[14], d[15]]);
            if fmt == 0xFFFE && d.len() >= 26 { fmt = u16::from_le_bytes([d[24], d[25]]); }
        } else if id == b"data" {
            let ch = ch.max(1) as usize;
            let s: Vec<f32> = match (fmt, bits) {
                (1, 16) => d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
                (1, 32) => d.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32 / 2147483648.0).collect(),
                (3, 32) => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                _ => bail!("{path}: unsupported WAV format {fmt}/{bits}"),
            };
            return Ok((s.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect(), sr));
        }
        p += 8 + n + (n & 1);
    }
    bail!("{path}: no data chunk")
}

pub const N_MELS: usize = 128;
const NFFT: usize = 512; const WIN: usize = 400; const HOP: usize = 160; const NB: usize = NFFT / 2 + 1;

pub struct MelFrontend { win: Vec<f64>, fb: Vec<f32>, tw: Vec<(f64, f64)> }

fn hz_to_mel(f: f64) -> f64 { if f < 1000.0 { 3.0 * f / 200.0 } else { 15.0 + (f / 1000.0).ln() / (6.4f64.ln() / 27.0) } }
fn mel_to_hz(m: f64) -> f64 { if m < 15.0 { 200.0 * m / 3.0 } else { 1000.0 * ((m - 15.0) * (6.4f64.ln() / 27.0)).exp() } }

impl MelFrontend {
    pub fn new() -> Self {
        // symmetric Hann(400) centred in the 512-point frame
        let mut win = vec![0.0; NFFT];
        for n in 0..WIN { win[(NFFT - WIN) / 2 + n] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos(); }
        // librosa slaney mel filterbank (htk=False, norm="slaney"), fmin 0, fmax 8000
        let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let hz: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
        let mut fb = vec![0f32; N_MELS * NB];
        for m in 0..N_MELS { for k in 0..NB {
            let f = k as f64 * 16000.0 / NFFT as f64;
            let w = ((f - hz[m]) / (hz[m + 1] - hz[m])).min((hz[m + 2] - f) / (hz[m + 2] - hz[m + 1])).max(0.0);
            fb[m * NB + k] = (w * 2.0 / (hz[m + 2] - hz[m])) as f32; } }
        let tw = (0..NFFT / 2).map(|k| { let a = -2.0 * std::f64::consts::PI * k as f64 / NFFT as f64; (a.cos(), a.sin()) }).collect();
        MelFrontend { win, fb, tw }
    }
    fn fft(&self, re: &mut [f64], im: &mut [f64]) {
        let n = re.len(); let mut j = 0;
        for i in 1..n { let mut bit = n >> 1; while j & bit != 0 { j ^= bit; bit >>= 1; } j |= bit; if i < j { re.swap(i, j); im.swap(i, j); } }
        let mut len = 2;
        while len <= n { let step = n / len;
            for s in (0..n).step_by(len) { for k in 0..len / 2 {
                let (c, si) = self.tw[k * step]; let (a, b) = (s + k, s + k + len / 2);
                let (xr, xi) = (re[b] * c - im[b] * si, re[b] * si + im[b] * c);
                re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; } }
            len <<= 1; }
    }
    /// Normalized log-mel features of the valid frames, [T][128] with T = len / 160.
    pub fn features(&self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len(); let t = n / HOP;
        let mut x = vec![0f64; n + NFFT];
        for i in 0..n { x[NFFT / 2 + i] = audio[i] as f64 - if i > 0 { 0.97 * audio[i - 1] as f64 } else { 0.0 }; }
        let nth = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(16);
        let per = t.div_ceil(nth).max(1);
        let mut out = vec![0f32; t * N_MELS];
        std::thread::scope(|sc| { for (ci, ch) in out.chunks_mut(per * N_MELS).enumerate() { let x = &x; sc.spawn(move || {
            let (mut re, mut im, mut pw) = (vec![0f64; NFFT], vec![0f64; NFFT], vec![0f32; NB]);
            for (fi, row) in ch.chunks_mut(N_MELS).enumerate() {
                let f = ci * per + fi;
                for i in 0..NFFT { re[i] = x[f * HOP + i] * self.win[i]; im[i] = 0.0; }
                self.fft(&mut re, &mut im);
                for k in 0..NB { pw[k] = (re[k] * re[k] + im[k] * im[k]) as f32; }
                for m in 0..N_MELS { let s: f32 = self.fb[m * NB..(m + 1) * NB].iter().zip(&pw).map(|(a, b)| a * b).sum(); row[m] = (s + 5.960464477539063e-8).ln(); }
            } }); } });
        for m in 0..N_MELS {
            let mean = (0..t).map(|f| out[f * N_MELS + m] as f64).sum::<f64>() / t as f64;
            let var = (0..t).map(|f| { let d = out[f * N_MELS + m] as f64 - mean; d * d }).sum::<f64>() / (t as f64 - 1.0).max(1.0);
            let sd = var.sqrt() + 1e-5;
            for f in 0..t { out[f * N_MELS + m] = ((out[f * N_MELS + m] as f64 - mean) / sd) as f32; }
        }
        out
    }
}

/// Minimal .npy reader (little-endian f32, C or Fortran order 2-D) -> (row-major data, shape).
pub fn read_npy(path: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let b = std::fs::read(path)?; let hl = u16::from_le_bytes([b[8], b[9]]) as usize; let h = std::str::from_utf8(&b[10..10 + hl])?;
    if !h.contains("'<f4'") { bail!("{path}: not f32") }
    let fortran = h.contains("'fortran_order': True");
    let sh = &h[h.find("'shape': (").unwrap() + 10..]; let sh = &sh[..sh.find(')').unwrap()];
    let shape: Vec<usize> = sh.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let d: Vec<f32> = b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    if fortran && shape.len() == 2 { let (r, c) = (shape[0], shape[1]); let mut o = vec![0f32; r * c];
        for i in 0..r { for j in 0..c { o[i * c + j] = d[j * r + i]; } } return Ok((o, shape)); }
    Ok((d, shape))
}
