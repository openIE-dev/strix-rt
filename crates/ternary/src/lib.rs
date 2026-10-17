//! Ternary (1.58-bit) weight formats for strix-rt: CPU reference dequant, Walsh-Hadamard
//! reference, safetensors reader, and SPIR-V for the GPU kernels (tern_*.comp).
//!
//! Formats (byte layouts copied from the producers' source; see docs/ternary.md):
//! * `Pq2_0`  PrismML ggml type 142: block 128 = { f16 d; u8 qs[32] } (34 B, 2.125 bpw).
//!            elem j = ((qs[j/4] >> 2*(j%4)) & 3) - 1  (00=-1, 01=0, 10=+1).
//! * `Ptq1_0` PrismML ggml type 143: block 128 = { u8 qs[24]; u8 qh[2]; f16 d } (28 B, 1.75 bpw).
//!            TQ1_0-style base-3 "fixed point" trits; elems 0..80 from qs[0..16] (5 trits each,
//!            trit-major), 80..120 from qs[16..24], 120..128 from qh[0..2] (4 trits each).
//!            trit n of byte b: ((u8)(b*3^n) * 3) >> 8, value-1.
//! * `Tq2_1`  KittenML ggml type 43: block 256 = { u8 qs[64]; f16 d[2] } (68 B, 2.125 bpw).
//!            elem g*128 + l*32 + m = ((qs[g*32+m] >> 2l) & 3) - 1, scale d[g].
//! * `Thrush5` moondream/kitten "thrush-ternary-v2" safetensors: row-major, ceil(K/5) B/row,
//!            elem i = base-3 digit (i%5) of byte i/5 (LSD first), w = scale[row, i/128]*(code-1),
//!            scales in a separate tensor. GPU kernel uses K%640==0 rows (128 B per 640 elems).
pub mod st;
pub mod gg;

pub const PQ2_0: u32 = 142;
pub const PTQ1_0: u32 = 143;
pub const TQ2_1: u32 = 43;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt { Pq2_0, Ptq1_0, Tq2_1, Thrush5 }

impl Fmt {
    pub fn from_ggml(t: u32) -> Option<Fmt> { match t { PQ2_0 => Some(Fmt::Pq2_0), PTQ1_0 => Some(Fmt::Ptq1_0), TQ2_1 => Some(Fmt::Tq2_1), _ => None } }
    /// (elements per block, bytes per block) for the ggml block formats.
    pub fn block(&self) -> (usize, usize) { match self { Fmt::Pq2_0 => (128, 34), Fmt::Ptq1_0 => (128, 28), Fmt::Tq2_1 => (256, 68), Fmt::Thrush5 => (5, 1) } }
    pub fn row_bytes(&self, k: usize) -> usize { let (e, b) = self.block(); k.div_ceil(e) * b }
    pub fn name(&self) -> &'static str { match self { Fmt::Pq2_0 => "PQ2_0", Fmt::Ptq1_0 => "PTQ1_0", Fmt::Tq2_1 => "TQ2_1", Fmt::Thrush5 => "thrush5" } }
    pub fn spv_tag(&self) -> &'static str { match self { Fmt::Pq2_0 => "pq2", Fmt::Ptq1_0 => "ptq1", Fmt::Tq2_1 => "tq21", Fmt::Thrush5 => "th5" } }
}

#[inline] pub fn f16(b: &[u8], o: usize) -> f32 { half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32() }

#[inline] fn trit_fp(b: u8, n: usize) -> i32 { const P3: [u8; 6] = [1, 3, 9, 27, 81, 243]; let q = b.wrapping_mul(P3[n]); ((q as u16 * 3) >> 8) as i32 - 1 }

/// Ternary codes (-1/0/+1) of one block plus its scale(s). Returns codes into `c`.
pub fn block_codes(f: Fmt, b: &[u8], c: &mut [i8], d: &mut [f32]) {
    match f {
        Fmt::Pq2_0 => { d[0] = f16(b, 0); for j in 0..128 { c[j] = (((b[2 + j / 4] >> (2 * (j % 4))) & 3) as i8) - 1; } }
        Fmt::Ptq1_0 => {
            let (qs, qh) = (&b[0..24], &b[24..26]); d[0] = f16(b, 26);
            let mut y = 0;
            for (lo, cnt) in [(0usize, 16usize), (16, 8)] { for n in 0..5 { for m in 0..cnt { c[y] = trit_fp(qs[lo + m], n) as i8; y += 1; } } }
            for n in 0..4 { for h in 0..2 { c[y] = trit_fp(qh[h], n) as i8; y += 1; } }
        }
        Fmt::Tq2_1 => { d[0] = f16(b, 64); d[1] = f16(b, 66);
            for g in 0..2 { for l in 0..4 { for m in 0..32 { c[g * 128 + l * 32 + m] = (((b[g * 32 + m] >> (2 * l)) & 3) as i8) - 1; } } } }
        Fmt::Thrush5 => unreachable!("use thrush_row"),
    }
}

/// Dequantize one row of a ggml-block format (K elements) into y.
pub fn dequant_row(f: Fmt, row: &[u8], y: &mut [f32]) {
    let (e, bs) = f.block();
    let mut c = [0i8; 256]; let mut d = [0f32; 2];
    for (bi, yb) in y.chunks_mut(e).enumerate() {
        block_codes(f, &row[bi * bs..(bi + 1) * bs], &mut c, &mut d);
        for (j, v) in yb.iter_mut().enumerate() { *v = d[j / 128] * c[j] as f32; }
    }
}

/// kittenml flat stream: trits of row `row` (K wide) start at global trit row*K.
pub fn thrush_row_flat(packed: &[u8], row: usize, k: usize, scales: &[f32], y: &mut [f32]) {
    for i in 0..k { let g = row * k + i; let mut b = packed[g / 5] as u32; for _ in 0..(g % 5) { b /= 3; } y[i] = scales[i / 128] * ((b % 3) as f32 - 1.0); }
}

/// Thrush5 row: codes from `packed` (ceil(K/5) bytes), scales[K/128] (already f32).
pub fn thrush_row(packed: &[u8], scales: &[f32], y: &mut [f32]) {
    for (i, v) in y.iter_mut().enumerate() {
        let mut b = packed[i / 5] as u32; for _ in 0..(i % 5) { b /= 3; }
        *v = scales[i / 128] * ((b % 3) as f32 - 1.0);
    }
}

/// Reference online transform used by Bonsai 2: y = H_n (perm(x) * signs) per n-block,
/// H_n the normalized Sylvester Walsh-Hadamard matrix (entry (-1)^popcount(r&c) / sqrt(n)).
/// `perm` = Some((hd, nk, rep)) applies the GDN v-grouped reorder [hd,nk,rep] -> [hd,rep,nk].
pub fn fwht_ref(x: &[f32], signs: Option<&[f32]>, n: usize, perm: Option<(usize, usize, usize)>) -> Vec<f32> {
    let k = x.len();
    let mut v: Vec<f32> = match perm {
        None => x.to_vec(),
        Some((hd, nk, rep)) => (0..k).map(|o| { let h = o % hd; let r = (o / hd) % rep; let kk = o / (hd * rep); x[h + hd * (kk + nk * r)] }).collect(),
    };
    if let Some(s) = signs { for (a, b) in v.iter_mut().zip(s) { *a *= b; } }
    let sc = 1.0 / (n as f32).sqrt();
    for blk in v.chunks_mut(n) {
        let mut h = 1; while h < n { for i in (0..n).step_by(2 * h) { for j in i..i + h { let (a, b) = (blk[j], blk[j + h]); blk[j] = a + b; blk[j + h] = a - b; } } h *= 2; }
        for a in blk.iter_mut() { *a *= sc; }
    }
    v
}

pub mod shaders {
    pub const FWHT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tern_fwht.spv"));
    pub const QUANTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tern_quantx.spv"));
    pub const GEMM_PQ2: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tern_gemm_pq2.spv"));
    pub const GEMM_PTQ1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tern_gemm_ptq1.spv"));
    macro_rules! g { ($($n:literal),*) => { pub fn gemv(tag: &str, t: u32, nr: u32) -> &'static [u8] { match format!("{tag}_t{t}_r{nr}").as_str() { $( $n => include_bytes!(concat!(env!("OUT_DIR"), "/tern_gemv_", $n, ".spv")), )* o => panic!("no gemv variant {o}") } } } }
    g!("pq2_t4_r1","pq2_t4_r2","pq2_t4_r4","pq2_t8_r1","pq2_t8_r2","pq2_t8_r4","pq2_t16_r1","pq2_t16_r2","pq2_t16_r4","pq2_t32_r1","pq2_t32_r2","pq2_t32_r4",
       "ptq1_t4_r1","ptq1_t4_r2","ptq1_t4_r4","ptq1_t8_r1","ptq1_t8_r2","ptq1_t8_r4","ptq1_t16_r1","ptq1_t16_r2","ptq1_t16_r4","ptq1_t32_r1","ptq1_t32_r2","ptq1_t32_r4",
       "tq21_t4_r1","tq21_t4_r2","tq21_t4_r4","tq21_t8_r1","tq21_t8_r2","tq21_t8_r4","tq21_t16_r1","tq21_t16_r2","tq21_t16_r4","tq21_t32_r1","tq21_t32_r2","tq21_t32_r4",
       "th5_t4_r1","th5_t4_r2","th5_t4_r4","th5_t8_r1","th5_t8_r2","th5_t8_r4","th5_t16_r1","th5_t16_r2","th5_t16_r4","th5_t32_r1","th5_t32_r2","th5_t32_r4");
}
