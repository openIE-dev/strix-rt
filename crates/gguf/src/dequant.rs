//! CPU reference dequantization (bit-exact with ggml's dequantize_row_*).
use crate::GgmlType;
use half::f16;
#[inline] fn h(b: &[u8], o: usize) -> f32 { f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32() }

#[inline] pub fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) } else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}
pub fn q4k_block(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0); let dmin = h(b, 2); let sc = &b[4..16]; let mut q = &b[16..144];
    let mut is = 0; let mut o = 0;
    for _ in 0..4 {
        let (s1, m1) = q4k_scale_min(is, sc); let (s2, m2) = q4k_scale_min(is + 1, sc);
        let (d1, mm1, d2, mm2) = (d * s1 as f32, dmin * m1 as f32, d * s2 as f32, dmin * m2 as f32);
        for l in 0..32 { y[o + l] = d1 * (q[l] & 0xF) as f32 - mm1; }
        for l in 0..32 { y[o + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2; }
        q = &q[32..]; is += 2; o += 64;
    }
}
pub fn q6k_block(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    for n in 0..2 {
        let ql = &b[n * 64..]; let qh = &b[128 + n * 32..]; let sc = &b[192 + n * 8..]; let yo = &mut y[n * 128..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            yo[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            yo[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            yo[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            yo[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}
pub const KV_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
pub fn q5k_block(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0); let dmin = h(b, 2); let sc = &b[4..16]; let qh = &b[16..48]; let mut ql = &b[48..176];
    let (mut is, mut o, mut u1, mut u2) = (0, 0, 1u8, 2u8);
    for _ in 0..4 {
        let (s1, m1) = q4k_scale_min(is, sc); let (s2, m2) = q4k_scale_min(is + 1, sc);
        for l in 0..32 { y[o + l] = d * s1 as f32 * ((ql[l] & 0xF) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 }) - dmin * m1 as f32; }
        for l in 0..32 { y[o + 32 + l] = d * s2 as f32 * ((ql[l] >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 }) - dmin * m2 as f32; }
        ql = &ql[32..]; is += 2; o += 64; u1 <<= 2; u2 <<= 2;
    }
}
pub fn iq4nl_block(b: &[u8], y: &mut [f32]) { let d = h(b, 0); for j in 0..16 { y[j] = d * KV_IQ4NL[(b[2 + j] & 0xF) as usize] as f32; y[j + 16] = d * KV_IQ4NL[(b[2 + j] >> 4) as usize] as f32; } }
pub fn iq4xs_block(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0); let sh = u16::from_le_bytes([b[2], b[3]]);
    for ib in 0..8 {
        let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32; let qs = &b[8 + 16 * ib..];
        for j in 0..16 { y[32 * ib + j] = dl * KV_IQ4NL[(qs[j] & 0xF) as usize] as f32; y[32 * ib + j + 16] = dl * KV_IQ4NL[(qs[j] >> 4) as usize] as f32; }
    }
}
pub fn q3k_block(b: &[u8], y: &mut [f32]) {
    let d_all = h(b, 108); let hm = &b[0..32];
    let mut aux = [0u32; 4]; for i in 0..3 { aux[i] = u32::from_le_bytes(b[96 + 4 * i..100 + 4 * i].try_into().unwrap()); }
    let (k1, k2) = (0x03030303u32, 0x0f0f0f0fu32); let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4); aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4); aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    let sc: Vec<i8> = aux.iter().flat_map(|a| a.to_le_bytes()).map(|v| v as i8).collect();
    let (mut is, mut o, mut m) = (0, 0, 1u8);
    for n in 0..2 { let q = &b[32 + 32 * n..]; let mut shift = 0;
        for _ in 0..4 {
            let dl = d_all * (sc[is] as i32 - 32) as f32; is += 1;
            for l in 0..16 { y[o] = dl * (((q[l] >> shift) & 3) as i32 - if hm[l] & m != 0 { 0 } else { 4 }) as f32; o += 1; }
            let dl = d_all * (sc[is] as i32 - 32) as f32; is += 1;
            for l in 0..16 { y[o] = dl * (((q[l + 16] >> shift) & 3) as i32 - if hm[l + 16] & m != 0 { 0 } else { 4 }) as f32; o += 1; }
            shift += 2; m <<= 1;
        } }
}
pub fn iq3s_block(b: &[u8], y: &mut [f32]) {
    use crate::iq3s_grid::IQ3S_GRID;
    let d = h(b, 0); let mut o = 0;
    for ib32 in (0..8).step_by(2) {
        let sc = b[106 + ib32 / 2];
        for half in 0..2 {
            let ib = ib32 + half; let db = d * (1 + 2 * if half == 0 { sc & 0xF } else { sc >> 4 }) as f32;
            let qs = &b[2 + 8 * ib..]; let qh = b[66 + ib] as usize; let signs = &b[74 + 4 * ib..];
            for l in 0..4 {
                let g1 = IQ3S_GRID[qs[2 * l] as usize | ((qh << (8 - 2 * l)) & 256)].to_le_bytes();
                let g2 = IQ3S_GRID[qs[2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256)].to_le_bytes();
                for j in 0..4 {
                    y[o + j] = db * g1[j] as f32 * if signs[l] & (1 << j) != 0 { -1.0 } else { 1.0 };
                    y[o + 4 + j] = db * g2[j] as f32 * if signs[l] & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                }
                o += 8;
            }
        }
    }
}
pub fn q8_0_block(b: &[u8], y: &mut [f32]) { let d = h(b, 0); for i in 0..32 { y[i] = d * (b[2 + i] as i8) as f32; } }

/// Dequantize one contiguous row of `y.len()` elements.
pub fn dequant_row(ty: GgmlType, src: &[u8], y: &mut [f32]) {
    let (be, bs) = ty.block();
    match ty {
        GgmlType::F32 => for (i, v) in y.iter_mut().enumerate() { *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()) },
        GgmlType::F16 => for (i, v) in y.iter_mut().enumerate() { *v = h(src, i * 2) },
        GgmlType::Other(30) => for (i, v) in y.iter_mut().enumerate() { *v = f32::from_bits((u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]) as u32) << 16) },
        GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 | GgmlType::Q5K | GgmlType::Other(20) | GgmlType::Other(23) | GgmlType::Other(11) | GgmlType::Other(21) => for (bi, yb) in y.chunks_mut(be).enumerate() {
            let b = &src[bi * bs..(bi + 1) * bs];
            match ty { GgmlType::Q4K => q4k_block(b, yb), GgmlType::Q6K => q6k_block(b, yb), GgmlType::Q5K => q5k_block(b, yb),
                GgmlType::Other(20) => iq4nl_block(b, yb), GgmlType::Other(23) => iq4xs_block(b, yb), GgmlType::Other(11) => q3k_block(b, yb), GgmlType::Other(21) => iq3s_block(b, yb), _ => q8_0_block(b, yb) }
        },
        t => panic!("dequant {t:?} not implemented"),
    }
}
