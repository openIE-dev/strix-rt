//! Faithful Rust port of stb_image's JPEG decoder (vendor/stb/stb_image.h as used by llama.cpp mtmd: `stbi_load_from_memory(.., req_comp = 3)`):
//! baseline + progressive Huffman decode, scalar `stbi__idct_block` (the SSE2/NEON IDCT is documented bit-identical), dequantisation,
//! `stbi__resample_row_{1,v_2,h_2,hv_2,generic}` (fancy upsampling), `stbi__YCbCr_to_RGB_row` fixed-point conversion (reduced precision,
//! identical in SIMD and scalar), Adobe/JFIF/RGB/CMYK/YCCK handling. Integer only, no threads, deterministic. Output is interleaved RGB8.
//! stb_image is by Sean Barrett and contributors (https://github.com/nothings/stb), released into the public domain or under the MIT license
//! (dual). This file is a translation of its JPEG path; see THIRD_PARTY_NOTICES.md.
//! Failure behaviour follows stb (errors -> Err; a bad marker after the first scan returns the partially decoded image like stb does).
use anyhow::{bail, Result};

const FAST_BITS: usize = 9;
const ZIG: [u8; 64 + 15] = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63];
const BMASK: [u32; 17] = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535];
const JBIAS: [i32; 16] = [0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767];

#[derive(Clone)] struct Huff { fast: [u8; 1 << FAST_BITS], code: [u16; 256], values: [u8; 256], size: [u8; 257], maxcode: [u32; 18], delta: [i32; 17] }
impl Huff { fn new() -> Self { Huff { fast: [255; 1 << FAST_BITS], code: [0; 256], values: [0; 256], size: [0; 257], maxcode: [0; 18], delta: [0; 17] } }
    fn build(&mut self, count: &[i32; 16]) -> Result<()> {
        let mut k = 0usize;
        for i in 0..16 { for _ in 0..count[i] { self.size[k] = (i + 1) as u8; k += 1; if k >= 257 { bail!("bad size list") } } }
        self.size[k] = 0;
        let mut code: u32 = 0; k = 0;
        for j in 1..=16usize {
            self.delta[j] = (k as u32).wrapping_sub(code) as i32;
            if self.size[k] as usize == j { while self.size[k] as usize == j { self.code[k] = code as u16; code += 1; k += 1; }
                if code - 1 >= (1u32 << j) { bail!("bad code lengths") } }
            self.maxcode[j] = code << (16 - j); code <<= 1;
        }
        self.maxcode[17] = 0xffff_ffff;
        self.fast = [255; 1 << FAST_BITS];
        for i in 0..k { let s = self.size[i] as usize; if s <= FAST_BITS { let c = (self.code[i] as usize) << (FAST_BITS - s); let m = 1usize << (FAST_BITS - s);
            for j in 0..m { self.fast[c + j] = i as u8; } } }
        Ok(())
    }
    fn fast_ac(&self) -> [i16; 1 << FAST_BITS] {
        let mut fa = [0i16; 1 << FAST_BITS];
        for i in 0..(1usize << FAST_BITS) { let fast = self.fast[i];
            if fast < 255 { let rs = self.values[fast as usize] as i32; let run = (rs >> 4) & 15; let magbits = rs & 15; let len = self.size[fast as usize] as i32;
                if magbits != 0 && len + magbits <= FAST_BITS as i32 {
                    let mut k = (((i as i32) << len) & ((1 << FAST_BITS) - 1)) >> (FAST_BITS as i32 - magbits); let m = 1 << (magbits - 1);
                    if k < m { k += ((!0u32 << magbits) as i32) + 1; }
                    if k >= -128 && k <= 127 { fa[i] = ((k * 256) + (run * 16) + (len + magbits)) as i16; } } } }
        fa
    }
}

struct Bits<'a> { d: &'a [u8], p: usize, cb: u32, nb: i32, nomore: bool, marker: u8 }
impl<'a> Bits<'a> {
    fn get8(&mut self) -> u8 { if self.p < self.d.len() { self.p += 1; self.d[self.p - 1] } else { 0 } }
    fn get16be(&mut self) -> i32 { let a = self.get8() as i32; (a << 8) + self.get8() as i32 }
    fn at_eof(&self) -> bool { self.p >= self.d.len() }
    fn skip(&mut self, n: i32) { if n == 0 { return } if n < 0 { self.p = self.d.len(); return } self.p = (self.p + n as usize).min(self.d.len()); }
    fn grow(&mut self) {
        loop {
            let b = if self.nomore { 0 } else { self.get8() } as u32;
            if b == 0xff { let mut c = self.get8(); while c == 0xff { c = self.get8() }
                if c != 0 { self.marker = c; self.nomore = true; return } }
            self.cb |= b.wrapping_shl((24 - self.nb) as u32); self.nb += 8;
            if self.nb > 24 { break }
        }
    }
    fn huff_decode(&mut self, h: &Huff) -> i32 {
        if self.nb < 16 { self.grow() }
        let c = ((self.cb >> (32 - FAST_BITS)) & ((1 << FAST_BITS) - 1)) as usize; let k = h.fast[c];
        if k < 255 { let s = h.size[k as usize] as i32; if s > self.nb { return -1 } self.cb <<= s; self.nb -= s; return h.values[k as usize] as i32 }
        let temp = self.cb >> 16; let mut k = FAST_BITS + 1;
        loop { if temp < h.maxcode[k] { break } k += 1; }
        if k == 17 { self.nb -= 16; return -1 }
        if k as i32 > self.nb { return -1 }
        let c = (((self.cb >> (32 - k)) & BMASK[k]) as i32).wrapping_add(h.delta[k]);
        if c < 0 || c >= 256 { return -1 }
        self.nb -= k as i32; self.cb <<= k; h.values[c as usize] as i32
    }
    fn extend_receive(&mut self, n: i32) -> i32 {
        if self.nb < n { self.grow() } if self.nb < n { return 0 }
        let sgn = (self.cb >> 31) as i32; let k = self.cb.rotate_left(n as u32); self.cb = k & !BMASK[n as usize]; let k = k & BMASK[n as usize]; self.nb -= n;
        (k as i32).wrapping_add(JBIAS[n as usize] & (sgn - 1))
    }
    fn get_bits(&mut self, n: i32) -> i32 {
        if self.nb < n { self.grow() } if self.nb < n { return 0 }
        let k = self.cb.rotate_left(n as u32); self.cb = k & !BMASK[n as usize]; let k = k & BMASK[n as usize]; self.nb -= n; k as i32
    }
    fn get_bit(&mut self) -> bool { if self.nb < 1 { self.grow() } if self.nb < 1 { return false } let k = self.cb; self.cb <<= 1; self.nb -= 1; k & 0x8000_0000 != 0 }
}

#[derive(Default, Clone)] struct Comp { id: i32, h: i32, v: i32, tq: usize, hd: usize, ha: usize, dc_pred: i32, x: i32, y: i32, w2: i32, h2: i32, data: Vec<u8>, coeff: Vec<i16>, coeff_w: i32 }

fn addints_valid(a: i32, b: i32) -> bool { if (a >= 0) != (b >= 0) { return true } if a < 0 && b < 0 { return a >= i32::MIN - b } a <= i32::MAX - b }
fn mul2shorts_valid(a: i32, b: i32) -> bool { if b == 0 || b == -1 { return true } if (a >= 0) == (b >= 0) { return a <= i16::MAX as i32 / b } if b < 0 { return a <= i16::MIN as i32 / b } a >= i16::MIN as i32 / b }
fn clamp8(x: i32) -> u8 { if (x as u32) > 255 { if x < 0 { 0 } else { 255 } } else { x as u8 } }
fn f2f(x: f32) -> i32 { ((x * 4096.0f32) as f64 + 0.5) as i32 }

struct Idct { c: [i32; 12] }
impl Idct { fn new() -> Self { Idct { c: [f2f(0.5411961), f2f(-1.847759065), f2f(0.765366865), f2f(1.175875602), f2f(-0.899976223), f2f(-2.562915447), f2f(-1.961570560), f2f(-0.390180644), f2f(0.298631336), f2f(2.053119869), f2f(3.072711026), f2f(1.501321110)] } }
    /// STBI__IDCT_1D: returns (x0,x1,x2,x3,t0,t1,t2,t3)
    #[inline] fn one(&self, s: [i32; 8]) -> [i32; 8] {
        let c = &self.c; let (s0, s1, s2, s3, s4, s5, s6, s7) = (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
        let mut p2 = s2; let mut p3 = s6; let mut p1 = p2.wrapping_add(p3).wrapping_mul(c[0]);
        let mut t2 = p1.wrapping_add(p3.wrapping_mul(c[1])); let mut t3 = p1.wrapping_add(p2.wrapping_mul(c[2]));
        p2 = s0; p3 = s4; let mut t0 = p2.wrapping_add(p3).wrapping_mul(4096); let mut t1 = p2.wrapping_sub(p3).wrapping_mul(4096);
        let x0 = t0.wrapping_add(t3); let x3 = t0.wrapping_sub(t3); let x1 = t1.wrapping_add(t2); let x2 = t1.wrapping_sub(t2);
        t0 = s7; t1 = s5; t2 = s3; t3 = s1;
        p3 = t0.wrapping_add(t2); let mut p4 = t1.wrapping_add(t3); p1 = t0.wrapping_add(t3); p2 = t1.wrapping_add(t2);
        let p5 = p3.wrapping_add(p4).wrapping_mul(c[3]);
        t0 = t0.wrapping_mul(c[8]); t1 = t1.wrapping_mul(c[9]); t2 = t2.wrapping_mul(c[10]); t3 = t3.wrapping_mul(c[11]);
        p1 = p5.wrapping_add(p1.wrapping_mul(c[4])); p2 = p5.wrapping_add(p2.wrapping_mul(c[5])); p3 = p3.wrapping_mul(c[6]); p4 = p4.wrapping_mul(c[7]);
        t3 = t3.wrapping_add(p1.wrapping_add(p4)); t2 = t2.wrapping_add(p2.wrapping_add(p3)); t1 = t1.wrapping_add(p2.wrapping_add(p4)); t0 = t0.wrapping_add(p1.wrapping_add(p3));
        [x0, x1, x2, x3, t0, t1, t2, t3]
    }
    fn block(&self, out: &mut [u8], off: usize, stride: usize, d: &[i16; 64]) {
        let mut v = [0i32; 64];
        for i in 0..8 {
            if d[i + 8] == 0 && d[i + 16] == 0 && d[i + 24] == 0 && d[i + 32] == 0 && d[i + 40] == 0 && d[i + 48] == 0 && d[i + 56] == 0 {
                let dc = d[i] as i32 * 4; for r in 0..8 { v[i + r * 8] = dc }
            } else {
                let r = self.one([d[i] as i32, d[i + 8] as i32, d[i + 16] as i32, d[i + 24] as i32, d[i + 32] as i32, d[i + 40] as i32, d[i + 48] as i32, d[i + 56] as i32]);
                let (x0, x1, x2, x3, t0, t1, t2, t3) = (r[0].wrapping_add(512), r[1].wrapping_add(512), r[2].wrapping_add(512), r[3].wrapping_add(512), r[4], r[5], r[6], r[7]);
                v[i] = x0.wrapping_add(t3) >> 10; v[i + 56] = x0.wrapping_sub(t3) >> 10; v[i + 8] = x1.wrapping_add(t2) >> 10; v[i + 48] = x1.wrapping_sub(t2) >> 10;
                v[i + 16] = x2.wrapping_add(t1) >> 10; v[i + 40] = x2.wrapping_sub(t1) >> 10; v[i + 24] = x3.wrapping_add(t0) >> 10; v[i + 32] = x3.wrapping_sub(t0) >> 10;
            }
        }
        let bias = 65536 + (128 << 17);
        for i in 0..8 {
            let r = self.one([v[i * 8], v[i * 8 + 1], v[i * 8 + 2], v[i * 8 + 3], v[i * 8 + 4], v[i * 8 + 5], v[i * 8 + 6], v[i * 8 + 7]]);
            let (x0, x1, x2, x3, t0, t1, t2, t3) = (r[0].wrapping_add(bias), r[1].wrapping_add(bias), r[2].wrapping_add(bias), r[3].wrapping_add(bias), r[4], r[5], r[6], r[7]);
            let o = off + i * stride;
            out[o] = clamp8(x0.wrapping_add(t3) >> 17); out[o + 7] = clamp8(x0.wrapping_sub(t3) >> 17); out[o + 1] = clamp8(x1.wrapping_add(t2) >> 17); out[o + 6] = clamp8(x1.wrapping_sub(t2) >> 17);
            out[o + 2] = clamp8(x2.wrapping_add(t1) >> 17); out[o + 5] = clamp8(x2.wrapping_sub(t1) >> 17); out[o + 3] = clamp8(x3.wrapping_add(t0) >> 17); out[o + 4] = clamp8(x3.wrapping_sub(t0) >> 17);
        }
    }
}

struct Jpeg<'a> {
    b: Bits<'a>, progressive: bool, spec_start: i32, spec_end: i32, succ_high: i32, succ_low: i32, eob_run: i32, restart_interval: i32, todo: i32,
    scan_n: usize, order: [usize; 4], img_x: i32, img_y: i32, img_n: usize, h_max: i32, v_max: i32, mcu_x: i32, mcu_y: i32,
    hdc: Vec<Huff>, hac: Vec<Huff>, fast_ac: Vec<[i16; 1 << FAST_BITS]>, dequant: [[u16; 64]; 4], comp: [Comp; 4], jfif: bool, app14: i32, rgb: i32, idct: Idct,
}

impl<'a> Jpeg<'a> {
    fn reset(&mut self) { self.b.nb = 0; self.b.cb = 0; self.b.nomore = false; for c in self.comp.iter_mut() { c.dc_pred = 0 } self.b.marker = 0xff;
        self.todo = if self.restart_interval != 0 { self.restart_interval } else { 0x7fff_ffff }; self.eob_run = 0; }
    fn get_marker(&mut self) -> u8 {
        if self.b.marker != 0xff { let x = self.b.marker; self.b.marker = 0xff; return x }
        let mut x = self.b.get8(); if x != 0xff { return 0xff }
        while x == 0xff { x = self.b.get8() } x
    }
    fn decode_block(&mut self, n: usize, data: &mut [i16; 64]) -> Result<()> {
        let (hd, ha, tq) = (self.comp[n].hd, self.comp[n].ha, self.comp[n].tq);
        let b = &mut self.b; let hdc = &self.hdc[hd]; let hac = &self.hac[ha]; let fac = &self.fast_ac[ha]; let dq = &self.dequant[tq];
        if b.nb < 16 { b.grow() }
        let t = b.huff_decode(hdc); if t < 0 || t > 15 { bail!("bad huffman code") }
        *data = [0; 64];
        let diff = if t != 0 { b.extend_receive(t) } else { 0 };
        if !addints_valid(self.comp[n].dc_pred, diff) { bail!("bad delta") }
        let dc = self.comp[n].dc_pred + diff; self.comp[n].dc_pred = dc;
        if !mul2shorts_valid(dc, dq[0] as i32) { bail!("can't merge dc and ac") }
        data[0] = (dc.wrapping_mul(dq[0] as i32)) as i16;
        let mut k = 1usize;
        loop {
            if b.nb < 16 { b.grow() }
            let c = ((b.cb >> (32 - FAST_BITS)) & ((1 << FAST_BITS) - 1)) as usize; let r = fac[c] as i32;
            if r != 0 {
                k += ((r >> 4) & 15) as usize; let s = r & 15; if s > b.nb { bail!("bad huffman code") }
                b.cb <<= s; b.nb -= s; let zig = ZIG[k] as usize; k += 1; data[zig] = ((r >> 8).wrapping_mul(dq[zig] as i32)) as i16;
            } else {
                let rs = b.huff_decode(hac); if rs < 0 { bail!("bad huffman code") }
                let s = rs & 15; let r = rs >> 4;
                if s == 0 { if rs != 0xf0 { break } k += 16; }
                else { k += r as usize; let zig = ZIG[k] as usize; k += 1; data[zig] = (b.extend_receive(s).wrapping_mul(dq[zig] as i32)) as i16; }
            }
            if k >= 64 { break }
        }
        Ok(())
    }
    fn prog_dc(&mut self, n: usize, off: usize) -> Result<()> {
        if self.spec_end != 0 { bail!("can't merge dc and ac") }
        let hd = self.comp[n].hd; let b = &mut self.b; if b.nb < 16 { b.grow() }
        if self.succ_high == 0 {
            for x in self.comp[n].coeff[off..off + 64].iter_mut() { *x = 0 }
            let t = b.huff_decode(&self.hdc[hd]); if t < 0 || t > 15 { bail!("bad dc") }
            let diff = if t != 0 { b.extend_receive(t) } else { 0 };
            let pred = self.comp[n].dc_pred; if !addints_valid(pred, diff) { bail!("bad delta") }
            let dc = pred + diff; self.comp[n].dc_pred = dc;
            if !mul2shorts_valid(dc, 1 << self.succ_low) { bail!("bad dc mul") }
            self.comp[n].coeff[off] = (dc.wrapping_mul(1 << self.succ_low)) as i16;
        } else if b.get_bit() { let c = &mut self.comp[n].coeff[off]; *c = c.wrapping_add((1i32 << self.succ_low) as i16); }
        Ok(())
    }
    fn prog_ac(&mut self, n: usize, off: usize) -> Result<()> {
        if self.spec_start == 0 { bail!("can't merge dc and ac") }
        let ha = self.comp[n].ha; let b = &mut self.b; let hac = &self.hac[ha]; let fac = &self.fast_ac[ha];
        let data = &mut self.comp[n].coeff[off..off + 64];
        if self.succ_high == 0 {
            let shift = self.succ_low;
            if self.eob_run != 0 { self.eob_run -= 1; return Ok(()) }
            let mut k = self.spec_start as usize;
            loop {
                if b.nb < 16 { b.grow() }
                let c = ((b.cb >> (32 - FAST_BITS)) & ((1 << FAST_BITS) - 1)) as usize; let r = fac[c] as i32;
                if r != 0 {
                    k += ((r >> 4) & 15) as usize; let s = r & 15; if s > b.nb { bail!("bad huffman code") }
                    b.cb <<= s; b.nb -= s; let zig = ZIG[k] as usize; k += 1; data[zig] = ((r >> 8).wrapping_mul(1 << shift)) as i16;
                } else {
                    let rs = b.huff_decode(hac); if rs < 0 { bail!("bad huffman code") }
                    let s = rs & 15; let r = rs >> 4;
                    if s == 0 {
                        if r < 15 { self.eob_run = 1 << r; if r != 0 { self.eob_run += b.get_bits(r) } self.eob_run -= 1; break }
                        k += 16;
                    } else { k += r as usize; let zig = ZIG[k] as usize; k += 1; data[zig] = (b.extend_receive(s).wrapping_mul(1 << shift)) as i16; }
                }
                if k > self.spec_end as usize { break }
            }
        } else {
            let bit = (1i32 << self.succ_low) as i16;
            let refine = |b: &mut Bits, p: &mut i16| { if *p != 0 && b.get_bit() && (*p & bit) == 0 { if *p > 0 { *p = p.wrapping_add(bit) } else { *p = p.wrapping_sub(bit) } } };
            if self.eob_run != 0 {
                self.eob_run -= 1;
                for k in self.spec_start..=self.spec_end { refine(b, &mut data[ZIG[k as usize] as usize]); }
            } else {
                let mut k = self.spec_start;
                loop {
                    let rs = b.huff_decode(hac); if rs < 0 { bail!("bad huffman code") }
                    let s = rs & 15; let mut r = rs >> 4; let sv: i16;
                    if s == 0 {
                        if r < 15 { self.eob_run = (1 << r) - 1; if r != 0 { self.eob_run += b.get_bits(r) } r = 64; }
                        sv = 0;
                    } else { if s != 1 { bail!("bad huffman code") } sv = if b.get_bit() { bit } else { bit.wrapping_neg() }; }
                    while k <= self.spec_end {
                        let p = &mut data[ZIG[k as usize] as usize]; k += 1;
                        if *p != 0 { if b.get_bit() && (*p & bit) == 0 { if *p > 0 { *p = p.wrapping_add(bit) } else { *p = p.wrapping_sub(bit) } } }
                        else { if r == 0 { *p = sv; break } r -= 1; }
                    }
                    if k > self.spec_end { break }
                }
            }
        }
        Ok(())
    }
    fn restart_check(&mut self) -> bool { // returns true to continue, false = "return 1" (bail with data so far)
        self.todo -= 1;
        if self.todo <= 0 { if self.b.nb < 24 { self.b.grow() } let m = self.b.marker; if !(0xd0..=0xd7).contains(&m) { return false } self.reset(); }
        true
    }
    fn parse_entropy(&mut self) -> Result<()> {
        self.reset();
        let idct = Idct::new();
        if !self.progressive {
            let mut data = [0i16; 64];
            if self.scan_n == 1 {
                let n = self.order[0]; let w = (self.comp[n].x + 7) >> 3; let h = (self.comp[n].y + 7) >> 3;
                for j in 0..h { for i in 0..w {
                    self.decode_block(n, &mut data)?;
                    let w2 = self.comp[n].w2 as usize; idct.block(&mut self.comp[n].data, w2 * j as usize * 8 + i as usize * 8, w2, &data);
                    if !self.restart_check() { return Ok(()) } } }
            } else {
                for j in 0..self.mcu_y { for i in 0..self.mcu_x {
                    for k in 0..self.scan_n { let n = self.order[k];
                        for y in 0..self.comp[n].v { for x in 0..self.comp[n].h {
                            let x2 = (i * self.comp[n].h + x) * 8; let y2 = (j * self.comp[n].v + y) * 8;
                            self.decode_block(n, &mut data)?;
                            let w2 = self.comp[n].w2 as usize; idct.block(&mut self.comp[n].data, w2 * y2 as usize + x2 as usize, w2, &data); } } }
                    if !self.restart_check() { return Ok(()) } } }
            }
        } else if self.scan_n == 1 {
            let n = self.order[0]; let w = (self.comp[n].x + 7) >> 3; let h = (self.comp[n].y + 7) >> 3;
            for j in 0..h { for i in 0..w {
                let off = 64 * (i + j * self.comp[n].coeff_w) as usize;
                if self.spec_start == 0 { self.prog_dc(n, off)? } else { self.prog_ac(n, off)? }
                if !self.restart_check() { return Ok(()) } } }
        } else {
            for j in 0..self.mcu_y { for i in 0..self.mcu_x {
                for k in 0..self.scan_n { let n = self.order[k];
                    for y in 0..self.comp[n].v { for x in 0..self.comp[n].h {
                        let x2 = i * self.comp[n].h + x; let y2 = j * self.comp[n].v + y; let off = 64 * (x2 + y2 * self.comp[n].coeff_w) as usize;
                        self.prog_dc(n, off)?; } } }
                if !self.restart_check() { return Ok(()) } } }
        }
        Ok(())
    }
    fn finish(&mut self) {
        if !self.progressive { return }
        let idct = Idct::new();
        for n in 0..self.img_n {
            let w = (self.comp[n].x + 7) >> 3; let h = (self.comp[n].y + 7) >> 3; let dq = self.dequant[self.comp[n].tq];
            for j in 0..h { for i in 0..w {
                let off = 64 * (i + j * self.comp[n].coeff_w) as usize; let mut d = [0i16; 64];
                for t in 0..64 { d[t] = self.comp[n].coeff[off + t].wrapping_mul(dq[t] as i16); } // short *= uint16: int multiply, truncated to short
                let w2 = self.comp[n].w2 as usize; idct.block(&mut self.comp[n].data, w2 * j as usize * 8 + i as usize * 8, w2, &d); } }
        }
    }
    /// Ok(true) = handled, Ok(false) = stb returned 0 (error)
    fn process_marker(&mut self, m: u8) -> bool {
        let b = &mut self.b;
        match m {
            0xff => return false,
            0xDD => { if b.get16be() != 4 { return false } self.restart_interval = b.get16be(); return true }
            0xDB => { let mut l = b.get16be() - 2;
                while l > 0 { let q = b.get8() as i32; let p = q >> 4; let sixteen = p != 0; let t = (q & 15) as usize;
                    if p != 0 && p != 1 { return false } if t > 3 { return false }
                    for i in 0..64 { self.dequant[t][ZIG[i] as usize] = (if sixteen { b.get16be() } else { b.get8() as i32 }) as u16; }
                    l -= if sixteen { 129 } else { 65 }; }
                return l == 0 }
            0xC4 => { let mut l = b.get16be() - 2;
                while l > 0 { let q = b.get8() as i32; let tc = q >> 4; let th = (q & 15) as usize; if tc > 1 || th > 3 { return false }
                    let mut sizes = [0i32; 16]; let mut n = 0i32; for i in 0..16 { sizes[i] = b.get8() as i32; n += sizes[i]; }
                    if n > 256 { return false } l -= 17;
                    let h = if tc == 0 { &mut self.hdc[th] } else { &mut self.hac[th] };
                    if h.build(&sizes).is_err() { return false }
                    for i in 0..n as usize { h.values[i] = b.get8(); }
                    if tc != 0 { self.fast_ac[th] = self.hac[th].fast_ac(); }
                    l -= n; }
                return l == 0 }
            _ => {}
        }
        if (0xE0..=0xEF).contains(&m) || m == 0xFE {
            let mut l = b.get16be(); if l < 2 { return false } l -= 2;
            if m == 0xE0 && l >= 5 { let tag = *b"JFIF\0"; let mut ok = true; for i in 0..5 { if b.get8() != tag[i] { ok = false } } l -= 5; if ok { self.jfif = true } }
            else if m == 0xEE && l >= 12 { let tag = *b"Adobe\0"; let mut ok = true; for i in 0..6 { if b.get8() != tag[i] { ok = false } } l -= 6;
                if ok { b.get8(); b.get16be(); b.get16be(); self.app14 = b.get8() as i32; l -= 6; } }
            b.skip(l); return true;
        }
        false
    }
    fn scan_header(&mut self) -> bool {
        let b = &mut self.b; let ls = b.get16be(); self.scan_n = b.get8() as usize;
        if self.scan_n < 1 || self.scan_n > 4 || self.scan_n > self.img_n { return false }
        if ls != 6 + 2 * self.scan_n as i32 { return false }
        for i in 0..self.scan_n { let id = b.get8() as i32; let q = b.get8() as i32;
            let mut which = 0; while which < self.img_n { if self.comp[which].id == id { break } which += 1; }
            if which == self.img_n { return false }
            self.comp[which].hd = (q >> 4) as usize; if self.comp[which].hd > 3 { return false }
            self.comp[which].ha = (q & 15) as usize; if self.comp[which].ha > 3 { return false }
            self.order[i] = which; }
        self.spec_start = b.get8() as i32; self.spec_end = b.get8() as i32; let aa = b.get8() as i32; self.succ_high = aa >> 4; self.succ_low = aa & 15;
        if self.progressive { if self.spec_start > 63 || self.spec_end > 63 || self.spec_start > self.spec_end || self.succ_high > 13 || self.succ_low > 13 { return false } }
        else { if self.spec_start != 0 { return false } if self.succ_high != 0 || self.succ_low != 0 { return false } self.spec_end = 63; }
        true
    }
    fn frame_header(&mut self) -> Result<()> {
        let b = &mut self.b; let lf = b.get16be(); if lf < 11 { bail!("bad SOF len") }
        if b.get8() != 8 { bail!("only 8-bit") }
        self.img_y = b.get16be(); if self.img_y == 0 { bail!("no header height") } self.img_x = b.get16be(); if self.img_x == 0 { bail!("0 width") }
        if self.img_y > (1 << 24) || self.img_x > (1 << 24) { bail!("too large") }
        let c = b.get8() as usize; if c != 3 && c != 1 && c != 4 { bail!("bad component count") } self.img_n = c;
        if lf != 8 + 3 * c as i32 { bail!("bad SOF len") }
        self.rgb = 0;
        for i in 0..c { let rgb = [b'R', b'G', b'B']; self.comp[i].id = b.get8() as i32; if c == 3 && self.comp[i].id == rgb[i] as i32 { self.rgb += 1 }
            let q = b.get8() as i32; self.comp[i].h = q >> 4; if self.comp[i].h == 0 || self.comp[i].h > 4 { bail!("bad H") }
            self.comp[i].v = q & 15; if self.comp[i].v == 0 || self.comp[i].v > 4 { bail!("bad V") }
            self.comp[i].tq = b.get8() as usize; if self.comp[i].tq > 3 { bail!("bad TQ") } }
        if (self.img_x as u64) * (self.img_y as u64) * (c as u64) >= (1 << 31) { bail!("too large") }
        let (mut hm, mut vm) = (1, 1); for i in 0..c { hm = hm.max(self.comp[i].h); vm = vm.max(self.comp[i].v); }
        for i in 0..c { if hm % self.comp[i].h != 0 { bail!("bad H") } if vm % self.comp[i].v != 0 { bail!("bad V") } }
        self.h_max = hm; self.v_max = vm; let (mw, mh) = (hm * 8, vm * 8);
        self.mcu_x = (self.img_x + mw - 1) / mw; self.mcu_y = (self.img_y + mh - 1) / mh;
        for i in 0..c { let cm = &mut self.comp[i];
            cm.x = (self.img_x * cm.h + hm - 1) / hm; cm.y = (self.img_y * cm.v + vm - 1) / vm;
            cm.w2 = self.mcu_x * cm.h * 8; cm.h2 = self.mcu_y * cm.v * 8;
            cm.data = vec![0u8; cm.w2 as usize * cm.h2 as usize];
            if self.progressive { cm.coeff_w = cm.w2 / 8; cm.coeff = vec![0i16; cm.w2 as usize * cm.h2 as usize]; } }
        Ok(())
    }
    fn decode_image(&mut self) -> Result<()> {
        self.restart_interval = 0;
        // header
        self.jfif = false; self.app14 = -1; self.b.marker = 0xff;
        let mut m = self.get_marker(); if m != 0xd8 { bail!("no SOI") }
        m = self.get_marker();
        while !(m == 0xc0 || m == 0xc1 || m == 0xc2) {
            if !self.process_marker(m) { bail!("bad marker in header") }
            m = self.get_marker();
            while m == 0xff { if self.b.at_eof() { bail!("no SOF") } m = self.get_marker(); }
        }
        self.progressive = m == 0xc2; self.frame_header()?;
        m = self.get_marker();
        while m != 0xd9 {
            if m == 0xda {
                if !self.scan_header() { bail!("bad SOS") }
                self.parse_entropy()?;
                if self.b.marker == 0xff { self.b.marker = self.skip_junk(); }
                m = self.get_marker(); if (0xd0..=0xd7).contains(&m) { m = self.get_marker(); }
            } else if m == 0xdc {
                let ld = self.b.get16be(); let nl = self.b.get16be(); if ld != 4 { bail!("bad DNL len") } if nl != self.img_y { bail!("bad DNL height") } m = self.get_marker();
            } else { if !self.process_marker(m) { return Ok(()) } m = self.get_marker(); }
        }
        if self.progressive { self.finish() }
        Ok(())
    }
    fn skip_junk(&mut self) -> u8 {
        while !self.b.at_eof() { let mut x = self.b.get8();
            while x == 0xff { if self.b.at_eof() { return 0xff } x = self.b.get8(); if x != 0 && x != 0xff { return x } } }
        0xff
    }
}

fn div4(x: i32) -> u8 { (x >> 2) as u8 }
fn div16(x: i32) -> u8 { (x >> 4) as u8 }
fn blinn(x: u8, y: u8) -> u8 { let t = x as u32 * y as u32 + 128; ((t + (t >> 8)) >> 8) as u8 }
fn compute_y(r: i32, g: i32, b: i32) -> u8 { ((r * 77 + g * 150 + 29 * b) >> 8) as u8 }
fn ycbcr_row(out: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8], count: usize, step: usize) {
    let f = |x: f32| ((x * 4096.0f32 + 0.5f32) as i32) << 8; let (c1, c2, c3, c4) = (f(1.40200), f(0.71414), f(0.34414), f(1.77200));
    for i in 0..count {
        let yf = ((y[i] as i32) << 20) + (1 << 19); let r_ = cr[i] as i32 - 128; let b_ = cb[i] as i32 - 128;
        let mut r = yf.wrapping_add(r_.wrapping_mul(c1));
        let mut g = yf.wrapping_add(r_.wrapping_mul(-c2)).wrapping_add((((b_.wrapping_mul(-c3)) as u32) & 0xffff_0000) as i32);
        let mut b = yf.wrapping_add(b_.wrapping_mul(c4));
        r >>= 20; g >>= 20; b >>= 20;
        out[i * step] = clamp8(r); out[i * step + 1] = clamp8(g); out[i * step + 2] = clamp8(b);
    }
}

/// stbi_load_from_memory(.., req_comp = 3) for a JPEG. Returns (RGB8, w, h).
pub fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let mut z = Jpeg { b: Bits { d: bytes, p: 0, cb: 0, nb: 0, nomore: false, marker: 0xff }, progressive: false, spec_start: 0, spec_end: 0, succ_high: 0, succ_low: 0, eob_run: 0, restart_interval: 0, todo: 0,
        scan_n: 0, order: [0; 4], img_x: 0, img_y: 0, img_n: 0, h_max: 1, v_max: 1, mcu_x: 0, mcu_y: 0, hdc: vec![Huff::new(); 4], hac: vec![Huff::new(); 4], fast_ac: vec![[0; 1 << FAST_BITS]; 4],
        dequant: [[0; 64]; 4], comp: Default::default(), jfif: false, app14: -1, rgb: 0, idct: Idct::new() };
    z.decode_image()?;
    let n = 3usize; let img_n = z.img_n; let (w, h) = (z.img_x as usize, z.img_y as usize);
    let is_rgb = img_n == 3 && (z.rgb == 3 || (z.app14 == 0 && !z.jfif));
    let decode_n = img_n;
    struct Rs { hs: i32, vs: i32, w_lores: usize, ystep: i32, ypos: i32, line0: usize, line1: usize, kind: u8 }
    let mut rs: Vec<Rs> = Vec::new(); let mut lines: Vec<Vec<u8>> = Vec::new();
    for k in 0..decode_n { let hs = z.h_max / z.comp[k].h; let vs = z.v_max / z.comp[k].v;
        let kind = if hs == 1 && vs == 1 { 0 } else if hs == 1 && vs == 2 { 1 } else if hs == 2 && vs == 1 { 2 } else if hs == 2 && vs == 2 { 3 } else { 4 };
        rs.push(Rs { hs, vs, w_lores: (w + hs as usize - 1) / hs as usize, ystep: vs >> 1, ypos: 0, line0: 0, line1: 0, kind }); lines.push(vec![0u8; w + 3 + 16]); }
    let mut out = vec![0u8; n * w * h];
    for j in 0..h {
        for k in 0..decode_n {
            let r = &mut rs[k]; let y_bot = r.ystep >= (r.vs >> 1); let (near, far) = if y_bot { (r.line1, r.line0) } else { (r.line0, r.line1) };
            let data = &z.comp[k].data; let lb = &mut lines[k]; let wl = r.w_lores;
            match r.kind {
                0 => { lb[..wl].copy_from_slice(&data[near..near + wl]); }
                1 => { for i in 0..wl { lb[i] = div4(3 * data[near + i] as i32 + data[far + i] as i32 + 2); } }
                2 => { let inp = &data[near..near + wl];
                    if wl == 1 { lb[0] = inp[0]; lb[1] = inp[0]; } else {
                        lb[0] = inp[0]; lb[1] = div4(inp[0] as i32 * 3 + inp[1] as i32 + 2);
                        let mut i = 1; while i < wl - 1 { let nn = 3 * inp[i] as i32 + 2; lb[i * 2] = div4(nn + inp[i - 1] as i32); lb[i * 2 + 1] = div4(nn + inp[i + 1] as i32); i += 1; }
                        lb[i * 2] = div4(inp[wl - 2] as i32 * 3 + inp[wl - 1] as i32 + 2); lb[i * 2 + 1] = inp[wl - 1]; } }
                3 => { let (inn, inf) = (&data[near..near + wl], &data[far..far + wl]);
                    if wl == 1 { let v = div4(3 * inn[0] as i32 + inf[0] as i32 + 2); lb[0] = v; lb[1] = v; } else {
                        let mut t1 = 3 * inn[0] as i32 + inf[0] as i32; lb[0] = div4(t1 + 2);
                        for i in 1..wl { let t0 = t1; t1 = 3 * inn[i] as i32 + inf[i] as i32; lb[i * 2 - 1] = div16(3 * t0 + t1 + 8); lb[i * 2] = div16(3 * t1 + t0 + 8); }
                        lb[wl * 2 - 1] = div4(t1 + 2); } }
                _ => { let hs = r.hs as usize; for i in 0..wl { for jj in 0..hs { lb[i * hs + jj] = data[near + i]; } } }
            }
            r.ystep += 1;
            if r.ystep >= r.vs { r.ystep = 0; r.line0 = r.line1; r.ypos += 1; if r.ypos < z.comp[k].y { r.line1 += z.comp[k].w2 as usize; } }
        }
        let o = &mut out[n * w * j..n * w * (j + 1)];
        let (c0, c1, c2) = (&lines[0], if decode_n > 1 { &lines[1] } else { &lines[0] }, if decode_n > 2 { &lines[2] } else { &lines[0] });
        if img_n == 3 {
            if is_rgb { for i in 0..w { o[i * 3] = c0[i]; o[i * 3 + 1] = c1[i]; o[i * 3 + 2] = c2[i]; } }
            else { ycbcr_row(o, c0, c1, c2, w, n); }
        } else if img_n == 4 {
            let c3 = &lines[3];
            if z.app14 == 0 { for i in 0..w { let m = c3[i]; o[i * 3] = blinn(c0[i], m); o[i * 3 + 1] = blinn(c1[i], m); o[i * 3 + 2] = blinn(c2[i], m); } }
            else if z.app14 == 2 { ycbcr_row(o, c0, c1, c2, w, n); for i in 0..w { let m = c3[i]; for c in 0..3 { o[i * 3 + c] = blinn(255 - o[i * 3 + c], m); } } }
            else { ycbcr_row(o, c0, c1, c2, w, n); }
        } else { for i in 0..w { o[i * 3] = c0[i]; o[i * 3 + 1] = c0[i]; o[i * 3 + 2] = c0[i]; } }
    }
    let _ = compute_y; // only used for n < 3
    Ok((out, w, h))
}
