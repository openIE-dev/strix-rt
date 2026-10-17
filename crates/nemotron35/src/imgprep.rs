//! Nemotron 3.5 VL real-image preprocessing, bit-matching llama.cpp mtmd for projector "nemotron_omni"
//! (mtmd_image_preprocessor_dyn_size + clip hparams: PAD_NONE, bicubic, patch 16 x merge 2):
//!   decode (PNG/JPEG, RGB8, alpha dropped) -> smart_resize target (align 32, min/max pixel budget, aspect preserved)
//!   -> Pillow-compatible separable bicubic (a = -0.5, 22-bit fixed point, integer only) -> x/255 -> (x - mean) / std.
//! Everything is integer / IEEE f32, so it is deterministic across runs and thread counts. Returns planar [3][h][w] f32.
//! Config (read from the mmproj gguf metadata, defaults below are that file's values): image_mean/std, min/max pixels.
use anyhow::{bail, Result};

pub const MEAN: [f32; 3] = [0.48145467042922974, 0.45782750844955444, 0.40821072459220886];
pub const STD: [f32; 3] = [0.2686295509338379, 0.2613025903701782, 0.27577710151672363];
pub const MIN_PIXELS: usize = 262144; pub const MAX_PIXELS: usize = 3407872; pub const ALIGN: usize = 32; // patch 16 * scale_factor 2

/// llama.cpp img_tool::calc_size_preserved_ratio ("smart_resize"), longest_edge disabled.
pub fn target_size(w: usize, h: usize) -> (usize, usize) {
    let f = ALIGN as f32;
    let round_f = |x: f32| ((x / f).round() as usize) * ALIGN; // std::round: half away from zero == f32::round
    let ceil_f = |x: f32| ((x / f).ceil() as usize) * ALIGN;
    let floor_f = |x: f32| ((x / f).floor() as usize) * ALIGN;
    let mut wb = ALIGN.max(round_f(w as f32)); let mut hb = ALIGN.max(round_f(h as f32));
    if hb * wb > MAX_PIXELS {
        let beta = ((h as f32) * (w as f32) / MAX_PIXELS as f32).sqrt();
        hb = ALIGN.max(floor_f(h as f32 / beta)); wb = ALIGN.max(floor_f(w as f32 / beta));
    } else if hb * wb < MIN_PIXELS {
        let beta = (MIN_PIXELS as f32 / ((h as f32) * (w as f32))).sqrt();
        hb = ceil_f(h as f32 * beta); wb = ceil_f(w as f32 * beta);
    }
    (wb, hb)
}

fn bicubic(x: f64) -> f64 { let x = x.abs(); let a = -0.5;
    if x < 1.0 { ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 } else if x < 2.0 { (((x - 5.0) * x + 8.0) * x - 4.0) * a } else { 0.0 } }

const PB: i32 = 22;
fn coeffs(in_sz: usize, out_sz: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let scale = in_sz as f64 / out_sz as f64; let fs = scale.max(1.0); let support = 2.0 * fs; let ks = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_sz); let mut w = vec![0i32; out_sz * ks]; let mut pre = vec![0f64; ks];
    for xx in 0..out_sz {
        let center = (xx as f64 + 0.5) * scale; let ss = 1.0 / fs;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize; let xmax = (((center + support + 0.5) as i64).min(in_sz as i64) as usize) - xmin;
        let mut ww = 0.0; for x in 0..xmax { let v = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss); pre[x] = v; ww += v; }
        for x in 0..ks { if x < xmax { if ww != 0.0 { pre[x] /= ww } } else { pre[x] = 0.0 } }
        let sc = (1u64 << PB) as f64;
        for x in 0..ks { let r = pre[x] * sc + if pre[x] < 0.0 { -0.5 } else { 0.5 }; w[xx * ks + x] = r as i32; } // trunc toward zero (Pillow)
        bounds.push((xmin, xmax));
    }
    (ks, bounds, w)
}
fn clip8(v: i32) -> u8 { v.clamp(0, 255) as u8 }

/// Pillow-compatible bicubic resize of interleaved RGB8 (horizontal pass first, then vertical, each rounded to u8 like Pillow).
pub fn resize_bicubic(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh { return src.to_vec() }
    let (ks, bd, wt) = coeffs(sw, dw); let half = 1i32 << (PB - 1);
    let mut mid = vec![0u8; dw * sh * 3];
    for y in 0..sh { for xx in 0..dw { let (xmin, cnt) = bd[xx]; let k = &wt[xx * ks..];
        let (mut s0, mut s1, mut s2) = (half, half, half);
        for x in 0..cnt { let p = &src[(y * sw + xmin + x) * 3..]; s0 += p[0] as i32 * k[x]; s1 += p[1] as i32 * k[x]; s2 += p[2] as i32 * k[x]; }
        let o = (y * dw + xx) * 3; mid[o] = clip8(s0 >> PB); mid[o + 1] = clip8(s1 >> PB); mid[o + 2] = clip8(s2 >> PB); } }
    let (ks, bd, wt) = coeffs(sh, dh); let re = dw * 3; let mut out = vec![0u8; re * dh]; let mut acc = vec![0i32; re];
    for yy in 0..dh { let (ymin, cnt) = bd[yy]; let k = &wt[yy * ks..]; acc.fill(half);
        for y in 0..cnt { let row = &mid[(ymin + y) * re..(ymin + y + 1) * re]; let w = k[y]; for i in 0..re { acc[i] += row[i] as i32 * w } }
        for i in 0..re { out[yy * re + i] = clip8(acc[i] >> PB) } }
    out
}

/// Decode PNG/JPEG file bytes to interleaved RGB8 (alpha dropped, 16-bit and gray expanded like stb_image 3-channel load).
pub fn decode(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    // JPEG: faithful port of stb_image's decoder (what llama.cpp mtmd uses), bit-exact. A panic on corrupt input becomes an error.
    if bytes.len() > 2 && bytes[0] == 0xff && bytes[1] == 0xd8 {
        return match std::panic::catch_unwind(|| crate::stbjpeg::decode_rgb(bytes)) { Ok(r) => r, Err(_) => bail!("jpeg decode panicked (corrupt data)") };
    }
    let im = image::load_from_memory(bytes).map_err(|e| anyhow::anyhow!("decode: {e}"))?.to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize); if w == 0 || h == 0 { bail!("empty image") }
    Ok((im.into_raw(), w, h))
}

/// Full preprocessing of RGB8 -> planar normalised f32 [3][h][w] and the (w, h) it was resized to.
pub fn preprocess_rgb(rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (tw, th) = target_size(w, h); let r = resize_bicubic(rgb, w, h, tw, th); let n = tw * th; let mut o = vec![0f32; 3 * n];
    for i in 0..n { for c in 0..3 { o[c * n + i] = (r[i * 3 + c] as f32 / 255.0f32 - MEAN[c]) / STD[c]; } }
    (o, tw, th)
}
pub fn preprocess_file(path: &str) -> Result<(Vec<f32>, usize, usize)> { let (rgb, w, h) = decode(&std::fs::read(path)?)?; Ok(preprocess_rgb(&rgb, w, h)) }
