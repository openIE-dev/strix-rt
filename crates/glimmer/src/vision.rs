//! Muse Glimmer vision encoder (mmproj GGUF, arch "clip"/muse-glimmer): ViT-G/14 (50 pre-LN layers, width 1536, 16 heads of 96, GELU(erf) MLP 8960, biased q/k/v/o,
//! learned 32x32 position table bilinearly resampled, 2D interleaved RoPE theta 1e4 (first half of each head: x position, second half: y), 1-indexed),
//! sparse attention: patches grouped in 32x32 windows (block-diagonal attention) except every 4th layer and the last, which are global;
//! post-LN, pixel-shuffle 2x2 (channel-outer), 3-layer GELU(erf) MLP 6144 -> 4096 -> 4096 -> 6656 (BF16). Preprocess: Pillow-style Lanczos resize
//! (fixed point, as llama.cpp mtmd), (x/255 - 0.5)/0.5. Patch embed + position add run on the CPU, everything else on the GPU (prefill GEMMs).
use anyhow::{bail, Result};
use ash::vk;
use prefill::{GemmOp, Pf};
use rtcore::{DsCache, Weights};
use vkrt::{push_u32, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
pub const D: u32 = 1536; pub const NH: u32 = 16; pub const HD: u32 = 96; pub const FF: u32 = 8960; pub const NL: u32 = 50; pub const PS: u32 = 14; pub const MERGE: u32 = 2;
pub const LDP: u32 = 3 * D; pub const MLP_IN: u32 = D * 4; pub const MLP_H: u32 = 4096; pub const OUT: u32 = 6656; const EPS: f32 = 1e-5; const NONE: u32 = u32::MAX;

/// Aspect-preserving grid (transformers get_aspect_ratio_preserving_size, as llama.cpp mtmd): returns (width, height) in pixels.
pub fn grid_size(w: usize, h: usize, max_tokens: usize) -> (usize, usize) {
    let hw = (PS * MERGE) as f64;
    let (mut nph, mut npw) = (h as f64 / hw, w as f64 / hw);
    let ratio = if nph > 0.0 { npw / nph } else { 1.0 };
    if nph * npw > max_tokens as f64 { nph = (max_tokens as f64 / ratio).sqrt(); npw = nph * ratio; }
    let hs = [nph.floor() as i64, nph.ceil() as i64]; let ws = [npw.floor() as i64, npw.ceil() as i64];
    let tar = h as f64 / w as f64; let (mut bh, mut bw, mut bd) = (-1i64, -1i64, 0.0f64);
    for a in 0..2 { for b in 0..2 {
        let (ph, pw) = (hs[a], ws[b]);
        if ph < 1 || pw < 1 || ph * pw > max_tokens as i64 { continue }
        let d = (ph as f64 / pw as f64 - tar).abs();
        if bh < 0 || d < bd || (d == bd && ph * pw > bh * bw) { bh = ph; bw = pw; bd = d; }
    } }
    if bh < 0 { bh = (nph.round() as i64).max(1); bw = (npw.round() as i64).max(1); }
    (bw as usize * hw as usize, bh as usize * hw as usize)
}

fn lanczos(x: f64) -> f64 { if (-3.0..3.0).contains(&x) { let sinc = |v: f64| if v == 0.0 { 1.0 } else { let p = v * std::f64::consts::PI; p.sin() / p }; sinc(x) * sinc(x / 3.0) } else { 0.0 } }
fn weights_1d(inn: usize, out: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    const PB: i32 = 22;
    let scale = inn as f64 / out as f64; let fs = scale.max(1.0); let support = 3.0 * fs; let ks = support.ceil() as usize * 2 + 1;
    let mut bounds = vec![]; let mut w = vec![0i32; out * ks];
    for xx in 0..out {
        let center = (xx as f64 + 0.5) * scale; let ss = 1.0 / fs;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize; let xmax = (((center + support + 0.5) as i64).min(inn as i64) as usize) - xmin;
        let mut pre = vec![0f64; ks]; let mut ww = 0.0;
        for x in 0..xmax { let v = lanczos((x as f64 + xmin as f64 - center + 0.5) * ss); pre[x] = v; ww += v; }
        for x in 0..xmax { if ww != 0.0 { pre[x] /= ww } }
        for x in 0..ks { let r = pre[x] * (1u64 << PB) as f64 + if pre[x] < 0.0 { -0.5 } else { 0.5 }; w[xx * ks + x] = r as i32; }
        bounds.push((xmin, xmax));
    }
    (ks, bounds, w)
}
/// Pillow-style Lanczos resize of interleaved RGB u8 (horizontal pass, then vertical), fixed point as llama.cpp.
pub fn resize_lanczos(src: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    const PB: i32 = 22; let clip = |v: i32| v.clamp(0, 255) as u8;
    let mut cur = src.to_vec(); let (mut cw, mut ch) = (sw, sh);
    if tw != sw {
        let (ks, b, w) = weights_1d(sw, tw); let mut out = vec![0u8; tw * sh * 3];
        for y in 0..sh { for xx in 0..tw { let (xmin, cnt) = b[xx]; let mut s = [1i32 << (PB - 1); 3];
            for x in 0..cnt { for c in 0..3 { s[c] = s[c].wrapping_add(cur[(y * sw + xmin + x) * 3 + c] as i32 * w[xx * ks + x]); } }
            for c in 0..3 { out[(y * tw + xx) * 3 + c] = clip(s[c] >> PB); } } }
        cur = out; cw = tw;
    }
    if th != sh {
        let (ks, b, w) = weights_1d(sh, th); let re = cw * 3; let mut out = vec![0u8; re * th];
        for yy in 0..th { let (ymin, cnt) = b[yy]; let mut acc = vec![1i32 << (PB - 1); re];
            for y in 0..cnt { let k = w[yy * ks + y]; let row = &cur[(ymin + y) * re..(ymin + y + 1) * re]; for i in 0..re { acc[i] = acc[i].wrapping_add(row[i] as i32 * k); } }
            for i in 0..re { out[yy * re + i] = clip(acc[i] >> PB); } }
        cur = out; ch = th;
    }
    let _ = ch; cur
}

pub struct Vision { pub ctx: Ctx, pub wt: Weights, ds: DsCache, pf: Pf, ln: Kernel, qkr: Kernel, flash: Kernel, unpad: Kernel, ew: Kernel, pub max_tokens: usize }
pub struct VisOut { pub emb: Vec<f32>, pub n_out: usize, pub gpu_s: f64, pub cpu_s: f64 }

impl Vision {
    pub fn load(path: &str) -> Result<Self> {
        let ctx = Ctx::new()?; let wt = Weights::load(&ctx, path)?;
        if wt.wref("v.blk.0.attn_q.weight")?.k != D { bail!("unexpected mmproj shapes") }
        Ok(Vision { pf: Pf::new(&ctx)?, ds: DsCache::default(), ln: ctx.kernel(spv!("vit_ln"), 4, 24, &[], None)?, qkr: ctx.kernel(spv!("vit_qkr"), 6, 24, &[], None)?, unpad: ctx.kernel(spv!("vit_unpad"), 2, 4, &[], None)?,
            flash: ctx.kernel(spv!("glm_vflash"), 4, 44, &[], Some(32))?, ew: ctx.kernel(spv!("vit_ew"), 5, 20, &[], None)?, max_tokens: 4096, wt, ctx })
    }
    /// rgb: interleaved u8 w x h. Returns [n_out][6656] f32 embeddings.
    pub fn encode(&self, rgb: &[u8], w: usize, h: usize) -> Result<VisOut> {
        let t0 = std::time::Instant::now();
        let (tw, th) = grid_size(w, h, self.max_tokens);
        let img = resize_lanczos(rgb, w, h, tw, th);
        let (gw, gh) = (tw / PS as usize, th / PS as usize); let n = gw * gh;
        let (nw, nh) = (gw / 2, gh / 2); let n_out = nw * nh;
        // planar normalised pixels
        let mut pix = vec![0f32; 3 * tw * th];
        for y in 0..th { for x in 0..tw { for c in 0..3 { pix[c * tw * th + y * tw + x] = (img[(y * tw + x) * 3 + c] as f32 / 255.0 - 0.5) / 0.5; } } }
        let fall = self.wt.f32b.as_f32(); let off = |s: &str| -> Result<usize> { Ok(self.wt.f(s)? as usize) };
        let (pw, pe) = (off("v.patch_embd.weight")?, off("v.position_embd.weight")?);
        // f32b is device-mapped memory (slow to read from the CPU): copy what the CPU stage needs once
        let wpv: Vec<f32> = fall[pw..pw + D as usize * 588].to_vec(); let ptab: Vec<f32> = fall[pe..pe + 1024 * D as usize].to_vec(); let (pw, pe) = (0usize, 0usize); let f = &ptab[..];
        // position table 32x32 -> gh x gw bilinear (ggml half-pixel, clamped)
        let side = 32usize; let mut pos = vec![0f32; n * D as usize];
        let (sfx, sfy) = (gw as f32 / side as f32, gh as f32 / side as f32);
        for y in 0..gh { for x in 0..gw {
            let (fy, fx) = ((y as f32 + 0.5) / sfy - 0.5, (x as f32 + 0.5) / sfx - 0.5);
            let (mut y0, mut y1, mut x0, mut x1) = (fy.floor() as i64, fy.floor() as i64 + 1, fx.floor() as i64, fx.floor() as i64 + 1);
            y0 = y0.clamp(0, side as i64 - 1); y1 = y1.clamp(0, side as i64 - 1); x0 = x0.clamp(0, side as i64 - 1); x1 = x1.clamp(0, side as i64 - 1);
            let dy = (fy - y0 as f32).clamp(0.0, 1.0); let dx = (fx - x0 as f32).clamp(0.0, 1.0);
            let r = |yy: i64, xx: i64| pe + (yy as usize * side + xx as usize) * D as usize;
            let o = (y * gw + x) * D as usize;
            for e in 0..D as usize { pos[o + e] = f[r(y0, x0) + e] * (1.0 - dx) * (1.0 - dy) + f[r(y0, x1) + e] * dx * (1.0 - dy) + f[r(y1, x0) + e] * (1.0 - dx) * dy + f[r(y1, x1) + e] * dx * dy; }
        } }
        // sparse window permutation
        let win = side; let (nwh, nww) = (gh.div_ceil(win), gw.div_ceil(win));
        let mut sp: Vec<u32> = vec![]; let mut slens: Vec<usize> = vec![];
        for wy in 0..nwh { for wx in 0..nww { let mut cnt = 0;
            for hh in 0..win { for ww in 0..win { let (gy, gx) = (wy * win + hh, wx * win + ww); if gy < gh && gx < gw { sp.push((gy * gw + gx) as u32); cnt += 1; } } }
            if cnt > 0 { slens.push(cnt) } } }
        let mut inv = vec![0u32; n]; let mut rpos = vec![0u32; 2 * n];
        for i in 0..n { let o = sp[i] as usize; rpos[2 * i] = (o % gw) as u32 + 1; rpos[2 * i + 1] = (o / gw) as u32 + 1; inv[o] = i as u32; }
        let mut rng = vec![0u32; 2 * n]; { let mut off = 0; for &s in &slens { for a in 0..s { rng[2 * (off + a)] = off as u32; rng[2 * (off + a) + 1] = s as u32; } off += s; } }
        let mut dsi = vec![0u32; n_out * 4]; { let mut k = 0; for oy in 0..nh { for ox in 0..nw { for ry in 0..2 { for rx in 0..2 { let g = (oy * 2 + ry) * gw + ox * 2 + rx; dsi[k] = inv[g]; k += 1; } } } } }
        // patch embed (CPU, threaded): x[perm i][o] = sum W[o][c][ky][kx] * pix + pos[orig]
        let wp = &wpv[pw..pw + D as usize * 588]; let mut xs = vec![0f32; n * D as usize];
        let nth = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8); let per = n.div_ceil(nth);
        std::thread::scope(|sc| { for (ci, ch) in xs.chunks_mut(per * D as usize).enumerate() { let (pix, pos, sp, wp) = (&pix, &pos, &sp, wp); sc.spawn(move || {
            let mut patch = [0f32; 588];
            for l in 0..ch.len() / D as usize { let i = ci * per + l; let orig = sp[i] as usize; let (py, px) = (orig / gw, orig % gw);
                for c in 0..3 { for ky in 0..14 { for kx in 0..14 { patch[c * 196 + ky * 14 + kx] = pix[c * tw * th + (py * 14 + ky) * tw + px * 14 + kx]; } } }
                for o in 0..D as usize { let wr = &wp[o * 588..(o + 1) * 588]; let mut s = 0f32; for k in 0..588 { s += wr[k] * patch[k]; }
                    ch[l * D as usize + o] = s + pos[orig * D as usize + o]; } } }); } });
        let cpu_s = t0.elapsed().as_secs_f64();
        // GPU
        let ctx = &self.ctx; let al = FF.max(MLP_IN);
        let fb = |nn: usize| ctx.buffer((nn.max(64) * 4) as u64);
        let (x, y, t, pr) = (fb(n * D as usize)?, fb(n * FF as usize)?, fb(n * D as usize)?, fb(n * LDP as usize)?);
        let a = ctx.buffer(((n + 256) * al as usize * 2) as u64)?;
        let hb = |_: u8| -> Result<Buffer> { let b = ctx.buffer(((n + 128) * 2048 * 2) as u64)?; unsafe { std::ptr::write_bytes(b.ptr as *mut u8, 0, b.size as usize); } Ok(b) };
        let (qh, kh, vh, oh) = (hb(0)?, hb(1)?, hb(2)?, hb(3)?);
        let (bp, br, bi, bo) = (ctx.buffer((2 * n * 4).max(64) as u64)?, ctx.buffer((2 * n * 4).max(64) as u64)?, ctx.buffer((n_out * 16).max(64) as u64)?, fb(1)?);
        x.as_f32_mut()[..xs.len()].copy_from_slice(&xs);
        unsafe { std::ptr::copy_nonoverlapping(rpos.as_ptr(), bp.ptr as *mut u32, rpos.len()); std::ptr::copy_nonoverlapping(rng.as_ptr(), br.ptr as *mut u32, rng.len()); std::ptr::copy_nonoverlapping(dsi.as_ptr(), bi.ptr as *mut u32, dsi.len()); }
        let wt = &self.wt; let set = |k: &Kernel, b: &[&Buffer]| self.ds.get(ctx, k, b);
        let gemm = |cb: vk::CommandBuffer, name: &str, m: u32, yb: &Buffer, ldy: u32, ycol: u32| -> Result<()> {
            let w = wt.wref(name)?; self.pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &a, lda: al, a_row0: 0, m, y: yb, ldy, y_col: ycol, acc: false })
        };
        let lnorm = |cb: vk::CommandBuffer, inp: &Buffer, w: &str, b: &str, mode: u32, out: &Buffer, lda: u32| -> Result<()> {
            let ds = set(&self.ln, &[inp, &a, &wt.f32b, out])?;
            ctx.dispatch3(cb, &self.ln, ds, &push_u32(&[D, lda, wt.f(w)?, wt.f(b)?, EPS.to_bits(), mode]), 1, n as u32, 1); Ok(())
        };
        let ew = |cb: vk::CommandBuffer, xb: &Buffer, yb: &Buffer, k: u32, mode: u32, b_off: u32, ycs: u32, rows: u32, gx: u32| -> Result<()> {
            let ds = set(&self.ew, &[xb, yb, &a, &wt.f32b, &bi])?;
            ctx.dispatch3(cb, &self.ew, ds, &push_u32(&[k, mode, b_off, ycs, al]), gx, rows, 1); Ok(())
        };
        let mut gpu = 0.0; let nn = n as u32;
        let wins_sparse: Vec<(usize, usize)> = { let mut o = 0; slens.iter().map(|&l| { let r = (o, l); o += l; r }).collect() };
        let per_sub = std::env::var("VIT_LAYERS_PER_SUB").ok().and_then(|v| v.parse().ok()).unwrap_or(5u32);
        let mut l0 = 0u32;
        // stage 0: pre_ln into x (in place semantic: separate: use t as scratch)
        { let mut err = Ok(()); gpu += ctx.run(|cb| { err = (|| -> Result<()> {
            lnorm(cb, &x, "v.pre_ln.weight", "v.pre_ln.bias", 1, &t, D)?; ctx.barrier(cb);
            // copy t -> x via ew? use ln output directly: swap roles below
            Ok(()) })(); })?; err?; }
        // x := pre_ln(x) (t holds it)
        let (mut xa, mut xb_) = (&t, &x); // xa = current residual, xb_ = scratch for sub-layer outputs
        while l0 < NL {
            let l1 = (l0 + per_sub).min(NL); let mut err = Ok(());
            gpu += ctx.run(|cb| { err = (|| -> Result<()> {
                for l in l0..l1 {
                    let p = |s: &str| format!("v.blk.{l}.{s}");
                    let global = l == NL - 1 || (l + 1) % 4 == 0;
                    lnorm(cb, xa, &p("ln1.weight"), &p("ln1.bias"), 0, &bo, al)?; ctx.barrier(cb);
                    gemm(cb, &p("attn_q.weight"), nn, &pr, LDP, 0)?; gemm(cb, &p("attn_k.weight"), nn, &pr, LDP, D)?; gemm(cb, &p("attn_v.weight"), nn, &pr, LDP, 2 * D)?; ctx.barrier(cb);
                    let ds = set(&self.qkr, &[&pr, &wt.f32b, &bp, &qh, &kh, &vh])?;
                    ctx.dispatch3(cb, &self.qkr, ds, &push_u32(&[LDP, wt.f(&p("attn_q.bias"))?, wt.f(&p("attn_k.bias"))?, wt.f(&p("attn_v.bias"))?, 10000f32.to_bits(), HD]), 3 * NH, nn, 1); ctx.barrier(cb);
                    let scale = (1.0f32 / (HD as f32).sqrt()).to_bits();
                    let wins: &[(usize, usize)] = if global { &[(0, n)] } else { &wins_sparse };
                    for &(off, len) in wins {
                        let ds = set(&self.flash, &[&qh, &kh, &vh, &oh])?;
                        ctx.dispatch(cb, &self.flash, ds, &push_u32(&[len as u32, len as u32, NH, 2048, 2048, 2048, 2048, scale, 1, 0, off as u32]), (len as u32).div_ceil(64) * NH); ctx.barrier(cb);
                    }
                    let ds = set(&self.unpad, &[&oh, &a])?;
                    ctx.dispatch3(cb, &self.unpad, ds, &push_u32(&[al]), D.div_ceil(256), nn, 1); ctx.barrier(cb);
                    gemm(cb, &p("attn_out.weight"), nn, xb_, D, 0)?; ctx.barrier(cb);
                    ew(cb, xa, xb_, D, 0, wt.f(&p("attn_out.bias"))?, 0, nn, D.div_ceil(256))?; ctx.barrier(cb);
                    lnorm(cb, xa, &p("ln2.weight"), &p("ln2.bias"), 0, &bo, al)?; ctx.barrier(cb);
                    gemm(cb, &p("ffn_up.weight"), nn, &y, FF, 0)?; ctx.barrier(cb);
                    ew(cb, xa, &y, FF, 1, wt.f(&p("ffn_up.bias"))?, FF, nn, FF.div_ceil(256))?; ctx.barrier(cb);
                    gemm(cb, &p("ffn_down.weight"), nn, xb_, D, 0)?; ctx.barrier(cb);
                    ew(cb, xa, xb_, D, 0, wt.f(&p("ffn_down.bias"))?, 0, nn, D.div_ceil(256))?; ctx.barrier(cb);
                }
                Ok(()) })(); })?; err?;
            l0 = l1;
        }
        let _ = &mut xb_; let _ = &mut xa;
        let mut err = Ok(());
        gpu += ctx.run(|cb| { err = (|| -> Result<()> {
            lnorm(cb, xa, "v.post_ln.weight", "v.post_ln.bias", 1, xb_, D)?; ctx.barrier(cb);
            ew(cb, xb_, xb_, D, 2, 0, 0, n_out as u32, (D * 4).div_ceil(256))?; ctx.barrier(cb);
            gemm(cb, "mm.0.weight", n_out as u32, &y, MLP_H, 0)?; ctx.barrier(cb);
            ew(cb, xb_, &y, MLP_H, 1, NONE, MLP_H, n_out as u32, MLP_H.div_ceil(256))?; ctx.barrier(cb);
            gemm(cb, "mm.1.weight", n_out as u32, &y, MLP_H, 0)?; ctx.barrier(cb);
            ew(cb, xb_, &y, MLP_H, 1, NONE, MLP_H, n_out as u32, MLP_H.div_ceil(256))?; ctx.barrier(cb);
            gemm(cb, "mm.2.weight", n_out as u32, &y, OUT, 0)?;
            Ok(()) })(); })?; err?;
        let emb = y.as_f32()[..n_out * OUT as usize].to_vec();
        Ok(VisOut { emb, n_out, gpu_s: gpu, cpu_s })
    }
}

/// Decode a PNG / JPEG file to interleaved RGB8 (alpha dropped). Returns (rgb, w, h).
pub fn load_image(path: &str) -> Result<(Vec<u8>, usize, usize)> {
    let raw = std::fs::read(path)?;
    if raw.starts_with(&[0x89, b'P', b'N', b'G']) {
        let mut dec = png::Decoder::new(std::io::Cursor::new(&raw[..])); dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut r = dec.read_info()?; let mut buf = vec![0u8; r.output_buffer_size()]; let info = r.next_frame(&mut buf)?;
        let (w, h) = (info.width as usize, info.height as usize); let n = info.color_type.samples();
        let rgb = match n { 3 => buf[..w * h * 3].to_vec(), 4 => buf.chunks(4).take(w * h).flat_map(|p| [p[0], p[1], p[2]]).collect(),
            1 => buf.iter().take(w * h).flat_map(|&g| [g, g, g]).collect(), 2 => buf.chunks(2).take(w * h).flat_map(|p| [p[0], p[0], p[0]]).collect(), _ => bail!("png samples {n}") };
        Ok((rgb, w, h))
    } else {
        let mut d = zune_jpeg::JpegDecoder::new_with_options(&raw, zune_core::options::DecoderOptions::default().jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB));
        let px = d.decode().map_err(|e| anyhow::anyhow!("jpeg: {e:?}"))?; let i = d.info().ok_or_else(|| anyhow::anyhow!("jpeg info"))?;
        Ok((px, i.width as usize, i.height as usize))
    }
}
