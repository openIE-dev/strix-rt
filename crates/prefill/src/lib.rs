//! Batched prompt processing (prefill) for strix-rt models.
//! * `Pf`: shared kernels (quantized-weight x f16 coop-matrix GEMM, norms, conv, scans, attention) and scratch buffers.
//! * `GemmOp` / `Pf::gemm`: the GEMM interface. A call covers a row range [row0, row0+rows) of one weight so a caller
//!   (e.g. an NPU split) can hand part of N to another engine and the GPU the rest.
//! * `qwen35::prefill` (Qwen3.8 / Bonsai trunk) and `lfm2moe::prefill` drive the batched forward and leave the model's
//!   KV cache / conv state / DeltaNet state exactly where the token-by-token path would, so decode continues.
pub mod qwen35;
pub mod lfm2moe;
pub mod qmm;
use anyhow::Result;
use ash::vk;
use rtcore::{qtype, DsCache, WRef};
use std::{cell::RefCell, collections::HashMap};
use vkrt::{push_u32, Buffer, Ctx, Kernel};

pub mod spv {
    macro_rules! s { ($n:ident, $f:literal) => { pub const $n: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $f, ".spv")); } }
    s!(GEMM, "pf_gemm"); s!(Q8, "pf_q8"); s!(QGEMM, "pf_qgemm"); s!(QGEMMH, "pf_qgemmh"); s!(PEAK_F16, "pf_peak_f16"); s!(H2F, "pf_h2f"); s!(TGEMM, "pf_tgemm"); s!(PEAK_S8, "pf_peak_s8");
    s!(RMSNORM, "pf_rmsnorm"); s!(GDN_CONV, "pf_gdn_conv"); s!(GDN_CONVST, "pf_gdn_convst"); s!(GDN_SCAN, "pf_gdn_scan"); s!(GDN_CHUNK, "pf_gdn_chunk"); s!(GDN_CHUNKP, "pf_gdn_chunkp"); s!(GDN_CHUNKCM, "pf_gdn_chunkcm");
    s!(GNORM, "pf_gnorm"); s!(ROPE64, "pf_rope64"); s!(ATTN64, "pf_attn64"); s!(SCONV, "pf_sconv"); s!(MOESORT, "pf_moesort"); s!(COMBNORM, "pf_combnorm"); s!(RSEL, "pf_rsel"); s!(MOESEL, "pf_moesel"); s!(F2H, "pf_f2h"); s!(SCONVST, "pf_sconvst"); s!(ROUTE, "pf_route"); s!(GATHER, "pf_gather"); s!(MOECOMB, "pf_moecomb"); s!(ROPE, "pf_rope"); s!(ATTN, "pf_attn"); s!(ATTNFA, "pf_attnfa"); s!(ATTNFA64, "pf_attnfa64"); s!(FACMP, "pf_facmp"); s!(SWIGLU, "pf_swiglu");
}

/// A buffers must be readable for whole BM-row tiles (activations padded by >= 128 rows): A is coop-loaded directly.
/// One GEMM: y[m, y_col + n] (+)= sum_k a[a_row0 + m, k] * W[row0 + n, k] for m < m_rows, n < rows.
pub struct GemmOp<'a> { pub w: WRef, pub row0: u32, pub rows: u32, pub a: &'a Buffer, pub lda: u32, pub a_row0: u32, pub m: u32, pub y: &'a Buffer, pub ldy: u32, pub y_col: u32, pub acc: bool }

pub struct Pf { pub gemm_k: RefCell<HashMap<u32, Box<Kernel>>>, pub qx: RefCell<Option<(u64, [Buffer; 3])>>, pub ds: DsCache, pub k: HashMap<&'static str, Kernel>, dummy: Buffer, gx: RefCell<Vec<Buffer>>, cmp: RefCell<Option<(Buffer, Buffer)>>, ncmp: std::cell::Cell<u32> }

/// (MI, NJ, WR) of a gemm variant key: 0x4000 = 32 x 256 (grouped mrows 0x8020), 0x200 = 16 x 256 tiles, 0x400 = 32 x 128 (PF_G32N=256: 32 x 256), 0x800 = 64 x 128, else the default 128 x 128.
fn tile_cfg(ty: u32) -> (u32, u32, u32) {
    if ty & 0x200 != 0 { (1, 2, 1) } else if ty & 0x4000 != 0 { (1, 4, 2) } else if ty & 0x400 != 0 { if env("PF_G32C", 0) == 1 { (2, 2, 1) } else if env("PF_G32C", 0) == 2 { (2, 1, 1) } else if env("PF_G32N", 128) == 128 { (1, 2, 2) } else { (1, 4, 2) } } else if ty & 0x800 != 0 { (1, 4, 4) } else if ty & 0x100 != 0 && ty & 0x2000 == 0 && matches!(ty & 0xf, 10 | 11 | 13) { (env("PF_SWI_MI", 4), 4, 4) } else if ty & 0x2100 != 0 { (2, 4, 4) } else { (env("PF_MI", 2), env("PF_NJ", 8), env("PF_WR", 8)) } }
// (0x2000 = 128 x 128 for narrow outputs; 0x100 = fused SwiGLU: its epilogue pairs gate/up waves across the wave columns, so it keeps 128 x 128 tiles)
/// Repack a quantized weight (any GGUF type gguf::dequant handles) into the T8 tiled int8 layout used by `gemm_t8`:
/// 128-row x 32-k chunks of int8 with one f16 absmax scale per (row, 32-k block). Rows are padded to 128.
/// Memory: 34/32 bytes per weight. Conversion runs on all CPU threads.
pub fn repack_t8(data: &[u8], ty: gguf::GgmlType, n: usize, k: usize) -> Vec<u8> {
    let (nt, kb) = (n.div_ceil(128), k / 32); let rb = data.len() / n;
    let mut out = vec![0u8; nt * kb * 4352];
    let nth = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8);
    let per = nt.div_ceil(nth);
    std::thread::scope(|sc| { for (ci, chunk) in out.chunks_mut(per * kb * 4352).enumerate() { sc.spawn(move || {
        let mut row = vec![0f32; k];
        for tl in 0..chunk.len() / (kb * 4352) { let t = ci * per + tl;
            for r in 0..128 { let gr = t * 128 + r; if gr >= n { break }
                gguf::dequant::dequant_row(ty, &data[gr * rb..(gr + 1) * rb], &mut row);
                for b in 0..kb { let blk = &row[b * 32..b * 32 + 32]; let am = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let d = half::f16::from_f32(am / 127.0); let di = if d.to_f32() > 0.0 { 1.0 / d.to_f32() } else { 0.0 };
                    let c = (tl * kb + b) * 4352;
                    for (e, v) in blk.iter().enumerate() { chunk[c + r * 32 + e] = ((v * di).round().clamp(-127.0, 127.0) as i8) as u8; }
                    chunk[c + 4096 + r * 2..c + 4096 + r * 2 + 2].copy_from_slice(&d.to_le_bytes()); } } } }); } });
    out
}
pub fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

impl Drop for Pf {
    fn drop(&mut self) {
        if let Some((_, o)) = self.cmp.borrow().as_ref() {
            let v = unsafe { std::slice::from_raw_parts(o.ptr as *const f32, 4096) };
            let n = (self.ncmp.get() as usize).min(4096);
            eprintln!("FA_CMP {} calls, max|diff| per call: {:?}", n, v[..n].iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
        }
    }
}
impl Pf {
    pub fn new(ctx: &Ctx) -> Result<Self> {
        let mut k = HashMap::new();
        // (name, spirv, bindings, push bytes, subgroup)
        let list: &[(&'static str, &[u8], u32, u32, &[u32])] = &[
            ("rmsnorm", spv::RMSNORM, 3, 20, &[]), ("gdn_conv", spv::GDN_CONV, 4, 32, &[]), ("gdn_convst", spv::GDN_CONVST, 2, 16, &[]),
            ("gdn_scan", spv::GDN_SCAN, 5, 40, &[]), ("gnorm", spv::GNORM, 4, 24, &[]), ("rope", spv::ROPE, 4, 52, &[256]),
            ("attn", spv::ATTN, 4, 36, &[]), ("swiglu", spv::SWIGLU, 2, 12, &[]),
            ("rope64", spv::ROPE64, 4, 40, &[]), ("attn64", spv::ATTN64, 4, 32, &[]), ("sconv", spv::SCONV, 4, 24, &[]), ("moesort", spv::MOESORT, 4, 24, &[]), ("combnorm", spv::COMBNORM, 6, 24, &[]), ("sconvst", spv::SCONVST, 2, 16, &[]),
            ("gather", spv::GATHER, 3, 16, &[]), ("f2h", spv::F2H, 2, 12, &[]), ("moecomb", spv::MOECOMB, 4, 12, &[])];
        for &(n, s, b, p, sc) in list { k.insert(n, ctx.kernel(s, b, p, sc, None)?); }
        // Bonsai (ternary, Hadamard-rotated): f16->f32, crates/ternary FWHT + per-token int8 quant, int8 coop-matrix GEMM
        // chunked GDN scan: spec (R rows per workgroup, C tokens per chunk)
        k.insert("gdn_chunk", ctx.kernel(spv::GDN_CHUNK, 6, 40, &[env("PF_GDN_R", 64), env("PF_GDN_C", 16)], None)?);
        k.insert("gdn_chunkcm", ctx.kernel(spv::GDN_CHUNKCM, 6, 40, &[env("PF_GDN_WG", 128)], Some(32))?);
        k.insert("gdn_chunkp", ctx.kernel(spv::GDN_CHUNKP, 4, 40, &[0, env("PF_GDN_C", 16)], None)?);
        k.insert("facmp", ctx.kernel(spv::FACMP, 3, 8, &[], None)?);
        k.insert("attnfa64", ctx.kernel(spv::ATTNFA64, 4, 32, &[env("PF_FA_QLO", 0)], Some(32))?);
        k.insert("attnfa", ctx.kernel(spv::ATTNFA, 4, 36, &[env("PF_FA_QLO", 0)], Some(32))?);
        k.insert("route", ctx.kernel(spv::ROUTE, 3, 28, &[], Some(32))?);
        k.insert("rsel", ctx.kernel(spv::RSEL, 2, 16, &[], Some(32))?);
        k.insert("moesel", ctx.kernel(spv::MOESEL, 5, 28, &[], Some(32))?);
        k.insert("h2f", ctx.kernel(spv::H2F, 2, 12, &[], None)?);
        k.insert("fwht", ctx.kernel(ternary::shaders::FWHT, 3, 28, &[], None)?);
        k.insert("quantx", ctx.kernel(ternary::shaders::QUANTX, 3, 8, &[], Some(32))?);
        k.insert("tgemm_pq2", ctx.kernel(spv::TGEMM, 4, 28, &[0], Some(32))?);
        k.insert("tgemm_ptq1", ctx.kernel(spv::TGEMM, 4, 28, &[1], Some(32))?);
        Ok(Pf { gemm_k: Default::default(), qx: Default::default(), ds: DsCache::default(), k, dummy: ctx.buffer(256)?, gx: Default::default(), cmp: Default::default(), ncmp: Default::default() })
    }
    fn gemm_kernel(&self, ctx: &Ctx, ty: u32) -> Result<&Kernel> {
        if !self.gemm_k.borrow().contains_key(&ty) {
            let (swi, t1, t2) = if ty & 0x100 != 0 { (1, ty & 0xf, (ty >> 4) & 0xf) } else { (0, ty & 0xff, 0) };   // 0x1000: grouped
            let (mi, nj, wr) = tile_cfg(ty);
            let grp = (ty >> 12) & 1;
            // cooperative K-slab (KS): fill all 256 lanes with sub-block decodes (BN=128 tiles -> KS=2); PF_KS overrides (1 = old)
            // fused SwiGLU (128 x 128, two waves share each A row-block) stages A through shared memory: 12.2 -> 16.6 TFLOPS at
            // M=4096 (qimage DiT), BF16/F16 weights only (Q4_K LLM prefill loses 12-15% with it: pp512 534 -> 472). Plain GEMMs keep direct A loads (PF_SWI_ADIR / PF_ADIR override)
            let (sg, adir) = (env("PF_SG", 32), if swi == 1 && matches!(t1, 10 | 11 | 13) { env("PF_SWI_ADIR", 0) } else { env("PF_ADIR", 1) });
            let bn = (256 / sg / wr) * 16 * nj;
            let ks = if adir == 0 { 1 } else { env("PF_KS", (256 / bn).clamp(1, 2)).clamp(1, (256 / bn).max(1)) };
            let kk = ctx.kernel(spv::GEMM, 5, 48, &[t1, mi, nj, wr, swi, t2, grp, env("PF_PROBE", 0), 256 / sg, env("PF_LS4", 4 * ks + 1), adir, ks], Some(sg))?; self.gemm_k.borrow_mut().insert(ty, Box::new(kk)); }
        let p: *const Kernel = &**self.gemm_k.borrow().get(&ty).unwrap();
        Ok(unsafe { &*p })
    }
    /// Record one GEMM (no barrier). `arena` is the buffer holding op.w, `f32b` the codebook buffer (rtcore::Weights::f32b).
    pub fn gemm(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, op: &GemmOp) -> Result<()> { self.gemm_tile(ctx, cb, arena, f32b, op, None) }
    /// gemm with an explicit tile variant (0 = 256 x 128, 0x2000 = 128 x 128, 0x800 = 64 x 128, 0x400 = 32 x 128/256, 0x200 = 16 x 256); None = automatic.
    pub fn gemm_tile(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, op: &GemmOp, tile: Option<u32>) -> Result<()> { self.gemm_split(ctx, cb, arena, f32b, op, tile, 1) }
    /// gemm_tile with split-K: `splits` k-slices per output tile, partial sums atomically added into y (y must hold zeros or, with acc, the addend).
    pub fn gemm_split(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, op: &GemmOp, tile: Option<u32>, splits: u32) -> Result<()> {
        let ty = qtype(op.w.ty)?;
        // small-M tiles (lm_head of the last token, per-expert groups of a grouped MoE GEMM)
        let v = if env("PF_MTILE", 1) == 0 { 0 } else if op.m <= 16 { 0x200 } else if op.m <= 32 { 0x400 } else if op.m <= 64 { 0x800 } else { 0 };
        // 256 x 128 tiles leave CUs idle on narrow outputs (e.g. ffn_down N=5120 at M=512: 80 workgroups): use 128 x 128 then
        let v = if v == 0 && op.m.div_ceil(256) * op.rows.div_ceil(128) < env("PF_MINWG", 0) { 0x2000 } else { v };
        let v = tile.unwrap_or(v);
        let k = self.gemm_kernel(ctx, ty | v)?;
        let row_bytes = op.w.bytes / (op.w.n as u64 * op.w.e.max(1) as u64);
        let w_off = op.w.off + op.row0 as u64 * row_bytes;
        anyhow::ensure!(w_off <= u32::MAX as u64 && op.lda % 8 == 0 && op.w.k % 64 == 0, "gemm: offset/alignment");
        let ds = self.ds.get(ctx, k, &[arena, op.a, op.y, f32b, &self.dummy])?;
        let push = push_u32(&[op.m, op.rows, op.w.k, w_off as u32, op.lda, op.ldy, op.y_col, op.acc as u32, op.a_row0, 0, 0, splits.max(1)]);
        let (mi, nj, wr) = tile_cfg(v); let (bm, bn) = (wr * 16 * mi, (256 / env("PF_SG", 32) / wr) * 16 * nj);
        let groups = op.m.div_ceil(bm) * op.rows.div_ceil(bn) * splits.max(1);
        ctx.dispatch(cb, k, ds, &push, groups);
        Ok(())
    }
    /// Fused SwiGLU GEMM: y16[m, y_col + n] = f16(silu(gate[n].a[m]) * (up[n].a[m])) for n in [row0, row0+rows).
    /// gate and up must live in the same arena. `y` is an f16 buffer with row stride ldy (halves).
    pub fn gemm_swiglu(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, gate: WRef, up: WRef, row0: u32, rows: u32,
                       a: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        self.gemm_swiglu_tile(ctx, cb, arena, f32b, gate, up, row0, rows, a, lda, m, y, ldy, y_col, 0)
    }
    /// gemm_swiglu with an explicit row-tile variant `v` (0 = default 256 x 128; 0x800 = 64-row tiles for small M).
    pub fn gemm_swiglu_tile(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, gate: WRef, up: WRef, row0: u32, rows: u32,
                       a: &Buffer, lda: u32, m: u32, y: &Buffer, ldy: u32, y_col: u32, v: u32) -> Result<()> {
        anyhow::ensure!(gate.arena == up.arena && gate.k == up.k && gate.n == up.n && lda % 8 == 0 && gate.k % 64 == 0, "gemm_swiglu: layout");
        let (t1, t2) = (qtype(gate.ty)?, qtype(up.ty)?);
        let k = self.gemm_kernel(ctx, 0x100 | t1 | (t2 << 4) | v)?;
        let rbg = gate.bytes / gate.n as u64; let rbu = up.bytes / up.n as u64;
        let (wo, wo2) = (gate.off + row0 as u64 * rbg, up.off + row0 as u64 * rbu);
        anyhow::ensure!(wo2 <= u32::MAX as u64 && wo <= u32::MAX as u64, "gemm_swiglu: offset");
        let ds = self.ds.get(ctx, k, &[arena, a, y, f32b, &self.dummy])?;
        let push = push_u32(&[m, rows, gate.k, wo as u32, lda, ldy, y_col, 0, 0, wo2 as u32, 0, 0]);
        let (mi, nj, wr) = tile_cfg(0x100 | t1 | (t2 << 4) | v); let (bm, bn) = (wr * 16 * mi, (8 / wr) * 16 * nj);
        ctx.dispatch(cb, k, ds, &push, m.div_ceil(bm) * rows.div_ceil(bn / 2));
        Ok(())
    }
    /// Grouped (MoE) GEMM over a tile table built on the GPU (pf_moesort): for each tile (e, r0, r1) and all n < w.n,
    /// y[r, y_col + n] = sum_k a[r, k] * W_e[n, k]. `w` is the stacked expert tensor; `mrows` selects the tile variant
    /// (16 / 32 / 64 / 128 rows, must match the table's tile height); `maxt` bounds the tile count.
    pub fn gemm_grouped(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, w: WRef, table: &Buffer, mrows: u32, maxt: u32,
                        a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        let v = match mrows { 16 => 0x200, 32 => 0x400, 0x8020 => 0x4400, 64 => 0x800, 128 => 0x2000, _ => anyhow::bail!("gemm_grouped: tile rows {mrows}") };
        let k = self.gemm_kernel(ctx, qtype(w.ty)? | v | 0x1000)?;
        let eb = w.bytes / w.e.max(1) as u64;
        anyhow::ensure!(w.off + w.bytes <= u32::MAX as u64 && lda % 8 == 0 && w.k % 64 == 0, "gemm_grouped: offset/alignment");
        let ds = self.ds.get(ctx, k, &[arena, a, y, f32b, table])?;
        let (mi, nj, wr) = tile_cfg(v); let (bm, bn) = (wr * 16 * mi, (256 / env("PF_SG", 32) / wr) * 16 * nj); debug_assert_eq!(bm, mrows & 0xff);
        let push = push_u32(&[0, w.n, w.k, w.off as u32, lda, ldy, y_col, 0, maxt, 0, eb as u32, 0]);
        ctx.dispatch(cb, k, ds, &push, maxt * w.n.div_ceil(bn));
        Ok(())
    }
    /// Grouped fused SwiGLU (MoE gate|up): y16[r, y_col + n] = f16(silu(G_e[n].a[r]) * (U_e[n].a[r])), tiles from `table`.
    pub fn gemm_grouped_swiglu(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, f32b: &Buffer, gate: WRef, up: WRef, table: &Buffer, mrows: u32, maxt: u32,
                               a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        let v = match mrows { 16 => 0x200, 32 => 0x400, 0x8020 => 0x4400, 64 => 0x800, 128 => 0x2000, _ => anyhow::bail!("gemm_grouped_swiglu: tile rows {mrows}") };
        anyhow::ensure!(gate.arena == up.arena && gate.k == up.k && gate.n == up.n && gate.bytes == up.bytes && gate.ty == up.ty, "gemm_grouped_swiglu: layout");
        anyhow::ensure!(up.off + up.bytes <= u32::MAX as u64 && gate.off + gate.bytes <= u32::MAX as u64 && lda % 8 == 0, "gemm_grouped_swiglu: offset");
        let t = qtype(gate.ty)?;
        let k = self.gemm_kernel(ctx, 0x100 | t | (t << 4) | v | 0x1000)?;
        let eb = gate.bytes / gate.e.max(1) as u64;
        let ds = self.ds.get(ctx, k, &[arena, a, y, f32b, table])?;
        let (mi, nj, wr) = tile_cfg(v); let bn = (256 / env("PF_SG", 32) / wr) * 16 * nj;
        let push = push_u32(&[0, gate.n, gate.k, gate.off as u32, lda, ldy, y_col, 0, maxt, up.off as u32, eb as u32, 0]);
        let _ = mi; ctx.dispatch(cb, k, ds, &push, maxt * gate.n.div_ceil(bn / 2));
        Ok(())
    }
    /// GEMM on a T8-repacked weight (see `repack_t8`): y[m, y_col + n] (+)= a[m] . W[n] for n < n_rows.
    pub fn gemm_t8(&self, ctx: &Ctx, cb: vk::CommandBuffer, wb: &Buffer, f32b: &Buffer, n_rows: u32, k: u32, a: &Buffer, lda: u32, m: u32,
                   y: &Buffer, ldy: u32, y_col: u32, acc: bool) -> Result<()> {
        let v = if env("PF_MTILE", 1) == 0 { 0 } else if m <= 16 { 0x200 } else if m <= 32 { 0x400 } else if m <= 64 { 0x800 } else { 0 };
        let kk = self.gemm_kernel(ctx, 12 | v)?;
        let ds = self.ds.get(ctx, kk, &[wb, a, y, f32b, &self.dummy])?;
        let (mi, nj, wr) = tile_cfg(v); let (bm, bn) = (wr * 16 * mi, (256 / env("PF_SG", 32) / wr) * 16 * nj);
        let push = push_u32(&[m, n_rows, k, 0, lda, ldy, y_col, acc as u32, 0, 0, 0, 0]);
        ctx.dispatch(cb, kk, ds, &push, m.div_ceil(bm) * n_rows.div_ceil(bn));
        Ok(())
    }
    /// Ternary (PQ2_0 / PTQ1_0) x int8 GEMM over weight rows [row0, row0+rows): y[m, y_col + n] (+)= ...
    /// xq/xs: per-token int8 activations + per-128-group scales (ternary quantx layout), padded to 64-row multiples.
    /// Same row-range contract as `gemm`, so an NPU split (npu::split) can take rows [0, nn) and the GPU the rest.
    pub fn tgemm(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, w: WRef, row0: u32, rows: u32, xq: &Buffer, xs: &Buffer, m: u32,
                 y: &Buffer, ldy: u32, y_col: u32, acc: bool) -> Result<()> {
        let name = match qtype(w.ty)? { 8 => "tgemm_pq2", 9 => "tgemm_ptq1", t => anyhow::bail!("tgemm: type {t}") };
        anyhow::ensure!(rows % 64 == 0 && w.k % 128 == 0, "tgemm: N {rows} / K {} alignment", w.k);
        let rb = w.bytes / w.n as u64; let wo = w.off + row0 as u64 * rb;
        anyhow::ensure!(wo <= u32::MAX as u64 && wo % 4 == 0, "tgemm: offset");
        self.run(ctx, name, &[arena, xq, xs, y], &[m, rows, w.k, wo as u32, ldy, y_col, acc as u32], m.div_ceil(64) * (rows / 64), cb)
    }
    pub fn kern(&self, n: &str) -> &Kernel { &self.k[n] }
    /// Gated DeltaNet scan over push[9] tokens (push layout of pf_gdn_scan). PF_GDN_CHUNK: 0 = sequential, 1 = chunked,
    /// 2 = chunked when M >= PF_GDN_CHUNK_MIN.
    pub fn gdn(&self, ctx: &Ctx, bufs: &[&Buffer], push: &[u32], cb: vk::CommandBuffer) -> Result<()> {
        let mode = env("PF_GDN_CHUNK", 0);
        if mode == 1 || (mode == 2 && push[9] >= env("PF_GDN_CHUNK_MIN", 64)) {
            // scratch for the pre-pass, grow-only; superseded buffers are kept alive since recorded work may still use them
            let cm = env("PF_GDN_CM", 1) != 0; let cc = if cm { 16 } else { env("PF_GDN_C", 16) }; let nch = push[9].div_ceil(cc); let need = (push[1] * nch * (2 * cc + 2 * cc * cc) * 4) as u64;
            if self.gx.borrow().last().map(|b| (b.size as u64) < need).unwrap_or(true) { self.gx.borrow_mut().push(ctx.buffer(need.max(1 << 20) as _)?); }
            let gx = self.gx.borrow(); let x = gx.last().unwrap();
            self.run(ctx, "gdn_chunkp", &[bufs[0], bufs[1], x, bufs[4]], push, push[1] * nch, cb)?; ctx.barrier(cb);
            if cm { self.run(ctx, "gdn_chunkcm", &[bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], x], push, push[1] * (128 / env("PF_GDN_WG", 128)), cb) } else { self.run(ctx, "gdn_chunk", &[bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], x], push, push[1] * (128 / env("PF_GDN_R", 64)), cb) }
        }
        else { self.run(ctx, "gdn_scan", bufs, push, push[1], cb) }
    }
    pub fn run(&self, ctx: &Ctx, name: &str, bufs: &[&Buffer], push: &[u32], groups: u32, cb: vk::CommandBuffer) -> Result<()> {
        // PF_FA_CMP=1 (debug): run pf_attn into the real output and pf_attnfa into a scratch copy; record max |diff| per call
        if name == "attn" && env("PF_FA_CMP", 0) == 1 {
            let n = push[8] * push[7];
            { let mut c = self.cmp.borrow_mut(); if c.as_ref().map(|c| c.0.size < bufs[3].size).unwrap_or(true) { *c = Some((ctx.buffer(bufs[3].size)?, ctx.buffer(4096 * 4)?)); } }
            let c = self.cmp.borrow(); let (sc, out) = c.as_ref().unwrap();
            let k = &self.k["attn"]; let ds = self.ds.get(ctx, k, bufs)?; ctx.dispatch(cb, k, ds, &push_u32(push), groups);
            let k = &self.k["attnfa"]; let ds = self.ds.get(ctx, k, &[bufs[0], bufs[1], bufs[2], sc])?; ctx.dispatch(cb, k, ds, &push_u32(push), push[8].div_ceil(64) * push[0]);
            ctx.barrier(cb);
            let i = self.ncmp.get(); self.ncmp.set(i + 1);
            let k = &self.k["facmp"]; let ds = self.ds.get(ctx, k, &[bufs[3], sc, out])?; ctx.dispatch(cb, k, ds, &push_u32(&[n, i % 4096]), 64);
            return Ok(());
        }
        // head-dim-256 causal attention: coopmat flash attention (default on since fa8/fa9 acceptance; PF_FA=0 -> scalar pf_attn) for prompts >= PF_FA_M tokens
        let (name, groups) = if name == "attn" && env("PF_FA", 1) == 1 && push[8] >= env("PF_FA_M", 1) { ("attnfa", push[8].div_ceil(64) * push[0]) }
            else if name == "attn64" && env("PF_FA64", 0) == 1 { ("attnfa64", push[7].div_ceil(64) * push[0]) } else { (name, groups) };
        let k = &self.k[name]; let ds = self.ds.get(ctx, k, bufs)?; ctx.dispatch(cb, k, ds, &push_u32(push), groups); Ok(())
    }
}
