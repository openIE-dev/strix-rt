//! Packed-LDS int8 GEMM (pf_qgemm, MMQ-style) for Q4_K weights, shared by all MoE prefill paths.
//! Activations are first quantized q8_1-style by `Pf::q8` (int8 + per-32 scale + scaled sum, transposed scales);
//! `Pf::qgemm` / `Pf::qgemm_grouped` then do one 256-deep k-step per Q4_K superblock with int8 coop-matrix MMAs.
//! Opt-in (PF_QMM=1 in the callers); not bitwise equal to the f16 path (activation quantization), so callers must
//! pass the teacher-forced agreement checks before any default change.
use crate::{env, spv, Pf};
use anyhow::Result;
use ash::vk;
use rtcore::{qtype, WRef};
use vkrt::{push_u32, Buffer, Ctx, Kernel};
const KQ8: u32 = 0x10_0000;
const KQG: u32 = 0x20_0000;
const KQH: u32 = 0x40_0000;
impl Pf {
    fn qk(&self, ctx: &Ctx, key: u32) -> Result<&Kernel> {
        if !self.gemm_k.borrow().contains_key(&key) {
            let k = if key == KQ8 { ctx.kernel(spv::Q8, 4, 16, &[], None)? } else if key & KQH != 0 { ctx.kernel(spv::QGEMMH, 6, 40, &[key & 0xf, (key >> 4) & 1, env("PF_QMM_PROBE", 0), (key >> 8) & 0xff, (key >> 16) & 3, (key >> 18) & 0xf, 2 - ((key >> 24) & 1), (key >> 25) & 1], Some(32))? } else { ctx.kernel(spv::QGEMM, 6, 40, &[key & 0xf, (key >> 4) & 1, env("PF_QMM_PROBE", 0)], Some(32))? };
            self.gemm_k.borrow_mut().insert(key, Box::new(k));
        }
        let p: *const Kernel = &**self.gemm_k.borrow().get(&key).unwrap();
        Ok(unsafe { &*p })
    }
    /// Quantize f16 rows a[0..m) (row stride lda halves, k columns) into xq (ldx x k int8), xs / xu ([k/32][ldx] f32).
    /// Rows [m, ldx) are zeroed. ldx must cover every row a qgemm tile may read (m rounded up + 64).
    pub fn q8(&self, ctx: &Ctx, cb: vk::CommandBuffer, a: &Buffer, lda: u32, m: u32, k: u32, ldx: u32, xq: &Buffer, xs: &Buffer, xu: &Buffer) -> Result<()> {
        anyhow::ensure!(k % 256 == 0 && lda % 8 == 0, "q8: k/lda alignment");
        let kk = self.qk(ctx, KQ8)?; let ds = self.ds.get(ctx, kk, &[a, xq, xs, xu])?;
        ctx.dispatch(cb, kk, ds, &push_u32(&[m, k, lda, ldx]), (ldx * (k / 32)).div_ceil(256));
        Ok(())
    }
    fn mi() -> u32 { env("PF_QMM_MI", 2).clamp(1, 2) }
    /// Plain GEMM y[m, y_col + n] = W[n] . a[m] for m < m_rows, from q8 activations.
    pub fn qgemm(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, w: WRef, m: u32, xq: &Buffer, xs: &Buffer, xu: &Buffer, ldx: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        anyhow::ensure!(qtype(w.ty)? == 0 && w.k % 256 == 0 && w.off % 16 == 0, "qgemm: Q4_K only, k%256, 16B offset");
        let mi = Self::mi(); let kk = self.qk(ctx, KQG | mi)?; let ds = self.ds.get(ctx, kk, &[arena, xq, xs, xu, &self.dummy, y])?;
        let push = push_u32(&[m, w.n, w.k, w.off as u32, ldx, ldy, y_col, 0, 0, 0]);
        ctx.dispatch(cb, kk, ds, &push, m.div_ceil(32 * mi) * w.n.div_ceil(128));
        Ok(())
    }
    /// Grouped MoE GEMM over the tile table of pf_moesel/moesort (tiles must be <= 32*MI rows: mrows of that table).
    pub fn qgemm_grouped(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, w: WRef, table: &Buffer, maxt: u32, xq: &Buffer, xs: &Buffer, xu: &Buffer, ldx: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        let eb = w.bytes / w.e.max(1) as u64;
        anyhow::ensure!(qtype(w.ty)? == 0 && w.k % 256 == 0 && w.off % 16 == 0 && eb % 16 == 0 && w.off + w.bytes <= u32::MAX as u64, "qgemm_grouped: Q4_K only / alignment");
        let mi = Self::mi(); let kk = self.qk(ctx, KQG | 0x10 | mi)?; let ds = self.ds.get(ctx, kk, &[arena, xq, xs, xu, table, y])?;
        let push = push_u32(&[0, w.n, w.k, w.off as u32, ldx, ldy, y_col, 0, maxt, eb as u32]);
        ctx.dispatch(cb, kk, ds, &push, maxt * w.n.div_ceil(128));
        Ok(())
    }
    /// Shared q8 scratch (xq, xs, xu) for up to ldx rows x k columns; grows on demand. Callers must not hold two at once.
    pub fn qbufs(&self, ctx: &Ctx, ldx: u32, k: u32) -> Result<[*const Buffer; 3]> {
        let need = ldx as u64 * k as u64;
        if self.qx.borrow().as_ref().map(|q| q.0 < need).unwrap_or(true) {
            if let Some((_, b)) = self.qx.borrow_mut().take() { for x in b { ctx.free_buffer(x); } }
            *self.qx.borrow_mut() = Some((need, [ctx.buffer(need)?, ctx.buffer(need / 8 + 256)?, ctx.buffer(need / 8 + 256)?]));
        }
        let q = self.qx.borrow(); let b = &q.as_ref().unwrap().1; Ok([&b[0], &b[1], &b[2]])
    }
    /// pf_qgemmh (packed Q4_K in LDS, f16 unpack in registers, f16 activations a with row stride lda halves): plain GEMM.
    pub fn qgemmh(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, w: WRef, m: u32, a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        let ty = Self::qh_ty(w)?; anyhow::ensure!(w.off % 16 == 0 && lda % 8 == 0, "qgemmh: alignment");
        let mi = Self::mi(); let (nj, kd) = Self::qh_cfg(ty); let wr = Self::qh_wr(); let kk = self.qk(ctx, KQH | ((2 - wr) << 24) | (kd << 18) | (nj << 16) | (ty << 8) | mi * (3 - wr))?; let ds = self.ds.get(ctx, kk, &[arena, a, &self.dummy, &self.dummy, &self.dummy, y])?;
        ctx.dispatch(cb, kk, ds, &push_u32(&[m, w.n, w.k, w.off as u32, lda, ldy, y_col, 0, 0, 0]), m.div_ceil(32 * mi) * w.n.div_ceil(64 * nj * (3 - wr)));
        Ok(())
    }
    /// pf_qgemmh grouped (MoE) over a tile table of qgemm_rows() rows.
    pub fn qgemmh_grouped(&self, ctx: &Ctx, cb: vk::CommandBuffer, arena: &Buffer, w: WRef, table: &Buffer, maxt: u32, a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32) -> Result<()> {
        let eb = w.bytes / w.e.max(1) as u64;
        let ty = Self::qh_ty(w)?; anyhow::ensure!(w.off % 16 == 0 && eb % 16 == 0 && lda % 8 == 0 && w.off + w.bytes <= u32::MAX as u64, "qgemmh_grouped: alignment");
        let mi = Self::mi(); let (nj, kd) = Self::qh_cfg(ty); let wr = Self::qh_wr(); let pers = env("PF_QH_PERS", 0); let kk = self.qk(ctx, KQH | ((pers.min(1)) << 25) | ((2 - wr) << 24) | (kd << 18) | (nj << 16) | (ty << 8) | 0x10 | mi * (3 - wr))?; let ds = self.ds.get(ctx, kk, &[arena, a, &self.dummy, &self.dummy, table, y])?;
        ctx.dispatch(cb, kk, ds, &push_u32(&[0, w.n, w.k, w.off as u32, lda, ldy, y_col, 0, maxt, eb as u32]), if pers > 0 { pers } else { maxt * w.n.div_ceil(64 * nj * (3 - wr)) });
        Ok(())
    }
    /// Weight formats pf_qgemmh takes: Q4_K (k % 256), Q8_0 / Q5_0 (k % 32; partial last 256-step masked).
    pub fn qh_ty(w: WRef) -> Result<u32> { let t = qtype(w.ty)?; anyhow::ensure!((t == 0 && w.k % 256 == 0) || ((t == 3 || t == 15) && w.k % 32 == 0), "qgemmh: format {t} k {}", w.k); Ok(t) }
    /// (NJ, KD) tile config of pf_qgemmh: BN = 64*NJ weight rows, k-step 32*KD. Q4_K: KD 8 (superblock). Q8_0/Q5_0:
    /// PF_QH_NJ / PF_QH_KD (defaults 2 / 8). Constraint KD >= 256/BN lanes per row (and even sub-blocks per lane for Q4_K).
    pub fn qh_cfg(ty: u32) -> (u32, u32) {
        let nj = env(if ty == 0 { "PF_QH4_NJ" } else { "PF_QH_NJ" }, 2).clamp(1, 2);
        let kd = if ty == 0 { 8 } else { match env("PF_QH_KD", 8) { 2 | 4 | 8 => env("PF_QH_KD", 8), _ => 8 } };
        let nj = if Self::qh_wr() == 1 { 1 } else { nj };   // WR=1 doubles BN: NJ=2 would need 69 KB LDS
        let lpr = 4 / nj; (nj, kd.max(lpr * if ty == 0 { 2 } else { 1 }))
    }
    /// Wave rows of pf_qgemmh (PF_QH_WR: 2 = default 2 x 4 waves, 1 = 1 x 8 waves, BN doubles, B unpacked once).
    pub fn qh_wr() -> u32 { env("PF_QH_WR", 2).clamp(1, 2) }
    pub fn qh_ok(w: WRef) -> bool { Self::qh_ty(w).is_ok() }
    /// Tile rows the grouped kernel expects in its table.
    pub fn qgemm_rows() -> u32 { 32 * Self::mi() }
}
