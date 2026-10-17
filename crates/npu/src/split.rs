//! NPU + GPU split linear for prefill: `Y[M][N] = X[M][K] * W^T` for a ternary PQ2_0 weight `W [N][K]`, with output
//! columns `[0, nn)` on the NPU and `[nn, N)` on the GPU, running at the same time and writing one shared output buffer.
//!
//! Pipeline (all on device; the NPU's A and C buffers are dma-bufs imported into Vulkan):
//!  1. GPU `rot_quant` phase 0 (per token x 1024-block): `xr = H_1024(s ⊙ x)` (Bonsai 2 basis; optional), x f32 or f16;
//!     writes the GPU half's per-128-group int8 directly (tern_quantx format) and xr in f16.
//!  2. `rot_quant` phase 1 (per token): per-token int8 A for the NPU (tiled), from the f16 xr.
//!  3. Concurrently: NPU int8 GEMM (weights folded to int8 per output column) and GPU `tern_gemm_split`
//!     (crates/ternary's coop-matrix PQ2_0 GEMM, writing Y[:, nn..N]).
//!  4. GPU `dequant_split`: Y[:, 0..nn] = C * sx[r] * sw[c].
//!
//! NPU weight format, "fold": row n of the PQ2_0 weight has ternary codes t and one f16 scale d_g per 128-group.
//! Stored as int8 `round(t * d_g / S_n)` with `S_n = max_g |d_g| / 127` (the per-column scale used in dequant).
//! The NPU build must match (M, K, nn) and the 128x64x64 tile (`wa_dc_dc2_oi32_..._MxKxNN_128x64x64`, see `build_dir`).
//!
//! Usage: one `SplitLinear` per shape (M, K, N, nn) opens one NPU context and owns the activation/C buffers;
//! `load(w)` makes a `SplitWeight` per layer (NPU int8 B BO + per-column scales + GPU PQ2_0 rows), so a model needs
//! one NPU context per distinct shape, not per layer. `forward(ctx, &weight, x, ldx, y, ldy, y_col)` blocks until Y is
//! written (three queue submits; the NPU half runs on a scoped thread while the GPU half runs).
use crate::{vkshare, Gemm};
use anyhow::{ensure, Result};
use std::time::Instant;
use ash::vk;
use vkrt::{push_u32, Buffer, Ctx, Kernel};

pub static ROT_QUANT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rot_quant.spv"));
pub static DEQUANT_SPLIT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant_split.spv"));
pub static TERN_GEMM_SPLIT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tern_gemm_split.spv"));
pub static NPU_PREP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/npu_prep.spv"));
const PQ2_BLOCK: usize = 34;

struct SendGemm(Gemm);
unsafe impl Sync for SendGemm {} // the XRT handles are used from one thread at a time (run() only inside forward's scope)
struct SendBo(crate::Bo);
unsafe impl Sync for SendBo {}

/// NPU build directory for a split shape (wa_dc dc2, int32 out, 128x64x64), under `$NPU_BUILDS` or ~/npu/kernel/builds.
pub fn build_dir(m: usize, k: usize, nn: usize) -> String {
    let root = std::env::var("NPU_BUILDS").unwrap_or_else(|_| format!("{}/npu/kernel/builds", std::env::var("HOME").unwrap_or_default()));
    format!("{root}/wa_dc_dc2_oi32_r1_st800_l24_at_bt_d221_8c_{m}x{k}x{nn}_128x64x64")
}

/// xclbin shared by all split shapes with this M and K (the dc2 core code depends on K, not on N): the
/// lexicographically first existing `{m}x{k}x*` build. Context switches between xclbins cost ~2.2 ms per NPU run.
/// NPU_SHARE=0 keeps one xclbin per shape.
pub fn shared_xclbin(m: usize, k: usize, own: &str) -> String {
    let own_x = format!("{own}/final.xclbin");
    if std::env::var("NPU_SHARE").map(|v| v == "0").unwrap_or(false) { return own_x }
    let d = build_dir(m, k, 0); let (root, name) = d.rsplit_once('/').unwrap(); let pre = name.split_once(&format!("{m}x{k}x")).map(|(p, _)| format!("{p}{m}x{k}x")).unwrap();
    let mut c: Vec<String> = std::fs::read_dir(root).map(|r| r.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with(&pre) && n.ends_with("_128x64x64")).collect()).unwrap_or_default();
    c.sort(); c.into_iter().map(|n| format!("{root}/{n}/final.xclbin")).find(|p| std::path::Path::new(p).exists()).unwrap_or(own_x)
}

/// One layer's weight prepared for a `SplitLinear`.
pub struct SplitWeight { b: Option<SendBo>, sw: Option<Buffer>, wg: Option<Buffer>, pub fold_ms: f64 }

/// Fold PQ2_0 rows [row0, row0+nn) into int8 B [K][nn] (row-major, NPU input) and per-column scales.
pub fn fold_pq2(w: &[u8], k: usize, row0: usize, nn: usize) -> (Vec<i8>, Vec<f32>) {
    let rb = k / 128 * PQ2_BLOCK; let g = k / 128;
    let mut b = vec![0i8; k * nn]; let mut sc = vec![0f32; nn];
    let (mut c, mut d) = ([0i8; 128], [0f32; 1]);
    for j in 0..nn {
        let row = &w[(row0 + j) * rb..(row0 + j + 1) * rb];
        let ds: Vec<f32> = (0..g).map(|gi| ternary::f16(row, gi * PQ2_BLOCK)).collect();
        let s = ds.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0; let s = if s > 0.0 { s } else { 1.0 };
        sc[j] = s;
        for gi in 0..g {
            ternary::block_codes(ternary::Fmt::Pq2_0, &row[gi * PQ2_BLOCK..(gi + 1) * PQ2_BLOCK], &mut c, &mut d);
            let q = (d[0] / s).round().clamp(-127.0, 127.0) as i32;
            for e in 0..128 { b[(gi * 128 + e) * nn + j] = (c[e] as i32 * q).clamp(-127, 127) as i8; }
        }
    }
    (b, sc)
}

/// Parallel fold of rows [row0, row0+nn) straight into the NPU's tiled B layout [K/64][nn/64][64][64]; returns scales.
fn fold_pq2_tiled(w: &[u8], k: usize, row0: usize, nn: usize, dst: &mut [u8]) -> Vec<f32> {
    let rb = k / 128 * PQ2_BLOCK; let g = k / 128;
    let nt = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(32);
    let mut colm = vec![0i8; nn * k]; let mut sc = vec![0f32; nn];
    let cpt = nn.div_ceil(nt);
    std::thread::scope(|s| { for (ci, (cm, scc)) in colm.chunks_mut(cpt * k).zip(sc.chunks_mut(cpt)).enumerate() { s.spawn(move || {
        let (mut c, mut d) = ([0i8; 128], [0f32; 1]);
        for (jj, col) in cm.chunks_mut(k).enumerate() { let j = ci * cpt + jj; let row = &w[(row0 + j) * rb..(row0 + j + 1) * rb];
            let ds: Vec<f32> = (0..g).map(|gi| ternary::f16(row, gi * PQ2_BLOCK)).collect();
            let smax = ds.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0; let smax = if smax > 0.0 { smax } else { 1.0 }; scc[jj] = smax;
            for gi in 0..g { ternary::block_codes(ternary::Fmt::Pq2_0, &row[gi * PQ2_BLOCK..(gi + 1) * PQ2_BLOCK], &mut c, &mut d);
                let q = (d[0] / smax).round().clamp(-127.0, 127.0) as i32;
                for e in 0..128 { col[gi * 128 + e] = (c[e] as i32 * q).clamp(-127, 127) as i8; } } } }); } });
    let blk = nn * 64; let colm = &colm;
    std::thread::scope(|s| { for (kbc, chunk) in dst.chunks_mut(blk * (k / 64).div_ceil(nt)).enumerate() { s.spawn(move || {
        let kb0 = kbc * (k / 64).div_ceil(nt);
        for (bi, b) in chunk.chunks_mut(blk).enumerate() { let kb = kb0 + bi;
            for nb in 0..nn / 64 { for i in 0..64 { for jj in 0..64 { b[(nb * 64 + i) * 64 + jj] = colm[(nb * 64 + jj) * k + kb * 64 + i] as u8; } } } } }); } });
    sc
}

/// Per-iteration timings (seconds).
#[derive(Default, Clone, Copy, Debug)]
pub struct SplitTimes { pub pre: f64, pub gemm: f64, pub npu: f64, pub gpu: f64, pub post: f64, pub total: f64 }

pub struct SplitLinear {
    pub m: usize, pub k: usize, pub n: usize, pub nn: usize, pub hadamard: bool,
    npu: Option<SendGemm>, a_vk: Option<Buffer>, c_vk: Option<Buffer>, sx: Buffer,
    xr: Buffer, xq: Buffer, xs: Buffer, signs: Buffer, bmax: Buffer,
    k_rot: Kernel, k_gemm: Kernel, k_deq: Kernel, k_prep: Kernel, dsc: rtcore::DsCache,
}

impl SplitLinear {
    /// Shape (m tokens, k inputs, n outputs), `nn` output columns on the NPU (0 = GPU only; multiple of 512),
    /// `npu_dir` the matching NPU build (`build_dir(m, k, nn)`), `signs`: k-wide ±1 Hadamard signs (None = no rotation).
    pub fn new(ctx: &Ctx, m: usize, k: usize, n: usize, nn: usize, npu_dir: &str, signs: Option<&[f32]>) -> Result<Self> {
        ensure!(m % 128 == 0 && k % 1024 == 0 && n % 64 == 0 && nn % 512 == 0 && nn <= n, "split: shape {m}x{k}x{n} nn={nn} unsupported");
        let sx = ctx.buffer((m * 4) as u64)?;
        let (mut npu, mut a_vk, mut c_vk) = (None, None, None);
        if nn > 0 {
            let g = Gemm::new_x(npu_dir, &shared_xclbin(m, k, npu_dir), (m, k, nn), (128, 64, 64), 4)?;
            a_vk = Some(unsafe { vkshare::import(ctx, g.a.export_dmabuf()?, (m * k) as u64)? });
            c_vk = Some(unsafe { vkshare::import(ctx, g.c.export_dmabuf()?, (m * nn * 4) as u64)? });
            npu = Some(SendGemm(g));
        }
        let xr = ctx.buffer((m * k * 2) as u64)?; let xq = ctx.buffer((m * k) as u64)?; let xs = ctx.buffer((m * k / 128 * 4) as u64)?;
        let signs_b = ctx.buffer((k * 4) as u64)?; let bmax = ctx.buffer((m * k / 1024 * 4) as u64)?;
        if let Some(s) = signs { ensure!(s.len() == k, "signs width"); signs_b.as_f32_mut().copy_from_slice(s); }
        let k_rot = ctx.kernel(ROT_QUANT, 8, 32, &[], None)?;
        let k_gemm = ctx.kernel(TERN_GEMM_SPLIT, 4, 28, &[], Some(32))?;
        let k_deq = ctx.kernel(DEQUANT_SPLIT, 4, 20, &[], None)?; let k_prep = ctx.kernel(NPU_PREP, 3, 28, &[], None)?;
        Ok(SplitLinear { m, k, n, nn, hadamard: signs.is_some(), npu, a_vk, c_vk, sx, xr, xq, xs, signs: signs_b, bmax, k_rot, k_gemm, k_deq, k_prep, dsc: Default::default() })
    }

    /// Prepare one PQ2_0 weight `w` ([n][k], ggml layout): fold rows [0, nn) to int8 for the NPU, copy rows [nn, n) for the GPU.
    pub fn load(&self, ctx: &Ctx, w: &[u8]) -> Result<SplitWeight> {
        let (k, n, nn) = (self.k, self.n, self.nn); let rb = k / 128 * PQ2_BLOCK;
        ensure!(w.len() >= n * rb, "split: weight too short");
        let t = Instant::now();
        let (mut b, mut sw) = (None, None);
        if let Some(g) = &self.npu {
            let mut bo = g.0.npu.alloc(k * nn, 4)?;
            let sc = fold_pq2_tiled(w, k, 0, nn, bo.bytes_mut());
            bo.to_device()?;
            let swb = ctx.buffer((nn * 4) as u64)?; swb.as_f32_mut().copy_from_slice(&sc);
            b = Some(SendBo(bo)); sw = Some(swb);
        }
        let ng = n - nn;
        let wg = if ng > 0 { let bb = ctx.buffer((ng * rb) as u64)?; bb.bytes_mut()[..ng * rb].copy_from_slice(&w[nn * rb..n * rb]); Some(bb) } else { None };
        Ok(SplitWeight { b, sw, wg, fold_ms: t.elapsed().as_secs_f64() * 1e3 })
    }
    /// NPU rows only (the caller runs the GPU half itself, e.g. prefill's tgemm with row0 = nn).
    pub fn load_npu(&self, ctx: &Ctx, w: &[u8]) -> Result<SplitWeight> {
        let (k, nn) = (self.k, self.nn); ensure!(w.len() >= self.n * (k / 128 * PQ2_BLOCK), "split: weight too short");
        let g = self.npu.as_ref().ok_or_else(|| anyhow::anyhow!("load_npu: nn = 0"))?;
        let t = Instant::now(); let mut bo = g.0.npu.alloc(k * nn, 4)?;
        let sc = fold_pq2_tiled(w, k, 0, nn, bo.bytes_mut()); bo.to_device()?;
        let swb = ctx.buffer((nn * 4) as u64)?; swb.as_f32_mut().copy_from_slice(&sc);
        Ok(SplitWeight { b: Some(SendBo(bo)), sw: Some(swb), wg: None, fold_ms: t.elapsed().as_secs_f64() * 1e3 })
    }
    pub fn nn(&self) -> usize { self.nn }

    // ---- recorded (non-blocking) interface, for callers that own the command stream (crates/prefill) ----
    /// Record: NPU A (per-token int8, tiled) + sx from an already-rotated f32 activation `xr` [rows][ldx], rows < n_rows live.
    pub fn record_prep(&self, ctx: &Ctx, cb: vk::CommandBuffer, xr: &Buffer, ldx: usize, n_rows: usize) -> Result<()> {
        let a = self.a_vk.as_ref().ok_or_else(|| anyhow::anyhow!("record_prep: nn = 0"))?;
        let ds = self.dsc.get(ctx, &self.k_prep, &[xr, a, &self.sx])?;
        ctx.dispatch(cb, &self.k_prep, ds, &push_u32(&[self.k as u32, 128, 64, n_rows as u32, ldx as u32, 0, self.k as u32]), self.m as u32); Ok(())
    }
    /// K-chunked prep: A from columns [x0, x0 + kv) of `xr` (kv <= k, rest zero), own per-token scale.
    pub fn record_prep_at(&self, ctx: &Ctx, cb: vk::CommandBuffer, xr: &Buffer, ldx: usize, n_rows: usize, x0: usize, kv: usize) -> Result<()> {
        let a = self.a_vk.as_ref().ok_or_else(|| anyhow::anyhow!("record_prep: nn = 0"))?;
        let ds = self.dsc.get(ctx, &self.k_prep, &[xr, a, &self.sx])?;
        ctx.dispatch(cb, &self.k_prep, ds, &push_u32(&[self.k as u32, 128, 64, n_rows as u32, ldx as u32, x0 as u32, kv as u32]), self.m as u32); Ok(())
    }
    /// Run the NPU half (blocking, ~ms); A must have been written by a completed submit.
    pub fn npu_run(&self, w: &SplitWeight) -> Result<f64> {
        let (g, b) = (self.npu.as_ref().unwrap(), w.b.as_ref().unwrap()); let t = Instant::now(); g.0.npu.run(&g.0.a, &b.0, &g.0.c)?; Ok(t.elapsed().as_secs_f64())
    }
    /// Record: y[:, y_col..y_col+nn] (+)= C * sx * sw (the NPU half's output), after npu_run returned.
    pub fn record_post(&self, ctx: &Ctx, cb: vk::CommandBuffer, w: &SplitWeight, y: &Buffer, ldy: usize, y_col: usize, acc: bool) -> Result<()> {
        let (m, nn) = (self.m, self.nn);
        let ds = self.dsc.get(ctx, &self.k_deq, &[self.c_vk.as_ref().unwrap(), &self.sx, w.sw.as_ref().unwrap(), y])?;
        let total4 = (m * nn / 4) as u32;
        ctx.dispatch(cb, &self.k_deq, ds, &push_u32(&[(nn / 4) as u32, total4, (ldy / 4) as u32, (y_col / 4) as u32, acc as u32]), total4.div_ceil(256)); Ok(())
    }

    pub fn free(&self, ctx: &Ctx, w: SplitWeight) { for b in [w.sw, w.wg].into_iter().flatten() { ctx.free_buffer(b); } }

    /// One forward: `x` f32 [m][k] (row stride ldx floats) -> `y` f32 with row stride ldy, columns y_col..y_col+n
    /// (ldy and y_col multiples of 4). Blocks until y is written.
    pub fn forward(&self, ctx: &Ctx, w: &SplitWeight, x: &Buffer, ldx: usize, y: &Buffer, ldy: usize, y_col: usize) -> Result<SplitTimes> {
        self.forward_x(ctx, w, x, ldx, false, y, ldy, y_col, false)
    }
    /// Same with f16 input `x` (e.g. crates/prefill's normed activation buffer `a`, row stride ldx halves).
    pub fn forward_f16(&self, ctx: &Ctx, w: &SplitWeight, x: &Buffer, ldx: usize, y: &Buffer, ldy: usize, y_col: usize) -> Result<SplitTimes> {
        self.forward_x(ctx, w, x, ldx, true, y, ldy, y_col, false)
    }
    /// General form: `xf16` selects the input type, `acc` adds into y (y += W x) instead of overwriting.
    pub fn forward_ex(&self, ctx: &Ctx, w: &SplitWeight, x: &Buffer, ldx: usize, xf16: bool, y: &Buffer, ldy: usize, y_col: usize, acc: bool) -> Result<SplitTimes> {
        self.forward_x(ctx, w, x, ldx, xf16, y, ldy, y_col, acc)
    }
    fn forward_x(&self, ctx: &Ctx, w: &SplitWeight, x: &Buffer, ldx: usize, xf16: bool, y: &Buffer, ldy: usize, y_col: usize, acc: bool) -> Result<SplitTimes> {
        ensure!(ldy % 4 == 0 && y_col % 4 == 0 && y_col + self.n <= ldy, "split: ldy/y_col");
        let (m, k, n, nn) = (self.m, self.k, self.n, self.nn); let ng = n - nn;
        let t0 = Instant::now();
        let a_bind = self.a_vk.as_ref().unwrap_or(&self.xq);
        let ds_rot = self.dsc.get(ctx, &self.k_rot, &[x, &self.signs, &self.xr, a_bind, &self.sx, &self.bmax, &self.xq, &self.xs])?;
        let ph = |phase: u32| push_u32(&[k as u32, 128, 64, self.hadamard as u32, ldx as u32, phase, xf16 as u32, (ng > 0) as u32]);
        ctx.run(|cb| {
            ctx.dispatch(cb, &self.k_rot, ds_rot, &ph(0), (m * k / 1024) as u32);
            if nn > 0 { ctx.barrier(cb); ctx.dispatch(cb, &self.k_rot, ds_rot, &ph(1), m as u32); }
        })?;
        let t1 = Instant::now();
        let mut tn = 0.0; let mut tg = 0.0;
        let gpu = || -> Result<f64> {
            if ng == 0 { return Ok(0.0) }
            let ds = self.dsc.get(ctx, &self.k_gemm, &[w.wg.as_ref().unwrap(), &self.xq, &self.xs, y])?;
            let pc = push_u32(&[m as u32, ng as u32, k as u32, 0, ldy as u32, (y_col + nn) as u32, acc as u32]);
            let t = Instant::now(); ctx.run(|cb| ctx.dispatch(cb, &self.k_gemm, ds, &pc, ((m / 64) * (ng / 64)) as u32))?; Ok(t.elapsed().as_secs_f64())
        };
        std::thread::scope(|s| -> Result<()> {
            let bw = w.b.as_ref();
            let h = self.npu.as_ref().map(|g| s.spawn(move || -> Result<f64> { let t = Instant::now(); g.0.npu.run(&g.0.a, &bw.unwrap().0, &g.0.c)?; Ok(t.elapsed().as_secs_f64()) }));
            tg = gpu()?;
            if let Some(h) = h { tn = h.join().unwrap()?; }
            Ok(())
        })?;
        let t2 = Instant::now();
        if nn > 0 {
            let ds = self.dsc.get(ctx, &self.k_deq, &[self.c_vk.as_ref().unwrap(), &self.sx, w.sw.as_ref().unwrap(), y])?;
            let total4 = (m * nn / 4) as u32;
            ctx.run(|cb| ctx.dispatch(cb, &self.k_deq, ds, &push_u32(&[(nn / 4) as u32, total4, (ldy / 4) as u32, (y_col / 4) as u32, acc as u32]), total4.div_ceil(256)))?;
        }
        let t3 = Instant::now();
        Ok(SplitTimes { pre: (t1 - t0).as_secs_f64(), gemm: (t2 - t1).as_secs_f64(), npu: tn, gpu: tg, post: (t3 - t2).as_secs_f64(), total: (t3 - t0).as_secs_f64() })
    }
    pub fn ops(&self) -> f64 { 2.0 * (self.m * self.k * self.n) as f64 }
}

/// Non-blocking submits on the context's queue (own command pool): `submit` returns a fence, `wait` blocks on one,
/// `sync` waits for everything and frees the command buffers. Queue order + a leading barrier orders work across submits.
pub struct Stream { pool: vk::CommandPool, pending: std::cell::RefCell<Vec<(vk::CommandBuffer, vk::Fence)>> }
impl Stream {
    pub fn new(ctx: &Ctx) -> Result<Self> {
        let pool = unsafe { ctx.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qfam), None)? };
        Ok(Stream { pool, pending: Default::default() })
    }
    pub fn submit<F: FnOnce(vk::CommandBuffer) -> Result<()>>(&self, ctx: &Ctx, f: F) -> Result<vk::Fence> {
        let d = &ctx.device;
        unsafe {
            let cb = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1))?[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            ctx.barrier(cb);
            let r = f(cb);
            d.end_command_buffer(cb)?;
            r?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            d.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cb])], fence)?;
            self.pending.borrow_mut().push((cb, fence));
            Ok(fence)
        }
    }
    /// Allocate + begin a command buffer (leading barrier recorded); pair with `end` and `submit_cb`.
    pub fn begin(&self, ctx: &Ctx) -> Result<vk::CommandBuffer> {
        let d = &ctx.device;
        unsafe {
            let cb = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1))?[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            ctx.barrier(cb); Ok(cb)
        }
    }
    pub fn end(&self, ctx: &Ctx, cb: vk::CommandBuffer) -> Result<()> { unsafe { ctx.device.end_command_buffer(cb)?; } Ok(()) }
    pub fn submit_cb(&self, ctx: &Ctx, cb: vk::CommandBuffer) -> Result<vk::Fence> {
        let d = &ctx.device;
        unsafe {
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            d.queue_submit(ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cb])], fence)?;
            self.pending.borrow_mut().push((cb, fence)); Ok(fence)
        }
    }
    pub fn wait(&self, ctx: &Ctx, f: vk::Fence) -> Result<()> { unsafe { ctx.device.wait_for_fences(&[f], true, u64::MAX)?; } Ok(()) }
    pub fn sync(&self, ctx: &Ctx) -> Result<()> {
        let d = &ctx.device;
        for (cb, f) in self.pending.borrow_mut().drain(..) { unsafe { d.wait_for_fences(&[f], true, u64::MAX)?; d.destroy_fence(f, None); d.free_command_buffers(self.pool, &[cb]); } }
        Ok(())
    }
}
