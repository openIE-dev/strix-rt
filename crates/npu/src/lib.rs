//! AMD XDNA2 NPU (amdxdna, RyzenAI-npu5) from Rust via a thin extern "C" shim over XRT (libxrt_coreutil).
//! See docs/npu.md.
use anyhow::{bail, Context, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};

extern "C" {
    fn npu_last_error() -> *const c_char;
    fn npu_open(dev: u32, xclbin: *const c_char, kname: *const c_char, instr: *const u32, n: usize) -> *mut c_void;
    fn npu_close(c: *mut c_void);
    fn npu_bo_alloc(c: *mut c_void, sz: usize, arg: c_int) -> *mut c_void;
    fn npu_bo_userptr(c: *mut c_void, p: *mut c_void, sz: usize, arg: c_int) -> *mut c_void;
    fn npu_bo_import_fd(c: *mut c_void, fd: c_int) -> *mut c_void;
    fn npu_bo_export_fd(b: *mut c_void) -> c_int;
    fn npu_bo_map(b: *mut c_void) -> *mut c_void;
    fn npu_bo_size(b: *mut c_void) -> usize;
    fn npu_bo_address(b: *mut c_void) -> u64;
    fn npu_bo_sync(b: *mut c_void, to_dev: c_int) -> c_int;
    fn npu_bo_free(b: *mut c_void);
    fn npu_run(c: *mut c_void, a: *mut c_void, b: *mut c_void, o: *mut c_void) -> c_int;
}
fn last_err() -> String { unsafe { CStr::from_ptr(npu_last_error()).to_string_lossy().into_owned() } }

/// A loaded xclbin + instruction stream on a hardware context (one kernel, IRON GEMM argument convention:
/// opcode, instr, ninstr, A, B, C, tmp, trace).
pub struct Npu { h: *mut c_void }
unsafe impl Send for Npu {}
impl Drop for Npu { fn drop(&mut self) { unsafe { npu_close(self.h) } } }

/// An XRT buffer object.
pub struct Bo { h: *mut c_void, pub size: usize, pub ptr: *mut u8 }
unsafe impl Send for Bo {}
impl Drop for Bo { fn drop(&mut self) { unsafe { npu_bo_free(self.h) } } }
impl Bo {
    fn wrap(h: *mut c_void, what: &str) -> Result<Bo> {
        if h.is_null() { bail!("{what}: {}", last_err()) }
        let size = unsafe { npu_bo_size(h) };
        let ptr = unsafe { npu_bo_map(h) } as *mut u8;
        if ptr.is_null() { bail!("{what}: map: {}", last_err()) }
        Ok(Bo { h, size, ptr })
    }
    pub fn bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr, self.size) } }
    pub fn bytes_mut(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) } }
    pub fn to_device(&self) -> Result<()> { if unsafe { npu_bo_sync(self.h, 1) } != 0 { bail!("sync to device: {}", last_err()) } Ok(()) }
    pub fn from_device(&self) -> Result<()> { if unsafe { npu_bo_sync(self.h, 0) } != 0 { bail!("sync from device: {}", last_err()) } Ok(()) }
    /// Export as a dma-buf fd (caller owns the fd).
    pub fn export_dmabuf(&self) -> Result<i32> { let fd = unsafe { npu_bo_export_fd(self.h) }; if fd < 0 { bail!("export: {}", last_err()) } Ok(fd) }
    pub fn device_address(&self) -> u64 { unsafe { npu_bo_address(self.h) } }
}

pub fn load_instr(path: &str) -> Result<Vec<u32>> {
    let b = std::fs::read(path).with_context(|| format!("read {path}"))?;
    if b.len() % 4 != 0 { bail!("{path}: length {} not a multiple of 4", b.len()) }
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

impl Npu {
    pub fn open(xclbin: &str, instr: &[u32], kernel: &str) -> Result<Npu> {
        let x = CString::new(xclbin)?; let k = CString::new(kernel)?;
        let h = unsafe { npu_open(0, x.as_ptr(), k.as_ptr(), instr.as_ptr(), instr.len()) };
        if h.is_null() { bail!("npu_open({xclbin}): {}", last_err()) }
        Ok(Npu { h })
    }
    /// Host-only BO bound to kernel argument `arg` (3 = A, 4 = B, 5 = C).
    pub fn alloc(&self, size: usize, arg: i32) -> Result<Bo> { Bo::wrap(unsafe { npu_bo_alloc(self.h, size, arg) }, "bo alloc") }
    /// Wrap existing user memory (page aligned) as a BO.
    pub unsafe fn userptr(&self, p: *mut u8, size: usize, arg: i32) -> Result<Bo> { Bo::wrap(npu_bo_userptr(self.h, p as *mut c_void, size, arg), "bo userptr") }
    /// Import a dma-buf fd as a BO.
    pub fn import_dmabuf(&self, fd: i32) -> Result<Bo> { Bo::wrap(unsafe { npu_bo_import_fd(self.h, fd) }, "bo import dma-buf") }
    /// Run once and wait. Errors unless the command reaches ERT_CMD_STATE_COMPLETED (4).
    pub fn run(&self, a: &Bo, b: &Bo, c: &Bo) -> Result<()> {
        let r = unsafe { npu_run(self.h, a.h, b.h, c.h) };
        if r != 4 { bail!("npu run state {r}: {}", last_err()) }
        Ok(())
    }
}

/// int8 GEMM C[M,N] = A[M,K] * B[K,N] on the whole-array dc2 design (wa_dc). A and B are uploaded pre-tiled
/// (A as [M/m][K/k][m][k], B as [K/k][N/n][k][n]) so every DDR read is contiguous; C comes back row-major.
pub struct Gemm { pub npu: Npu, pub mm: usize, pub kk: usize, pub nn: usize, pub m: usize, pub k: usize, pub n: usize,
    pub out_bytes: usize, pub a: Bo, pub b: Bo, pub c: Bo }

impl Gemm {
    pub fn new(build_dir: &str, mkn: (usize, usize, usize), tile: (usize, usize, usize), out_bytes: usize) -> Result<Gemm> {
        Self::new_x(build_dir, &format!("{build_dir}/final.xclbin"), mkn, tile, out_bytes)
    }
    /// As `new`, with the instruction stream from `build_dir` run on `xclbin` (another build of the same core code,
    /// e.g. the same M and K; opens of one xclbin share a hardware context, see xrt_shim.cpp).
    pub fn new_x(build_dir: &str, xclbin: &str, (mm, kk, nn): (usize, usize, usize), (m, k, n): (usize, usize, usize), out_bytes: usize) -> Result<Gemm> {
        if mm % m != 0 || kk % k != 0 || nn % n != 0 { bail!("problem {mm}x{kk}x{nn} not a multiple of tile {m}x{k}x{n}") }
        let instr = load_instr(&format!("{build_dir}/insts.txt"))?;
        let npu = Npu::open(xclbin, &instr, "MLIR_AIE")?;
        let a = npu.alloc(mm * kk, 3)?; let b = npu.alloc(kk * nn, 4)?; let c = npu.alloc(mm * nn * out_bytes, 5)?;
        Ok(Gemm { npu, mm, kk, nn, m, k, n, out_bytes, a, b, c })
    }
    /// Pack row-major A [M,K] into the A BO and sync.
    pub fn set_a(&mut self, a: &[i8]) -> Result<()> {
        let (mm, kk, m, k) = (self.mm, self.kk, self.m, self.k);
        assert_eq!(a.len(), mm * kk);
        let dst = self.a.bytes_mut(); let mut o = 0;
        for mb in 0..mm / m { for kb in 0..kk / k { for i in 0..m {
            let s = (mb * m + i) * kk + kb * k;
            for j in 0..k { dst[o + j] = a[s + j] as u8; } o += k;
        } } }
        self.a.to_device()
    }
    /// Pack row-major B [K,N] into the B BO and sync.
    pub fn set_b(&mut self, b: &[i8]) -> Result<()> {
        let (kk, nn, k, n) = (self.kk, self.nn, self.k, self.n);
        assert_eq!(b.len(), kk * nn);
        let dst = self.b.bytes_mut(); let mut o = 0;
        for kb in 0..kk / k { for nb in 0..nn / n { for i in 0..k {
            let s = (kb * k + i) * nn + nb * n;
            for j in 0..n { dst[o + j] = b[s + j] as u8; } o += n;
        } } }
        self.b.to_device()
    }
    /// Run on the already-synced A/B BOs (no C sync).
    pub fn run(&self) -> Result<()> { self.npu.run(&self.a, &self.b, &self.c) }
    pub fn c_bytes(&self) -> Result<&[u8]> { self.c.from_device()?; Ok(self.c.bytes()) }
    pub fn c_i32(&self) -> Result<&[i32]> {
        assert_eq!(self.out_bytes, 4);
        self.c.from_device()?;
        Ok(unsafe { std::slice::from_raw_parts(self.c.ptr as *const i32, self.mm * self.nn) })
    }
    pub fn ops(&self) -> f64 { 2.0 * self.mm as f64 * self.kk as f64 * self.nn as f64 }
}

/// Exact int32 dot for C[r][c] from row-major A/B (reference for sampled verification).
pub fn ref_dot(a: &[i8], b: &[i8], kk: usize, nn: usize, r: usize, c: usize) -> i32 {
    (0..kk).map(|k| a[r * kk + k] as i32 * b[k * nn + c] as i32).sum()
}

/// Symmetric per-token (per-row) int8 quantization: q = round(x / s), s = max|x_row| / 127.
pub fn quantize_rows(x: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols]; let mut s = vec![0f32; rows];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 }; s[r] = sc;
        for (d, v) in q[r * cols..(r + 1) * cols].iter_mut().zip(row) { *d = (v / sc).round().clamp(-127.0, 127.0) as i8; }
    }
    (q, s)
}

/// Per-output-channel (per-column) int8 quantization of a weight matrix W [K,N].
pub fn quantize_cols(w: &[f32], kk: usize, nn: usize) -> (Vec<i8>, Vec<f32>) {
    let mut s = vec![0f32; nn];
    for k in 0..kk { for c in 0..nn { s[c] = s[c].max(w[k * nn + c].abs()); } }
    for v in s.iter_mut() { *v = if *v > 0.0 { *v / 127.0 } else { 1.0 }; }
    let q = (0..kk * nn).map(|i| (w[i] / s[i % nn]).round().clamp(-127.0, 127.0) as i8).collect();
    (q, s)
}

fn nthreads() -> usize { std::env::var("NPU_HOST_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)) }

/// Batched prompt-processing linear layer on the NPU: Y[M,N] = X[M,K] (f32, quantized per token) x Wq[K,N]
/// (int8 with per-column scales, already uploaded via `set_b`). Needs an int32-output build.
/// Host work (per-token quantize straight into the tiled A BO, dequantize) is split across threads.
pub fn linear_f32(g: &mut Gemm, x: &[f32], w_scale: &[f32]) -> Result<Vec<f32>> {
    let (mm, kk, nn, m, k) = (g.mm, g.kk, g.nn, g.m, g.k);
    assert_eq!(x.len(), mm * kk);
    let nt = nthreads().min(mm / m).max(1);
    let mut sx = vec![0f32; mm];
    // quantize row r and write it into the tiled layout [M/m][K/k][m][k]; each row-block mb is a disjoint slab
    let a_base = g.a.ptr as usize;
    std::thread::scope(|sc| {
        for (t, sxc) in sx.chunks_mut(((mm / m + nt - 1) / nt) * m).enumerate() {
            let r0 = t * ((mm / m + nt - 1) / nt) * m;
            sc.spawn(move || {
                let dst = a_base as *mut i8;
                for (ri, sv) in sxc.iter_mut().enumerate() {
                    let r = r0 + ri; let row = &x[r * kk..(r + 1) * kk];
                    let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 }; *sv = sc;
                    let (mb, i) = (r / m, r % m);
                    for kb in 0..kk / k {
                        let o = ((mb * (kk / k) + kb) * m + i) * k;
                        for j in 0..k { unsafe { *dst.add(o + j) = (row[kb * k + j] / sc).round().clamp(-127.0, 127.0) as i8; } }
                    }
                }
            });
        }
    });
    g.a.to_device()?; g.run()?;
    let acc = g.c_i32()?;
    let mut y = vec![0f32; mm * nn];
    let rows_per = (mm + nt - 1) / nt;
    std::thread::scope(|sc| {
        for (t, yc) in y.chunks_mut(rows_per * nn).enumerate() {
            let (acc, sx) = (&acc, &sx);
            sc.spawn(move || {
                for (i, v) in yc.iter_mut().enumerate() { let r = t * rows_per + i / nn; let c = i % nn; *v = acc[r * nn + c] as f32 * sx[r] * w_scale[c]; }
            });
        }
    });
    Ok(y)
}

/// int8 activations x int4 (ternary-capable) weights with one f32 scale per (128-wide K group, output column),
/// f32 output, on the wa_q4 design (kernel `aie_kernels/aie2p/mm_q4.cc`, k = group = 128).
/// B tile layout (host packed, contiguous per tile, tiles ordered [K/k][N/n]):
///   int4 blocks [k/16][n/16][16][16] (two per byte, low nibble = even column), then f32 scale[n].
pub struct GemmQ4 { pub npu: Npu, pub mm: usize, pub kk: usize, pub nn: usize, pub m: usize, pub k: usize, pub n: usize,
    pub a: Bo, pub b: Bo, pub c: Bo, pub qint: bool }

impl GemmQ4 {
    pub fn tile_bytes(k: usize, n: usize, qint: bool) -> usize { k * n / 2 + if qint { 2 } else { 4 } * n }
    /// `qint` selects the integer-epilogue build (-DQ4_INT): int16 fixed-point group scales, int32 C.
    pub fn new(build_dir: &str, (mm, kk, nn): (usize, usize, usize), (m, k, n): (usize, usize, usize), qint: bool) -> Result<GemmQ4> {
        if mm % (4 * m) != 0 || kk % k != 0 || nn % (8 * n) != 0 { bail!("bad tiling") }
        let instr = load_instr(&format!("{build_dir}/insts.txt"))?;
        let npu = Npu::open(&format!("{build_dir}/final.xclbin"), &instr, "MLIR_AIE")?;
        let a = npu.alloc(mm * kk, 3)?;
        let b = npu.alloc((kk / k) * (nn / n) * Self::tile_bytes(k, n, qint), 4)?;
        let c = npu.alloc(mm * nn * 4, 5)?;
        Ok(GemmQ4 { npu, mm, kk, nn, m, k, n, a, b, c, qint })
    }
    pub fn set_a(&mut self, a: &[i8]) -> Result<()> {
        let (mm, kk, m, k) = (self.mm, self.kk, self.m, self.k);
        let dst = self.a.bytes_mut(); let mut o = 0;
        for mb in 0..mm / m { for kb in 0..kk / k { for i in 0..m {
            let s = (mb * m + i) * kk + kb * k; for j in 0..k { dst[o + j] = a[s + j] as u8; } o += k; } } }
        self.a.to_device()
    }
    /// `w` row-major [K, N] with values in -8..=7 (ternary uses -1..=1); `scale` row-major [K/k, N].
    pub fn set_b(&mut self, w: &[i8], scale: &[f32]) -> Result<()> {
        let (kk, nn, k, n) = (self.kk, self.nn, self.k, self.n);
        let tb = Self::tile_bytes(k, n, self.qint);
        let dst = self.b.bytes_mut();
        for kb in 0..kk / k { for nb in 0..nn / n {
            let base = (kb * (nn / n) + nb) * tb; let mut o = base;
            let pack = std::env::var("Q4PACK").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
            for bi in 0..k / 16 { for bj in 0..n / 16 {
                // element (i = k within block, j = n within block) -> nibble index within the 256-element block
                let mut blk = [0u8; 128];
                for i in 0..16 { for j in 0..16 {
                    let v = (w[(kb * k + bi * 16 + i) * nn + nb * n + bj * 16 + j] as u8) & 0xF;
                    let e = match pack { 1 => j * 16 + i, _ => i * 16 + j };
                    blk[e / 2] |= if e % 2 == 0 { v } else { v << 4 };
                } }
                dst[o..o + 128].copy_from_slice(&blk); o += 128;
            } }
            if self.qint { for j in 0..n { dst[o..o + 2].copy_from_slice(&(scale[kb * nn + nb * n + j].round() as i16).to_le_bytes()); o += 2; } }
            else { for j in 0..n { dst[o..o + 4].copy_from_slice(&scale[kb * nn + nb * n + j].to_le_bytes()); o += 4; } }
            debug_assert_eq!(o, base + tb);
        } }
        self.b.to_device()
    }
    pub fn run(&self) -> Result<()> { self.npu.run(&self.a, &self.b, &self.c) }
    /// Integer-epilogue helper: per-column S = max_g scale, q = round(scale / S * 32767) (pass q as `scale` to set_b).
    pub fn int_scales(scale: &[f32], groups: usize, nn: usize) -> (Vec<f32>, Vec<f32>) {
        let mut sc = vec![0f32; nn];
        for g in 0..groups { for c in 0..nn { sc[c] = sc[c].max(scale[g * nn + c].abs()); } }
        let q = (0..groups * nn).map(|i| (scale[i] / sc[i % nn] * 32767.0).round()).collect();
        (q, sc.iter().map(|s| s * 32768.0 / 32767.0).collect())
    }
    pub fn c_i32(&self) -> Result<&[i32]> { self.c.from_device()?; Ok(unsafe { std::slice::from_raw_parts(self.c.ptr as *const i32, self.mm * self.nn) }) }
    pub fn c_f32(&self) -> Result<&[f32]> { self.c.from_device()?; Ok(unsafe { std::slice::from_raw_parts(self.c.ptr as *const f32, self.mm * self.nn) }) }
    pub fn ops(&self) -> f64 { 2.0 * self.mm as f64 * self.kk as f64 * self.nn as f64 }
}

/// Vulkan side of NPU sharing: import an NPU BO's dma-buf as a storage buffer, plus the compiled shaders.
pub mod vkshare {
    use anyhow::{anyhow, Result};
    use ash::vk;
    use vkrt::{Buffer, Ctx};
    pub static QUANT_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_tiled.spv"));
    pub static DEQUANT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant.spv"));
    /// Import a dma-buf fd (ownership passes to Vulkan) as a STORAGE|TRANSFER_SRC|DST buffer.
    pub unsafe fn import(ctx: &Ctx, fd: i32, size: u64) -> Result<Buffer> {
        let d = &ctx.device; let ht = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let efd = ash::khr::external_memory_fd::Device::new(&ctx.instance, d);
        let mut fp = vk::MemoryFdPropertiesKHR::default();
        (efd.fp().get_memory_fd_properties_khr)(d.handle(), ht, fd, &mut fp).result().map_err(|e| anyhow!("fd props: {e:?}"))?;
        let mut emb = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
        let buf = d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).push_next(&mut emb), None)?;
        let req = d.get_buffer_memory_requirements(buf);
        let bits = req.memory_type_bits & fp.memory_type_bits;
        let idx = (0..ctx.mem_props.memory_type_count).find(|i| bits & (1 << i) != 0).ok_or_else(|| anyhow!("no memory type"))?;
        let mut imp = vk::ImportMemoryFdInfoKHR::default().handle_type(ht).fd(fd);
        let mut ded = vk::MemoryDedicatedAllocateInfo::default().buffer(buf);
        let mem = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size.max(size)).memory_type_index(idx).push_next(&mut imp).push_next(&mut ded), None).map_err(|e| anyhow!("allocate_memory(import): {e:?}"))?;
        d.bind_buffer_memory(buf, mem, 0)?;
        Ok(Buffer { buf, mem, size, ptr: std::ptr::null_mut(), flags: ctx.mem_props.memory_types[idx as usize].property_flags })
    }
}
pub mod split;
