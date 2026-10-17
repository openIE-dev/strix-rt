//! One big float arena for every decoder tensor. CPU mode: host Vec. GPU mode: a single host-visible Vulkan buffer
//! (UMA), so CPU fallback ops read/write GPU tensors in place after a flush. Frees of storage touched by the
//! pending (unsubmitted) GPU batch are deferred until that batch completes.
use anyhow::{anyhow, Result};
use ash::vk;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use vkrt::{push_u32, Buffer, Ctx, Kernel};

pub struct Store { pub off: usize, pub len: usize, pub touched: Cell<u64> }
impl Drop for Store { fn drop(&mut self) { free(self.off, self.len, self.touched.get()) } }

struct Alloc { free: BTreeMap<usize, usize>, pending: Vec<(usize, usize)>, peak: usize, used: usize }
pub struct Gpu {
    pub ctx: Ctx, pub buf: Buffer, pool: vk::CommandPool, cb: Cell<vk::CommandBuffer>, fence: vk::Fence, open: Cell<bool>,
    pub ew: Kernel, pub gemm: Kernel, pub sdpa: Kernel, pub rownorm: Kernel, pub gemm_cm: Kernel, pub sdpa_cm: Kernel, pub misc: Kernel, ds: [vk::DescriptorSet; 7], pub n_disp: Cell<usize>, pub n_flush: Cell<usize>, pub t_wait: Cell<f64>, qp: vk::QueryPool,
}
thread_local! {
    static BASE: Cell<(*mut f32, usize)> = const { Cell::new((std::ptr::null_mut(), 0)) };
    static AL: RefCell<Option<Alloc>> = const { RefCell::new(None) };
    static BATCH: Cell<u64> = const { Cell::new(1) };
    static PENDING: Cell<bool> = const { Cell::new(false) };
    static HOSTVEC: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    pub static GPU: RefCell<Option<&'static Gpu>> = const { RefCell::new(None) };
    pub static LABEL: RefCell<String> = const { RefCell::new(String::new()) };
    static QLAB: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    pub static GPROF: RefCell<std::collections::HashMap<String, (f64, usize)>> = RefCell::new(Default::default());
}
const ALIGN: usize = 64;
pub fn init(cap_floats: usize, gpu: bool) -> Result<()> {
    if gpu {
        let ctx = Ctx::new()?; let buf = ctx.buffer((cap_floats * 4) as u64)?;
        let d = &ctx.device;
        let pool = unsafe { d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qfam).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)? };
        let cb = unsafe { d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0] };
        let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let qp = unsafe { d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(16384), None)? };
        macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
        let ew = ctx.kernel(spv!("ew"), 1, 128, &[], None)?; let gemm = ctx.kernel(spv!("gemm"), 1, 120, &[], None)?;
        let sdpa = ctx.kernel(spv!("sdpa"), 1, 120, &[], None)?; let rownorm = ctx.kernel(spv!("rownorm"), 1, 56, &[], None)?; let gemm_cm = ctx.kernel(spv!("gemm_cm"), 1, 136, &[], Some(32))?; let sdpa_cm = ctx.kernel(spv!("sdpa_cm"), 1, 120, &[], Some(32))?; let misc = ctx.kernel(spv!("misc"), 1, 96, &[], None)?;
        let ds = [ctx.bind(&ew, &[&buf])?, ctx.bind(&gemm, &[&buf])?, ctx.bind(&sdpa, &[&buf])?, ctx.bind(&rownorm, &[&buf])?, ctx.bind(&gemm_cm, &[&buf])?, ctx.bind(&sdpa_cm, &[&buf])?, ctx.bind(&misc, &[&buf])?];
        BASE.with(|b| b.set((buf.ptr as *mut f32, cap_floats)));
        let g = Box::leak(Box::new(Gpu { ctx, buf, pool, cb: Cell::new(cb), fence, open: Cell::new(false), ew, gemm, sdpa, rownorm, gemm_cm, sdpa_cm, misc, ds, n_disp: Cell::new(0), n_flush: Cell::new(0), t_wait: Cell::new(0.0), qp }));
        GPU.with(|x| *x.borrow_mut() = Some(g));
    } else {
        HOSTVEC.with(|v| { let mut v = v.borrow_mut(); *v = vec![0f32; cap_floats]; BASE.with(|b| b.set((v.as_mut_ptr(), cap_floats))); });
    }
    let mut free = BTreeMap::new(); free.insert(0, cap_floats);
    AL.with(|a| *a.borrow_mut() = Some(Alloc { free, pending: vec![], peak: 0, used: 0 }));
    Ok(())
}
pub fn gpu() -> Option<&'static Gpu> { GPU.with(|g| *g.borrow()) }
pub fn alloc(n: usize) -> Store {
    let len = n.max(1).div_ceil(ALIGN) * ALIGN;
    let try_take = || AL.with(|a| { let mut a = a.borrow_mut(); let a = a.as_mut().expect("arena not initialised");
        let (&o, &l) = a.free.iter().find(|(_, &l)| l >= len)?;
        a.free.remove(&o); if l > len { a.free.insert(o + len, l - len); } a.used += len; a.peak = a.peak.max(a.used); Some(o) });
    // frees of buffers used by the open command buffer are deferred to the next flush; on exhaustion flush and retry
    let off = match try_take() { Some(o) => o, None => { flush().expect("flush"); try_take().unwrap_or_else(|| panic!("arena out of memory ({} floats requested, {} used)", len, AL.with(|a| a.borrow().as_ref().unwrap().used))) } };
    Store { off, len, touched: Cell::new(0) }
}
fn insert_free(a: &mut Alloc, off: usize, len: usize) {
    let (mut o, mut l) = (off, len);
    if let Some((&po, &pl)) = a.free.range(..o).next_back() { if po + pl == o { a.free.remove(&po); o = po; l += pl; } }
    if let Some(&nl) = a.free.get(&(o + l)) { a.free.remove(&(o + l)); l += nl; }
    a.free.insert(o, l);
}
fn free(off: usize, len: usize, touched: u64) {
    let defer = PENDING.with(|p| p.get()) && touched == BATCH.with(|b| b.get());
    let _ = AL.try_with(|a| { if let Ok(mut a) = a.try_borrow_mut() { if let Some(a) = a.as_mut() { a.used -= len; if defer { a.pending.push((off, len)) } else { insert_free(a, off, len) } } } });
}
pub fn peak_mb() -> f64 { AL.with(|a| a.borrow().as_ref().map_or(0, |a| a.peak)) as f64 * 4.0 / 1e6 }
/// CPU access to a store: flush first if the pending GPU batch touched it.
pub fn cpu_slice(s: &Store) -> &'static mut [f32] {
    if PENDING.with(|p| p.get()) && s.touched.get() == BATCH.with(|b| b.get()) { flush().expect("gpu flush") }
    let (p, cap) = BASE.with(|b| b.get()); assert!(s.off + s.len <= cap);
    unsafe { std::slice::from_raw_parts_mut(p.add(s.off), s.len) }
}
pub fn touch(s: &Store) { s.touched.set(BATCH.with(|b| b.get())) }
pub fn flush() -> Result<()> {
    let g = match gpu() { Some(g) => g, None => return Ok(()) };
    if !g.open.get() { return Ok(()) }
    let d = &g.ctx.device; let cb = g.cb.get();
    unsafe {
        d.end_command_buffer(cb)?;
        let t0 = std::time::Instant::now();
        d.reset_fences(&[g.fence])?;
        d.queue_submit(g.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cb])], g.fence)?;
        d.wait_for_fences(&[g.fence], true, u64::MAX).map_err(|e| anyhow!("gpu wait: {e:?}"))?;
        d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
        g.t_wait.set(g.t_wait.get() + t0.elapsed().as_secs_f64());
    }
    let labs = QLAB.with(|q| std::mem::take(&mut *q.borrow_mut()));
    if !labs.is_empty() { let mut ts = vec![0u64; labs.len() + 1];
        unsafe { d.get_query_pool_results(g.qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
        GPROF.with(|p| { let mut p = p.borrow_mut(); for (i, l) in labs.iter().enumerate() { let e = p.entry(l.clone()).or_default(); e.0 += (ts[i + 1] - ts[i]) as f64 * g.ctx.info.ts_period_ns as f64 * 1e-9; e.1 += 1; } }); }
    g.open.set(false); g.n_flush.set(g.n_flush.get() + 1);
    PENDING.with(|p| p.set(false)); BATCH.with(|b| b.set(b.get() + 1));
    AL.with(|a| { let mut a = a.borrow_mut(); let a = a.as_mut().unwrap(); let p = std::mem::take(&mut a.pending); for (o, l) in p { insert_free(a, o, l) } });
    Ok(())
}
/// record one dispatch (kernel index 0 ew, 1 gemm, 2 sdpa, 3 rownorm) followed by a full compute barrier
pub fn dispatch(ki: usize, push: &[u32], x: u32, y: u32, z: u32) {
    let g = gpu().unwrap(); let d = &g.ctx.device; let cb = g.cb.get();
    let gp = std::env::var("DEC_GPROF").is_ok();
    if !g.open.get() { unsafe { d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        if gp { d.cmd_reset_query_pool(cb, g.qp, 0, 16384); d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, g.qp, 0); } }; g.open.set(true); }
    let k = [&g.ew, &g.gemm, &g.sdpa, &g.rownorm, &g.gemm_cm, &g.sdpa_cm, &g.misc][ki];
    g.ctx.dispatch3(cb, k, g.ds[ki], &push_u32(push), x, y, z); g.ctx.barrier(cb);
    if gp { QLAB.with(|q| { let mut q = q.borrow_mut(); if q.len() < 16383 { q.push(LABEL.with(|l| l.borrow().clone())); unsafe { d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, g.qp, q.len() as u32) } } }); }
    PENDING.with(|p| p.set(true)); g.n_disp.set(g.n_disp.get() + 1);
}
