//! NPU_SPLIT=1: Bonsai 2 ternary projections run as NPU+GPU N-splits (crates/npu `npu::split`).
//! Eligible: attn_qkv, attn_gate, attn_q, attn_output, ffn_gate/up/down (PQ2_0, Hadamard-rotated). ssm_out (GDN
//! v-grouped permutation) and the small k/v/alpha/beta projections stay GPU-only.
//! No extra rotate/quant: the NPU A operand is built from prefill's own rotated f32 activation (b.xr) by one small
//! kernel recorded into the current command buffer; the GPU rows [nn, N) run through `Pf::tgemm` on prefill's xq/xs.
//! Recording cuts the command stream at each split: [.. prep] | [GPU half] | [dequant NPU half, ..]. Execution
//! submits a segment (fence) and the GPU half, waits for the segment, runs the NPU while the GPU half runs, then
//! submits the next segment.
use anyhow::{bail, Result};
use ash::vk;
use npu::split::{build_dir, SplitLinear, SplitWeight, Stream};
use std::{cell::{Cell, RefCell}, collections::HashMap};
use vkrt::{Buffer, Ctx};

/// NPU column share per (K, N) at chunk size m (rounds 8/9; FFN re-tuned whole-model in round 11). None = GPU only.
pub fn nn_for(k: u32, n: u32, m: u32) -> Option<u32> {
    // NPU_NN=KxN:nn,... overrides the NPU column share per shape (nn=0: GPU only); needs a {m}x{K}x{nn} build
    if let Ok(v) = std::env::var("NPU_NN") { for e in v.split(',') { if let Some((s, c)) = e.trim().split_once(':') {
        if s == format!("{k}x{n}") { if let Ok(c) = c.parse::<u32>() { return (c > 0).then_some(c) } } } } }
    // Round 12 (f16 GPU baseline, PF_TF16=1): the NPU only pays off at M >= 2048 and only on the FFN; everything else
    // stays on the GPU. Override with NPU_NN (e.g. the round-11 int8-baseline shares) to split other shapes.
    // round 16: at M < 2048 only gate/up (4096 NPU cols) pay off (pp512 ~1.04x); ffn_down stays on the GPU
    if m < 2048 && crate::env("PF_TF16", 1) != 0 { return if (k, n) == (5120, 17408) { Some(4096) } else { None } }
    Some(match (k, n) { (5120, 17408) => if m >= 2048 { 6144 } else { 8704 }, (17408, 5120) => if m >= 2048 { 2048 } else { 2560 },
        (5120, 10240) | (5120, 6144) | (5120, 12288) | (6144, 5120) if crate::env("PF_TF16", 1) != 0 => return None,
        (5120, 10240) => 5632, (5120, 6144) => 3584, (5120, 12288) => 6656, (6144, 5120) => if m >= 2048 { 3072 } else { 2560 }, _ => return None })
}
const NAMES: [&str; 7] = ["attn_qkv", "attn_gate", "attn_q", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];

struct Seg { cb: vk::CommandBuffer, half: Option<(vk::CommandBuffer, Vec<String>)>, bk: usize }
/// NPU_TB=3: one block-layer recorded by `rec_layer`, executed step by step (`sub_seg` / `npu_seg` / `sub_last`).
pub struct TbLayer { segs: Vec<Seg>, last: vk::CommandBuffer, f: Vec<vk::Fence> }
impl TbLayer { pub fn nseg(&self) -> usize { self.segs.len() } }
type Key = (u32, u32, u8); // (K, N, slot): ffn_up gets its own SplitLinear (slot 1) so gate/up NPU runs can share one sync point
/// A split whose NPU run is deferred into the next cut (NPU_FUSE): GPU half already recorded into `hb`.
/// One NPU GEMM of a split weight: whole K, or (NPU_KCHUNK) a K-chunk [x0, x0+kv) run on the K=5120 xclbin.
struct Part { key: Key, sw: SplitWeight, x0: u32, kv: u32 }
const KC: u32 = 5120; // K-chunk width: ffn_down (K=17408) runs as 4 chunks on the same xclbin as the K=5120 shapes
struct Post { name: String, y: *const Buffer, ldy: u32, y_col: u32, acc: bool }
struct Pend { hb: vk::CommandBuffer, post: Post }
pub struct NpuSplit { lin: Vec<HashMap<Key, SplitLinear>>, pub bk: Cell<usize>, w: HashMap<String, Vec<Part>>, st: Stream, pend: RefCell<Option<Pend>>, fuse: bool,
    segs: RefCell<Vec<Seg>>, pub cur: Cell<vk::CommandBuffer>, pub t_npu: Cell<f64>, pub t_wait: Cell<f64>, pub calls: Cell<u32>, pub load_s: f64,
    pub per: RefCell<std::collections::BTreeMap<Key, (f64, u32)>> }
impl NpuSplit {
    pub fn new(m: &qwen38::Model, mmax: u32, bon: &super::Bon, tb: bool, nblk: usize) -> Result<Self> {
        let mut mk: Vec<(Key, usize, usize, usize, usize, String, Option<Vec<f32>>)> = vec![];
        let t0 = std::time::Instant::now(); let ctx = &m.ctx;
        let (mut lin, mut w) = (HashMap::new(), HashMap::new());
        // NPU_FUSE=1 (default): ffn_gate and ffn_up (same input) prep together, GPU halves in one submit, NPU runs back to back
        let fuse = crate::env("NPU_FUSE", 1) != 0; let kchunk = crate::env("NPU_KCHUNK", 1) != 0;
        // NPU_SPLIT_OPS=attn_qkv,ffn_gate,... restricts the split to those projections (default: all of NAMES)
        let only: Option<Vec<String>> = std::env::var("NPU_SPLIT_OPS").ok().filter(|v| !v.is_empty()).map(|v| v.split(',').map(|x| x.trim().to_string()).collect());
        for l in 0..m.cfg.n_layer as usize { for nm in NAMES {
            if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == nm)) { continue }
            if tb && !nm.starts_with("ffn_") { continue } // NPU_TB: the whole FFN of half A on the NPU, nothing else
            let name = format!("blk.{l}.{nm}.weight");
            let Some(t) = m.wt.g.tensor(&name) else { continue };
            if t.ty != gguf::GgmlType::Other(142) { continue }
            let (k, n) = (t.dims[0] as u32, t.dims[1] as u32);
            let Some(nn) = (if tb { if std::env::var("NPU_NN").is_ok() { nn_for(k, n, mmax) } else { Some(n) } } else { nn_for(k, n, mmax) }) else { continue };
            if !bon.rot(&name) { bail!("NPU_SPLIT: {name} is not Hadamard-rotated (unsupported)") }
            // NPU_KCHUNK=1 (default): K=17408 (ffn_down) as zero-padded K=5120 chunks -> one xclbin, no hw_context switches
            if kchunk && k > KC && k % 128 == 0 {
                let data = m.wt.g.data(t); let rb = data.len() / n as usize; let blk = rb / (k as usize / 128); let cb = KC as usize / 128;
                let mut parts = vec![];
                for (ci, x0) in (0..k).step_by(KC as usize).enumerate() {
                    let kv = KC.min(k - x0); let key: Key = (k, n, 2 + ci as u8);
                    if !lin.contains_key(&key) { let d = build_dir(mmax as usize, KC as usize, nn as usize); lin.insert(key, SplitLinear::new(ctx, mmax as usize, KC as usize, nn as usize, nn as usize, &d, None)?); mk.push((key, mmax as usize, KC as usize, nn as usize, nn as usize, d, None)); }
                    // rows [0, nn) restricted to K blocks [x0/128, (x0+kv)/128), zero blocks (scale 0) beyond kv
                    let mut sub = vec![0u8; nn as usize * cb * blk]; let (b0, nb) = (x0 as usize / 128, kv as usize / 128);
                    for j in 0..nn as usize { sub[j * cb * blk..(j * cb + nb) * blk].copy_from_slice(&data[j * rb + b0 * blk..j * rb + (b0 + nb) * blk]); }
                    let sw = lin[&key].load_npu(ctx, &sub)?; parts.push(Part { key, sw, x0, kv });
                }
                w.insert(name, parts); continue;
            }
            let key: Key = (k, n, (fuse && nm == "ffn_up") as u8);
            if !lin.contains_key(&key) {
                let sg = bon.sg.iter().find(|(wd, _)| *wd == k).map(|(_, b)| b.as_f32()[..k as usize].to_vec()).ok_or_else(|| anyhow::anyhow!("no signs {k}"))?;
                let d = build_dir(mmax as usize, k as usize, nn as usize); lin.insert(key, SplitLinear::new(ctx, mmax as usize, k as usize, n as usize, nn as usize, &d, Some(&sg))?); mk.push((key, mmax as usize, k as usize, n as usize, nn as usize, d, Some(sg)));
            }
            let sw = lin[&key].load_npu(ctx, m.wt.g.data(t))?;
            w.insert(name, vec![Part { key, sw, x0: 0, kv: k }]);
        } }
        // NPU_TB=3: a second set of NPU A/C buffers (same weights) so block B can prep while block A's GEMM runs
        let mut lins = vec![lin];
        for _ in 1..nblk { let mut l2 = HashMap::new(); for (key, a, b, c, d, dir, sg) in &mk { l2.insert(*key, SplitLinear::new(ctx, *a, *b, *c, *d, dir, sg.as_deref())?); } lins.push(l2); }
        Ok(NpuSplit { lin: lins, bk: Cell::new(0), w, st: Stream::new(ctx)?, pend: RefCell::new(None), fuse, segs: Default::default(), cur: Cell::new(vk::CommandBuffer::null()),
            t_npu: Cell::new(0.0), t_wait: Cell::new(0.0), calls: Cell::new(0), per: Default::default(), load_s: t0.elapsed().as_secs_f64() })
    }
    /// Per-(K, N) NPU time: "KxN nn: calls, ms/call".
    pub fn stats(&self) -> String { self.per.borrow().iter().map(|(k, (t, c))| format!("{}x{}{} nn={}: {} x {:.2} ms", k.0, k.1, if k.2 == 1 { "/up".to_string() } else if k.2 >= 2 { format!("/kc{}", k.2 - 2) } else { String::new() }, self.lin[0][k].nn(), c, t * 1e3 / *c as f64)).collect::<Vec<_>>().join(", ") }
    /// NPU columns of `name` if it is split.
    pub fn nn(&self, name: &str) -> Option<u32> { self.w.get(name).map(|v| self.lin[self.bk.get()][&v[0].key].nn() as u32) }
    /// At a split GEMM: record the NPU A prep (from rotated f32 `xr` [n][k]) into the current cb and close it, record
    /// the GPU half (`gpu`) into its own cb, open the next cb with the NPU half's dequant into y[:, y_col..y_col+nn].
    pub fn cut(&self, ctx: &Ctx, name: &str, xr: &Buffer, n: u32, y: &Buffer, ldy: u32, y_col: u32, acc: bool,
               gpu: impl FnOnce(vk::CommandBuffer) -> Result<()>) -> Result<()> {
        let cb = self.cur.get();
        for pt in &self.w[name] { self.lin[self.bk.get()][&pt.key].record_prep_at(ctx, cb, xr, pt.key.0 as usize, n as usize, pt.x0 as usize, pt.kv as usize)?; }
        // y outlives the recording (prefill scratch buffers); the post is recorded within this same `run`
        let post = Post { name: name.to_string(), y: y as *const Buffer, ldy, y_col, acc };
        let fuse_gate = self.fuse && name.ends_with("ffn_gate.weight") && self.pend.borrow().is_none();
        let (hb, mut posts) = match self.pend.borrow_mut().take() { Some(p) => (p.hb, vec![p.post]), None => (self.st.begin(ctx)?, vec![]) };
        gpu(hb)?;
        if fuse_gate { *self.pend.borrow_mut() = Some(Pend { hb, post }); return Ok(()) }  // gate: keep cb open for up's prep
        posts.push(post);
        let names: Vec<String> = posts.iter().map(|p| p.name.clone()).collect();
        self.st.end(ctx, cb)?; self.st.end(ctx, hb)?;
        self.segs.borrow_mut().push(Seg { cb, half: Some((hb, names)), bk: self.bk.get() });
        let nb = self.st.begin(ctx)?;
        for p in posts { for (i, pt) in self.w[&p.name].iter().enumerate() {
            if i > 0 { ctx.barrier(nb) }  // K-chunks accumulate into the same y columns
            self.lin[self.bk.get()][&pt.key].record_post(ctx, nb, &pt.sw, unsafe { &*p.y }, p.ldy as usize, p.y_col as usize, p.acc || i > 0)?; } }
        ctx.barrier(nb);
        self.cur.set(nb); Ok(())
    }
    /// Record a submit group with `f` (which may cut), then execute it with the NPU overlapped. Blocks until done.
    pub fn rec_layer(&self, ctx: &Ctx, bk: usize, f: impl FnOnce(vk::CommandBuffer)) -> Result<TbLayer> {
        self.bk.set(bk); self.cur.set(self.st.begin(ctx)?); f(self.cur.get());
        let last = self.cur.get(); self.st.end(ctx, last)?; self.bk.set(0);
        Ok(TbLayer { segs: self.segs.borrow_mut().drain(..).collect(), last, f: vec![] })
    }
    pub fn sub_seg(&self, ctx: &Ctx, ly: &mut TbLayer, i: usize) -> Result<()> {
        let s = &ly.segs[i]; let f = self.st.submit_cb(ctx, s.cb)?; if let Some((hb, _)) = &s.half { self.st.submit_cb(ctx, *hb)?; }
        if ly.f.len() != i { bail!("sub_seg out of order") } ly.f.push(f); Ok(())
    }
    pub fn npu_seg(&self, ctx: &Ctx, ly: &TbLayer, i: usize) -> Result<()> {
        let s = &ly.segs[i];
        let t = std::time::Instant::now(); self.st.wait(ctx, ly.f[i])?; self.t_wait.set(self.t_wait.get() + t.elapsed().as_secs_f64());
        if let Some((_, names)) = &s.half { for name in names { for pt in &self.w[name] { let key = &pt.key;
            let dt = self.lin[s.bk][key].npu_run(&pt.sw)?; self.t_npu.set(self.t_npu.get() + dt);
            { let mut p = self.per.borrow_mut(); let e = p.entry(*key).or_insert((0.0, 0)); e.0 += dt; e.1 += 1; } self.calls.set(self.calls.get() + 1); } } }
        Ok(())
    }
    pub fn sub_last(&self, ctx: &Ctx, ly: &TbLayer) -> Result<()> { self.st.submit_cb(ctx, ly.last)?; Ok(()) }
    pub fn sub(&self, ctx: &Ctx, cb: vk::CommandBuffer) -> Result<()> { self.st.submit_cb(ctx, cb)?; Ok(()) }
    pub fn sync(&self, ctx: &Ctx) -> Result<()> { self.st.sync(ctx) }
    /// Record a plain GPU command buffer on this stream (submitted later by `run_tb`).
    pub fn rec(&self, ctx: &Ctx, f: impl FnOnce(vk::CommandBuffer)) -> Result<vk::CommandBuffer> {
        let cb = self.st.begin(ctx)?; f(cb); self.st.end(ctx, cb)?; Ok(cb)
    }
    /// Token-block pipeline (NPU_TB): like `run`, but `extra[i]` (half B's GPU work) is queued right after segment i,
    /// so the GPU runs it while the NPU runs segment i's NPU GEMMs of half A.
    pub fn run_tb(&self, ctx: &Ctx, f: impl FnOnce(vk::CommandBuffer), extra: Vec<vk::CommandBuffer>, sync: bool) -> Result<()> {
        self.cur.set(self.st.begin(ctx)?);
        f(self.cur.get());
        let last = self.cur.get(); self.st.end(ctx, last)?;
        let segs: Vec<Seg> = self.segs.borrow_mut().drain(..).collect();
        let mut ex = extra.into_iter();
        for s in segs.iter() {
            let fa = self.st.submit_cb(ctx, s.cb)?;
            if let Some((hb, names)) = &s.half {
                self.st.submit_cb(ctx, *hb)?;
                if let Some(e) = ex.next() { self.st.submit_cb(ctx, e)?; }
                let t = std::time::Instant::now(); self.st.wait(ctx, fa)?; self.t_wait.set(self.t_wait.get() + t.elapsed().as_secs_f64());
                for name in names { for pt in &self.w[name] { let key = &pt.key;
                let dt = self.lin[s.bk][key].npu_run(&pt.sw)?; self.t_npu.set(self.t_npu.get() + dt);
                { let mut p = self.per.borrow_mut(); let e = p.entry(*key).or_insert((0.0, 0)); e.0 += dt; e.1 += 1; } self.calls.set(self.calls.get() + 1); } }
            }
        }
        self.st.submit_cb(ctx, last)?;
        for e in ex { self.st.submit_cb(ctx, e)?; }
        // no host sync between layers: the next layer is recorded while the GPU drains this one (queue order + the
        // leading barrier of every command buffer keep the dependencies); fences are reaped at the next sync
        if sync { self.st.sync(ctx) } else { Ok(()) }
    }
    pub fn run(&self, ctx: &Ctx, f: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        self.cur.set(self.st.begin(ctx)?);
        f(self.cur.get());
        let last = self.cur.get(); self.st.end(ctx, last)?;
        let segs: Vec<Seg> = self.segs.borrow_mut().drain(..).collect();
        for s in segs.iter() {
            let fa = self.st.submit_cb(ctx, s.cb)?;
            if let Some((hb, names)) = &s.half {
                self.st.submit_cb(ctx, *hb)?;
                let t = std::time::Instant::now(); self.st.wait(ctx, fa)?; self.t_wait.set(self.t_wait.get() + t.elapsed().as_secs_f64());
                for name in names { for pt in &self.w[name] { let key = &pt.key;
                let dt = self.lin[s.bk][key].npu_run(&pt.sw)?; self.t_npu.set(self.t_npu.get() + dt);
                { let mut p = self.per.borrow_mut(); let e = p.entry(*key).or_insert((0.0, 0)); e.0 += dt; e.1 += 1; } self.calls.set(self.calls.get() + 1); } }
            }
        }
        self.st.submit_cb(ctx, last)?;
        self.st.sync(ctx)
    }
}
