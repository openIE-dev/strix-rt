//! GPU-driven decode: embedding lookup, fused-norm GEMVs, fused q|k|v, f16 KV cache, argmax and token feedback all on
//! the GPU. One reusable command buffer = one token step; the host submits many steps in one vkQueueSubmit and waits once.
//! aux buffer layout (u32 words): [0..64) f32 sum-of-squares slots, [64..66) u64 argmax key, [80] pos, [81] prompt length.
use crate::{Model, WRef};
use anyhow::{bail, Result};
use ash::vk;
use gguf::GgmlType;
use vkrt::{push_u32, shaders, Buffer, Kernel};

pub struct Gk { pub k: Kernel, pub rows_per_wg: u32 }

pub struct Fast { pub small: Gk, pub swi: Gk, pub down: Gk, pub head: Gk, embed: Kernel, rope: Kernel, attn: Kernel, argmax: Kernel, sample: Kernel,
    pub x: Buffer, pub xw: Buffer, pub qkv: Buffer, pub ao: Buffer, pub hb: Buffer, pub logits: Buffer, pub kc: Buffer, pub vc: Buffer, pub toks: Buffer, pub aux: Buffer,
    pub step: vk::CommandBuffer }

#[derive(Default, Clone, Copy)]
struct G { n: u32, k: u32, flags: u32, y_off: u32, norm_off: u32, sq_in: u32, sq_out: u32, r1: u32, r2: u32, types: u32, w: [u32; 3], b: [u32; 3], wup: u32, eps: f32 }
impl G { fn bytes(&self) -> Vec<u8> { push_u32(&[self.n, self.k, self.flags, self.y_off, self.norm_off, self.sq_in, self.sq_out, self.r1, self.r2, self.types,
    self.w[0], self.w[1], self.w[2], self.b[0], self.b[1], self.b[2], self.wup, self.eps.to_bits(), 0, 0]) } }

fn unit_off(w: &WRef) -> Result<u32> { Ok(match w.ty { GgmlType::Q4K => (w.off / 4) as u32, GgmlType::Q6K => (w.off / 2) as u32, t => bail!("unsupported {t:?}") }) }

impl Fast {
    pub fn new(m: &Model, _ls: u32, sg: u32) -> Result<Self> {
        let c = &m.ctx; let cfg = &m.cfg;
        // per-op (lanes per row, workgroup size), from logs/fast_stage_sweep.log; override with e.g. GEMV_DOWN=8,64
        let mk = |var: &str, def: (u32, u32), swiglu: bool| -> Result<Gk> {
            let (lpr, ls) = std::env::var(var).ok().and_then(|v| { let p: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect(); (p.len() == 2).then(|| (p[0], p[1])) }).unwrap_or(def);
            let lpr = lpr.min(sg);
            Ok(Gk { k: c.kernel(shaders::gemv3(lpr, swiglu), 6, 80, &[ls], Some(sg))?, rows_per_wg: ls / lpr })
        };
        let f = |n: u64| c.buffer(n * 4);
        let kv = (cfg.n_layer * cfg.max_ctx * cfg.n_kv * 128) as u64;
        let mut s = Fast {
            small: mk("GEMV_SMALL", (8, 64), false)?, swi: mk("GEMV_SWIGLU", (16, 64), true)?, down: mk("GEMV_DOWN", (8, 64), false)?, head: mk("GEMV_HEAD", (32, 128), false)?,
            embed: c.kernel(shaders::EMBED_Q4K, 6, 12, &[], None)?, rope: c.kernel(shaders::ROPE_KV16, 4, 16, &[], None)?,
            attn: c.kernel(shaders::ATTN16, 5, 16, &[], None)?, argmax: c.kernel(shaders::ARGMAX, 2, 4, &[], None)?, sample: c.kernel(shaders::SAMPLE, 2, 0, &[], None)?,
            x: f(cfg.dim as u64)?, xw: f(cfg.dim as u64)?, qkv: f((cfg.dim + 2 * cfg.n_kv * 128) as u64)?, ao: f(cfg.dim as u64)?, hb: f(cfg.ffn as u64)?, logits: f(cfg.vocab as u64)?,
            kc: c.buffer(kv * 2)?, vc: c.buffer(kv * 2)?, toks: f(cfg.max_ctx as u64 + 1)?, aux: f(128)?, step: vk::CommandBuffer::null() };
        let mut err = Ok(());
        s.step = c.record_raw(|cb| { err = s.record(m, cb); })?;
        err?;
        Ok(s)
    }

    fn gemv(&self, m: &Model, cb: vk::CommandBuffer, gk: &Gk, segs: &[(WRef, Option<String>)], up: Option<WRef>, x: &Buffer, y: &Buffer, acc: bool, scale_slot: Option<u32>, sq_out: Option<(u32, String)>) -> Result<()> {
        let arena = segs[0].0.arena;
        if segs.iter().any(|s| s.0.arena != arena) || up.map(|u| u.arena != arena).unwrap_or(false) { bail!("fused segments must share an arena") }
        let mut g = G { k: segs[0].0.k, eps: m.cfg.eps, ..Default::default() };
        let mut r = 0; let mut bounds = [0u32; 3];
        for (i, (w, bias)) in segs.iter().enumerate() {
            g.w[i] = unit_off(w)?; if w.ty == GgmlType::Q6K { g.types |= 1 << i; }
            g.b[i] = match bias { Some(b) => m.fo[b], None => u32::MAX };
            r += w.n; bounds[i] = r;
        }
        g.n = r; g.r1 = bounds[0]; g.r2 = if segs.len() > 1 { bounds[1] } else { r };
        if let Some(u) = up { g.wup = unit_off(&u)?; }
        if acc { g.flags |= 1; }
        if let Some(slot) = scale_slot { g.flags |= 4; g.sq_in = slot; }
        if let Some((slot, nw)) = &sq_out { g.flags |= 8 | 16; g.sq_out = *slot; g.norm_off = m.fo[nw]; }
        let ds = m.set6(&gk.k, &[&m.arenas[arena], x, y, &m.f32b, &self.aux, &self.xw])?;
        m.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), r.div_ceil(gk.rows_per_wg));
        Ok(())
    }

    fn record(&self, m: &Model, cb: vk::CommandBuffer) -> Result<()> { self.record_prof(m, cb, None) }
    /// prof: (query pool, labels) - writes a timestamp after every stage barrier.
    pub fn record_prof(&self, m: &Model, cb: vk::CommandBuffer, prof: Option<(vk::QueryPool, &std::cell::RefCell<Vec<&'static str>>)>) -> Result<()> {
        let (c, cfg) = (&m.ctx, &m.cfg);
        let cur = std::cell::Cell::new("start");
        if let Some((qp, labels)) = prof { unsafe { c.device.cmd_reset_query_pool(cb, qp, 0, 1024); c.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 0); } labels.borrow_mut().push("start"); }
        let bar = || { c.barrier(cb); if let Some((qp, labels)) = prof { let mut l = labels.borrow_mut(); let i = l.len() as u32; l.push(cur.get());
            unsafe { c.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, i); } } };
        bar(); // orders this step after the previous step in the same queue
        let te = m.wref("token_embd.weight")?;
        if te.ty != GgmlType::Q4K { bail!("embed kernel expects Q4_K token_embd") }
        let ds = m.set6(&self.embed, &[&m.arenas[te.arena], &self.x, &self.aux, &self.toks, &m.f32b, &self.xw])?;
        cur.set("embed"); c.dispatch(cb, &self.embed, ds, &push_u32(&[(te.off / 4) as u32, cfg.dim, m.fo["blk.0.attn_norm.weight"]]), 1); bar();
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let cache_off = l * cfg.max_ctx * cfg.n_kv * 128;
            cur.set("qkv"); let segs = [(m.wref(&n("attn_q.weight"))?, Some(n("attn_q.bias"))), (m.wref(&n("attn_k.weight"))?, Some(n("attn_k.bias"))), (m.wref(&n("attn_v.weight"))?, Some(n("attn_v.bias")))];
            self.gemv(m, cb, &self.small, &segs, None, &self.xw, &self.qkv, false, Some(2 * l), None)?; bar();
            cur.set("rope"); let ds = m.set(&self.rope, &[&self.qkv, &self.kc, &self.vc, &self.aux])?;
            c.dispatch(cb, &self.rope, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, cfg.theta.to_bits()]), cfg.n_head + cfg.n_kv); bar();
            cur.set("attn"); let ds = m.set(&self.attn, &[&self.qkv, &self.kc, &self.vc, &self.ao, &self.aux])?;
            c.dispatch(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, cfg.max_ctx]), cfg.n_head); bar();
            cur.set("o_proj"); self.gemv(m, cb, &self.small, &[(m.wref(&n("attn_output.weight"))?, None)], None, &self.ao, &self.x, true, None, Some((2 * l + 1, n("ffn_norm.weight"))))?; bar();
            cur.set("gate_up"); self.gemv(m, cb, &self.swi, &[(m.wref(&n("ffn_gate.weight"))?, None)], Some(m.wref(&n("ffn_up.weight"))?), &self.xw, &self.hb, false, Some(2 * l + 1), None)?; bar();
            cur.set("down"); self.gemv(m, cb, &self.down, &[(m.wref(&n("ffn_down.weight"))?, None)], None, &self.hb, &self.x, true, None, Some((2 * l + 2, if l + 1 < cfg.n_layer { format!("blk.{}.attn_norm.weight", l + 1) } else { "output_norm.weight".into() })))?; bar();
        }
        cur.set("lm_head"); self.gemv(m, cb, &self.head, &[(m.wref("output.weight")?, None)], None, &self.xw, &self.logits, false, Some(2 * cfg.n_layer), None)?; bar();
        cur.set("argmax"); let ds = m.set(&self.argmax, &[&self.logits, &self.aux])?;
        c.dispatch(cb, &self.argmax, ds, &push_u32(&[cfg.vocab]), 64); bar();
        let ds = m.set(&self.sample, &[&self.aux, &self.toks])?;
        cur.set("sample"); c.dispatch(cb, &self.sample, ds, &[], 1); bar();
        Ok(())
    }

    /// Greedy generation fully on the GPU. Returns (generated ids, prompt GPU s, prompt wall s, decode GPU s, decode wall s, decode steps).
    pub fn generate(&self, m: &Model, prompt: &[u32], n_new: usize) -> Result<(Vec<u32>, f64, f64, f64, f64, usize)> {
        let toks = unsafe { std::slice::from_raw_parts_mut(self.toks.ptr as *mut u32, self.toks.size as usize / 4) };
        let au = unsafe { std::slice::from_raw_parts_mut(self.aux.ptr as *mut u32, 128) };
        toks[..prompt.len()].copy_from_slice(prompt);
        au[80] = 0; au[81] = prompt.len() as u32;
        let t = std::time::Instant::now();
        let pg = m.ctx.submit_repeated(self.step, prompt.len())?; // prompt + first generated token
        let pw = t.elapsed().as_secs_f64();
        let steps = n_new.saturating_sub(1);
        let t = std::time::Instant::now();
        let dg = if steps > 0 { m.ctx.submit_repeated(self.step, steps)? } else { 0.0 };
        let dw = t.elapsed().as_secs_f64();
        Ok((toks[prompt.len()..prompt.len() + n_new].to_vec(), pg, pw, dg, dw, steps))
    }
}

impl Fast {
    /// One profiled step at the current state; returns per-stage GPU microseconds summed over layers.
    pub fn profile(&self, m: &Model) -> Result<Vec<(&'static str, f64)>> {
        let c = &m.ctx;
        let qp = unsafe { c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(1024), None)? };
        let labels = std::cell::RefCell::new(vec![]);
        let mut err = Ok(());
        let cb = c.record_raw(|cb| { err = self.record_prof(m, cb, Some((qp, &labels))); })?; err?;
        c.submit_repeated(cb, 1)?;
        let l = labels.borrow(); let mut ts = vec![0u64; l.len()];
        unsafe { c.device.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; }
        let mut agg: Vec<(&'static str, f64)> = vec![];
        for i in 1..l.len() { let us = (ts[i] - ts[i - 1]) as f64 * c.info.ts_period_ns as f64 / 1e3;
            match agg.iter_mut().find(|a| a.0 == l[i]) { Some(a) => a.1 += us, None => agg.push((l[i], us)) } }
        Ok(agg)
    }
}
