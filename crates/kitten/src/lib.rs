//! KittenTTS 2 speech LM (GGUF arch "qwen3", 1.7B, TQ2_1 ternary linears, F16 tied embedding) on the generic gemvq path.
//! Prompt layout follows kitten-tts-2-cpp: position 0 is the voice's speaker vector (no token), then prompt ids[1..].
//! Decoding is host-driven: per step the GPU writes logits, the host applies the fork's repetition/run penalties.
use anyhow::{bail, Result};
use ash::vk;
use rtcore::{qtype, DsCache, GkQ, Weights, WRef, GQ};
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel};

pub struct Cfg { pub n_layer: u32, pub dim: u32, pub ffn: u32, pub n_head: u32, pub n_kv: u32, pub hd: u32, pub vocab: u32, pub eps: f32, pub theta: f32, pub max_ctx: u32 }
pub struct Model {
    pub ctx: Ctx, pub wt: Weights, pub cfg: Cfg, ds: DsCache,
    g_proj: GkQ, g_out: GkQ, g_swi: GkQ, g_down: GkQ, g_head: GkQ,
    embed_any: Kernel, rmsprep: Kernel, rope: Kernel, attn: Kernel, comb: Kernel, attnf: Kernel, fuse_attn: bool, attnr: Kernel, fuse_rope: bool, xq8: Kernel, nc: u32,
    pub x: Buffer, pub xw: Buffer, pub pr: Buffer, pub ao: Buffer, pub hb: Buffer, pub logits: Buffer, kc: Buffer, vc: Buffer, part: Buffer, pub toks: Buffer, pub aux: Buffer,
    step_tok: vk::CommandBuffer, step_vec: vk::CommandBuffer,
    /// KT_MT: tokens per submit with the GPU sampler (0 = host sampling, one submit per token)
    samp: Kernel, mt: usize, step_multi: std::cell::RefCell<Option<(String, vk::CommandBuffer)>>,
    /// ids whose K/V currently sit in the cache (index 0 = speaker-vector bits); a causal cache stays valid for the common prefix
    cached: std::cell::RefCell<Vec<u32>>,
    /// first vocab row computed by the LM head (0 = full vocabulary); KT_HEAD=full disables the speech-only head
    pub head_base: u32,
    fence: vk::Fence, gputime: bool,
    /// KT_LMPROF: timestamp after every barrier while recording the profiled step
    prof_qp: std::cell::Cell<Option<(vk::QueryPool, u32)>>,
}
fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
pub fn slot(s: u32) -> u32 { if s < 128 { s } else { 256 + s } }

/// kitten-tts-2-cpp sampling.h token map subset + penalties
pub struct Penalties { pub base: u32, pub count: u32, pub speech_end: u32, pub stop: u32, pub repetition: f32, pub window: usize, pub run_penalty: f32, pub grace: usize }

impl Model {
    pub fn load(path: &str, max_ctx: u32) -> Result<Self> {
        let ctx = Ctx::new()?;
        let wt = Weights::load(&ctx, path)?;
        let g = &wt.g; let a = "qwen3";
        let u = |k: &str| g.u64(&format!("{a}.{k}")).map(|v| v as u32);
        let cfg = Cfg { n_layer: u("block_count")?, dim: u("embedding_length")?, ffn: u("feed_forward_length")?, n_head: u("attention.head_count")?, n_kv: u("attention.head_count_kv")?,
            hd: u("attention.key_length")?, vocab: wt.wref("token_embd.weight")?.n, eps: g.f32(&format!("{a}.attention.layer_norm_rms_epsilon"))?,
            theta: g.f32(&format!("{a}.rope.freq_base"))?, max_ctx };
        if 2 * cfg.n_layer + 1 > 128 { bail!("too many layers") }
        let f = |n: u64| ctx.buffer(n.max(16) * 4);
        let nc = env("ATTN_NC", 16); let sg = env("GEMV_SG", 32);
        let kvn = (cfg.n_layer * max_ctx * cfg.n_kv * cfg.hd) as u64;
        let pr_n = (cfg.n_head + 2 * cfg.n_kv) * cfg.hd;
        let mut m = Model {
            g_proj: GkQ::new("GEMV_PROJ", (16, 64), false, sg), g_out: GkQ::new("GEMV_OUT", (32, 128), false, sg), g_swi: GkQ::new("GEMV_SWIGLU", (32, 128), true, sg),
            g_down: GkQ::new("GEMV_DOWN", (16, 64), false, sg), g_head: GkQ::new("GEMV_HEAD", (32, 256), false, sg),
            embed_any: ctx.kernel(shaders::EMBED_ANY, 5, 24, &[], None)?, rmsprep: ctx.kernel(shaders::RMSPREP, 4, 20, &[], None)?,
            rope: ctx.kernel(shaders::ROPE_QKN, 5, 56, &[cfg.hd], Some(32))?, attn: ctx.kernel(shaders::ATTN_SPLIT, 5, 36, &[cfg.hd], None)?, comb: ctx.kernel(shaders::ATTN_COMBINE, 3, 20, &[cfg.hd], None)?,
            attnf: ctx.kernel(include_bytes!(concat!(env!("OUT_DIR"), "/kt_attn.spv")), 6, 36, &[cfg.hd], None)?, fuse_attn: std::env::var("KT_ATTNFUSE").map(|v| v != "0").unwrap_or(true), attnr: ctx.kernel(include_bytes!(concat!(env!("OUT_DIR"), "/kt_attn_rope.spv")), 9, 64, &[cfg.hd], Some(32))?, fuse_rope: cfg.hd == 64 && std::env::var("KT_ROPEFUSE").map(|v| v != "0").unwrap_or(true),
            xq8: ctx.kernel(shaders::XQ8, 2, 12, &[], Some(32))?, nc,
            x: f(cfg.dim as u64)?, xw: f(cfg.dim as u64)?, pr: f(pr_n as u64)?, ao: f((cfg.n_head * cfg.hd) as u64)?, hb: f(cfg.ffn as u64)?, logits: f(cfg.vocab as u64)?,
            kc: ctx.buffer(kvn * 2)?, vc: ctx.buffer(kvn * 2)?, part: f((cfg.n_head * nc * (2 + cfg.hd)) as u64)?, toks: f(max_ctx as u64 + 4)?, aux: f(49152)?,
            step_tok: vk::CommandBuffer::null(), step_vec: vk::CommandBuffer::null(), samp: ctx.kernel(include_bytes!(concat!(env!("OUT_DIR"), "/kt_sample.spv")), 3, 48, &[], None)?, mt: env("KT_MT", 8) as usize, step_multi: Default::default(), ds: DsCache::default(), cached: Default::default(), fence: vk::Fence::null(), prof_qp: Default::default(), gputime: std::env::var("KT_GPUTIME").is_ok(), head_base: if std::env::var("KT_HEAD").map(|v| v == "full").unwrap_or(false) { 0 } else { 151669 }, cfg, wt, ctx };
        m.au()[49088..49088 + 64].fill(0);
        m.fence = unsafe { m.ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let mut err = Ok(()); let cb = m.ctx.record_raw(|cb| { err = m.record(cb, false); })?; err?; m.step_tok = cb;
        let mut err = Ok(()); let cb = m.ctx.record_raw(|cb| { err = m.record(cb, true); })?; err?; m.step_vec = cb;
        Ok(m)
    }
    fn set(&self, k: &Kernel, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> { self.ds.get(&self.ctx, k, bufs) }
    fn gemv(&self, cb: vk::CommandBuffer, gq: &GkQ, segs: &[WRef], up: Option<WRef>, x: &Buffer, y: &Buffer, acc: bool, scale: Option<u32>, sq_out: Option<(u32, u32)>) -> Result<()> {
        let tern = |w: &WRef| matches!(qtype(w.ty), Ok(8) | Ok(9) | Ok(12));
        let mut qf = 0;
        if std::env::var("TERN_Q8").map(|v| v != "0").unwrap_or(true) && segs.iter().chain(up.iter()).all(tern) {
            let ds = self.set(&self.xq8, &[x, &self.aux])?;
            self.ctx.dispatch3(cb, &self.xq8, ds, &push_u32(&[segs[0].k, 0, 0]), segs[0].k / 128, 1, 1); self.ctx.barrier(cb); qf = 32;
        }
        let (mut g, arena) = GQ::segs(segs, up, &[], self.cfg.eps)?;
        let gk = gq.get(&self.ctx, &g, segs.len())?;
        g.flags |= qf | 64;
        if acc { g.flags |= 1; }
        if let Some(s) = scale { g.flags |= 4; g.sq_in = slot(s); }
        if let Some((s, nw)) = sq_out { g.flags |= 8 | 16; g.sq_out = slot(s); g.norm_off = nw; }
        let ds = self.set(&gk.k, &[&self.wt.arenas[arena], x, y, &self.wt.f32b, &self.aux, &self.xw])?;
        self.ctx.dispatch(cb, &gk.k, ds, &g.bytes(), g.n.div_ceil(gk.rows_per_wg));
        Ok(())
    }
    fn record(&self, cb: vk::CommandBuffer, from_vec: bool) -> Result<()> {
        let (c, cfg, wt) = (&self.ctx, &self.cfg, &self.wt);
        let bar = || { c.barrier(cb); if let Some((qp, i)) = self.prof_qp.get() { unsafe { c.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, i) }; self.prof_qp.set(Some((qp, i + 1))); } };
        bar();
        if !from_vec {
            let te = wt.wref("token_embd.weight")?;
            let ds = self.set(&self.embed_any, &[&wt.arenas[te.arena], &self.x, &self.aux, &self.toks, &wt.f32b])?;
            c.dispatch(cb, &self.embed_any, ds, &push_u32(&[te.off as u32, cfg.dim, qtype(te.ty)?, 160, 0, 0]), 1); bar();
        }
        let ds = self.set(&self.rmsprep, &[&self.x, &self.xw, &self.aux, &wt.f32b])?;
        c.dispatch(cb, &self.rmsprep, ds, &push_u32(&[cfg.dim, wt.f("blk.0.attn_norm.weight")?, 128, 0, 0]), 1); bar();
        let (qn, kn) = (cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd);
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let cache_off = l * cfg.max_ctx * cfg.n_kv * cfg.hd;
            self.gemv(cb, &self.g_proj, &[wt.wref(&n("attn_q.weight"))?, wt.wref(&n("attn_k.weight"))?, wt.wref(&n("attn_v.weight"))?], None, &self.xw, &self.pr, false, Some(2 * l), None)?; bar();
            if self.fuse_rope && self.fuse_attn {
                let ds = self.set(&self.attnr, &[&self.pr, &self.kc, &self.vc, &self.part, &self.aux, &self.ao, &self.kc, &self.vc, &wt.f32b])?;
                c.dispatch3(cb, &self.attnr, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0,
                    cfg.theta.to_bits(), wt.f(&n("attn_q_norm.weight"))?, wt.f(&n("attn_k_norm.weight"))?, cfg.eps.to_bits(), qn, qn + kn, cfg.hd]), cfg.n_head, self.nc, 1); bar();
            } else {
            let ds = self.set(&self.rope, &[&self.pr, &self.kc, &self.vc, &self.aux, &wt.f32b])?;
            c.dispatch(cb, &self.rope, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, cfg.theta.to_bits(), wt.f(&n("attn_q_norm.weight"))?, wt.f(&n("attn_k_norm.weight"))?, cfg.eps.to_bits(),
                cfg.hd, qn, qn + kn, cfg.hd, 160, 0, 0]), cfg.n_head + cfg.n_kv); bar();
            if self.fuse_attn {
                let ds = self.set(&self.attnf, &[&self.pr, &self.kc, &self.vc, &self.part, &self.aux, &self.ao])?;
                c.dispatch3(cb, &self.attnf, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0]), cfg.n_head, self.nc, 1); bar();
            } else {
            let ds = self.set(&self.attn, &[&self.pr, &self.kc, &self.vc, &self.part, &self.aux])?;
            c.dispatch3(cb, &self.attn, ds, &push_u32(&[cfg.n_head, cfg.n_kv, cache_off, self.nc, cfg.hd, 160, (1.0f32 / (cfg.hd as f32).sqrt()).to_bits(), 0, 0]), cfg.n_head, self.nc, 1); bar();
            let ds = self.set(&self.comb, &[&self.part, &self.ao, &self.pr])?;
            c.dispatch(cb, &self.comb, ds, &push_u32(&[self.nc, cfg.hd, 0, 0, 0]), cfg.n_head); bar();
            }
            }
            self.gemv(cb, &self.g_out, &[wt.wref(&n("attn_output.weight"))?], None, &self.ao, &self.x, true, None, Some((2 * l + 1, wt.f(&n("ffn_norm.weight"))?)))?; bar();
            self.gemv(cb, &self.g_swi, &[wt.wref(&n("ffn_gate.weight"))?], Some(wt.wref(&n("ffn_up.weight"))?), &self.xw, &self.hb, false, Some(2 * l + 1), None)?; bar();
            let next = if l + 1 < cfg.n_layer { wt.f(&format!("blk.{}.attn_norm.weight", l + 1))? } else { wt.f("output_norm.weight")? };
            self.gemv(cb, &self.g_down, &[wt.wref(&n("ffn_down.weight"))?], None, &self.hb, &self.x, true, None, Some((2 * l + 2, next)))?; bar();
        }
        let mut head = wt.wref("output.weight").or_else(|_| wt.wref("token_embd.weight"))?;
        // speech-only head: rows [head_base, vocab) = speech_start/end, text_start, start, stop, audio codes, specials
        if self.head_base > 0 { head.off += self.head_base as u64 * head.bytes / head.n as u64; head.bytes -= self.head_base as u64 * head.bytes / head.n as u64; head.n -= self.head_base; }
        self.gemv(cb, &self.g_head, &[head], None, &self.xw, &self.logits, false, Some(2 * cfg.n_layer), None)?; bar();
        Ok(())
    }
    /// per-segment GPU time of one token step (segments end at each barrier), averaged over `n` runs at position `pos`
    pub fn lm_profile(&self, pos: u32, n: usize) -> Result<Vec<f64>> {
        let d = &self.ctx.device;
        let qp = unsafe { d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(1024), None)? };
        let mut err = Ok(()); let mut cnt = 0;
        let cb = self.ctx.record_raw(|cb| { unsafe { d.cmd_reset_query_pool(cb, qp, 0, 1024); d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 0); }
            self.prof_qp.set(Some((qp, 1))); err = self.record(cb, false); cnt = self.prof_qp.get().unwrap().1; self.prof_qp.set(None); })?; err?;
        let mut acc = vec![0f64; cnt as usize - 1];
        for _ in 0..n { self.au()[160] = pos; self.submit(cb)?; let mut ts = vec![0u64; cnt as usize];
            unsafe { d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
            for i in 1..cnt as usize { acc[i - 1] += (ts[i] - ts[i - 1]) as f64 * self.ctx.info.ts_period_ns as f64 * 1e-9 / n as f64 } }
        Ok(acc)
    }
    fn au(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.aux.ptr as *mut u32, 49152) } }
    fn tk(&self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.toks.ptr as *mut u32, self.toks.size as usize / 4) } }
    /// Run position `pos` with token `tok` (or the speaker vector when `vec` is given). Returns GPU seconds.
    pub fn step(&self, pos: u32, tok: u32, vec: Option<&[f32]>) -> Result<f64> {
        self.au()[160] = pos;
        match vec { Some(v) => { self.x.as_f32_mut()[..v.len()].copy_from_slice(v); self.submit(self.step_vec) }
                    None => { self.tk()[pos as usize] = tok; self.submit(self.step_tok) } }
    }
    /// one pre-recorded step: persistent fence, no per-step query pool / command buffers (KT_GPUTIME=1: timestamped path, returns GPU s)
    fn submit(&self, cb: vk::CommandBuffer) -> Result<f64> {
        if self.gputime { return self.ctx.submit_repeated(cb, 1) }
        let d = &self.ctx.device; let t = std::time::Instant::now();
        unsafe { d.queue_submit(self.ctx.queue, &[vk::SubmitInfo::default().command_buffers(&[cb])], self.fence)?; d.wait_for_fences(&[self.fence], true, u64::MAX)?; d.reset_fences(&[self.fence])?; }
        Ok(t.elapsed().as_secs_f64())
    }
    /// Batched prompt: all positions [start, start+ids.len()) recorded into one command buffer (one submit, one wait).
    /// Each position updates aux[160] in-stream (vkCmdUpdateBuffer) and replays the step body; logits hold the last position.
    pub fn prefill_batch(&self, start: usize, ids: &[u32]) -> Result<f64> {
        if ids.is_empty() { return Ok(0.0) }
        for (i, &id) in ids.iter().enumerate() { self.tk()[start + i] = id; }
        let d = &self.ctx.device; let mut err = Ok(());
        let full = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)];
        let t = self.ctx.run(|cb| { for i in 0..ids.len() { if err.is_err() { return }
            unsafe {
                d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &full, &[], &[]);
                d.cmd_update_buffer(cb, self.aux.buf, 160 * 4, &((start + i) as u32).to_le_bytes());
                d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &full, &[], &[]);
            }
            err = self.record(cb, false); } })?;
        err?; self.au()[160] = (start + ids.len() - 1) as u32; Ok(t)
    }
    /// K steps in one command buffer: [GPU sampler -> step body] x K (sampler writes toks[pos+1] and advances aux[160])
    fn multi_cb(&self, p: &Penalties) -> Result<vk::CommandBuffer> {
        let key = format!("{} {} {} {} {} {} {} {} {}", p.base, p.count, p.speech_end, p.stop, p.repetition, p.window, p.run_penalty, p.grace, self.mt);
        if let Some((k, cb)) = &*self.step_multi.borrow() { if *k == key { return Ok(*cb) } }
        let nv = self.cfg.vocab - self.head_base; if nv > 8192 || p.window == 0 || p.window > 256 { bail!("sampler: head too large or window unsupported") }
        let ds = self.set(&self.samp, &[&self.logits, &self.toks, &self.aux])?;
        let push = push_u32(&[self.head_base, nv, 0, p.window as u32, p.grace as u32, p.base, p.count, p.speech_end, p.stop, p.base + 4299, p.run_penalty.to_bits(), p.repetition.to_bits()]);
        let mut err = Ok(());
        let cond = vkrt::HAS_COND.load(std::sync::atomic::Ordering::Relaxed) && std::env::var("KT_COND").map(|v| v != "0").unwrap_or(true);
        let cr = if cond { Some(ash::ext::conditional_rendering::Device::new(&self.ctx.instance, &self.ctx.device)) } else { None };
        let cb = self.ctx.record_raw(|cb| { for _ in 0..self.mt { if err.is_err() { return } self.ctx.barrier(cb); self.ctx.dispatch(cb, &self.samp, ds, &push, 1);
            // steps after a stop token are skipped (predicate aux[162] cleared by the sampler)
            if let Some(cr) = &cr { unsafe {
                let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT)];
                self.ctx.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT, vk::DependencyFlags::empty(), &mb, &[], &[]);
                (cr.fp().cmd_begin_conditional_rendering_ext)(cb, &vk::ConditionalRenderingBeginInfoEXT::default().buffer(self.aux.buf).offset(162 * 4)); } }
            err = self.record(cb, false);
            if let Some(cr) = &cr { unsafe { (cr.fp().cmd_end_conditional_rendering_ext)(cb) } } } })?; err?;
        *self.step_multi.borrow_mut() = Some((key, cb)); Ok(cb)
    }
    /// fork-equivalent greedy decode: speaker at pos 0, prompt[1..], then argmax of penalized logits.
    /// Returns (generated ids incl. terminator, prefill wall s, decode wall s, decode GPU s).
    pub fn generate(&self, prompt: &[u32], speaker: &[f32], max_new: usize, p: &Penalties) -> Result<(Vec<u32>, f64, f64, f64)> {
        if prompt.len() + max_new + 1 > self.cfg.max_ctx as usize { bail!("MAX_CTX too small") }
        let t = std::time::Instant::now();
        // KV prefix reuse: the voice reference block (speaker, ref text, ref audio) is a fixed prefix per voice
        let key: Vec<u32> = std::iter::once(speaker.iter().fold(0x811c9dc5u32, |h, v| (h ^ v.to_bits()).wrapping_mul(16777619))).chain(prompt[1..].iter().copied()).collect();
        let reuse = if std::env::var("KT_NOCACHE").is_ok() { 0 } else { let c = self.cached.borrow(); c.iter().zip(&key).take_while(|(a, b)| a == b).count().min(key.len() - 1) };
        let mut pg = vec![];
        if reuse == 0 { self.step(0, 0, Some(speaker))?; }
        let from = 1.max(reuse);
        if std::env::var("KT_BATCH").map(|v| v != "0").unwrap_or(true) { pg.push(self.prefill_batch(from, &prompt[from..])?); pg.resize(prompt.len() - from, 0.0); }
        else { for (i, &id) in prompt.iter().enumerate().skip(from) { pg.push(self.step(i as u32, id, None)?); } }
        if std::env::var("KT_PROF").is_ok() { eprintln!("prefill: reused {reuse} cached positions, ran {}", pg.len() + (reuse == 0) as usize); }
        *self.cached.borrow_mut() = key.clone();
        let pw = t.elapsed().as_secs_f64();
        let mut hist: Vec<u32> = prompt.to_vec(); let mut out = vec![]; let mut gpu = 0.0; let mut tstep = 0.0;
        let t = std::time::Instant::now();
        if self.mt > 0 && !self.gputime && prompt.len() + max_new + self.mt + 1 <= self.cfg.max_ctx as usize {
            let cb = self.multi_cb(p)?; let mut pos = prompt.len() - 1; self.au()[160] = pos as u32; self.au()[161] = prompt.len() as u32; self.au()[162] = 1;
            'outer: loop { let ts = std::time::Instant::now(); self.submit(cb)?; tstep += ts.elapsed().as_secs_f64();
                for i in 1..=self.mt { let tok = self.tk()[pos + i]; out.push(tok); if tok == p.speech_end || tok == p.stop || out.len() >= max_new { break 'outer } }
                pos += self.mt; }
            if std::env::var("KT_PROF").is_ok() { eprintln!("decode (GPU sampler, {} tok/submit): submit wall {:.3}s, host rest {:.3}s", self.mt, tstep, t.elapsed().as_secs_f64() - tstep); }
            self.cached.borrow_mut().extend(out.iter().take(out.len().saturating_sub(1)));
            return Ok((out, pw, t.elapsed().as_secs_f64(), tstep))
        }
        for _ in 0..max_new {
            let hb = self.head_base as usize; let mut sc = vec![f32::NEG_INFINITY; self.cfg.vocab as usize];
            sc[hb..].copy_from_slice(&self.logits.as_f32()[..self.cfg.vocab as usize - hb]);
            penalize(&mut sc, &hist, prompt.len(), p);
            let mut best = 0usize; for i in 1..sc.len() { if sc[i] > sc[best] { best = i } }
            let tok = best as u32; out.push(tok); hist.push(tok);
            if tok == p.speech_end || tok == p.stop { break }
            let ts = std::time::Instant::now(); gpu += self.step((hist.len() - 1) as u32, tok, None)?; tstep += ts.elapsed().as_secs_f64();
        }
        if std::env::var("KT_PROF").is_ok() { eprintln!("decode: step() wall {:.3}s, GPU {:.3}s, host rest {:.3}s", tstep, gpu, t.elapsed().as_secs_f64() - tstep); }
        self.cached.borrow_mut().extend(out.iter().take(out.len().saturating_sub(1)));
        Ok((out, pw, t.elapsed().as_secs_f64(), gpu))
    }
}
/// sampling.h process(): run penalty, windowed repetition penalty (temperature/top-k/top-p do not change the argmax)
pub fn penalize(sc: &mut [f32], hist: &[u32], prompt_size: usize, p: &Penalties) {
    if let Some(&last) = hist.last() { if p.run_penalty != 0.0 {
        let run = hist.iter().rev().take_while(|&&t| t == last).count();
        if last >= p.base && last < p.base + p.count && run >= p.grace { sc[last as usize] -= (run - p.grace + 1) as f32 * p.run_penalty; } } }
    if p.repetition != 1.0 {
        let begin = if p.window > 0 { prompt_size.max(hist.len().saturating_sub(p.window)) } else { 0 };
        let mut seen = std::collections::HashSet::new();
        for &id in &hist[begin..] {
            if id == p.speech_end || id == p.stop || id == p.base + 4299 || !seen.insert(id) { continue }
            let s = &mut sc[id as usize]; *s = if *s < 0.0 { *s * p.repetition } else { *s / p.repetition };
        }
    }
}
pub mod dec;
pub mod arena;
pub mod gexec;
pub mod json;
pub mod tok;
pub mod text;
pub mod math_table;
/// 32-bit float mono WAV (format 3), like kitten-tts-2-cpp
pub fn write_wav(path: &str, x: &[f32], sr: u32) -> Result<()> {
    let mut b = Vec::with_capacity(44 + x.len() * 4); let n = (x.len() * 4) as u32;
    b.extend(b"RIFF"); b.extend((36 + n).to_le_bytes()); b.extend(b"WAVEfmt "); b.extend(16u32.to_le_bytes()); b.extend(3u16.to_le_bytes()); b.extend(1u16.to_le_bytes());
    b.extend(sr.to_le_bytes()); b.extend((sr * 4).to_le_bytes()); b.extend(4u16.to_le_bytes()); b.extend(32u16.to_le_bytes()); b.extend(b"data"); b.extend(n.to_le_bytes());
    for v in x { b.extend(v.to_le_bytes()) } std::fs::write(path, b)?; Ok(())
}
