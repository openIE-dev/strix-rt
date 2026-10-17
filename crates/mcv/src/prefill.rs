//! Batched prefill for the MiniCPM-V-4.7 text tower, built from crates/prefill: the qwen35 trunk (rmsnorm, Gated DeltaNet
//! conv/scan/gnorm, gated attention, Pf GEMMs on the Q4_K/Q6_K weights) with a 256-expert grouped MoE (router GEMM ->
//! mcv_psel select+group -> gather -> grouped gate|up SwiGLU -> grouped down -> moecomb), the sigmoid-gated shared
//! expert (GEMM SwiGLU -> down -> mcv_shadd), and per-row interleaved M-RoPE (mcv_prope). Writes the same KV cache /
//! recurrent state as the decode path, then hands off to Llm::decode.
use crate::llm::{Item, Llm};
use anyhow::{bail, Result};
use ash::vk;
use gguf::dequant::dequant_row;
use prefill::{GemmOp, Pf};
use rtcore::WRef;
use vkrt::{push_u32, Buffer, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }
fn env(n: &str, d: u32) -> u32 { std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
const LDR: u32 = 264; // router logits row stride (256 experts + shared gate at column 256)

pub struct Prefill {
    pub pf: Pf, pub mmax: u32, al: u32, ldp_max: u32,
    prope: Kernel, psel: Kernel, shadd: Kernel, gscan: Kernel, gscan_r: u32, gmv: Kernel, gscan4: Kernel, gmvd: Kernel, gmv2: Kernel, qp: vk::QueryPool,
    x: Buffer, a: Buffer, a2: Buffer, p: Buffer, cv: Buffer, o: Buffer, pos: Buffer,
    rw: Buffer, rl: Buffer, r: Buffer, g: Buffer, gt: Buffer, gt2: Buffer, ag: Buffer, hg: Buffer, yd: Buffer, hsh: Buffer, ysh: Buffer,
}

impl Prefill {
    pub fn new(m: &Llm, mmax: u32) -> Result<Self> {
        let (c, ctx) = (&m.cfg, &m.ctx);
        let conv_ch = 2 * c.nk * c.d_state + c.nv * c.d_state;
        let al = c.dim.max(c.nv * c.d_state).max(c.n_head * c.hd).max(c.ff_sh).next_multiple_of(8);
        let ldp_max = (conv_ch + c.nv * c.d_state + 2 * c.nv).max(c.n_head * c.hd * 2 + 2 * c.n_kv * c.hd);
        let mm = mmax as u64; let nk = mm * c.top_k as u64; let ne = c.n_exp as u64;
        let f = |n: u64| ctx.buffer(n.max(16) * 4); let h = |n: u64| ctx.buffer((n.max(32) * 2).next_multiple_of(16));
        // router weights as f16 [n_layer][n_exp + 1][dim]: expert rows then the shared-expert gate row
        let rows = c.n_exp + 1; let lw = (rows * c.dim) as usize;
        let rw = h((c.n_layer as usize * lw) as u64)?;
        { let fb = m.wt.f32b.as_f32_mut(); let dst = unsafe { std::slice::from_raw_parts_mut(rw.ptr as *mut half::f16, c.n_layer as usize * lw) };
          for l in 0..c.n_layer as usize { let o1 = m.wt.f(&format!("blk.{l}.ffn_gate_inp.weight"))? as usize; let o2 = m.wt.f(&format!("blk.{l}.ffn_gate_inp_shexp.weight"))? as usize;
            let d = c.dim as usize; let ex = c.n_exp as usize * d;
            for i in 0..ex { dst[l * lw + i] = half::f16::from_f32(fb[o1 + i]) }
            for i in 0..d { dst[l * lw + ex + i] = half::f16::from_f32(fb[o2 + i]) } } }
        let tab = 1 + 3 * (nk + ne);
        let qp = unsafe { ctx.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(6 * c.n_layer), None)? };
        Ok(Prefill { pf: Pf::new(ctx)?, mmax, al, ldp_max, qp,
            prope: ctx.kernel(spv!("mcv_prope"), 5, 52, &[c.hd], Some(32))?, psel: ctx.kernel(spv!("mcv_psel"), 5, 24, &[], Some(32))?, shadd: ctx.kernel(spv!("mcv_shadd"), 3, 16, &[], None)?,
            gscan: ctx.kernel(spv!("mcv_gscan"), 5, 40, &[env("PF_GSCAN_R", 0).max(16), env("PF_GSCAN_PM", 0)], None)?, gscan_r: env("PF_GSCAN_R", 0), gmv2: ctx.kernel(spv!("mcv_gmv2_swiglu"), 4, 28, &[env("PF_GMV2_T", 8)], None)?, gmvd: ctx.kernel(spv!("mcv_gmv_down"), 4, 24, &[env("PF_GMVD_T", 8)], Some(32))?, gscan4: ctx.kernel(spv!("mcv_gscan4"), 5, 40, &[env("PF_GSCAN_CH", 16)], Some(32))?, gmv: ctx.kernel(spv!("mcv_gmv_swiglu"), 4, 28, &[env("PF_GMV_T", 4), env("PF_GMV_NOLDS", 0), env("PF_GMV_RR", 1)], Some(32))?,
            x: f(mm * c.dim as u64)?, a: h((mm + 256) * al as u64)?, a2: h((mm + 256) * al as u64)?, p: f(mm * ldp_max as u64)?, cv: f(mm * conv_ch as u64)?,
            o: f(mm * (c.nv * c.d_state) as u64)?, pos: f(mm * 4)?, rw, rl: f(mm * LDR as u64)?, r: f(2 * nk)?, g: f(2 * nk)?, gt: f(tab)?, gt2: f(tab)?,
            ag: h((nk + 256) * c.dim as u64)?, hg: h((nk + 256) * c.ff_exp as u64)?, yd: f(nk * c.dim as u64)?, hsh: h((mm + 256) * c.ff_sh as u64)?, ysh: f(mm * c.dim as u64)? })
    }

    fn disp(&self, m: &Llm, cb: vk::CommandBuffer, k: &Kernel, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<()> {
        let ds = self.pf.ds.get(&m.ctx, k, bufs)?; m.ctx.dispatch(cb, k, ds, &push_u32(push), groups); Ok(())
    }

    /// Record the forward of rows [0, n) of self.x at KV positions pos0.. (rope positions in self.pos).
    fn record(&self, m: &Llm, cb: vk::CommandBuffer, n: u32, pos0: u32) -> Result<()> {
        let (c, ctx, wt, pf, al) = (&m.cfg, &m.ctx, &m.wt, &self.pf, self.al);
        let conv_ch = 2 * c.nk * c.d_state + c.nv * c.d_state; let ds_ = c.d_state;
        let bar = || ctx.barrier(cb);
        let tson = env("PF_TS", 0) > 0; let ts2 = env("PF_TS", 0) == 2; // PF_TS=2: gdn 'mixer' = input GEMMs only // per-phase GPU timestamps (single chunk only)
        if tson { unsafe { ctx.device.cmd_reset_query_pool(cb, self.qp, 0, 6 * c.n_layer) } }
        let ts = |l: usize, i: u32| if tson { unsafe { ctx.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.qp, l as u32 * 6 + i) } };
        let grid = |t: u32| t.div_ceil(256);
        // small-M prompts: 256-row tiles leave most CUs idle on N <= 8192 outputs; use narrower row tiles (PF_SMT tile code, PF_SMT_M max rows)
        let smt = if n <= env("PF_SMT_M", 160) { Some(env("PF_SMT", 0x800)) } else if n <= env("PF_SMT2_M", 320) { Some(0x2000) } else { None }; // 128-row tiles: pp226 1200 -> 1241
        let gemm_w = |w: WRef, a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> {
            pf.gemm_tile(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m: n, y, ldy, y_col, acc }, smt) };
        // split-K for the small-M dense GEMMs (too few output tiles to fill 40 CUs): PF_SPLITK slices, partials atomically added
        let sk = if smt.is_some() { env("PF_SPLITK", 1) } else { 1 };
        let gemm_k = |name: &str, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> { let w = wt.wref(name)?;
            pf.gemm_split(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &self.a, lda: al, a_row0: 0, m: n, y, ldy, y_col, acc }, smt, sk) };
        let zero_p = |ldp: u32| { if sk > 1 { unsafe { ctx.device.cmd_fill_buffer(cb, self.p.buf, 0, (n * ldp * 4) as u64, 0);
            let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            ctx.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &mb, &[], &[]); } } };
        let gemm = |name: &str, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> { gemm_w(wt.wref(name)?, &self.a, al, y, ldy, y_col, acc) };
        let skip = std::env::var("PF_SKIPK").unwrap_or_default(); // profiling: comma list of kernels to drop (results become wrong)
        let run = |k: &str, b: &[&Buffer], p: &[u32], g: u32| if skip.split(',').any(|x| x == k) { Ok(()) } else { pf.run(ctx, k, b, p, g, cb) };
        let norm = |wn: &str| -> Result<()> { run("rmsnorm", &[&self.x, &self.a, &wt.f32b], &[c.dim, al, wt.f(wn)?, c.eps.to_bits(), 0], n)?; bar(); Ok(()) };
        let nk = n * c.top_k; let d = c.dim; let fe = c.ff_exp;
        let wide = nk / c.n_exp >= env("PF_WIDE_RPE", 64);
        let (bm, bm2) = (env("PF_GBM", if wide { 32 } else { 16 }), env("PF_GBM2", 32)); // sweep: 16/32 best at pp91..482
        let gmr = if wide && bm == 32 { 0x8020 } else { bm }; let gmr2 = if wide && bm2 == 32 { 0x8020 } else { bm2 };
        // small prompts: multi-row GEMV for gate|up (mcv_gmv_swiglu, 8-row tiles) instead of the dequant-bound coopmat GEMM
        let (bm0, gmr0, bm20, gmr20) = (bm, gmr, bm2, gmr2);
        let gmv_all = n <= env("PF_GMV_M", 600);
        let gmvd_all = n <= env("PF_GMVD_M", 0);
        for l in 0..c.n_layer as usize {
            let nm = |x: &str| format!("blk.{l}.{x}");
            let li = m.lidx[l]; ts(l, 0);
            // per-layer MoE kernel choice: the GEMV paths handle Q4_K gate|up (K=2048) and Q6_K down (K=512) only
            let (gw0, dw0) = (wt.wref(&nm("ffn_gate_exps.weight"))?, wt.wref(&nm("ffn_down_exps.weight"))?);
            let gmv = gmv_all && gw0.ty == gguf::GgmlType::Q4K && gw0.k == 2048; let gmvd = gmvd_all && dw0.ty == gguf::GgmlType::Q6K && dw0.k == 512;
            let g2 = gmv && n > env("PF_GMV2_M", 160); // row-per-thread variant
            let (bm, gmr) = if g2 { (env("PF_GMV2_T", 8), 0) } else if gmv { (env("PF_GMV_T", 4), 0) } else { (bm0, gmr0) };
            let (bm2, gmr2) = if gmvd { (env("PF_GMVD_T", 8), 0) } else { (bm20, gmr20) };
            let (mt, mt2) = (nk / bm + c.n_exp, nk / bm2 + c.n_exp);
            if env("PF_PROF", 0) > 1 { eprintln!("  layer {l}: gate {:?} down {:?} gmv {gmv} gmvd {gmvd}", gw0.ty, dw0.ty); }
            for _ in 0..env("PF_XBAR", 0) { run("rmsnorm", &[&self.x, &self.a, &wt.f32b], &[c.dim, al, wt.f(&nm("attn_norm.weight"))?, c.eps.to_bits(), 0], 1)?; bar(); } // profiling: cost of a tiny dispatch + barrier
            norm(&nm("attn_norm.weight"))?;
            let part = env("PF_PART", 7);
            if part & 1 == 0 {} else if c.recr[l] {
                let ldp = conv_ch + c.nv * ds_ + 2 * c.nv; let (zo, ao, bo) = (conv_ch, conv_ch + c.nv * ds_, conv_ch + c.nv * ds_ + c.nv);
                zero_p(ldp);
                gemm_k(&nm("attn_qkv.weight"), &self.p, ldp, 0, false)?; gemm_k(&nm("attn_gate.weight"), &self.p, ldp, zo, false)?;
                gemm_k(&nm("ssm_alpha.weight"), &self.p, ldp, ao, false)?; gemm_k(&nm("ssm_beta.weight"), &self.p, ldp, bo, false)?; bar(); if ts2 { ts(l, 1) }
                let st_off = li * conv_ch * (c.d_conv - 1);
                run("gdn_conv", &[&self.p, &m.cst, &self.cv, &wt.f32b], &[c.nk, st_off, wt.f(&nm("ssm_conv1d.weight"))?, c.d_conv, c.eps.to_bits(), ldp, conv_ch, conv_ch / 128], n * (conv_ch / 128))?; bar();
                run("gdn_convst", &[&self.p, &m.cst], &[st_off, c.d_conv, ldp, n], conv_ch / 128)?;
                let sp = [c.nk, c.nv, li * c.nv * ds_ * ds_ / 4, ao, bo, wt.f(&nm("ssm_dt.bias"))?, wt.f(&nm("ssm_a"))?, ldp, conv_ch, n];
                for _ in 0..env("PF_XSCAN", 0) { run("gdn_scan", &[&self.cv, &self.p, &m.sst, &self.o, &wt.f32b], &sp, c.nv)?; bar(); } // profiling: extra scans
                if self.gscan_r == 4 { self.disp(m, cb, &self.gscan4, &[&self.cv, &self.p, &m.sst, &self.o, &wt.f32b], &sp, c.nv * 2)?; }
                else if self.gscan_r == 0 { if !skip.split(',').any(|x| x == "gdn_scan") { pf.gdn(ctx, &[&self.cv, &self.p, &m.sst, &self.o, &wt.f32b], &sp, cb)?; } }
                else if !skip.split(',').any(|x| x == "gdn_scan") { self.disp(m, cb, &self.gscan, &[&self.cv, &self.p, &m.sst, &self.o, &wt.f32b], &sp, c.nv * (128 / self.gscan_r))?; }
                bar();
                run("gnorm", &[&self.o, &self.p, &self.a, &wt.f32b], &[c.nv, ldp, zo, wt.f(&nm("ssm_norm.weight"))?, c.eps.to_bits(), al], n * c.nv)?; bar();
                gemm_k(&nm("ssm_out.weight"), &self.x, c.dim, 0, true)?; bar();
            } else {
                let (qn, kn) = (c.n_head * c.hd * 2, c.n_kv * c.hd); let ldp = qn + 2 * kn;
                let cache_off = li * c.max_ctx * c.n_kv * c.hd;
                gemm(&nm("attn_q.weight"), &self.p, ldp, 0, false)?; gemm(&nm("attn_k.weight"), &self.p, ldp, qn, false)?; gemm(&nm("attn_v.weight"), &self.p, ldp, qn + kn, false)?; bar();
                self.disp(m, cb, &self.prope, &[&self.p, &m.kc, &m.vc, &wt.f32b, &self.pos], &[c.n_head, c.n_kv, cache_off, c.theta.to_bits(), wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?,
                    c.eps.to_bits(), 2 * c.hd, qn, qn + kn, c.n_rot, pos0, ldp], n * (c.n_head + c.n_kv))?; bar();
                run("attn", &[&self.p, &m.kc, &m.vc, &self.a], &[c.n_head, c.n_kv, cache_off, 2 * c.hd, pos0, (1.0f32 / (c.hd as f32).sqrt()).to_bits(), ldp, al, n], n.div_ceil(16) * c.n_head)?; bar();
                gemm(&nm("attn_output.weight"), &self.x, c.dim, 0, true)?; bar();
            }
            if !(ts2 && c.recr[l]) { ts(l, 1); }
            if part & 6 == 0 { ts(l, 2); ts(l, 3); ts(l, 4); ts(l, 5); continue }
            // MoE FFN + shared expert, both from the normed rows in self.a
            norm(&nm("post_attention_norm.weight"))?;
            let rw = WRef { arena: 0, off: 0, ty: gguf::GgmlType::F16, k: d, n: c.n_exp + 1, e: 1, bytes: 0 };
            let rw = WRef { off: l as u64 * ((c.n_exp + 1) * d * 2) as u64, bytes: ((c.n_exp + 1) * d * 2) as u64, ..rw };
            pf.gemm_tile(ctx, cb, &self.rw, &wt.f32b, &GemmOp { w: rw, row0: 0, rows: rw.n, a: &self.a, lda: al, a_row0: 0, m: n, y: &self.rl, ldy: LDR, y_col: 0, acc: false }, smt)?;
            let (sg, su) = (wt.wref(&nm("ffn_gate_shexp.weight"))?, wt.wref(&nm("ffn_up_shexp.weight"))?);
            pf.gemm_swiglu_tile(ctx, cb, &wt.arenas[sg.arena], &wt.f32b, sg, su, 0, c.ff_sh, &self.a, al, n, &self.hsh, c.ff_sh, 0, smt.unwrap_or(0))?; bar();
            self.disp(m, cb, &self.psel, &[&self.rl, &self.r, &self.g, &self.gt, &self.gt2], &[n, c.top_k, c.n_exp, bm, bm2, LDR], 1)?;
            for _ in 0..env("PF_XPSEL", 0) { bar(); self.disp(m, cb, &self.psel, &[&self.rl, &self.r, &self.g, &self.gt, &self.gt2], &[n, c.top_k, c.n_exp, bm, bm2, LDR], 1)?; } // profiling: psel cost
            gemm_w(wt.wref(&nm("ffn_down_shexp.weight"))?, &self.hsh, c.ff_sh, &self.ysh, d, 0, false)?; bar();
            run("gather", &[&self.a, &self.ag, &self.g], &[d / 2, al / 2, d / 2, nk], grid(d / 2 * nk))?; bar(); ts(l, 2);
            let (gw, uw, dw) = (wt.wref(&nm("ffn_gate_exps.weight"))?, wt.wref(&nm("ffn_up_exps.weight"))?, wt.wref(&nm("ffn_down_exps.weight"))?);
            if g2 { self.disp(m, cb, &self.gmv2, &[&wt.arenas[gw.arena], &self.ag, &self.hg, &self.gt], &[gw.off as u32, uw.off as u32, (gw.bytes / gw.e as u64) as u32, fe, d, fe, mt], mt * fe.div_ceil(256))?; }
            else if gmv { anyhow::ensure!(gw.k == 2048 && fe % 64 == 0 && uw.off + uw.bytes <= u32::MAX as u64, "gmv: shape");
                self.disp(m, cb, &self.gmv, &[&wt.arenas[gw.arena], &self.ag, &self.hg, &self.gt], &[gw.off as u32, uw.off as u32, (gw.bytes / gw.e as u64) as u32, fe, d, fe, mt], mt * (fe / 64))?; }
            else { pf.gemm_grouped_swiglu(ctx, cb, &wt.arenas[gw.arena], &wt.f32b, gw, uw, &self.gt, gmr, mt, &self.ag, d, &self.hg, fe, 0)?; }
            bar(); ts(l, 3);
            if gmvd { anyhow::ensure!(dw.k == 512 && d % 64 == 0 && dw.off + dw.bytes <= u32::MAX as u64, "gmvd: shape");
                self.disp(m, cb, &self.gmvd, &[&wt.arenas[dw.arena], &self.hg, &self.yd, &self.gt2], &[dw.off as u32, (dw.bytes / dw.e as u64) as u32, d, fe, d, mt2], mt2 * (d / 64))?; }
            else { pf.gemm_grouped(ctx, cb, &wt.arenas[dw.arena], &wt.f32b, dw, &self.gt2, gmr2, mt2, &self.hg, fe, &self.yd, d, 0)?; }
            bar(); ts(l, 4);
            run("moecomb", &[&self.x, &self.yd, &self.r, &self.g], &[d, n, c.top_k], grid(d * n))?; bar(); // moecomb and shadd both RMW x
            self.disp(m, cb, &self.shadd, &[&self.x, &self.ysh, &self.rl], &[d, n, LDR, c.n_exp], grid(d * n))?; bar(); ts(l, 5);
        }
        Ok(())
    }

    /// Prefill a whole prompt (tokens / injected embeddings with M-RoPE positions), leave m ready for Llm::decode
    /// (toks[n] = greedy prediction, KV pos n, rope pos = KV index - (n - n_pos)). Returns (prediction, wall seconds).
    pub fn run(&self, m: &Llm, items: &[Item], mpos: &[(u32, u32, u32)], n_pos: u32) -> Result<(u32, f64)> {
        let (c, ctx, wt) = (&m.cfg, &m.ctx, &m.wt);
        let n = items.len();
        if n + 2 > c.max_ctx as usize { bail!("context too long") }
        let te = wt.g.tensor("token_embd.weight").ok_or_else(|| anyhow::anyhow!("no token_embd"))?;
        let data = wt.g.data(te); let d = c.dim as usize; let rb = data.len() / c.vocab as usize;
        let t0 = std::time::Instant::now();
        let mut s0 = 0usize;
        { let au = m.au(); au[128] = 0; au[129] = 0; }
        while s0 < n {
            let k = (n - s0).min(self.mmax as usize); let tst = std::time::Instant::now();
            let x = &mut self.x.as_f32_mut()[..k * d];
            let ps = unsafe { std::slice::from_raw_parts_mut(self.pos.ptr as *mut u32, k * 4) };
            for i in 0..k { let row = &mut x[i * d..(i + 1) * d];
                match &items[s0 + i] { Item::Tok(id) => dequant_row(te.ty, &data[*id as usize * rb..(*id as usize + 1) * rb], row), Item::Emb(e) => row.copy_from_slice(&e[..d]) }
                let (a, b, cc) = mpos[s0 + i]; ps[i * 4] = a; ps[i * 4 + 1] = b; ps[i * 4 + 2] = cc; }
            let last = s0 + k == n; if env("PF_PROF", 0) > 0 { eprintln!("  stage {:.1} ms", tst.elapsed().as_secs_f64() * 1e3); }
            let mut err = Ok(()); let tr = std::time::Instant::now();
            let cb = ctx.record_raw(|cb| { err = (|| -> Result<()> {
                if s0 == 0 { unsafe { // zero recurrent state on the GPU (a host memset of the 64 MB scan state costs ~35 ms)
                    for b in [&m.cst, &m.sst] { ctx.device.cmd_fill_buffer(cb, b.buf, 0, vk::WHOLE_SIZE, 0); }
                    let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                    ctx.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &mb, &[], &[]); } }
                self.record(m, cb, k as u32, s0 as u32)?;
                if last {
                    self.pf.run(ctx, "rmsnorm", &[&self.x, &self.a, &wt.f32b], &[c.dim, self.al, wt.f("output_norm.weight")?, c.eps.to_bits(), k as u32 - 1], 1, cb)?; ctx.barrier(cb);
                    let w = wt.wref("output.weight")?;
                    self.pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &self.a, lda: self.al, a_row0: 0, m: 1, y: &m.logits, ldy: w.n, y_col: 0, acc: false })?;
                    ctx.barrier(cb); let ds = m.set(&m.argmax, &[&m.logits, &m.aux])?; // GPU argmax (host reads of the 1 MB logits cost ~35 ms)
                    ctx.dispatch(cb, &m.argmax, ds, &push_u32(&[c.vocab]), 64);
                }
                Ok(()) })(); })?;
            err?; let trs = tr.elapsed().as_secs_f64(); let tg = std::time::Instant::now();
            ctx.submit_repeated(cb, 1)?;
            if env("PF_PROF", 0) > 0 { eprintln!("  chunk {s0}+{k}: record {:.1} ms, submit+wait {:.1} ms", trs * 1e3, tg.elapsed().as_secs_f64() * 1e3); }
            if env("PF_TS", 0) > 0 { let nl = c.n_layer as usize; let mut t = vec![0u64; 6 * nl];
                unsafe { ctx.device.get_query_pool_results(self.qp, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; }
                let mut acc = [0f64; 5]; let (mut mix_r, mut mix_a) = (0f64, 0f64);
                for l in 0..nl { for i in 0..5 { let v = (t[l * 6 + i + 1] - t[l * 6 + i]) as f64 * ctx.info.ts_period_ns as f64 * 1e-6; acc[i] += v; if i == 0 { if c.recr[l] { mix_r += v } else { mix_a += v } } } }
                let gaps: f64 = (1..nl).map(|l| (t[l * 6] - t[l * 6 - 1]) as f64 * ctx.info.ts_period_ns as f64 * 1e-6).sum();
                eprintln!("  TS ms: mixer {:.1} (gdn {:.1}, attn {:.1}) | router+shared+gather {:.1} | grouped gate|up {:.1} | grouped down {:.1} | comb {:.1} | gaps {:.1}", acc[0], mix_r, mix_a, acc[1], acc[2], acc[3], acc[4], gaps); }
            s0 += k;
        }
        let tam = std::time::Instant::now(); let pred = !m.au()[128]; // argmax_w key: u64 aux[64]
        let dt = t0.elapsed().as_secs_f64(); if env("PF_PROF", 0) > 0 { eprintln!("  argmax {:.1} ms, total {:.1} ms", tam.elapsed().as_secs_f64() * 1e3, dt * 1e3); }
        let (au, tk) = (m.au(), m.tk());
        tk[n] = pred; au[160] = n as u32; au[161] = n as u32; au[162] = 0; au[169] = 0;
        au[164] = n as u32 - n_pos; au[167] = 2;
        Ok((pred, dt))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Cross-prompt batched prefill (strix-serve1): one pass runs one chunk of items from each of several sequences.
// Row-wise work (norms, projection GEMMs, router, shared expert, grouped MoE, combine) runs once over the packed rows;
// DeltaNet conv/scan/gated norm and M-RoPE + attention run per segment on the scratch Prefill `t` (segment rows copied
// in / out) against that sequence's own state. Size-dependent kernel choices are pinned (no small-M tiles / split-K,
// no MoE GEMV, fixed MoE tiles) so a sequence's numerics do not depend on what it is packed with.
pub struct PSeg<'a> { pub items: &'a [Item<'a>], pub mpos: &'a [(u32, u32, u32)], pub pos0: u32, pub kc: &'a Buffer, pub vc: &'a Buffer, pub cst: &'a Buffer, pub sst: &'a Buffer, pub last: bool }
impl Prefill {
    fn xbar(&self, m: &Llm, cb: vk::CommandBuffer) {
        let a = vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ;
        let st = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let mb = [vk::MemoryBarrier::default().src_access_mask(a).dst_access_mask(a)];
        unsafe { m.ctx.device.cmd_pipeline_barrier(cb, st, st, vk::DependencyFlags::empty(), &mb, &[], &[]) };
    }
    fn cp(&self, m: &Llm, cb: vk::CommandBuffer, a: &Buffer, b: &Buffer, ao: u64, bo: u64, n: u64) {
        if n > 0 { unsafe { m.ctx.device.cmd_copy_buffer(cb, a.buf, b.buf, &[vk::BufferCopy { src_offset: ao, dst_offset: bo, size: n }]) } }
    }
    fn record_m(&self, t: &Prefill, m: &Llm, cb: vk::CommandBuffer, segs: &[PSeg], r0: &[u32], n: u32) -> Result<()> {
        let (c, ctx, wt, pf, al) = (&m.cfg, &m.ctx, &m.wt, &self.pf, self.al);
        let conv_ch = 2 * c.nk * c.d_state + c.nv * c.d_state; let ds_ = c.d_state;
        let bar = || ctx.barrier(cb);
        let grid = |t: u32| t.div_ceil(256);
        let gemm_w = |w: WRef, a: &Buffer, lda: u32, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> {
            pf.gemm_tile(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a, lda, a_row0: 0, m: n, y, ldy, y_col, acc }, None) };
        let gemm = |name: &str, y: &Buffer, ldy: u32, y_col: u32, acc: bool| -> Result<()> { gemm_w(wt.wref(name)?, &self.a, al, y, ldy, y_col, acc) };
        let run = |k: &str, b: &[&Buffer], p: &[u32], g: u32| pf.run(ctx, k, b, p, g, cb);
        let trun = |k: &str, b: &[&Buffer], p: &[u32], g: u32| t.pf.run(ctx, k, b, p, g, cb);
        let norm = |wn: &str| -> Result<()> { run("rmsnorm", &[&self.x, &self.a, &wt.f32b], &[c.dim, al, wt.f(wn)?, c.eps.to_bits(), 0], n)?; bar(); Ok(()) };
        let nk = n * c.top_k; let d = c.dim; let fe = c.ff_exp;
        let (bm, bm2) = (env("PF_GBM", 16), env("PF_GBM2", 32)); let (gmr, gmr2) = (bm, bm2);
        let (mt, mt2) = (nk / bm + c.n_exp, nk / bm2 + c.n_exp);
        if t.al != al { bail!("prefill scratch row stride mismatch") }
        for l in 0..c.n_layer as usize {
            let nm = |x: &str| format!("blk.{l}.{x}");
            let li = m.lidx[l];
            norm(&nm("attn_norm.weight"))?;
            if c.recr[l] {
                let ldp = conv_ch + c.nv * ds_ + 2 * c.nv; let (zo, ao, bo) = (conv_ch, conv_ch + c.nv * ds_, conv_ch + c.nv * ds_ + c.nv);
                gemm(&nm("attn_qkv.weight"), &self.p, ldp, 0, false)?; gemm(&nm("attn_gate.weight"), &self.p, ldp, zo, false)?;
                gemm(&nm("ssm_alpha.weight"), &self.p, ldp, ao, false)?; gemm(&nm("ssm_beta.weight"), &self.p, ldp, bo, false)?; self.xbar(m, cb);
                let st_off = li * conv_ch * (c.d_conv - 1);
                for (g, &r) in segs.iter().zip(r0) { let k = g.items.len() as u32;
                    self.cp(m, cb, &self.p, &t.p, r as u64 * ldp as u64 * 4, 0, k as u64 * ldp as u64 * 4); self.xbar(m, cb);
                    trun("gdn_conv", &[&t.p, g.cst, &t.cv, &wt.f32b], &[c.nk, st_off, wt.f(&nm("ssm_conv1d.weight"))?, c.d_conv, c.eps.to_bits(), ldp, conv_ch, conv_ch / 128], k * (conv_ch / 128))?; bar();
                    trun("gdn_convst", &[&t.p, g.cst], &[st_off, c.d_conv, ldp, k], conv_ch / 128)?;
                    let sp = [c.nk, c.nv, li * c.nv * ds_ * ds_ / 4, ao, bo, wt.f(&nm("ssm_dt.bias"))?, wt.f(&nm("ssm_a"))?, ldp, conv_ch, k];
                    if self.gscan_r == 4 { t.disp(m, cb, &t.gscan4, &[&t.cv, &t.p, g.sst, &t.o, &wt.f32b], &sp, c.nv * 2)?; }
                    else if self.gscan_r == 0 { t.pf.gdn(ctx, &[&t.cv, &t.p, g.sst, &t.o, &wt.f32b], &sp, cb)?; }
                    else { t.disp(m, cb, &t.gscan, &[&t.cv, &t.p, g.sst, &t.o, &wt.f32b], &sp, c.nv * (128 / self.gscan_r))?; }
                    bar();
                    trun("gnorm", &[&t.o, &t.p, &t.a, &wt.f32b], &[c.nv, ldp, zo, wt.f(&nm("ssm_norm.weight"))?, c.eps.to_bits(), al], k * c.nv)?; self.xbar(m, cb);
                    self.cp(m, cb, &t.a, &self.a, 0, r as u64 * al as u64 * 2, k as u64 * al as u64 * 2); self.xbar(m, cb); }
                gemm(&nm("ssm_out.weight"), &self.x, c.dim, 0, true)?; bar();
            } else {
                let (qn, kn) = (c.n_head * c.hd * 2, c.n_kv * c.hd); let ldp = qn + 2 * kn;
                let cache_off = li * c.max_ctx * c.n_kv * c.hd;
                gemm(&nm("attn_q.weight"), &self.p, ldp, 0, false)?; gemm(&nm("attn_k.weight"), &self.p, ldp, qn, false)?; gemm(&nm("attn_v.weight"), &self.p, ldp, qn + kn, false)?; self.xbar(m, cb);
                for (g, &r) in segs.iter().zip(r0) { let k = g.items.len() as u32;
                    self.cp(m, cb, &self.p, &t.p, r as u64 * ldp as u64 * 4, 0, k as u64 * ldp as u64 * 4);
                    self.cp(m, cb, &self.pos, &t.pos, r as u64 * 16, 0, k as u64 * 16); self.xbar(m, cb);
                    t.disp(m, cb, &t.prope, &[&t.p, g.kc, g.vc, &wt.f32b, &t.pos], &[c.n_head, c.n_kv, cache_off, c.theta.to_bits(), wt.f(&nm("attn_q_norm.weight"))?, wt.f(&nm("attn_k_norm.weight"))?,
                        c.eps.to_bits(), 2 * c.hd, qn, qn + kn, c.n_rot, g.pos0, ldp], k * (c.n_head + c.n_kv))?; bar();
                    trun("attn", &[&t.p, g.kc, g.vc, &t.a], &[c.n_head, c.n_kv, cache_off, 2 * c.hd, g.pos0, (1.0f32 / (c.hd as f32).sqrt()).to_bits(), ldp, al, k], k.div_ceil(16) * c.n_head)?; self.xbar(m, cb);
                    self.cp(m, cb, &t.a, &self.a, 0, r as u64 * al as u64 * 2, k as u64 * al as u64 * 2); self.xbar(m, cb); }
                gemm(&nm("attn_output.weight"), &self.x, c.dim, 0, true)?; bar();
            }
            norm(&nm("post_attention_norm.weight"))?;
            let rw = WRef { arena: 0, off: l as u64 * ((c.n_exp + 1) * d * 2) as u64, ty: gguf::GgmlType::F16, k: d, n: c.n_exp + 1, e: 1, bytes: ((c.n_exp + 1) * d * 2) as u64 };
            pf.gemm_tile(ctx, cb, &self.rw, &wt.f32b, &GemmOp { w: rw, row0: 0, rows: rw.n, a: &self.a, lda: al, a_row0: 0, m: n, y: &self.rl, ldy: LDR, y_col: 0, acc: false }, None)?;
            let (sg, su) = (wt.wref(&nm("ffn_gate_shexp.weight"))?, wt.wref(&nm("ffn_up_shexp.weight"))?);
            pf.gemm_swiglu_tile(ctx, cb, &wt.arenas[sg.arena], &wt.f32b, sg, su, 0, c.ff_sh, &self.a, al, n, &self.hsh, c.ff_sh, 0, 0)?; bar();
            self.disp(m, cb, &self.psel, &[&self.rl, &self.r, &self.g, &self.gt, &self.gt2], &[n, c.top_k, c.n_exp, bm, bm2, LDR], 1)?;
            gemm_w(wt.wref(&nm("ffn_down_shexp.weight"))?, &self.hsh, c.ff_sh, &self.ysh, d, 0, false)?; bar();
            run("gather", &[&self.a, &self.ag, &self.g], &[d / 2, al / 2, d / 2, nk], grid(d / 2 * nk))?; bar();
            let (gw, uw, dw) = (wt.wref(&nm("ffn_gate_exps.weight"))?, wt.wref(&nm("ffn_up_exps.weight"))?, wt.wref(&nm("ffn_down_exps.weight"))?);
            pf.gemm_grouped_swiglu(ctx, cb, &wt.arenas[gw.arena], &wt.f32b, gw, uw, &self.gt, gmr, mt, &self.ag, d, &self.hg, fe, 0)?; bar();
            pf.gemm_grouped(ctx, cb, &wt.arenas[dw.arena], &wt.f32b, dw, &self.gt2, gmr2, mt2, &self.hg, fe, &self.yd, d, 0)?; bar();
            run("moecomb", &[&self.x, &self.yd, &self.r, &self.g], &[d, n, c.top_k], grid(d * n))?; bar();
            self.disp(m, cb, &self.shadd, &[&self.x, &self.ysh, &self.rl], &[d, n, LDR, c.n_exp], grid(d * n))?; bar();
        }
        Ok(())
    }
    /// One packed pass over `segs` (sum of lengths <= self.mmax, each <= t.mmax); segments with pos0 == 0 get their
    /// recurrent state zeroed first. Returns (segment index, greedy prediction, logits) for every segment with `last`.
    pub fn run_multi(&self, t: &Prefill, m: &Llm, segs: &[PSeg]) -> Result<Vec<(usize, u32, Vec<f32>)>> {
        let (c, ctx, wt) = (&m.cfg, &m.ctx, &m.wt);
        let nt: usize = segs.iter().map(|g| g.items.len()).sum();
        if nt == 0 || nt > self.mmax as usize || segs.iter().any(|g| g.items.len() > t.mmax as usize) { bail!("run_multi: {nt} rows") }
        let te = wt.g.tensor("token_embd.weight").ok_or_else(|| anyhow::anyhow!("no token_embd"))?;
        let data = wt.g.data(te); let d = c.dim as usize; let rb = data.len() / c.vocab as usize;
        { let x = &mut self.x.as_f32_mut()[..nt * d];
          let ps = unsafe { std::slice::from_raw_parts_mut(self.pos.ptr as *mut u32, nt * 4) };
          let mut i = 0;
          for g in segs { for (it, &(a, b, cc)) in g.items.iter().zip(g.mpos) { let row = &mut x[i * d..(i + 1) * d];
              match it { Item::Tok(id) => dequant_row(te.ty, &data[*id as usize * rb..(*id as usize + 1) * rb], row), Item::Emb(e) => row.copy_from_slice(&e[..d]) }
              ps[i * 4] = a; ps[i * 4 + 1] = b; ps[i * 4 + 2] = cc; i += 1; } } }
        let r0: Vec<u32> = segs.iter().scan(0u32, |a, g| { let r = *a; *a += g.items.len() as u32; Some(r) }).collect();
        let mut err = Ok(());
        ctx.run(|cb| { err = (|| -> Result<()> {
            let mut z = false;
            for g in segs { if g.pos0 == 0 { z = true; unsafe { for b in [g.cst, g.sst] { ctx.device.cmd_fill_buffer(cb, b.buf, 0, vk::WHOLE_SIZE, 0); } } } }
            if z { self.xbar(m, cb); }
            self.record_m(t, m, cb, segs, &r0, nt as u32) })(); })?;
        err?;
        let mut out = vec![];
        for (i, (g, &r)) in segs.iter().zip(&r0).enumerate() { if !g.last { continue }
            let mut err = Ok(());
            ctx.run(|cb| { err = (|| -> Result<()> {
                self.pf.run(ctx, "rmsnorm", &[&self.x, &self.a, &wt.f32b], &[c.dim, self.al, wt.f("output_norm.weight")?, c.eps.to_bits(), r + g.items.len() as u32 - 1], 1, cb)?; ctx.barrier(cb);
                let w = wt.wref("output.weight")?;
                self.pf.gemm(ctx, cb, &wt.arenas[w.arena], &wt.f32b, &GemmOp { w, row0: 0, rows: w.n, a: &self.a, lda: self.al, a_row0: 0, m: 1, y: &m.logits, ldy: w.n, y_col: 0, acc: false }) })(); })?;
            err?;
            let lg = m.logits.as_f32()[..c.vocab as usize].to_vec();
            let am = lg.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
            out.push((i, am as u32, lg)); }
        Ok(out)
    }
}
