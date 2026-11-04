//! Reflection Beam-1.0-501B-A23B (NVFP4 safetensors) decode with disk-streamed routed experts.
//! Resident (GPU-visible UMA memory): attention + dense-MLP BF16 weights, router, norms, lm_head. Streamed per token per layer:
//! the 8 selected NVFP4 experts (gate/up/down, ~22 MB each) are pread from NVMe (page cache) into slot buffers right after the router.
//! One host sync per MoE layer: submit [experts(L-1) + attention(L) + router(L)], read router logits, pick top-8, stream experts.
use anyhow::{anyhow, bail, Result};
use ash::vk;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex, mpsc};
extern "C" { fn readahead(fd: i32, offset: i64, count: usize) -> isize; fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32; }
use vkrt::{push_u32, shaders, Buffer, Ctx, Kernel};

macro_rules! spv { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv")) } }

pub const NL: usize = 52; pub const DIM: usize = 5120; pub const NEXP: usize = 256; pub const TOPK: usize = 8;
pub const NH: usize = 48; pub const NKV: usize = 8; pub const HD: usize = 128; pub const VOCAB: usize = 151936;
pub const FF_DENSE: usize = 20480; pub const FF_EXP: usize = 2560; pub const N_DENSE: usize = 3;
const EPS: f32 = 1e-6;
pub const NSLOT: usize = 128; pub const PF_T: usize = 16;
// slot layout (bytes): down w | gate w | up w | down s | gate s | up s
const WP: usize = DIM * (FF_EXP / 2);               // down weight bytes = 5120*1280
const SP: usize = DIM * (FF_EXP / 16);              // down scale bytes = 5120*160
const SLOT: usize = 3 * WP + 3 * SP;                // 22,118,400

pub fn is_full(l: usize) -> bool { matches!(l, 5 | 11 | 17 | 23 | 29 | 35 | 41 | 47 | 51) }

pub struct Ent { pub file: usize, pub off: u64, pub len: u64, pub dtype: String, pub dims: Vec<u64> }
pub struct Store { pub files: Vec<File>, pub ents: HashMap<String, Ent> }
impl Store {
    pub fn open(dir: &str) -> Result<Self> {
        let idx = std::fs::read_to_string(format!("{dir}/beam.idx")).map_err(|e| anyhow!("beam.idx missing in {dir} (run tools/beam_index.py): {e}"))?;
        let (mut files, mut fidx, mut ents) = (vec![], HashMap::<String, usize>::new(), HashMap::new());
        for l in idx.lines() {
            let p: Vec<&str> = l.split(' ').collect(); if p.len() < 6 { continue }
            let f = *fidx.entry(p[2].to_string()).or_insert_with(|| { files.push(File::open(format!("{dir}/{}", p[2])).expect("open shard")); files.len() - 1 });
            ents.insert(p[0].to_string(), Ent { file: f, off: p[3].parse()?, len: p[4].parse()?, dtype: p[1].to_string(), dims: p[5].split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect() });
        }
        Ok(Store { files, ents })
    }
    pub fn ent(&self, n: &str) -> Result<&Ent> { self.ents.get(n).ok_or_else(|| anyhow!("tensor {n} not found")) }
    pub fn read_into(&self, n: &str, dst: &mut [u8]) -> Result<()> { let e = self.ent(n)?; if dst.len() as u64 != e.len { bail!("{n}: size {} != {}", e.len, dst.len()) } self.files[e.file].read_exact_at(dst, e.off)?; Ok(()) }
    pub fn f32s(&self, n: &str) -> Result<Vec<f32>> {
        let e = self.ent(n)?; let mut b = vec![0u8; e.len as usize]; self.files[e.file].read_exact_at(&mut b, e.off)?;
        Ok(match e.dtype.as_str() {
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            d => bail!("dtype {d}") })
    }
}

struct LayerW { attn: Buffer, mlp: Option<Buffer>, f_in: u32, f_pa: u32, f_pf: u32, f_po: u32, f_qn: u32, f_kn: u32, f_gb: u32 }
#[derive(Clone, Copy)] struct ExpLoc { file: usize, w: u64, s: u64, ws2: [f32; 3] }

#[derive(Default, Clone, Debug)]
pub struct Stats { pub gpu_s: f64, pub disk_s: f64, pub cpu_s: f64, pub bytes: u64, pub wall_s: f64 }

pub struct PfStats { pub hit: [u64; 4], pub tot: [u64; 4], pub issued: u64 }
struct PfShared { router: Vec<Vec<f32>>, bias: Vec<Vec<f32>>, fnorm: Vec<f32>, pf_off: Vec<u32>, exps: Vec<Vec<ExpLoc>>, fds: Vec<i32>, pred: Mutex<HashMap<(usize, usize), Vec<usize>>>, top: usize, issued: std::sync::atomic::AtomicU64 }
struct PfJob { l: usize, x: Vec<f32> }

pub struct Model {
    pub ctx: Ctx, pub st: Store, k: Vec<Kernel>, lw: Vec<LayerW>, f32b: Buffer, rtr: Buffer, rbias: Vec<Vec<f32>>, head: Buffer, f_final: u32,
    x: Buffer, xn: Buffer, br: Buffer, qkv: Buffer, ao: Buffer, g: Buffer, u: Buffer, h: Buffer, rl: Buffer, meta: Buffer, slots: Buffer, logits: Buffer, kc: Buffer, vc: Buffer,
    exps: Vec<Vec<ExpLoc>>, embed: (usize, u64), pub max_ctx: usize, dsc: RefCell<HashMap<(usize, Vec<u64>), vk::DescriptorSet>>, pub stats: RefCell<Stats>, groups: bool,
    pub trace: RefCell<Vec<(usize, [usize; 8])>>,
    dfiles: Option<Vec<File>>, file_len: Vec<u64>, qfiles: Vec<File>, qmode: bool, wp: usize, sp: usize, slot: usize, act: RefCell<Vec<f64>>, actn: RefCell<Vec<u64>>, ebits: Vec<Vec<u8>>, mixed: bool,
    cache: Vec<Buffer>, ck: usize, cstate: RefCell<Vec<Vec<(usize, u64)>>>, ctick: RefCell<u64>, chm: RefCell<[u64; 2]>,
    pf: Option<(mpsc::Sender<PfJob>, Arc<PfShared>, usize)>, pub pfs: RefCell<PfStats>,
}
// kernel ids
const K_GEMV: usize = 0; const K_RMS: usize = 1; const K_RMSRES: usize = 2; const K_QKV: usize = 3; const K_ATTN: usize = 4; const K_RLOG: usize = 5; const K_SWIGLU: usize = 6; const K_GU: usize = 7; const K_DN: usize = 8;

impl Model {
    pub fn load(dir: &str, max_ctx: usize) -> Result<Self> {
        if max_ctx > 4096 { bail!("max_ctx <= 4096 (attention kernel keeps all scores in shared memory)") }
        let ctx = Ctx::new()?; let st = Store::open(dir)?;
        let t0 = std::time::Instant::now();
        let k = vec![
            ctx.kernel(spv!("bm_gemv"), 3, 24, &[], None)?, ctx.kernel(shaders::RMSNORM, 3, 12, &[], None)?, ctx.kernel(spv!("bm_rmsres"), 3, 20, &[], None)?,
            ctx.kernel(spv!("bm_qkv"), 4, 32, &[], None)?, ctx.kernel(spv!("bm_attn"), 5, 20, &[], None)?, ctx.kernel(spv!("bm_rlog"), 3, 12, &[], Some(32))?,
            ctx.kernel(spv!("bm_swiglu"), 2, 4, &[], None)?, ctx.kernel(spv!("bm_fp4gu"), 5, 28, &[], Some(32))?, ctx.kernel(spv!("bm_fp4dn"), 5, 28, &[], Some(32))?,
        ];
        // f32 small tensors
        let mut f: Vec<f32> = vec![]; let mut lw = vec![];
        let push = |f: &mut Vec<f32>, v: Vec<f32>| { let o = f.len() as u32; f.extend(v); o };
        let sz = |v: u64| v;
        let nl: usize = std::env::var("BEAM_NL").ok().and_then(|v| v.parse().ok()).unwrap_or(NL);
        for l in 0..nl {
            let p = format!("model.layers.{l}.");
            let (f_in, f_pa, f_pf, f_po) = (push(&mut f, st.f32s(&format!("{p}input_layernorm.weight"))?), push(&mut f, st.f32s(&format!("{p}post_attention_layernorm.weight"))?),
                push(&mut f, st.f32s(&format!("{p}pre_feedforward_layernorm.weight"))?), push(&mut f, st.f32s(&format!("{p}post_feedforward_layernorm.weight"))?));
            let (f_qn, f_kn, f_gb) = (push(&mut f, st.f32s(&format!("{p}self_attn.q_norm.weight"))?), push(&mut f, st.f32s(&format!("{p}self_attn.k_norm.weight"))?), push(&mut f, st.f32s(&format!("{p}self_attn.gate_bias"))?));
            let attn = ctx.buffer(sz(13107200 * 16))?;
            let ab = attn.bytes_mut(); let mut o = 0usize;
            for n in ["q_proj", "gate_proj", "k_proj", "v_proj", "o_proj"] { let e = st.ent(&format!("{p}self_attn.{n}.weight"))?.len as usize; st.read_into(&format!("{p}self_attn.{n}.weight"), &mut ab[o..o + e])?; o += e; }
            assert_eq!(o, ab.len());
            let mlp = if l < N_DENSE { let b = ctx.buffer(629145600)?; let mb = b.bytes_mut(); let mut o = 0usize;
                for n in ["gate_proj", "up_proj", "down_proj"] { let nm = format!("{p}mlp.{n}.weight"); let e = st.ent(&nm)?.len as usize; st.read_into(&nm, &mut mb[o..o + e])?; o += e; } Some(b) } else { None };
            lw.push(LayerW { attn, mlp, f_in, f_pa, f_pf, f_po, f_qn, f_kn, f_gb });
            if l % 10 == 0 { eprintln!("[load] layer {l} {:.1}s", t0.elapsed().as_secs_f64()); }
        }
        let f_final = push(&mut f, st.f32s("model.norm.weight")?);
        let f32b = ctx.buffer(f.len() as u64 * 4)?; f32b.as_f32_mut().copy_from_slice(&f); let fcopy = f.clone(); let mut rcpu: Vec<Vec<f32>> = vec![vec![]; NL];
        let rtr = ctx.buffer((NL - N_DENSE) as u64 * (NEXP * DIM) as u64 * 4)?; let mut rbias = vec![vec![]; NL]; let mut exps = vec![vec![]; NL];
        for l in N_DENSE..nl {
            let w = st.f32s(&format!("model.layers.{l}.mlp.gate.weight"))?; rcpu[l] = w.clone(); let o = (l - N_DENSE) * NEXP * DIM; rtr.as_f32_mut()[o..o + w.len()].copy_from_slice(&w);
            rbias[l] = st.f32s(&format!("model.layers.{l}.mlp.gate.e_score_correction_bias"))?;
            for e in 0..NEXP {
                let q = |p: &str, s: &str| format!("model.layers.{l}.mlp.experts.{e}.{p}_proj.{s}");
                let (d, g, u) = (st.ent(&q("down", "weight"))?, st.ent(&q("gate", "weight"))?, st.ent(&q("up", "weight"))?);
                let (ds, gs, us) = (st.ent(&q("down", "weight_scale"))?, st.ent(&q("gate", "weight_scale"))?, st.ent(&q("up", "weight_scale"))?);
                if d.len as usize != WP || g.off != d.off + WP as u64 || u.off != g.off + WP as u64 || ds.len as usize != SP || gs.off != ds.off + SP as u64 || us.off != gs.off + SP as u64 { bail!("expert {l}.{e} not contiguous (down,gate,up)") }
                let r = |p: &str| -> Result<f32> { Ok(st.f32s(&q(p, "weight_scale_2"))?[0]) };
                exps[l].push(ExpLoc { file: d.file, w: d.off, s: ds.off, ws2: [r("gate")?, r("up")?, r("down")?] });
            }
        }
        eprintln!("[load] router+expert index {:.1}s", t0.elapsed().as_secs_f64());
        let head = ctx.buffer((VOCAB * DIM * 2) as u64)?; st.read_into("lm_head.weight", head.bytes_mut())?;
        let em = st.ent("model.embed_tokens.weight")?; let embed = (em.file, em.off);
        let pf_depth: usize = std::env::var("BEAM_PF_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let pf = if pf_depth > 0 {
            let sh = Arc::new(PfShared { router: rcpu, bias: rbias.clone(), fnorm: fcopy, pf_off: lw.iter().map(|w| w.f_pf).collect(), exps: exps.clone(), fds: st.files.iter().map(|f| f.as_raw_fd()).collect(), pred: Mutex::new(HashMap::new()),
                top: std::env::var("BEAM_PF_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(12), issued: Default::default() });
            let (tx, rx) = mpsc::channel::<PfJob>(); let rx = Arc::new(Mutex::new(rx));
            for _ in 0..2 { let (rx, sh) = (rx.clone(), sh.clone()); std::thread::spawn(move || loop {
                let job = match rx.lock().unwrap().recv() { Ok(j) => j, Err(_) => break };
                pf_predict(&sh, job, pf_depth); }); }
            Some((tx, sh, pf_depth))
        } else { None };
        if std::env::var("BEAM_EVICT").is_ok() { for f in &st.files { unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4); } } eprintln!("[load] page cache evicted (fadvise DONTNEED)"); }
        let dfiles = if std::env::var("BEAM_DIRECT").is_ok() { use std::os::unix::fs::OpenOptionsExt; let idx = std::fs::read_to_string(format!("{dir}/beam.idx"))?; let mut names: Vec<String> = vec![]; for l in idx.lines() { let n = l.split(' ').nth(2).unwrap().to_string(); if !names.contains(&n) { names.push(n) } }
            // same file order as Store::open (first appearance in beam.idx)
            Some(names.iter().map(|n| std::fs::OpenOptions::new().read(true).custom_flags(0x4000).open(format!("{dir}/{n}")).unwrap()).collect::<Vec<_>>()) } else { None };
        let file_len: Vec<u64> = st.files.iter().map(|f| f.metadata().map(|m| m.len()).unwrap_or(0)).collect();
        let qbits: u32 = qdir().and_then(|d| d.rsplit('q').next().and_then(|b| b.trim_matches('/').parse().ok())).unwrap_or(0);
        let qhi: Option<String> = std::env::var("BEAM_QHI").ok().map(|d| d.replace('~', &std::env::var("HOME").unwrap_or_default()));
        let hb: u32 = qhi.as_ref().and_then(|d| d.trim_end_matches('/').rsplit('q').next().and_then(|b| b.parse().ok())).unwrap_or(qbits); let mixed = qhi.is_some() && qbits > 0;
        let sbits = if mixed { qbits.max(hb) } else { qbits };
        let (wp, sp) = if qbits > 0 { (13107200 * sbits as usize / 8, 13107200 / 32 * 2) } else { (WP, SP) }; let slot = 3 * wp + 3 * sp;
        let abm: u32 = std::env::var("BEAM_ABM").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let ck: usize = std::env::var("BEAM_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let cache: Vec<Buffer> = if ck == 0 { vec![] } else { (0..NL).map(|l| ctx.buffer(if l >= N_DENSE && l < nl { (ck * slot) as u64 } else { 16 })).collect::<Result<Vec<_>>>()? };
        if ck > 0 { eprintln!("[load] expert cache {ck} slots/layer = {:.1} GB", (ck * slot * (nl - N_DENSE)) as f64 / 1e9); }
        let mut qfiles: Vec<File> = vec![]; let mut exps = exps; let mut ebits: Vec<Vec<u8>> = vec![vec![qbits as u8; NEXP]; NL];
        if qbits > 0 { let qd = qdir().unwrap(); let qd = qd.replace('~', &std::env::var("HOME").unwrap_or_default());
            for l in 0..NL { if l < N_DENSE || l >= nl { qfiles.push(File::open("/dev/null")?); continue }
                qfiles.push(File::open(format!("{qd}/L{l:02}.bin"))?); }
            if let Some(h) = &qhi { for l in 0..NL { if l < N_DENSE || l >= nl { qfiles.push(File::open("/dev/null")?) } else { qfiles.push(File::open(format!("{h}/L{l:02}.bin"))?) } } }
            let mut hot: Vec<Vec<bool>> = vec![vec![false; NEXP]; NL];
            if mixed { let hf = std::env::var("BEAM_QHOT").expect("BEAM_QHOT"); for line in std::fs::read_to_string(&hf)?.lines() { let f: Vec<usize> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect(); if f.len() > 1 { for &e in &f[1..] { hot[f[0]][e] = true; } } } }
            let (wl, wh) = (13107200 * qbits as usize / 8, 13107200 * hb as usize / 8); let (slo, shi) = (3 * wl + 3 * 819200, 3 * wh + 3 * 819200);
            for l in N_DENSE..nl.min(NL) { ebits[l] = (0..NEXP).map(|e| if mixed && hot[l][e] { hb as u8 } else { qbits as u8 }).collect();
                exps[l] = (0..NEXP).map(|e| if mixed && hot[l][e] { ExpLoc { file: NL + l, w: (e * shi) as u64, s: (e * shi + 3 * wh) as u64, ws2: [1.0; 3] } } else { ExpLoc { file: l, w: (e * slo) as u64, s: (e * slo + 3 * wl) as u64, ws2: [1.0; 3] } }).collect(); }
            if mixed { let nh: usize = hot.iter().map(|v| v.iter().filter(|&&x| x).count()).sum(); eprintln!("[load] mixed precision: {nh} hot experts at {hb}-bit, rest {qbits}-bit"); }
            eprintln!("[load] quantized experts {qbits}-bit from {qd}: {slot} B/expert ({:.3} of NVFP4)", slot as f64 / SLOT as f64); }
        let mut k = k; if qbits > 0 { k[K_GU] = ctx.kernel(spv!("bm_qgu"), 5, 60, &[qbits, mixed as u32, abm, cblut().is_some() as u32], Some(32))?; k[K_DN] = ctx.kernel(spv!("bm_qdn"), 5, 60, &[qbits, mixed as u32, abm, cblut().is_some() as u32], Some(32))?; }
        let (dfiles, file_len) = if qbits > 0 { use std::os::unix::fs::OpenOptionsExt; let qd = qdir().unwrap().replace('~', &std::env::var("HOME").unwrap_or_default());
            let mut paths: Vec<String> = (0..NL).map(|l| if l < N_DENSE || l >= nl { "/dev/null".into() } else { format!("{qd}/L{l:02}.bin") }).collect();
            if let Some(h) = &qhi { paths.extend((0..NL).map(|l| if l < N_DENSE || l >= nl { "/dev/null".to_string() } else { format!("{h}/L{l:02}.bin") })); }
            let fl = qfiles.iter().map(|f| f.metadata().map(|m| m.len()).unwrap_or(0)).collect();
            (if std::env::var("BEAM_DIRECT").is_ok() { Some(paths.iter().map(|p| if p == "/dev/null" { File::open(p).unwrap() } else { std::fs::OpenOptions::new().read(true).custom_flags(0x4000).open(p).unwrap() }).collect()) } else { None }, fl) } else { (dfiles, file_len) };
        let fb = |n: usize| ctx.buffer((n * 4).max(64) as u64);
        let m = Model { x: fb(DIM)?, xn: fb(DIM)?, br: fb(DIM)?, qkv: fb(NH * HD * 2 + NKV * HD * 2)?, ao: fb(NH * HD)?, g: fb(FF_DENSE)?, u: fb(FF_DENSE)?, h: fb(TOPK * FF_EXP)?,
            rl: ctx.buffer_readback(NEXP as u64 * 4)?, meta: fb(TOPK * 8)?, slots: ctx.buffer((slot * NSLOT) as u64)?, logits: ctx.buffer_readback((VOCAB * 4) as u64)?,
            kc: ctx.buffer((NL * max_ctx * NKV * HD * 4) as u64)?, vc: ctx.buffer((NL * max_ctx * NKV * HD * 4) as u64)?,
            k, lw, f32b, rtr, rbias, head, f_final, exps, embed, max_ctx, dsc: Default::default(), stats: Default::default(), groups: std::env::var("BEAM_GROUPS").is_ok(), trace: Default::default(), dfiles, file_len, qfiles, qmode: qbits > 0, wp, sp, slot, act: RefCell::new(if std::env::var("BEAM_ACTOUT").is_ok() { vec![0.0; NL * (DIM + FF_EXP)] } else { vec![] }), actn: RefCell::new(vec![0; 2 * NL]), ebits, mixed, cache, ck, cstate: RefCell::new(vec![vec![(0, 0); ck]; NL]), ctick: Default::default(), chm: Default::default(), pf, pfs: RefCell::new(PfStats { hit: [0; 4], tot: [0; 4], issued: 0 }), st, ctx };
        eprintln!("[load] done {:.1}s", t0.elapsed().as_secs_f64());
        Ok(m)
    }
    fn ds(&self, k: usize, bufs: &[&Buffer]) -> Result<vk::DescriptorSet> {
        use ash::vk::Handle;
        let key = (k, bufs.iter().map(|b| b.buf.as_raw()).collect::<Vec<_>>());
        if let Some(d) = self.dsc.borrow().get(&key) { return Ok(*d) }
        let d = self.ctx.bind(&self.k[k], bufs)?; self.dsc.borrow_mut().insert(key, d); Ok(d)
    }
    fn bar(&self, cb: vk::CommandBuffer) { self.ctx.barrier(cb) }
    #[allow(clippy::too_many_arguments)]
    fn gemv(&self, cb: vk::CommandBuffer, w: &Buffer, w_off4: u32, n: usize, k: usize, x: &Buffer, x_off: u32, y: &Buffer, y_off: u32, acc: bool) -> Result<()> {
        let ds = self.ds(K_GEMV, &[w, x, y])?;
        self.ctx.dispatch(cb, &self.k[K_GEMV], ds, &push_u32(&[n as u32, k as u32, w_off4, x_off, y_off, acc as u32]), n as u32); Ok(())
    }
    fn rms(&self, cb: vk::CommandBuffer, x: &Buffer, y: &Buffer, w_off: u32) -> Result<()> {
        let ds = self.ds(K_RMS, &[x, y, &self.f32b])?; self.ctx.dispatch(cb, &self.k[K_RMS], ds, &push_u32(&[DIM as u32, w_off, EPS.to_bits()]), 1); self.bar(cb); Ok(())
    }
    fn rmsres(&self, cb: vk::CommandBuffer, w_off: u32, i: usize) -> Result<()> {
        let a = (1.0 / (i as f64).sqrt()) as f32; let b = (((i - 1) as f64) / (i as f64)).sqrt() as f32;
        let ds = self.ds(K_RMSRES, &[&self.br, &self.x, &self.f32b])?;
        self.ctx.dispatch(cb, &self.k[K_RMSRES], ds, &push_u32(&[DIM as u32, w_off, EPS.to_bits(), a.to_bits(), b.to_bits()]), 1); self.bar(cb); Ok(())
    }
    fn rec_attn(&self, cb: vk::CommandBuffer, l: usize, pos: usize) -> Result<()> {
        let w = &self.lw[l]; let c = &self.ctx;
        self.rms(cb, &self.x, &self.xn, w.f_in)?;
        let (qo, go, ko, vo, oo) = (0u32, 3932160u32, 7864320u32, 8519680u32, 9175040u32);
        self.gemv(cb, &w.attn, qo, NH * HD, DIM, &self.xn, 0, &self.qkv, 0, false)?;
        self.gemv(cb, &w.attn, ko, NKV * HD, DIM, &self.xn, 0, &self.qkv, 6144, false)?;
        self.gemv(cb, &w.attn, vo, NKV * HD, DIM, &self.xn, 0, &self.qkv, 7168, false)?;
        self.gemv(cb, &w.attn, go, NH * HD, DIM, &self.xn, 0, &self.qkv, 8192, false)?; self.bar(cb);
        let full = is_full(l); let kv_off = (l * self.max_ctx * NKV * HD) as u32;
        let qscale = if full { ((0.1f64) * (((pos + 1) as f64) / 2048.0).ln_1p()).exp() as f32 } else { 1.0 };
        let ds = self.ds(K_QKV, &[&self.qkv, &self.kc, &self.vc, &self.f32b])?;
        c.dispatch(cb, &self.k[K_QKV], ds, &push_u32(&[pos as u32, (!full) as u32, qscale.to_bits(), kv_off, w.f_qn, w.f_kn, 10000.0f32.to_bits(), EPS.to_bits()]), 64); self.bar(cb);
        let lo = if full { 0 } else { pos.saturating_sub(511) } as u32;
        let ds = self.ds(K_ATTN, &[&self.qkv, &self.kc, &self.vc, &self.ao, &self.f32b])?;
        c.dispatch(cb, &self.k[K_ATTN], ds, &push_u32(&[pos as u32, lo, kv_off, w.f_gb, (1.0f32 / (HD as f32).sqrt()).to_bits()]), NH as u32); self.bar(cb);
        self.gemv(cb, &w.attn, oo, DIM, NH * HD, &self.ao, 0, &self.br, 0, false)?; self.bar(cb);
        self.rmsres(cb, w.f_pa, 2 * (l + 1))?;
        self.rms(cb, &self.x, &self.xn, w.f_pf)?;
        if let Some(m) = &w.mlp {
            let (gu, up, dn) = (0u32, 13107200u32, 26214400u32);
            self.gemv(cb, m, gu, FF_DENSE, DIM, &self.xn, 0, &self.g, 0, false)?; self.gemv(cb, m, up, FF_DENSE, DIM, &self.xn, 0, &self.u, 0, false)?; self.bar(cb);
            let ds = self.ds(K_SWIGLU, &[&self.g, &self.u])?; c.dispatch(cb, &self.k[K_SWIGLU], ds, &push_u32(&[FF_DENSE as u32]), (FF_DENSE as u32).div_ceil(256)); self.bar(cb);
            self.gemv(cb, m, dn, DIM, FF_DENSE, &self.g, 0, &self.br, 0, false)?; self.bar(cb);
            self.rmsres(cb, w.f_po, 2 * (l + 1) + 1)?;
        } else {
            let ds = self.ds(K_RLOG, &[&self.f32b_router(), &self.xn, &self.rl])?;
            c.dispatch(cb, &self.k[K_RLOG], ds, &push_u32(&[((l - N_DENSE) * NEXP * DIM) as u32, DIM as u32, NEXP as u32]), (NEXP / 8) as u32); self.bar(cb);
        }
        Ok(())
    }
    fn f32b_router(&self) -> &Buffer { &self.rtr }
    fn rec_exp(&self, cb: vk::CommandBuffer, l: usize, cached: bool) -> Result<()> {
        let c = &self.ctx; let sb = if cached && self.ck > 0 { &self.cache[l] } else { &self.slots };
        let mut pc = push_u32(&[0, 0, self.slot as u32, self.wp as u32, (3 * self.wp) as u32, self.sp as u32, TOPK as u32]);
        let lut: [f32; 8] = cblut().map(|t| t[l]).unwrap_or([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]); for v in lut { pc.extend_from_slice(&v.to_le_bytes()); }
        let mut p = pc.clone(); p[0..4].copy_from_slice(&(FF_EXP as u32).to_le_bytes()); p[4..8].copy_from_slice(&(DIM as u32).to_le_bytes());
        let ds = self.ds(K_GU, &[sb, sb, &self.xn, &self.h, &self.meta])?;
        c.dispatch3(cb, &self.k[K_GU], ds, &p, (FF_EXP / 8) as u32, TOPK as u32, 1); self.bar(cb);
        let mut p = pc; p[0..4].copy_from_slice(&(DIM as u32).to_le_bytes()); p[4..8].copy_from_slice(&(FF_EXP as u32).to_le_bytes());
        let ds = self.ds(K_DN, &[sb, sb, &self.h, &self.br, &self.meta])?;
        c.dispatch(cb, &self.k[K_DN], ds, &p, (DIM / 8) as u32); self.bar(cb);
        self.rmsres(cb, self.lw[l].f_po, 2 * (l + 1) + 1)
    }
    fn select(&self, l: usize) -> ([usize; TOPK], [f32; TOPK]) {
        let lg = &self.rl.as_f32()[..NEXP]; let b = &self.rbias[l];
        let p: Vec<f32> = lg.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let mut s: Vec<f32> = (0..NEXP).map(|i| p[i] + b[i]).collect();
        if self.groups { // DeepSeek-V3 style: 8 groups of 32, keep the 4 best by sum of top-2
            let mut gs: Vec<(f32, usize)> = (0..8).map(|g| { let mut v: Vec<f32> = s[g * 32..g * 32 + 32].to_vec(); v.sort_by(|a, b| b.partial_cmp(a).unwrap()); (v[0] + v[1], g) }).collect();
            gs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            for &(_, g) in &gs[4..] { for i in 0..32 { s[g * 32 + i] = f32::NEG_INFINITY; } }
        }
        let mut ix: Vec<usize> = (0..NEXP).collect(); ix.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap().then(a.cmp(&b)));
        let mut sel = [0usize; TOPK]; let mut w = [0f32; TOPK]; let mut sum = 0.0f32;
        for i in 0..TOPK { sel[i] = ix[i]; w[i] = p[ix[i]]; sum += w[i]; }
        for v in w.iter_mut() { *v = *v / sum.max(6.1e-5) * 2.5; }
        if let Some((_, sh, depth)) = &self.pf { let mut st = self.pfs.borrow_mut(); let mut pr = sh.pred.lock().unwrap();
            for d in 1..=(*depth).min(3) { if let Some(p) = pr.remove(&(l, d)) { st.tot[d] += TOPK as u64; st.hit[d] += sel.iter().filter(|e| p.contains(e)).count() as u64; } } }
        self.trace.borrow_mut().push((l, sel));
        (sel, w)
    }
    fn read_list(&self, l: usize, list: &[usize]) { let ix: Vec<usize> = (0..list.len()).collect(); self.read_to(l, list, self.slots.ptr as usize, &ix) }
    fn read_to(&self, l: usize, list: &[usize], base: usize, ix: &[usize]) {
        let t = std::time::Instant::now();
        let ch: usize = std::env::var("BEAM_RD_CH").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 20);
        let nthr: usize = std::env::var("BEAM_RD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let direct = self.dfiles.is_some();
        // pieces: (file idx, file offset, len, dst address)
        let mut pieces: Vec<(usize, u64, usize, usize)> = vec![];
        for (i, &e) in list.iter().enumerate() { let loc = self.exps[l][e]; let dst = base + ix[i] * self.slot; let wpe = if self.mixed { 1638400 * self.ebits[l][e] as usize } else { self.wp };
            for (off, len, d) in [(loc.w, 3 * wpe, dst), (loc.s, 3 * self.sp, dst + 3 * wpe)] { let mut o = 0; while o < len { let n = ch.min(len - o); pieces.push((loc.file, off + o as u64, n, d + o)); o += n; } } }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (files, dfs, flen, pcs) = (if self.qmode { &self.qfiles } else { &self.st.files }, self.dfiles.as_ref(), &self.file_len, &pieces);
        std::thread::scope(|sc| { for _ in 0..nthr { sc.spawn(|| {
            let mut tmp: Option<(*mut u8, std::alloc::Layout)> = None;
            loop { let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed); if k >= pcs.len() { break }
                let (fi, off, len, dst) = pcs[k];
                if !direct { let b = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) }; files[fi].read_exact_at(b, off).expect("read expert"); }
                else { let a = off & !4095; let end = (off + len as u64 + 4095) & !4095; let n = (end - a) as usize;
                    if tmp.map(|t| t.1.size() < n).unwrap_or(true) { if let Some((p, lay)) = tmp { unsafe { std::alloc::dealloc(p, lay) } } let lay = std::alloc::Layout::from_size_align(n.max(ch + 8192), 4096).unwrap(); tmp = Some((unsafe { std::alloc::alloc(lay) }, lay)); }
                    let tb = unsafe { std::slice::from_raw_parts_mut(tmp.unwrap().0, n) };
                    let got = dfs.unwrap()[fi].read_at(tb, a).expect("direct read"); assert!(got as u64 >= (off - a) + len as u64 || a + got as u64 >= flen[fi], "short direct read");
                    unsafe { std::ptr::copy_nonoverlapping(tb.as_ptr().add((off - a) as usize), dst as *mut u8, len) }; } }
            if let Some((p, lay)) = tmp { unsafe { std::alloc::dealloc(p, lay) } } }); } });
        let mut s = self.stats.borrow_mut(); s.disk_s += t.elapsed().as_secs_f64(); s.bytes += list.iter().map(|&e| if self.mixed { 3 * 1638400 * self.ebits[l][e] as usize + 3 * self.sp } else { self.slot }).sum::<usize>() as u64;
    }
    fn set_meta(&self, l: usize, sel: &[usize; TOPK], w: &[f32; TOPK], slot_of: &dyn Fn(usize) -> usize) {
        let m = self.meta.as_f32_mut();
        for (i, &e) in sel.iter().enumerate() { let ws = self.exps[l][e].ws2; m[i * 8] = ws[0]; m[i * 8 + 1] = ws[1]; m[i * 8 + 2] = ws[2]; m[i * 8 + 3] = w[i]; m[i * 8 + 4] = f32::from_bits(slot_of(e) as u32); m[i * 8 + 5] = f32::from_bits(self.ebits[l][e] as u32); }
    }
    fn stream(&self, l: usize, sel: &[usize; TOPK], w: &[f32; TOPK]) {
        if self.ck == 0 { self.read_list(l, sel); self.set_meta(l, sel, w, &|e| sel.iter().position(|&x| x == e).unwrap()); return }
        let mut cs = self.cstate.borrow_mut(); let st = &mut cs[l]; let mut tk = self.ctick.borrow_mut();
        let mut slot_of = [0usize; TOPK]; let mut miss: Vec<usize> = vec![]; let mut mix: Vec<usize> = vec![];
        for (i, &e) in sel.iter().enumerate() { *tk += 1; if let Some(s) = st.iter().position(|x| x.0 == e + 1) { st[s].1 = *tk; slot_of[i] = s; } else { miss.push(i); } }
        let mut hm = self.chm.borrow_mut(); hm[0] += (TOPK - miss.len()) as u64; hm[1] += TOPK as u64;
        for &i in &miss { let mut v = 0; for s in 0..st.len() { if st[s].1 < st[v].1 { v = s } } *tk += 1; st[v] = (sel[i] + 1, *tk); slot_of[i] = v; mix.push(v); }
        drop(tk); drop(hm);
        let lst: Vec<usize> = miss.iter().map(|&i| sel[i]).collect();
        if !lst.is_empty() { self.read_to(l, &lst, self.cache[l].ptr as usize, &mix); }
        self.set_meta(l, sel, w, &|e| slot_of[sel.iter().position(|&x| x == e).unwrap()]);
    }
    /// Batched prefill of up to PF_T tokens at positions pos0..: per layer, attention token by token, then the UNION of the chunk's
    /// routed experts is streamed once and each token runs its own 8. Leaves logits of the last token in self.logits.
    pub fn prefill_chunk(&self, toks: &[u32], pos0: usize) -> Result<()> {
        let t0 = std::time::Instant::now(); let n = toks.len(); assert!(n <= PF_T && pos0 + n <= self.max_ctx);
        let mut xs: Vec<Vec<f32>> = vec![]; let mut row = vec![0u8; DIM * 2];
        for &tk in toks { let (fi, eo) = self.embed; self.st.files[fi].read_exact_at(&mut row, eo + tk as u64 * (DIM * 2) as u64)?;
            xs.push(row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()); }
        let mut gpu = 0.0;
        for l in 0..self.lw.len() {
            let mut xn_t: Vec<Vec<f32>> = vec![]; let mut sels: Vec<([usize; TOPK], [f32; TOPK])> = vec![];
            for t in 0..n {
                self.x.as_f32_mut()[..DIM].copy_from_slice(&xs[t]);
                let mut err = Ok(()); gpu += self.ctx.run(|cb| { err = self.rec_attn(cb, l, pos0 + t); })?; err?;
                xs[t].copy_from_slice(&self.x.as_f32()[..DIM]);
                if l >= N_DENSE { xn_t.push(self.xn.as_f32()[..DIM].to_vec()); sels.push(self.select(l)); }
            }
            if l < N_DENSE { continue }
            let mut uni: Vec<usize> = vec![]; for (s, _) in &sels { for &e in s { if !uni.contains(&e) { uni.push(e) } } }
            self.read_list(l, &uni);
            { let mut st = self.stats.borrow_mut(); st.bytes = st.bytes; }
            for t in 0..n {
                self.x.as_f32_mut()[..DIM].copy_from_slice(&xs[t]); self.xn.as_f32_mut()[..DIM].copy_from_slice(&xn_t[t]);
                self.set_meta(l, &sels[t].0, &sels[t].1, &|e| uni.iter().position(|&x| x == e).unwrap());
                let mut err = Ok(()); gpu += self.ctx.run(|cb| { err = self.rec_exp(cb, l, false); })?; err?;
                xs[t].copy_from_slice(&self.x.as_f32()[..DIM]);
            }
        }
        self.x.as_f32_mut()[..DIM].copy_from_slice(&xs[n - 1]);
        let mut err = Ok(()); gpu += self.ctx.run(|cb| { err = (|| { self.rms(cb, &self.x, &self.xn, self.f_final)?; self.gemv(cb, &self.head, 0, VOCAB, DIM, &self.xn, 0, &self.logits, 0, false) })(); })?; err?;
        let mut s = self.stats.borrow_mut(); s.gpu_s += gpu; s.wall_s += t0.elapsed().as_secs_f64();
        Ok(())
    }
    /// One decode step: embeds `tok` at `pos`, runs all layers, leaves logits (pre-softcap) in self.logits. Returns after GPU idle.
    pub fn step(&self, tok: u32, pos: usize) -> Result<()> {
        if pos >= self.max_ctx { bail!("pos >= max_ctx") }
        let t0 = std::time::Instant::now();
        let (fi, eo) = self.embed; let mut row = vec![0u8; DIM * 2];
        self.st.files[fi].read_exact_at(&mut row, eo + tok as u64 * (DIM * 2) as u64)?;
        let x = self.x.as_f32_mut(); for (i, c) in row.chunks_exact(2).enumerate() { x[i] = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16); }
        let mut pend: Option<usize> = None; let mut gpu = 0.0;
        for l in 0..self.lw.len() {
            let mut err = Ok(());
            gpu += self.ctx.run(|cb| { err = (|| { if let Some(p) = pend { self.rec_exp(cb, p, true)?; } self.rec_attn(cb, l, pos) })(); })?; err?;
            if let Some((tx, _, _)) = &self.pf { let x = self.x.as_f32()[..DIM].to_vec(); let _ = tx.send(PfJob { l, x }); }
            self.act_x(l);
            if let Some(p) = pend { self.act_h(p); }
            if l >= N_DENSE { let (sel, w) = self.select(l); self.stream(l, &sel, &w); pend = Some(l); } else { pend = None; }
        }
        let mut err = Ok(());
        gpu += self.ctx.run(|cb| { err = (|| { if let Some(p) = pend { self.rec_exp(cb, p, true)?; }
            self.rms(cb, &self.x, &self.xn, self.f_final)?;
            self.gemv(cb, &self.head, 0, VOCAB, DIM, &self.xn, 0, &self.logits, 0, false) })(); })?; err?;
        if let Some(p) = pend { self.act_h(p); }
        if std::env::var("BEAM_DEBUG").is_ok() { let x = self.x.as_f32(); let r = (x[..DIM].iter().map(|v| (v*v) as f64).sum::<f64>() / DIM as f64).sqrt(); eprintln!("[dbg] pos {pos} x rms {r:.4} nan {}", x[..DIM].iter().any(|v| !v.is_finite())); }
        let mut s = self.stats.borrow_mut(); s.gpu_s += gpu; s.wall_s += t0.elapsed().as_secs_f64();
        Ok(())
    }
    /// Top-n (token, logit after softcap 30*tanh(l/30)) of the last step.
    fn act_x(&self, l: usize) { let mut a = self.act.borrow_mut(); if a.is_empty() || l < N_DENSE { return } let xn = &self.xn.as_f32()[..DIM]; for i in 0..DIM { a[l * (DIM + FF_EXP) + i] += (xn[i] as f64).powi(2); } self.actn.borrow_mut()[l] += 1; }
    fn act_h(&self, l: usize) { let mut a = self.act.borrow_mut(); if a.is_empty() { return } let h = &self.h.as_f32()[..TOPK * FF_EXP]; for e in 0..TOPK { for i in 0..FF_EXP { a[l * (DIM + FF_EXP) + DIM + i] += (h[e * FF_EXP + i] as f64).powi(2); } } self.actn.borrow_mut()[NL + l] += TOPK as u64; }
    /// writes NL x (DIM + FF_EXP) f32: mean x^2 of the FFN input and mean h^2 (down input, pooled over routed experts)
    pub fn dump_act(&self, path: &str) -> Result<()> { let a = self.act.borrow(); let n = self.actn.borrow(); let mut out = Vec::with_capacity(a.len() * 4);
        for l in 0..NL { for i in 0..DIM + FF_EXP { let c = if i < DIM { n[l] } else { n[NL + l] }; out.extend_from_slice(&((a[l * (DIM + FF_EXP) + i] / c.max(1) as f64) as f32).to_le_bytes()); } }
        std::fs::write(path, out)?; Ok(()) }
    pub fn cache_report(&self) -> String { let h = *self.chm.borrow(); format!("cache {} slots/layer: hit {}/{} = {:.3}", self.ck, h[0], h[1], h[0] as f64 / h[1].max(1) as f64) }
    pub fn reset_stats(&self) { *self.chm.borrow_mut() = [0; 2]; *self.stats.borrow_mut() = Default::default(); }
    pub fn top(&self, n: usize) -> Vec<(u32, f32)> {
        let l = &self.logits.as_f32()[..VOCAB]; let mut ix: Vec<u32> = (0..VOCAB as u32).collect();
        ix.select_nth_unstable_by(n, |&a, &b| l[b as usize].partial_cmp(&l[a as usize]).unwrap()); ix.truncate(n);
        ix.sort_by(|&a, &b| l[b as usize].partial_cmp(&l[a as usize]).unwrap());
        ix.into_iter().map(|i| (i, 30.0 * (l[i as usize] / 30.0).tanh())).collect()
    }
    /// log-softmax denominator over softcapped logits (for logprobs)
    pub fn logsumexp(&self) -> f64 {
        let l = &self.logits.as_f32()[..VOCAB]; let v: Vec<f64> = l.iter().map(|&x| (30.0 * (x / 30.0).tanh()) as f64).collect();
        let m = v.iter().cloned().fold(f64::MIN, f64::max); m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
    }
}

fn pf_predict(sh: &PfShared, job: PfJob, depth: usize) {
    let ss: f32 = job.x.iter().map(|v| v * v).sum::<f32>() / DIM as f32; let sc = 1.0 / (ss + EPS).sqrt();
    for d in 1..=depth { let j = job.l + d; if j >= NL { break } if j < N_DENSE { continue }
        let w = &sh.fnorm[sh.pf_off[j] as usize..sh.pf_off[j] as usize + DIM];
        let z: Vec<f32> = job.x.iter().zip(w).map(|(a, b)| a * sc * b).collect();
        let r = &sh.router[j]; let b = &sh.bias[j];
        let mut s: Vec<(f32, usize)> = (0..NEXP).map(|e| { let dot: f32 = r[e * DIM..(e + 1) * DIM].iter().zip(&z).map(|(a, b)| a * b).sum(); (1.0 / (1.0 + (-dot).exp()) + b[e], e) }).collect();
        s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); let top: Vec<usize> = s.iter().take(sh.top).map(|x| x.1).collect();
        for &e in &top { let loc = sh.exps[j][e]; let fd = sh.fds[loc.file];
            unsafe { readahead(fd, loc.w as i64, 3 * WP); readahead(fd, loc.s as i64, 3 * SP); } }
        sh.issued.fetch_add(top.len() as u64, std::sync::atomic::Ordering::Relaxed);
        sh.pred.lock().unwrap().insert((j, d), top);
    }
}

/// Quantized expert dir. BEAM_QDIR=cb (or =default) selects the default quant path: per-layer 3-bit codebook on all
/// expert layers (cbBall, ~/models/new/beam/cbBall/q3; best measured: top1 0.950 KL 0.024 vs nvfp4). A `lut.txt` inside the
/// dir is loaded as the codebook LUT unless BEAM_CBLUT overrides it. Unset = full-precision nvfp4 experts.
fn qdir() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    std::env::var("BEAM_QDIR").ok().map(|d| match d.as_str() { "cb" | "default" => format!("{home}/models/new/beam/cbBall/q3"), _ => d.replace('~', &home) })
}

/// Per-layer 3-bit codebook (BEAM_CBLUT file: lines "layer v0 .. v7"); layers not listed use uniform q - 3.5.
fn cblut() -> Option<&'static Vec<[f32; 8]>> {
    static T: std::sync::OnceLock<Option<Vec<[f32; 8]>>> = std::sync::OnceLock::new();
    T.get_or_init(|| { let f = std::env::var("BEAM_CBLUT").ok().or_else(|| qdir().map(|d| format!("{d}/lut.txt")).filter(|f| std::path::Path::new(f).exists()))?; let mut t = vec![[-3.5f32, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]; NL];
        for line in std::fs::read_to_string(f).ok()?.lines() { let v: Vec<f32> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect(); if v.len() == 9 { let l = v[0] as usize; for i in 0..8 { t[l][i] = v[1 + i]; } } }
        eprintln!("[load] codebook LUT from BEAM_CBLUT"); Some(t) }).as_ref()
}
