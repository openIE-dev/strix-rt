//! k6 dit   --ref DIR : step-0 DiT parity vs the diffusers dumps (text blocks, selected blocks, velocity grids)
//! k6 gen   --ref DIR [--steps 10] [--out DIR] : full PiFlow sampling from the reference noise; latent parity + timing
use anyhow::Result;
use k6::*;
fn arg(a: &[String], k: &str, d: &str) -> String { a.iter().position(|x| x == k).map(|i| a[i + 1].clone()).unwrap_or(d.to_string()) }
fn rd(dir: &str, n: &str) -> Result<(Vec<f32>, Vec<usize>)> { npy::read(&format!("{dir}/{n}.npy")) }
fn rep(name: &str, a: &[f32], b: &[f32]) { let (c, r, m) = err_cmp(a, b); println!("  {name:28} n={:<9} cos {c:.6}  rel {r:.3e}  max|d| {m:.4}", b.len()); }
fn envn(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect(); let mode = a.get(1).cloned().unwrap_or_default();
    let refd = arg(&a, "--ref", &format!("{}/k6_ref/out", std::env::var("HOME")?));
    let mdir = arg(&a, "--model", &format!("{}/models/k6-pro-distill/transformer", std::env::var("HOME")?));
    if mode == "sched2" {   // CPU: every scheduler step on reference inputs/grids (out2 dumps) vs the next reference input
        let (raw, sig) = sched::schedule(10); println!("raw {raw:?}");
        for i in 0..9 { let (vin, vs) = rd(&refd, &format!("st{i}_in_vis"))?; let (ain, _) = rd(&refd, &format!("st{i}_in_aud"))?;
            let (rv, _) = rd(&refd, &format!("st{i}_v_vis"))?; let (ra, _) = rd(&refd, &format!("st{i}_v_aud"))?;
            let (nv1, _) = rd(&refd, &format!("st{}_in_vis", i + 1))?; let (na1, _) = rd(&refd, &format!("st{}_in_aud", i + 1))?;
            let (tt, _) = rd(&refd, &format!("st{i}_t"))?;
            let nv = vs[1] * vs[2] * vs[3]; let cin = vin.len() / nv;
            let mut x: Vec<f32> = (0..nv).flat_map(|j| vin[j * cin..j * cin + 16].to_vec()).collect(); let mut xa = ain.clone();
            sched::step(&mut x, &rv, CV, NG, &raw, &sig, i); sched::step(&mut xa, &ra, CA, NG, &raw, &sig, i);
            let rx: Vec<f32> = (0..nv).flat_map(|j| nv1[j * cin..j * cin + 16].to_vec()).collect();
            println!("step {i} t_ref {:?} (cin {cin})", &tt[..tt.len().min(2)]); rep("vis next", &x, &rx); rep("aud next", &xa, &na1);
            if i == 0 { let c0: Vec<f32> = (0..nv).flat_map(|j| vin[j * cin + 16..j * cin + cin].to_vec()).collect(); let c1: Vec<f32> = (0..nv).flat_map(|j| nv1[j * cin + 16..j * cin + cin].to_vec()).collect(); rep("cond chans 0 vs 1", &c1, &c0); } }
        return Ok(()) }
    if mode == "sched" {   // CPU: scheduler step 0 on the reference step-0 grids
        let (vin, vs) = rd(&refd, "s0_in_vis")?; let (ain, _) = rd(&refd, "s0_in_aud")?; let (rv, _) = rd(&refd, "s0_v_vis")?; let (ra, _) = rd(&refd, "s0_v_aud")?;
        let nv = vs[1] * vs[2] * vs[3]; let mut x: Vec<f32> = (0..nv).flat_map(|i| vin[i * 33..i * 33 + 16].to_vec()).collect(); let mut xa = ain.clone();
        let (raw, sig) = sched::schedule(10); sched::step(&mut x, &rv, CV, NG, &raw, &sig, 0); sched::step(&mut xa, &ra, CA, NG, &raw, &sig, 0);
        npy::write("/tmp/k6_s1_vis.npy", &x, &[nv, 16])?; npy::write("/tmp/k6_s1_aud.npy", &xa, &[xa.len() / 40, 40])?; println!("raw {raw:?}\nsig {sig:?}"); return Ok(()) }
    let ctx = vkrt::Ctx::new()?; let pf = prefill::Pf::new(&ctx)?; let kn = kern(&ctx)?;
    if mode == "te" {   // text encoders: Qwen2.5-VL (GPU) + CLIP-L (CPU) parity vs the reference embeddings
        let home = std::env::var("HOME")?; let m = format!("{home}/models/k6-pro-distill");
        let prompt = arg(&a, "--prompt", "A red fox trotting through fresh snow in a pine forest at sunrise, soft crunching footsteps, birds chirping.");
        let tq = qimage::tok::Tok::load(&format!("{m}/text_encoder/tokenizer.json"))?; let ids = tq.encode(&te::TEMPLATE.replace("{}", &prompt));
        let clip = te::Clip::load(&ctx, &format!("{m}/text_encoder_2"), &format!("{m}/tokenizer_2/tokenizer.json"))?; let cids = clip.tokenize(&prompt);
        if let Ok(s) = std::fs::read_to_string(format!("{refd}/te_ids.json")) { let j = kev::json::parse(&s)?;
            let get = |k: &str| -> Vec<u32> { match j.get(k) { Some(kev::json::J::Arr(v)) => v.iter().map(|x| match x { kev::json::J::Num(t) => t.parse().unwrap_or(0), _ => 0 }).collect(), _ => vec![] } };
            println!("qwen ids match ref: {} ({} tokens)  clip ids match ref: {} ({} tokens)", get("qwen") == ids, ids.len(), get("clip") == cids, cids.len()); }
        let q = te::Qwen::load(&ctx, &arg(&a, "--te", &format!("{m}/text_encoder_al")))?; println!("qwen loaded {:.2}s, clip {:.2}s", q.load_s, clip.load_s);
        let mut tq_ = vec![]; let mut tc = vec![]; let mut pe = vec![]; let mut pooled = vec![];
        for _ in 0..envn("K6_REPS", 1) { let t0 = std::time::Instant::now(); let (e, g) = q.encode(&ctx, &pf, &kn, &ids, te::CROP)?; tq_.push((t0.elapsed().as_secs_f64(), g)); pe = e;
            let t1 = std::time::Instant::now(); pooled = clip.pooled(&cids); tc.push(t1.elapsed().as_secs_f64()); }
        println!("qwen encode (wall, gpu) s {tq_:?}; clip pooled (cpu) s {tc:?}");
        if let Ok((r, _)) = rd(&refd, "prompt_embeds_trim") { if r.len() == pe.len() { rep("qwen prompt_embeds", &pe, &r); } else { println!("len {} vs ref {}", pe.len(), r.len()); } }
        if let Ok((r, _)) = rd(&refd, "pooled") { rep("clip pooled", &pooled, &r); }
        npy::write("/tmp/k6_pe.npy", &pe, &[1, pe.len() / 3584, 3584])?; npy::write("/tmp/k6_pooled.npy", &pooled, &[1, 768])?; return Ok(()) }
    if mode == "aud" {   // audio decode: --alat audio_latents.npy (1,40,L) -> mel + wav (parity vs --ref dir aref_mel/aref_wav)
        let (z, zs) = npy::read(&arg(&a, "--alat", &format!("{refd}/audio_latents.npy")))?;
        let home = std::env::var("HOME")?; let au = audio::Audio::load(&ctx, &format!("{home}/models/k6-pro-distill/audio_vae"), &format!("{home}/models/k6-pro-distill/vocoder"))?;
        println!("audio loaded {:.2}s", au.load_s); let l = zs[2] as u32; let zz: Vec<f32> = z.iter().map(|v| v / 0.417).collect();
        let mut ts = vec![]; let mut res = None;
        for _ in 0..envn("K6_REPS", 1) { let t0 = std::time::Instant::now(); let r = au.decode(&ctx, &pf, &kn, &zz, l)?; ts.push((t0.elapsed().as_secs_f64(), r.2)); res = Some(r); }
        let (mel, wav, _) = res.unwrap(); println!("audio decode (wall, gpu) s: {ts:?} -> mel {} frames, {} samples", mel.len() / 128, wav.len());
        let ad = arg(&a, "--aref", &format!("{home}/k6_ref/aud"));
        if let Ok((rm, _)) = npy::read(&format!("{ad}/aref_mel.npy")) { rep("mel", &mel, &rm); }
        if let Ok((rw, _)) = npy::read(&format!("{ad}/aref_wav.npy")) { if rw.len() == wav.len() { rep("wav", &wav, &rw); } else { println!("wav len {} vs ref {}", wav.len(), rw.len()); } }
        npy::write(&format!("{ad}/ours_mel.npy"), &mel, &[128, mel.len() / 128])?; npy::write(&format!("{ad}/ours_wav.npy"), &wav, &[wav.len()])?;
        npy::write_wav(&format!("{ad}/ours.wav"), &wav, 1, 44100)?; return Ok(()) }
    if mode == "vae" {   // video VAE decode: --lat latents.npy (1,16,T,H,W) [--dump DIR] [--out video.npy]
        let (lat, ls) = npy::read(&arg(&a, "--lat", &format!("{refd}/latents.npy")))?;
        let vdir = arg(&a, "--vae", &format!("{}/models/k6-pro-distill/vae", std::env::var("HOME")?));
        let v = vae::Vae::load(&ctx, &vdir)?; println!("vae loaded {:.2}s ({} convs)", v.load_s, 0);
        let dump = a.iter().position(|x| x == "--dump").map(|i| a[i + 1].clone());
        let t0 = std::time::Instant::now();
        let (vid, g) = v.decode_auto(&ctx, &pf, &kn, &lat, ls[2] as u32, ls[3] as u32, ls[4] as u32, dump.as_deref())?;
        let (tn, hn, wn) = (4 * (ls[2] - 1) + 1, 8 * ls[3], 8 * ls[4]);
        println!("vae decode {:.2}s wall, {:.2}s gpu -> 3x{tn}x{hn}x{wn}", t0.elapsed().as_secs_f64(), g);
        npy::write(&arg(&a, "--out", "/tmp/k6_video.npy"), &vid, &[3, tn, hn, wn])?; return Ok(()) }
    let t0 = std::time::Instant::now(); let m = Model::load(&ctx, &mdir)?;
    println!("loaded in {:.2}s (zero copy {}, st {:.2}s)", t0.elapsed().as_secs_f64(), m.st.zero_copy, m.st.load_s);
    let (pe, pes) = rd(&refd, "prompt_embeds_trim")?; let lt = pes[1] as u32; let (pooled, _) = rd(&refd, "pooled")?;
    let (vin, vs) = rd(&refd, "s0_in_vis")?; let (ain, as_) = rd(&refd, "s0_in_aud")?;
    let (t, h, w) = (vs[1] as u32, vs[2] as u32, vs[3] as u32); let na = as_[1] as u32;
    println!("grid T {t} H {h} W {w} -> {} video tokens, {na} audio tokens, {lt} text tokens", t * h * w / 4);
    let te = std::time::Instant::now(); let text = m.text_embed(&pe); println!("text embed (cpu) {:.2}s", te.elapsed().as_secs_f64());
    let s = State::new(&ctx, &m, t, h, w, na, lt, &text)?;
    if mode == "dit" {
        let (tt, _) = rd(&refd, "s0_t")?;
        let nd = |n: &str| std::path::Path::new(&format!("{refd}/{n}.npy")).exists();
        let out = forward(&ctx, &pf, &kn, &m, &s, &vin, &ain, tt[0], &pooled, |tag, s| {
            if tag == "pre" { println!("== text blocks");
                if let Ok((r, _)) = rd(&refd, "s0_vtext") { rep("video text (after 4 blocks)", &s.tx[0].as_f32()[..r.len()], &r) }
                if let Ok((r, _)) = rd(&refd, "s0_atext") { rep("audio text (after 4 blocks)", &s.tx[1].as_f32()[..r.len()], &r) } }
            else if let Some(b) = tag.strip_prefix("blk") { if nd(&format!("s0_blk{b}_vis")) {
                let (r, _) = rd(&refd, &format!("s0_blk{b}_vis")).unwrap(); rep(&format!("block {b} video"), &s.xv.as_f32()[..r.len()], &r);
                let (r, _) = rd(&refd, &format!("s0_blk{b}_aud")).unwrap(); rep(&format!("block {b} audio"), &s.xa.as_f32()[..r.len()], &r); } }
        })?;
        println!("== step-0 outputs (t = {})", tt[0]);
        let (rv, _) = rd(&refd, "s0_v_vis")?; let (ra, _) = rd(&refd, "s0_v_aud")?;
        if std::env::var("K6_NB").is_err() { rep("video grid [T,H,W,160]", &out.vis, &rv); rep("audio grid [Na,400]", &out.aud, &ra); }
        println!("  forward wall {:.2}s gpu {:.2}s", out.wall, out.gpu);
    } else if mode == "tf" {   // teacher-forced: every step's reference inputs -> compare grids
        let steps: Vec<usize> = arg(&a, "--steps", "0,1,2,5,9").split(',').map(|x| x.parse().unwrap()).collect();
        for &i in &steps {
            let (xv, _) = rd(&refd, &format!("st{i}_in_vis"))?; let (xa, _) = rd(&refd, &format!("st{i}_in_aud"))?; let (tt, _) = rd(&refd, &format!("st{i}_t"))?;
            let nv = xv.len() / 16; let mut inp = vec![0f32; nv * 33]; for e in 0..nv { inp[e * 33..e * 33 + 16].copy_from_slice(&xv[e * 16..e * 16 + 16]); }
            let o = forward(&ctx, &pf, &kn, &m, &s, &inp, &xa, tt[0], &pooled, |_, _| {})?;
            let (rv, _) = rd(&refd, &format!("st{i}_v_vis"))?; let (ra, _) = rd(&refd, &format!("st{i}_v_aud"))?;
            println!("step {i} t {}", tt[0]); rep("video grid", &o.vis, &rv); rep("audio grid", &o.aud, &ra);
        }
    } else if mode == "gen" {
        let steps: usize = arg(&a, "--steps", "10").parse()?; let outd = arg(&a, "--out", "/tmp/k6_gen"); std::fs::create_dir_all(&outd)?;
        let (raw, sig) = sched::schedule(steps);
        let nv = (t * h * w) as usize; let mut x: Vec<f32> = (0..nv).flat_map(|i| vin[i * 33..i * 33 + 16].to_vec()).collect();
        let mut xa = ain.clone(); let mut inp = vec![0f32; nv * 33];
        let tg = std::time::Instant::now(); let mut gpu = 0.0; let mut sw = vec![];
        for i in 0..steps {
            for e in 0..nv { inp[e * 33..e * 33 + 16].copy_from_slice(&x[e * 16..e * 16 + 16]); }
            let st = std::time::Instant::now();
            let o = forward(&ctx, &pf, &kn, &m, &s, &inp, &xa, sig[i] * 1000.0, &pooled, |_, _| {})?;
            sched::step(&mut x, &o.vis, CV, NG, &raw, &sig, i); sched::step(&mut xa, &o.aud, CA, NG, &raw, &sig, i);
            for v in x.iter_mut().chain(xa.iter_mut()) { *v = bf(*v) }   // pipeline keeps latents in bf16
            if let Ok((rn, _)) = rd(&refd, &format!("st{}_in_vis", i + 1)) { rep(&format!("x after step {i} (vis)"), &x, &rn); }
            if let Ok((rn, _)) = rd(&refd, &format!("st{}_in_aud", i + 1)) { rep(&format!("x after step {i} (aud)"), &xa, &rn); }
            if std::env::var("K6_TF_X").is_ok() { if let Ok((rn, _)) = rd(&refd, &format!("st{}_in_vis", i + 1)) { x.copy_from_slice(&rn); } }
            gpu += o.gpu; sw.push(st.elapsed().as_secs_f64());
            println!("  step {i}: t {:.2} forward {:.2}s (gpu {:.2}s) step total {:.2}s", sig[i] * 1000.0, o.wall, o.gpu, sw[i]);
        }
        let wall = tg.elapsed().as_secs_f64();
        // reference latents are (B, C, T, H, W) and (B, C, Na)
        let to_cthw = |x: &[f32], c: usize, n: usize| -> Vec<f32> { let mut o = vec![0f32; x.len()]; for e in 0..n { for ch in 0..c { o[ch * n + e] = x[e * c + ch] } } o };
        let xv = to_cthw(&x, CV, nv); let xac = to_cthw(&xa, CA, na as usize);
        npy::write(&format!("{outd}/latents.npy"), &xv, &[1, CV, t as usize, h as usize, w as usize])?;
        npy::write(&format!("{outd}/audio_latents.npy"), &xac, &[1, CA, na as usize])?;
        println!("== {steps}-step PiFlow sampling from the reference noise");
        if let Ok((r, _)) = rd(&refd, "latents") { rep("video latents", &xv, &r) }
        if let Ok((r, _)) = rd(&refd, "audio_latents") { rep("audio latents", &xac, &r) }
        let mut s2 = sw.clone(); s2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("RESULT dit_wall_s {wall:.2} gpu_s {gpu:.2} step_median_s {:.2} steps {steps} out {outd}", s2[s2.len() / 2]);
    }
    Ok(())
}
