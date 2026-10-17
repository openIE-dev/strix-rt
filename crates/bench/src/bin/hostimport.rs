//! Does RADV accept host pointers from a file-backed mmap (VK_EXT_external_memory_host)?
//! Imports each ffn_gate tensor (page-aligned enclosing range) of q7.gguf zero-copy, runs the Q4_K GEMV on it,
//! and compares GB/s + results against the same tensors copied into DL|HV buffers.
//! MODE env: shared_ro (MAP_SHARED PROT_READ, default) | private_rw (MAP_PRIVATE PROT_READ|PROT_WRITE) | anon (anonymous copy, control)
use anyhow::{bail, Result};
use gguf::{GgmlType, Gguf};
use std::time::Instant;
use vkrt::{push_u32, shaders, Ctx};

fn main() -> Result<()> {
    let path = std::env::var("MODEL").unwrap_or_else(|_| format!("{}/models/q7.gguf", std::env::var("HOME").unwrap()));
    let mode = std::env::var("MODE").unwrap_or("shared_ro".into());
    let tname = std::env::var("TENSOR").unwrap_or("ffn_gate.weight".into());
    let g = Gguf::open(&path)?; // parse only; we make our own mapping below
    let ctx = Ctx::new()?;
    println!("host import ext: {}, mode {mode}", ctx.has_host_import);
    let f = std::fs::File::open(&path)?;
    let len = f.metadata()?.len() as usize;
    let maplen = len.div_ceil(4096) * 4096;
    use std::os::fd::AsRawFd;
    let base = unsafe {
        let (prot, flags, fd) = match mode.as_str() {
            "shared_ro" => (libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd()),
            "private_rw" => (libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE, f.as_raw_fd()),
            "anon" => (libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1),
            m => bail!("bad MODE {m}"),
        };
        let p = libc::mmap(std::ptr::null_mut(), maplen, prot, flags, fd, 0);
        if p == libc::MAP_FAILED { bail!("mmap failed") }
        p as *mut u8
    };
    if mode == "anon" { unsafe { std::ptr::copy_nonoverlapping(g.mmap.as_ptr(), base, len) } }
    let ts: Vec<_> = g.tensors.iter().filter(|t| t.name.ends_with(&tname) && t.ty == GgmlType::Q4K).collect();
    // import
    let t0 = Instant::now();
    let mut imp = vec![];
    for t in &ts {
        let s = g.data_offset + t.offset as usize; let e = s + t.n_bytes();
        let ps = s / 4096 * 4096; let pe = e.div_ceil(4096) * 4096;
        let b = unsafe { ctx.import_host(base.add(ps), (pe - ps) as u64) };
        match b { Ok(b) => imp.push((b, (s - ps) as u64)), Err(e) => { println!("IMPORT FAILED for {}: {e}", t.name); return Ok(()) } }
    }
    let imp_s = t0.elapsed().as_secs_f64();
    let tot: usize = ts.iter().map(|t| t.n_bytes()).sum();
    println!("imported {} tensors, {:.1} MiB in {:.3}s ({:.2} GB/s), memory flags {:?}", ts.len(), tot as f64 / 1048576.0, imp_s, tot as f64 / imp_s / 1e9, imp[0].0.flags);
    // copies
    let t0 = Instant::now();
    let cp: Vec<_> = ts.iter().map(|t| { let b = ctx.buffer(t.n_bytes() as u64).unwrap(); b.bytes_mut()[..t.n_bytes()].copy_from_slice(g.data(t)); b }).collect();
    println!("copied into DL|HV buffers in {:.3}s", t0.elapsed().as_secs_f64());
    let (k, n) = (ts[0].dims[0] as u32, ts[0].dims[1] as u32);
    let xb = ctx.buffer(k as u64 * 4)?; let yb = ctx.buffer(n as u64 * 4)?; let y2 = ctx.buffer(n as u64 * 4)?; let fb = ctx.buffer(256)?;
    let mut seed = 1u64; for v in xb.as_f32_mut() { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *v = ((seed >> 40) as f32 / 16777216.0) * 2.0 - 1.0; }
    let kern = ctx.kernel(shaders::GEMV2_Q4K, 4, 28, &[64, 1], Some(32))?;
    let groups = n.div_ceil(2);
    let p = |off: u64| push_u32(&[n, k, (off / 4) as u32, 0, u32::MAX, 0, 0]);
    let dsi: Vec<_> = imp.iter().map(|(b, _)| ctx.bind(&kern, &[b, &xb, &yb, &fb]).unwrap()).collect();
    let dsc: Vec<_> = cp.iter().map(|b| ctx.bind(&kern, &[b, &xb, &y2, &fb]).unwrap()).collect();
    ctx.run(|cb| ctx.dispatch(cb, &kern, dsi[0], &p(imp[0].1), groups))?;
    ctx.run(|cb| ctx.dispatch(cb, &kern, dsc[0], &p(0), groups))?;
    let maxdiff = yb.as_f32().iter().zip(y2.as_f32()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("imported vs copied result max |diff| = {maxdiff:e}");
    // per-submit overhead with these BOs resident (RADV puts all BOs on the CS BO list)
    let empty = ctx.record(|_| {})?;
    for (label, keep) in [("with imports/copies alive", true)] { let _ = keep;
        let t = Instant::now(); for _ in 0..200 { ctx.submit(&empty)?; } println!("empty submit+wait ({label}): {:.1} us", t.elapsed().as_secs_f64() / 200.0 * 1e6); }
    let iters = 112u32;
    for (label, ds, offs) in [("imported(mmap)", &dsi, imp.iter().map(|x| x.1).collect::<Vec<_>>()), ("copied", &dsc, vec![0; cp.len()])] {
        let rec = ctx.record(|cb| for i in 0..iters { let j = i as usize % ds.len(); ctx.dispatch(cb, &kern, ds[j], &p(offs[j]), groups); ctx.barrier(cb); })?;
        ctx.submit(&rec)?;
        let mut best = f64::MAX; for _ in 0..3 { best = best.min(ctx.submit(&rec)?); }
        let per = best / iters as f64;
        let gbs = (ts[0].n_bytes() as f64 + k as f64 * 4.0 + n as f64 * 4.0) / per / 1e9;
        println!("GEMV {label}: {n}x{k} Q4_K, {:.1} us/call, {:.1} GB/s ({:.1}% of 256), cycling {} tensors", per * 1e6, gbs, gbs / 2.56, ds.len());
    }
    if std::env::var("DROP_IMPORTS").is_ok() {
        for (b, _) in imp { ctx.free_buffer(b); }
        let t = Instant::now(); for _ in 0..200 { ctx.submit(&empty)?; } println!("empty submit+wait (imports freed, copies alive): {:.1} us", t.elapsed().as_secs_f64() / 200.0 * 1e6);
    }
    Ok(())
}
