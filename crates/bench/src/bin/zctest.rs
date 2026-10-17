//! Clean zero-copy comparison in its own process (nothing else imported):
//! MODE=userptr|dmabuf|dmabuf_whole|copy ; reports GPU VA alignment, GEMV GB/s (28 x ffn_gate), empty-submit cost.
use anyhow::Result;
use gguf::Gguf;
use vkrt::{push_u32, shaders, Buffer, Ctx};
fn main() -> Result<()> {
    let mode = std::env::var("MODE").unwrap_or("dmabuf".into());
    let g = Gguf::open(format!("{}/models/q7.gguf", std::env::var("HOME")?))?; let ctx = Ctx::new()?;
    let ts: Vec<_> = g.tensors.iter().filter(|t| t.name.ends_with("ffn_gate.weight")).collect();
    let (k, n) = (ts[0].dims[0] as u32, ts[0].dims[1] as u32);
    let xb = ctx.buffer(k as u64 * 4)?; let yb = ctx.buffer(n as u64 * 4)?; let fb = ctx.buffer(256)?;
    for (i, v) in xb.as_f32_mut().iter_mut().enumerate() { *v = ((i * 7919) % 1000) as f32 / 500.0 - 1.0; }
    let kern = ctx.kernel(shaders::GEMV2_Q4K, 4, 28, &[64, 1], Some(32))?;
    let al = 2usize << 20; let flen = g.mmap.len().div_ceil(4096) * 4096; let base = g.mmap.as_ptr();
    let rng = |t: &gguf::TensorInfo| { let s = g.data_offset + t.offset as usize; (s, s / al * al, ((s + t.n_bytes()).div_ceil(al) * al).min(flen)) };
    let t0 = std::time::Instant::now();
    let bufs: Vec<(Buffer, u64)> = match mode.as_str() {
        "copy" => ts.iter().map(|t| { let b = ctx.buffer(t.n_bytes() as u64).unwrap(); b.bytes_mut()[..t.n_bytes()].copy_from_slice(g.data(t)); (b, 0) }).collect(),
        "userptr" => ts.iter().map(|t| { let (s, ps, pe) = rng(t); (unsafe { ctx.import_host(base.add(ps) as *mut u8, (pe - ps) as u64).unwrap() }, (s - ps) as u64) }).collect(),
        "dmabuf" => ts.iter().map(|t| { let (s, ps, pe) = rng(t); let fd = unsafe { vkrt::pagecache::export(base.add(ps), pe - ps).unwrap() }; (unsafe { ctx.import_dmabuf(fd, (pe - ps) as u64).unwrap() }, (s - ps) as u64) }).collect(),
        _ => { // one dma-buf for the whole file, shared by all tensors
            let fd = unsafe { vkrt::pagecache::export(base, flen)? }; let b = unsafe { ctx.import_dmabuf(fd, flen as u64)? };
            let va = ctx.device_address(&b); println!("whole-file BO VA {va:#x} (mod 2MiB = {:#x})", va % (2 << 20));
            let offs: Vec<u64> = ts.iter().map(|t| (g.data_offset + t.offset as usize) as u64).collect();
            let rec_b = b; return run(&ctx, &kern, &[(rec_b, 0)], &offs, &xb, &yb, &fb, n, k, ts[0].n_bytes(), t0); }
    };
    let offs: Vec<u64> = bufs.iter().map(|b| b.1).collect();
    for (b, _) in bufs.iter().take(3) { let va = ctx.device_address(b); println!("BO VA {va:#x} (mod 2MiB = {:#x}) flags {:?}", va % (2 << 20), b.flags); }
    run(&ctx, &kern, &bufs, &offs, &xb, &yb, &fb, n, k, ts[0].n_bytes(), t0)
}
fn run(ctx: &Ctx, kern: &vkrt::Kernel, bufs: &[(Buffer, u64)], offs: &[u64], xb: &Buffer, yb: &Buffer, fb: &Buffer, n: u32, k: u32, wbytes: usize, t0: std::time::Instant) -> Result<()> {
    println!("setup {:.3}s", t0.elapsed().as_secs_f64());
    let ds: Vec<_> = bufs.iter().map(|(b, _)| ctx.bind(kern, &[b, xb, yb, fb]).unwrap()).collect();
    let iters = 112u32;
    let rec = ctx.record(|cb| for i in 0..iters { let j = i as usize % offs.len(); let d = ds[j.min(ds.len() - 1)];
        ctx.dispatch(cb, kern, d, &push_u32(&[n, k, (offs[j] / 4) as u32, 0, u32::MAX, 0, 0]), n.div_ceil(2)); ctx.barrier(cb); })?;
    ctx.submit(&rec)?; let mut best = f64::MAX; for _ in 0..5 { best = best.min(ctx.submit(&rec)?); }
    let gbs = (wbytes as f64 + (k + n) as f64 * 4.0) / (best / iters as f64) / 1e9;
    let ck: f64 = yb.as_f32().iter().map(|v| *v as f64).sum();
    let empty = ctx.record(|_| {})?; let t = std::time::Instant::now(); for _ in 0..200 { ctx.submit(&empty)?; }
    println!("{} GEMV {:.1} GB/s checksum {ck:.4}; empty submit {:.1} us", std::env::var("MODE").unwrap_or_default(), gbs, t.elapsed().as_secs_f64() / 200.0 * 1e6);
    Ok(())
}
