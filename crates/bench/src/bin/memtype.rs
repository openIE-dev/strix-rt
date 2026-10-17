//! Why are imported pages slower? Same Q4_K GEMV (28 x ffn_gate, 1020 MiB) over different backings:
//! copied DL|HV, copied HOST_CACHED (snooped GTT), anon import 4K pages, anon import THP (2M), file mmap import (needs shim).
use anyhow::Result;
use gguf::{GgmlType, Gguf};
use vkrt::{push_u32, shaders, Buffer, Ctx};

fn anon(len: usize, huge: bool) -> *mut u8 { unsafe {
    let al = 2usize << 20;
    let p = libc::mmap(std::ptr::null_mut(), len + al, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) as usize;
    let p = (p + al - 1) / al * al;
    libc::madvise(p as *mut _, len, if huge { libc::MADV_HUGEPAGE } else { libc::MADV_NOHUGEPAGE });
    p as *mut u8 } }

fn main() -> Result<()> {
    let path = format!("{}/models/q7.gguf", std::env::var("HOME")?);
    let g = Gguf::open(&path)?; let ctx = Ctx::new()?;
    let ts: Vec<_> = g.tensors.iter().filter(|t| t.name.ends_with("ffn_gate.weight")).collect();
    let (k, n) = (ts[0].dims[0] as u32, ts[0].dims[1] as u32);
    let xb = ctx.buffer(k as u64 * 4)?; let yb = ctx.buffer(n as u64 * 4)?; let fb = ctx.buffer(256)?;
    for (i, v) in xb.as_f32_mut().iter_mut().enumerate() { *v = ((i * 7919) % 1000) as f32 / 500.0 - 1.0; }
    let kern = ctx.kernel(shaders::GEMV2_Q4K, 4, 28, &[64, 1], Some(32))?;
    let bench = |label: &str, bufs: &[(Buffer, u64)]| -> Result<()> {
        let ds: Vec<_> = bufs.iter().map(|(b, _)| ctx.bind(&kern, &[b, &xb, &yb, &fb]).unwrap()).collect();
        let iters = 112u32;
        let rec = ctx.record(|cb| for i in 0..iters { let j = i as usize % ds.len(); ctx.dispatch(cb, &kern, ds[j], &push_u32(&[n, k, (bufs[j].1 / 4) as u32, 0, u32::MAX, 0, 0]), n.div_ceil(2)); ctx.barrier(cb); })?;
        ctx.submit(&rec)?; let mut best = f64::MAX; for _ in 0..5 { best = best.min(ctx.submit(&rec)?); }
        let per = best / iters as f64; let gbs = (ts[0].n_bytes() as f64 + (k + n) as f64 * 4.0) / per / 1e9;
        let ck: f64 = yb.as_f32().iter().map(|v| *v as f64).sum();
        println!("{label:<44} flags {:<48} {:.1} GB/s  (checksum {ck:.4})", format!("{:?}", bufs[0].0.flags), gbs); Ok(())
    };
    let copy = |cached: bool| -> Vec<(Buffer, u64)> { ts.iter().map(|t| { let b = if cached { ctx.buffer_readback(t.n_bytes() as u64).unwrap() } else { ctx.buffer(t.n_bytes() as u64).unwrap() }; b.bytes_mut()[..t.n_bytes()].copy_from_slice(g.data(t)); (b, 0) }).collect() };
    bench("copied DL|HV", &copy(false))?;
    bench("copied HOST_CACHED (GTT, snooped)", &copy(true))?;
    for huge in [false, true] {
        let v: Vec<_> = ts.iter().map(|t| { let len = t.n_bytes().div_ceil(2 << 20) * (2 << 20); let p = anon(len, huge);
            unsafe { std::ptr::copy_nonoverlapping(g.data(t).as_ptr(), p, t.n_bytes()); (ctx.import_host(p, len as u64).unwrap(), 0u64) } }).collect();
        bench(if huge { "anon import, THP (MADV_HUGEPAGE)" } else { "anon import, 4K (MADV_NOHUGEPAGE)" }, &v)?;
    }
    if std::env::var("LD_PRELOAD").is_ok() {
        let v: Vec<_> = ts.iter().map(|t| { let s = g.data_offset + t.offset as usize; let e = s + t.n_bytes(); let ps = s / 4096 * 4096; let pe = e.div_ceil(4096) * 4096;
            (unsafe { ctx.import_host(g.mmap.as_ptr().add(ps) as *mut u8, (pe - ps) as u64).unwrap() }, (s - ps) as u64) }).collect();
        bench("file mmap import (page cache, shim)", &v)?;
        drop(v);
        let v: Vec<_> = ts.iter().map(|t| { let s = g.data_offset + t.offset as usize; let e = s + t.n_bytes(); let al = 2usize << 20; let ps = s / al * al; let pe = (e.div_ceil(al) * al).min(g.mmap.len().div_ceil(4096) * 4096);
            (unsafe { ctx.import_host(g.mmap.as_ptr().add(ps) as *mut u8, (pe - ps) as u64).unwrap() }, (s - ps) as u64) }).collect();
        println!("mmap base 2MiB-aligned: {}", g.mmap.as_ptr() as usize % (2 << 20) == 0);
        bench("file mmap import, 2MiB-aligned file ranges", &v)?;
        let empty = ctx.record(|_| {})?; let t = std::time::Instant::now(); for _ in 0..200 { ctx.submit(&empty)?; }
        println!("  empty submit with these userptr imports alive: {:.1} us", t.elapsed().as_secs_f64() / 200.0 * 1e6);
        for (b, _) in v { ctx.free_buffer(b); }
        if vkrt::pagecache::available() {
            let t0 = std::time::Instant::now();
            let v: Vec<_> = ts.iter().map(|t| { let s = g.data_offset + t.offset as usize; let e = s + t.n_bytes(); let al = 2usize << 20; let ps = s / al * al; let pe = (e.div_ceil(al) * al).min(g.mmap.len().div_ceil(4096) * 4096);
                let fd = unsafe { vkrt::pagecache::export(g.mmap.as_ptr().add(ps), pe - ps).unwrap() };
                if t.name.starts_with("blk.0.") { println!("  dma-buf importable type bits: {:#x}", ctx.dmabuf_type_bits(fd)); }
                (unsafe { ctx.import_dmabuf(fd, (pe - ps) as u64).unwrap() }, (s - ps) as u64) }).collect();
            println!("  strixpc export+import of {} ranges: {:.3}s", v.len(), t0.elapsed().as_secs_f64());
            bench("file mmap -> strixpc dma-buf, 2MiB-aligned", &v)?;
            let t = std::time::Instant::now(); for _ in 0..200 { ctx.submit(&empty)?; }
            println!("  empty submit with dma-buf imports alive: {:.1} us", t.elapsed().as_secs_f64() / 200.0 * 1e6);
            for (b, _) in v { ctx.free_buffer(b); }
            let t = std::time::Instant::now(); for _ in 0..200 { ctx.submit(&empty)?; }
            println!("  empty submit after freeing them: {:.1} us", t.elapsed().as_secs_f64() / 200.0 * 1e6);
        }
        // 2 MiB-aligned file mapping + MADV_COLLAPSE (read-only file THP), then import
        let f = std::fs::File::open(&path)?; use std::os::fd::AsRawFd;
        let len = g.mmap.len().div_ceil(2 << 20) * (2 << 20);
        let base = unsafe { let r = anon(len, false) as *mut libc::c_void;
            libc::mmap(r, len, libc::PROT_READ, libc::MAP_SHARED | libc::MAP_FIXED, f.as_raw_fd(), 0) as *mut u8 };
        let fhp = || std::fs::read_to_string("/proc/meminfo").unwrap().lines().find(|l| l.starts_with("FileHugePages")).unwrap().to_string();
        println!("before collapse: {}", fhp());
        let t = std::time::Instant::now();
        let r = unsafe { libc::madvise(base as *mut _, len, libc::MADV_HUGEPAGE); libc::madvise(base as *mut _, len, 25 /* MADV_COLLAPSE */) };
        println!("MADV_COLLAPSE whole file: ret {r} ({}) in {:.2}s; after: {}", if r != 0 { std::io::Error::last_os_error().to_string() } else { "ok".into() }, t.elapsed().as_secs_f64(), fhp());
        let v: Vec<_> = ts.iter().map(|t| { let s = g.data_offset + t.offset as usize; let e = s + t.n_bytes(); let ps = s / 4096 * 4096; let pe = e.div_ceil(4096) * 4096;
            (unsafe { ctx.import_host(base.add(ps), (pe - ps) as u64).unwrap() }, (s - ps) as u64) }).collect();
        bench("file mmap import after MADV_COLLAPSE", &v)?;
    }
    Ok(())
}
