//! Can O_DIRECT reads land straight in Vulkan-mapped DL|HV (UMA) memory? Measures cold read GB/s into it.
use anyhow::Result;
use std::{os::fd::AsRawFd, os::unix::fs::OpenOptionsExt, time::Instant};
fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or(format!("{}/strix-fio.dat", std::env::var("HOME")?));
    let ctx = vkrt::Ctx::new()?;
    let size: u64 = 4 << 30;
    for (label, buf) in [("DL|HV (vkAllocateMemory, mapped)", ctx.buffer(size)?), ("HV|HC|HCached (readback type)", ctx.buffer_readback(size)?)] {
        let f = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path)?;
        let chunk = 8usize << 20; let t = Instant::now(); let mut done = 0usize;
        while done < size as usize {
            let r = unsafe { libc::pread(f.as_raw_fd(), buf.ptr.add(done) as *mut libc::c_void, chunk, done as i64) };
            if r < 0 { println!("{label}: O_DIRECT pread FAILED: {}", std::io::Error::last_os_error()); break }
            if r == 0 { break } done += r as usize;
        }
        if done > 0 { let s = t.elapsed().as_secs_f64(); println!("{label} flags {:?}: O_DIRECT pread 8MiB x qd1 -> {:.2} GiB in {:.3}s = {:.2} GB/s", buf.flags, done as f64 / (1u64 << 30) as f64, s, done as f64 / s / 1e9); }
    }
    // buffered pread (page cache -> copy) into DL|HV, cold; and O_DIRECT into anonymous memory that is then imported (stock RADV accepts anon userptr)
    let buf = ctx.buffer(size)?;
    unsafe { libc::posix_fadvise(std::fs::File::open(&path)?.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED); }
    let f = std::fs::File::open(&path)?; let chunk = 8usize << 20; let t = Instant::now(); let mut done = 0usize;
    while done < size as usize { let r = unsafe { libc::pread(f.as_raw_fd(), buf.ptr.add(done) as *mut libc::c_void, chunk, done as i64) }; if r <= 0 { break } done += r as usize; }
    let s = t.elapsed().as_secs_f64(); println!("buffered pread into DL|HV (cold, qd1 8MiB): {:.2} GB/s", done as f64 / s / 1e9);
    let anon = unsafe { libc::mmap(std::ptr::null_mut(), size as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE, -1, 0) } as *mut u8;
    unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED); }
    let fd = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&path)?; let t = Instant::now(); let mut done = 0usize;
    while done < size as usize { let r = unsafe { libc::pread(fd.as_raw_fd(), anon.add(done) as *mut libc::c_void, chunk, done as i64) }; if r <= 0 { println!("anon O_DIRECT failed {}", std::io::Error::last_os_error()); break } done += r as usize; }
    let s = t.elapsed().as_secs_f64(); println!("O_DIRECT pread into anonymous memory (qd1 8MiB): {:.2} GB/s", done as f64 / s / 1e9);
    let t = Instant::now(); let imp = unsafe { ctx.import_host(anon, size)? }; println!("import of 4GiB anon buffer (stock RADV, no shim): {:.3}s, flags {:?}", t.elapsed().as_secs_f64(), imp.flags);
    Ok(())
}
