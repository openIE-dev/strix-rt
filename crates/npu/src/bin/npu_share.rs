//! npu_share <build_dir> M K N m k n : zero-copy NPU <-> GPU (Vulkan) <-> strixpc page-cache sharing tests.
//! Each test moves data only through shared pages (no host memcpy between the two devices) and verifies a GEMM.
use anyhow::{anyhow, Result};
use ash::vk;
use npu::*;
use std::time::Instant;
use vkrt::{Buffer, Ctx};

fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }
fn ms(t: Instant) -> f64 { t.elapsed().as_secs_f64() * 1e3 }

/// dma-buf import with TRANSFER_SRC|DST (vkrt's import is SRC only). Takes ownership of fd on success.
unsafe fn vk_import(ctx: &Ctx, fd: i32, size: u64) -> Result<Buffer> {
    let d = &ctx.device; let ht = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let efd = ash::khr::external_memory_fd::Device::new(&ctx.instance, d);
    let mut fp = vk::MemoryFdPropertiesKHR::default();
    (efd.fp().get_memory_fd_properties_khr)(d.handle(), ht, fd, &mut fp).result().map_err(|e| anyhow!("fd props: {e:?}"))?;
    let mut emb = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
    let buf = d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).push_next(&mut emb), None)?;
    let req = d.get_buffer_memory_requirements(buf);
    let bits = req.memory_type_bits & fp.memory_type_bits;
    let idx = (0..ctx.mem_props.memory_type_count).find(|i| bits & (1 << i) != 0).ok_or_else(|| anyhow!("no mem type (fd bits {:#x} buf bits {:#x})", fp.memory_type_bits, req.memory_type_bits))?;
    let mut imp = vk::ImportMemoryFdInfoKHR::default().handle_type(ht).fd(fd);
    let mut ded = vk::MemoryDedicatedAllocateInfo::default().buffer(buf);
    let mem = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size.max(size)).memory_type_index(idx).push_next(&mut imp).push_next(&mut ded), None).map_err(|e| anyhow!("allocate_memory(import): {e:?}"))?;
    d.bind_buffer_memory(buf, mem, 0)?;
    println!("  vk import: fd type bits {:#x}, using type {idx} {:?}", fp.memory_type_bits, ctx.mem_props.memory_types[idx as usize].property_flags);
    Ok(Buffer { buf, mem, size, ptr: std::ptr::null_mut(), flags: ctx.mem_props.memory_types[idx as usize].property_flags })
}

/// Exportable Vulkan allocation; returns (buffer, dma-buf fd).
unsafe fn vk_export(ctx: &Ctx, size: u64) -> Result<(Buffer, i32)> {
    let d = &ctx.device; let ht = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let mut emb = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
    let buf = d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).push_next(&mut emb), None)?;
    let req = d.get_buffer_memory_requirements(buf);
    let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mt = |i: u32| ctx.mem_props.memory_types[i as usize].property_flags;
    let pick = std::env::var("VK_EXPORT_TYPE").ok().and_then(|v| v.parse::<u32>().ok());
    let idx = pick.or_else(|| (0..ctx.mem_props.memory_type_count).find(|&i| req.memory_type_bits & (1 << i) != 0 && mt(i).contains(want) && !mt(i).contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)))
        .or_else(|| (0..ctx.mem_props.memory_type_count).find(|&i| req.memory_type_bits & (1 << i) != 0)).unwrap();
    let mut exp = vk::ExportMemoryAllocateInfo::default().handle_types(ht);
    let mut ded = vk::MemoryDedicatedAllocateInfo::default().buffer(buf);
    let mem = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx).push_next(&mut exp).push_next(&mut ded), None).map_err(|e| anyhow!("allocate_memory(export): {e:?}"))?;
    d.bind_buffer_memory(buf, mem, 0)?;
    let efd = ash::khr::external_memory_fd::Device::new(&ctx.instance, d);
    let fd = efd.get_memory_fd(&vk::MemoryGetFdInfoKHR::default().memory(mem).handle_type(ht)).map_err(|e| anyhow!("get_memory_fd: {e:?}"))?;
    println!("  vk export: type {idx} {:?}", mt(idx));
    Ok((Buffer { buf, mem, size, ptr: std::ptr::null_mut(), flags: mt(idx) }, fd))
}

fn gpu_copy(ctx: &Ctx, src: &Buffer, dst: &Buffer, size: u64) -> Result<f64> {
    ctx.run(|cb| unsafe { ctx.device.cmd_copy_buffer(cb, src.buf, dst.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]) })
}

struct Ref { a: Vec<i8>, b: Vec<i8>, kk: usize, nn: usize, mm: usize }
impl Ref {
    fn check(&self, c: &[u8], tag: &str) -> bool {
        let mut s = 7u64; let mut bad = 0; let ns = 4000;
        for _ in 0..ns { let r = lcg(&mut s) as usize % self.mm; let cc = lcg(&mut s) as usize % self.nn;
            if c[r * self.nn + cc] as i8 != ref_dot(&self.a, &self.b, self.kk, self.nn, r, cc) as i8 { bad += 1; } }
        println!("  [{tag}] verify {ns} samples: {bad} mismatches -> {}", if bad == 0 { "PASS" } else { "FAIL" });
        bad == 0
    }
}

fn time_runs(npu: &Npu, a: &Bo, b: &Bo, c: &Bo, n: usize, ops: f64, tag: &str) -> Result<()> {
    for _ in 0..3 { npu.run(a, b, c)?; }
    let t = Instant::now(); for _ in 0..n { npu.run(a, b, c)?; }
    let avg = t.elapsed().as_secs_f64() / n as f64;
    println!("  [{tag}] GEMM run avg {:.0} us = {:.2} TOPS ({n} iters)", avg * 1e6, ops / avg / 1e12);
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (dir, mm, kk, nn, m, k, n) = (&a[1], p(2), p(3), p(4), p(5), p(6), p(7));
    let mut g = Gemm::new(dir, (mm, kk, nn), (m, k, n), 1)?;
    let ops = g.ops();
    let mut s = 1u64;
    let r = Ref { a: (0..mm * kk).map(|_| lcg(&mut s) as i8).collect(), b: (0..kk * nn).map(|_| lcg(&mut s) as i8).collect(), kk, nn, mm };
    g.set_a(&r.a)?; g.set_b(&r.b)?;
    let (asz, csz) = ((mm * kk) as u64, (mm * nn) as u64);
    let atiled = g.a.bytes().to_vec();
    time_runs(&g.npu, &g.a, &g.b, &g.c, 20, ops, "native BOs")?;
    g.run()?; let c_ref = g.c_bytes()?.to_vec(); r.check(&c_ref, "native");

    let ctx = Ctx::new()?;
    println!("GPU: {} ({}) dmabuf={}", ctx.info.name, ctx.info.driver_info, ctx.has_dmabuf);
    let rb = ctx.buffer_readback(asz.max(csz))?;

    println!("T1 NPU BO -> dma-buf -> Vulkan (GPU reads NPU output C)");
    let t1 = (|| -> Result<()> {
        let t = Instant::now(); let fd = g.c.export_dmabuf()?; let te = ms(t);
        let t = Instant::now(); let vb = unsafe { vk_import(&ctx, fd, csz)? }; let ti = ms(t);
        println!("  export {te:.3} ms, import {ti:.3} ms");
        unsafe { std::ptr::write_bytes(rb.ptr, 0, csz as usize) };
        let gt = gpu_copy(&ctx, &vb, &rb, csz)?;
        let same = unsafe { std::slice::from_raw_parts(rb.ptr, csz as usize) } == &c_ref[..];
        println!("  GPU copy of NPU C: {:.3} ms ({:.1} GB/s), bytes identical to NPU output: {same}", gt * 1e3, csz as f64 / gt / 1e9);
        unsafe { std::ptr::write_bytes(g.c.ptr, 0, csz as usize) }; g.c.to_device()?; g.run()?;
        gpu_copy(&ctx, &vb, &rb, csz)?;
        r.check(unsafe { std::slice::from_raw_parts(rb.ptr, csz as usize) }, "T1 GPU view of fresh NPU C (no host sync)");
        ctx.free_buffer(vb); Ok(())
    })(); if let Err(e) = t1 { println!("  T1 FAILED: {e:#}"); }

    println!("T2 NPU BO -> dma-buf -> Vulkan (GPU writes NPU input A, NPU computes)");
    let t2 = (|| -> Result<()> {
        let fd = g.a.export_dmabuf()?; let vb = unsafe { vk_import(&ctx, fd, asz)? };
        let src = ctx.buffer(asz)?; unsafe { std::ptr::copy_nonoverlapping(atiled.as_ptr(), src.ptr, asz as usize) };
        g.a.bytes_mut().fill(0); g.a.to_device()?;
        let gt = gpu_copy(&ctx, &src, &vb, asz)?;
        println!("  GPU wrote A into NPU BO: {:.3} ms ({:.1} GB/s)", gt * 1e3, asz as f64 / gt / 1e9);
        g.run()?; r.check(g.c_bytes()?, "T2 NPU on GPU-written A (no host sync)");
        ctx.free_buffer(vb); ctx.free_buffer(src); Ok(())
    })(); if let Err(e) = t2 { println!("  T2 FAILED: {e:#}"); }
    g.set_a(&r.a)?;

    println!("T3 Vulkan export dma-buf -> NPU import (NPU writes C into GPU memory)");
    let t3 = (|| -> Result<()> {
        let (vb, fd) = unsafe { vk_export(&ctx, csz)? };
        let t = Instant::now(); let bo = g.npu.import_dmabuf(fd)?; println!("  NPU import {:.3} ms, BO size {}", ms(t), bo.size);
        g.npu.run(&g.a, &g.b, &bo)?;
        gpu_copy(&ctx, &vb, &rb, csz)?;
        r.check(unsafe { std::slice::from_raw_parts(rb.ptr, csz as usize) }, "T3 GPU reads NPU-written C");
        time_runs(&g.npu, &g.a, &g.b, &bo, 20, ops, "C in Vulkan memory")?;
        drop(bo); ctx.free_buffer(vb); Ok(())
    })(); if let Err(e) = t3 { println!("  T3 FAILED: {e:#}"); }

    println!("T4 strixpc (pinned user pages) dma-buf -> NPU import as A");
    let t4 = (|| -> Result<()> {
        if !vkrt::pagecache::available() { anyhow::bail!("/dev/strixpc missing") }
        let len = (asz as usize + (2 << 20) - 1) & !((2 << 20) - 1);
        let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE, -1, 0) } as *mut u8;
        if p as isize == -1 { anyhow::bail!("mmap") }
        unsafe { std::ptr::copy_nonoverlapping(atiled.as_ptr(), p, asz as usize) };
        let t = Instant::now(); let fd = unsafe { vkrt::pagecache::export(p, len)? }; let te = ms(t);
        let t = Instant::now(); let bo = g.npu.import_dmabuf(fd); let ti = ms(t);
        let bo = bo?; println!("  strixpc export {te:.3} ms, NPU import {ti:.3} ms, BO size {}", bo.size);
        g.npu.run(&bo, &g.b, &g.c)?; r.check(g.c_bytes()?, "T4 A from strixpc pages");
        time_runs(&g.npu, &bo, &g.b, &g.c, 20, ops, "A via strixpc")?;
        Ok(())
    })(); if let Err(e) = t4 { println!("  T4 FAILED: {e:#}"); }

    println!("T5 userptr BO (anonymous mmap pages) as A");
    let t5 = (|| -> Result<()> {
        let len = (asz as usize + 4095) & !4095;
        let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE, -1, 0) } as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(atiled.as_ptr(), p, asz as usize) };
        let t = Instant::now(); let bo = unsafe { g.npu.userptr(p, len, 3)? }; println!("  userptr BO create {:.3} ms", ms(t));
        bo.to_device()?; g.npu.run(&bo, &g.b, &g.c)?; r.check(g.c_bytes()?, "T5 A userptr");
        time_runs(&g.npu, &bo, &g.b, &g.c, 20, ops, "A via userptr")?;
        Ok(())
    })(); if let Err(e) = t5 { println!("  T5 FAILED: {e:#}"); }


    println!("T6 userptr BO over a file-backed MAP_SHARED page-cache mapping as A (no strixpc)");
    let path = std::env::var("PC_FILE").unwrap_or("/tmp/npu_share_a.bin".into());
    std::fs::write(&path, &atiled)?;
    for (prot, tag) in [(libc::PROT_READ, "PROT_READ"), (libc::PROT_READ | libc::PROT_WRITE, "PROT_READ|WRITE")] {
        let t6 = (|| -> Result<()> {
            let f = std::fs::OpenOptions::new().read(true).write(true).open(&path)?;
            use std::os::fd::AsRawFd;
            let len = (asz as usize + 4095) & !4095;
            let p = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_SHARED, f.as_raw_fd(), 0) } as *mut u8;
            if p as isize == -1 { anyhow::bail!("mmap file") }
            let t = Instant::now(); let bo = unsafe { g.npu.userptr(p, len, 3)? }; println!("  [{tag}] userptr BO create {:.3} ms", ms(t));
            bo.to_device()?; g.npu.run(&bo, &g.b, &g.c)?; r.check(g.c_bytes()?, &format!("T6 A page-cache {tag}"));
            time_runs(&g.npu, &bo, &g.b, &g.c, 20, ops, &format!("A via page-cache userptr {tag}"))?;
            Ok(())
        })(); if let Err(e) = t6 { println!("  T6 [{tag}] FAILED: {e:#}"); }
    }
    println!("sync costs on native BOs:");
    let t = Instant::now(); for _ in 0..5 { g.c.from_device()?; } println!("  C ({} MB) from_device sync {:.2} ms", csz >> 20, ms(t) / 5.0);
    let t = Instant::now(); for _ in 0..5 { g.a.to_device()?; } println!("  A ({} MB) to_device sync {:.2} ms", asz >> 20, ms(t) / 5.0);
    Ok(())
}
