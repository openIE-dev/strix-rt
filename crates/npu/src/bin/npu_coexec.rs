//! npu_coexec <i32-out NPU build for M x K x Nn> M K Nn Ng m k n [iters]
//! Co-execution of one prefill linear split along N: the NPU computes Nn columns (int8 x int8 -> int32, dc2 build),
//! the GPU computes Ng columns with crates/ternary's coop-matrix GEMM (int8 x PQ2_0 ternary -> f32), concurrently.
//! Reports each engine alone, then both at once (median wall per iteration, plus each engine's own time while overlapped).
//! Inputs are synthetic (perf measurement); each side's correctness is covered by npu_gemm/npu_e2e and tern_gemm.
use npu::*; use std::{sync::mpsc, time::Instant};
use vkrt::{push_u32, Ctx};
fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 33 }
// (median, min): median is robust to GPU contention spikes from other processes
fn stats(v: &[f64]) -> (f64, f64) { let mut w = v.to_vec(); w.sort_by(|a, b| a.partial_cmp(b).unwrap()); (w[w.len() / 2], w[0]) }
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (dir, mm, kk, nn, ng, m, k, n) = (a[1].clone(), p(2), p(3), p(4), p(5), p(6), p(7), p(8));
    let iters = if a.len() > 9 { p(9) } else { 30 };
    let ntot = nn + ng; let ops = |cols: usize| 2.0 * (mm * kk * cols) as f64;
    // ---- NPU worker thread owns the XRT objects ----
    let (go_tx, go_rx) = mpsc::channel::<bool>(); let (done_tx, done_rx) = mpsc::channel::<f64>();
    let npu_on = nn > 0;
    let h = if npu_on { Some(std::thread::spawn(move || -> anyhow::Result<()> {
        let mut g = Gemm::new(&dir, (mm, kk, nn), (m, k, n), 4)?;
        let mut s = 3u64;
        let av: Vec<i8> = (0..mm * kk).map(|_| lcg(&mut s) as i8).collect(); let bv: Vec<i8> = (0..kk * nn).map(|_| lcg(&mut s) as i8).collect();
        g.set_a(&av)?; g.set_b(&bv)?; for _ in 0..3 { g.run()?; }
        done_tx.send(0.0)?;
        while let Ok(true) = go_rx.recv() { let t = Instant::now(); g.run()?; done_tx.send(t.elapsed().as_secs_f64())?; }
        Ok(())
    })) } else { None };
    if npu_on { done_rx.recv()?; }
    // ---- GPU side ----
    let ctx = Ctx::new()?;
    let (gpu_run, mut gpu_free): (Box<dyn Fn() -> anyhow::Result<f64>>, Vec<vkrt::Buffer>) = if ng > 0 {
        let kg = ctx.kernel(ternary::shaders::GEMM_PQ2, 4, 16, &[], Some(32))?;
        let rb = kk / 128 * 34;
        let wb = ctx.buffer((ng * rb) as u64)?; { let w = wb.bytes_mut(); let mut s = 7u64;
            for (i, b) in w.iter_mut().enumerate() { *b = if i % 34 == 0 { 0x00 } else if i % 34 == 1 { 0x3C } else { lcg(&mut s) as u8 }; } } // f16 scale 1.0
        let xq = ctx.buffer((mm * kk) as u64)?; { let mut s = 9u64; for b in xq.bytes_mut() { *b = lcg(&mut s) as u8; } }
        let xs = ctx.buffer((mm * kk / 128 * 4) as u64)?; for v in xs.as_f32_mut() { *v = 1.0; }
        let yb = ctx.buffer((mm * ng * 4) as u64)?;
        let ds = ctx.bind(&kg, &[&wb, &xq, &xs, &yb])?; let groups = ((mm / 64) * (ng / 64)) as u32;
        let pc = push_u32(&[mm as u32, ng as u32, kk as u32, 0]);
        let ctxr = &ctx as *const Ctx as usize; let kgb = Box::new(kg);
        let f = move || -> anyhow::Result<f64> { let c = unsafe { &*(ctxr as *const Ctx) }; let t = Instant::now(); c.run(|cb| c.dispatch(cb, &kgb, ds, &pc, groups))?; Ok(t.elapsed().as_secs_f64()) };
        (Box::new(f), vec![wb, xq, xs, yb])
    } else { (Box::new(|| Ok(0.0)), vec![]) };
    for _ in 0..3 { gpu_run()?; }
    let npu_run = || -> anyhow::Result<f64> { if npu_on { go_tx.send(true)?; Ok(done_rx.recv()?) } else { Ok(0.0) } };
    // alone
    let (mut tn, mut tg) = (vec![], vec![]);
    for _ in 0..iters { tn.push(npu_run()?); } for _ in 0..iters { tg.push(gpu_run()?); }
    // together: one iteration = both engines started at once, wall until both done
    let (mut tw, mut tn2, mut tg2) = (vec![], vec![], vec![]);
    for _ in 0..iters {
        let t = Instant::now();
        if npu_on { go_tx.send(true)?; }
        let gt = gpu_run()?;
        let nt = if npu_on { done_rx.recv()? } else { 0.0 };
        tw.push(t.elapsed().as_secs_f64()); tn2.push(nt); tg2.push(gt);
    }
    let (an, bn) = stats(&tn); let (ag, bg) = stats(&tg); let (aw, bw) = stats(&tw); let (an2, _) = stats(&tn2); let (ag2, _) = stats(&tg2);
    let tops = |o: f64, t: f64| if t > 0.0 { o / t / 1e12 } else { 0.0 };
    println!("{mm}x{kk}x{ntot} split NPU {nn} / GPU {ng} ({:.1}% NPU), {iters} iters", 100.0 * nn as f64 / ntot as f64);
    println!("  alone:    NPU {:.3} ms ({:.2} TOPS, best {:.3}) | GPU {:.3} ms ({:.2} TOPS, best {:.3})", an * 1e3, tops(ops(nn), an), bn * 1e3, ag * 1e3, tops(ops(ng), ag), bg * 1e3);
    println!("  together: wall {:.3} ms (best {:.3}) = {:.2} TOPS combined | NPU {:.3} ms ({:+.1}% vs alone) | GPU {:.3} ms ({:+.1}% vs alone)",
        aw * 1e3, bw * 1e3, tops(ops(ntot), aw), an2 * 1e3, 100.0 * (an2 / an.max(1e-12) - 1.0), ag2 * 1e3, 100.0 * (ag2 / ag.max(1e-12) - 1.0));
    drop(go_tx); if let Some(h) = h { h.join().unwrap()?; }
    for b in gpu_free.drain(..) { ctx.free_buffer(b); }
    Ok(())
}
