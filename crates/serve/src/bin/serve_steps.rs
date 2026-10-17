//! serve_steps <model.gguf> [maxm=8] [pos=600]: GPU ms of one record_multi decode step for m = 1..maxm sequences
//! (slots at KV depth pos), plus slot_load copy time. Scratch state; output tokens meaningless.
use anyhow::Result;
use qwen38::multi::{POSW, TOKW};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let maxm: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8); let pos: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(600);
    std::env::set_var("NCOL", maxm.max(5).to_string());
    let m = qwen38::Model::load(&a[1], 2048)?;
    let slots: Vec<_> = (0..maxm).map(|_| m.new_slot()).collect::<Result<_>>()?; let parts = m.new_parts()?; m.multi_init();
    let t = std::time::Instant::now(); m.slot_load(&slots[0], pos)?; println!("slot_load({pos}): {:.2} ms wall", t.elapsed().as_secs_f64() * 1e3);
    let au = m.au();
    for n in 1..=maxm {
        let sl: Vec<_> = slots[..n].iter().collect();
        let mut err = Ok(()); let cb = m.ctx.record_raw(|cb| { err = m.record_multi(cb, &sl, &parts); })?; err?;
        for j in 0..n { au[POSW as usize + j] = pos; au[TOKW as usize + j] = 1000 + j as u32; }
        m.ctx.submit_repeated(cb, 1)?;
        let mut best = f64::MAX; for _ in 0..5 { best = best.min(m.ctx.submit_repeated(cb, 1)?); }
        println!("m={n}: {:.2} ms/step GPU -> {:.1} tok/s aggregate", best * 1e3, n as f64 / best);
        // SPOW=N: N back-to-back steps with package power (amdgpu hwmon power1_input) sampled every 20 ms -> W and J/token
        if let Some(ns) = std::env::var("SPOW").ok().and_then(|v| v.parse::<usize>().ok()) {
            let h = std::fs::read_dir("/sys/class/drm/card1/device/hwmon")?.next().unwrap()?.path().join("power1_input");
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)); let st2 = stop.clone(); let h2 = h.clone();
            let th = std::thread::spawn(move || { let mut v = vec![]; while !st2.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(x) = std::fs::read_to_string(&h2) { v.push(x.trim().parse::<f64>().unwrap_or(0.0) / 1e6) } std::thread::sleep(std::time::Duration::from_millis(20)); } v });
            let t = std::time::Instant::now(); m.ctx.submit_repeated(cb, ns)?; let w = t.elapsed().as_secs_f64();
            stop.store(true, std::sync::atomic::Ordering::Relaxed); let v = th.join().unwrap(); let pw = v.iter().sum::<f64>() / v.len().max(1) as f64;
            println!("  m={n} power: {ns} steps {:.2} ms/step wall, {:.1} W avg ({} samples) -> {:.3} J/token", w / ns as f64 * 1e3, pw, v.len(), pw * w / (ns * n) as f64);
        }
    }
    Ok(())
}
