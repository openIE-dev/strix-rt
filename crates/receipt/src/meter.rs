//! In-process energy meter for one slice: samples a sysfs power file (microwatts, e.g. amdgpu hwmon power1_input = the
//! SMU's PPT socket power) on a background thread and integrates (trapezoid). Gross joules, not net of the idle floor.
//! JouleClaw provenance: ModelBased (SMU telemetry is firmware-modelled, not a shunt).
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
use std::time::{Duration, Instant};

/// Find the amdgpu hwmon power file, or honour KEV_POWER=<path>.
pub fn detect() -> Option<String> {
    if let Ok(p) = std::env::var("KEV_POWER") { return Some(p) }
    for e in std::fs::read_dir("/sys/class/hwmon").ok()?.flatten() {
        let d = e.path();
        if std::fs::read_to_string(d.join("name")).map(|n| n.trim() == "amdgpu").unwrap_or(false) {
            for f in ["power1_input", "power1_average"] { let p = d.join(f); if p.exists() { return Some(p.to_string_lossy().into()) } }
        }
    }
    None
}

fn read_w(p: &str) -> Option<f64> { std::fs::read_to_string(p).ok()?.trim().parse::<f64>().ok().map(|uw| uw / 1e6) }

pub struct Meter { stop: Arc<AtomicBool>, h: std::thread::JoinHandle<(f64, usize)>, t0: Instant }

impl Meter {
    pub fn start(path: &str, period: Duration) -> Option<Meter> {
        let p = path.to_string(); let w0 = read_w(&p)?;
        let stop = Arc::new(AtomicBool::new(false)); let s2 = stop.clone(); let t0 = Instant::now();
        let h = std::thread::spawn(move || {
            let (mut j, mut last_w, mut last_t, mut n) = (0.0, w0, Instant::now(), 1usize);
            loop {
                let done = s2.load(Ordering::Relaxed);
                if !done { std::thread::sleep(period) }
                let now = Instant::now();
                if let Some(w) = read_w(&p) { j += 0.5 * (w + last_w) * (now - last_t).as_secs_f64(); last_w = w; last_t = now; n += 1 }
                if done { return (j, n) }
            }
        });
        Some(Meter { stop, h, t0 })
    }
    /// (joules, seconds, samples)
    pub fn stop(self) -> (f64, f64, usize) {
        self.stop.store(true, Ordering::Relaxed); let (j, n) = self.h.join().unwrap_or((f64::NAN, 0)); (j, self.t0.elapsed().as_secs_f64(), n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn integrates_constant_power() {
        let p = std::env::temp_dir().join(format!("meter-{}", std::process::id()));
        std::fs::write(&p, "20000000\n").unwrap(); // 20 W
        let m = Meter::start(p.to_str().unwrap(), Duration::from_millis(2)).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let (j, s, n) = m.stop();
        assert!(n > 5 && (j - 20.0 * s).abs() < 20.0 * 0.05, "j={j} s={s} n={n}");
        let _ = std::fs::remove_file(p);
    }
    #[test] fn missing_file_is_none() { assert!(Meter::start("/nonexistent/power", Duration::from_millis(1)).is_none()) }
}
