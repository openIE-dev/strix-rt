//! Stand-alone energy recorder for NVIDIA Jetson (Thor and friends), using only standard sysfs hwmon files.
//! A Rust sibling of tools/energy/record.py with the same CLI and JSON shape:
//!
//!   jrecord record [--hz N] [--idle S] [--json F] [--csv F] -- CMD [ARGS...]
//!       Run CMD, sample power at N Hz (default 20) while it runs, integrate to joules (trapezoid), write JSON/CSV.
//!   jrecord snapshot
//!       Print one JSON line with the current readings.
//!
//! Channels (each reported only if its sysfs files exist and read):
//!   VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0   INA3221 rails (inN_input mV x currN_input mA)
//!   ina238_power                             INA238 power1_input (uW), the module input rail
//! Like the SMU numbers on the AMD box, these are board rails, NOT wall power: fans, SSD and PSU losses are not in them.
//! Method used throughout the repo: run the same workload with a short and a long token count and take
//! (J_long - J_short) / (N_long - N_short); load, prefill and warm-up are in both runs and cancel.

use std::io::Write as _;
use std::time::{Duration, Instant};

struct Chan { name: String, source: Source }
enum Source { Vi { v: String, i: String }, Pw { p: String } }

fn read_f(p: &str) -> Option<f64> { std::fs::read_to_string(p).ok()?.trim().parse().ok() }

impl Chan {
    fn watts(&self) -> Option<f64> {
        match &self.source {
            Source::Vi { v, i } => Some(read_f(v)? / 1000.0 * (read_f(i)? / 1000.0)), // mV * mA -> W
            Source::Pw { p } => Some(read_f(p)? / 1e6),                               // uW -> W
        }
    }
    fn path(&self) -> &str { match &self.source { Source::Vi { v, .. } => v, Source::Pw { p } => p } }
}

fn discover() -> Vec<Chan> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("/sys/class/hwmon") else { return out };
    for e in rd.flatten() {
        let d = e.path(); let ds = d.to_string_lossy().to_string();
        let name = std::fs::read_to_string(d.join("name")).unwrap_or_default();
        match name.trim() {
            "ina3221" => for ch in 1..=3u32 {
                let (lbl, v, i) = (format!("{ds}/in{ch}_label"), format!("{ds}/in{ch}_input"), format!("{ds}/curr{ch}_input"));
                if let Ok(l) = std::fs::read_to_string(&lbl) {
                    if read_f(&v).is_some() && read_f(&i).is_some() {
                        out.push(Chan { name: l.trim().to_string(), source: Source::Vi { v, i } });
                    }
                }
            },
            "ina238" => {
                let p = format!("{ds}/power1_input");
                if read_f(&p).is_some() { out.push(Chan { name: "ina238_power".into(), source: Source::Pw { p } }); }
            }
            _ => {}
        }
    }
    out
}

fn snapshot(chans: &[Chan]) -> String {
    let fields: Vec<String> = chans.iter().filter_map(|c| c.watts().map(|w| format!("\"{}\": {:.3}", c.name, w))).collect();
    format!("{{{}}}", fields.join(", "))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let chans = discover();
    if chans.is_empty() { eprintln!("jrecord: no ina3221/ina238 hwmon rails found"); std::process::exit(1) }
    match args.first().map(|s| s.as_str()) {
        Some("snapshot") => println!("{}", snapshot(&chans)),
        Some("record") => record(&chans, &args[1..]),
        _ => { eprintln!("usage: jrecord record [--hz N] [--idle S] [--json F] [--csv F] -- CMD [ARGS...] | jrecord snapshot"); std::process::exit(2) }
    }
}

fn record(chans: &[Chan], args: &[String]) {
    let (mut hz, mut idle, mut json_f, mut csv_f) = (20.0f64, 0.0f64, None::<String>, None::<String>);
    let mut cmd = Vec::new(); let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hz" => hz = it.next().and_then(|v| v.parse().ok()).unwrap_or(20.0),
            "--idle" => idle = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            "--json" => json_f = it.next().cloned(),
            "--csv" => csv_f = it.next().cloned(),
            "--" => { cmd = it.cloned().collect(); break }
            other => { eprintln!("jrecord: unknown arg {other}"); std::process::exit(2) }
        }
    }
    if cmd.is_empty() { eprintln!("jrecord: no command after --"); std::process::exit(2) }

    // samples[i] = (t_s, per-channel W at that instant)
    let period = Duration::from_secs_f64(1.0 / hz);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let samples = std::sync::Mutex::new(Vec::<(f64, Vec<f64>)>::new());
    let t0 = Instant::now();
    let status = std::thread::scope(|s| {
        s.spawn(|| {
            let mut next = Instant::now();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let t = t0.elapsed().as_secs_f64();
                let w: Vec<f64> = chans.iter().map(|c| c.watts().unwrap_or(f64::NAN)).collect();
                samples.lock().unwrap().push((t, w));
                next += period; let now = Instant::now();
                if next > now { std::thread::sleep(next - now) } else { next = now }
            }
        });
        if idle > 0.0 { std::thread::sleep(Duration::from_secs_f64(idle)) }
        let st = std::process::Command::new(&cmd[0]).args(&cmd[1..]).status();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        st
    });
    let samples = samples.into_inner().unwrap();
    let dur = t0.elapsed().as_secs_f64();
    match &status { Ok(s) if s.success() => {}, Ok(s) => eprintln!("jrecord: CMD exited {s}"), Err(e) => eprintln!("jrecord: CMD failed: {e}") }

    // trapezoid per channel, skipping NaN gaps
    let nc = chans.len();
    let mut joules = vec![0.0f64; nc];
    for k in 1..samples.len() {
        let (ta, wa) = &samples[k - 1]; let (tb, wb) = &samples[k];
        for c in 0..nc { if wa[c].is_finite() && wb[c].is_finite() { joules[c] += (wa[c] + wb[c]) * 0.5 * (tb - ta) } }
    }
    let avg: Vec<f64> = (0..nc).map(|c| if dur > 0.0 { joules[c] / dur } else { 0.0 }).collect();

    if let Some(f) = &csv_f {
        let mut s = String::from("t_s"); for c in chans { s += &format!(",{}_w", c.name.to_lowercase()) } s += "\n";
        for (t, w) in &samples { s += &format!("{t:.4}"); for x in w { s += &format!(",{x:.3}") } s += "\n"; }
        std::fs::write(f, s).unwrap_or_else(|e| eprintln!("jrecord: csv write: {e}"));
    }
    let brk: Vec<String> = (0..nc).map(|c| format!(
        "{{\"fabric\": \"{}\", \"joules\": {:.3}, \"avg_w\": {:.3}, \"source\": \"{}\"}}", chans[c].name, joules[c], avg[c], chans[c].path())).collect();
    let out = format!("{{\"duration_s\": {dur:.3}, \"hz\": {hz}, \"samples\": {}, \"breakdown\": {{\"channels\": [{}]}}}}\n", samples.len(), brk.join(", "));
    if let Some(f) = &json_f { std::fs::write(f, &out).unwrap_or_else(|e| eprintln!("jrecord: json write: {e}")) }
    print!("{out}"); std::io::stdout().flush().ok();
    std::process::exit(status.ok().and_then(|s| s.code()).unwrap_or(1));
}
