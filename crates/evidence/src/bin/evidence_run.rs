//! Write one seeded discovery report. Does not start a service or load a model.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use evidence::probe::{SysfsProbe, UnavailableProbe};
use evidence::strategy::Budget;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let mut seed = 7u64;
    let mut out = String::from("evidence-run.json");
    let mut power: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "--out" => out = args.next().unwrap_or(out),
            "--power" => power = args.next(),
            _ => {}
        }
    }
    let report = if let Some(path) = power {
        let mut probe = SysfsProbe { path };
        evidence::run_with(
            seed,
            &mut probe,
            Budget {
                max_reads: 8,
                max_bytes: 1_000_000,
            },
        )
    } else {
        let mut probe = UnavailableProbe;
        evidence::run_with(
            seed,
            &mut probe,
            Budget {
                max_reads: 8,
                max_bytes: 1_000_000,
            },
        )
    };
    let json = report.to_json();
    if let Err(e) = fs::write(&out, &json) {
        eprintln!("write {out}: {e}");
        return ExitCode::from(1);
    }
    println!(
        "wrote {out} seed={seed} telemetry={} conditions={}",
        report.telemetry_availability,
        report.conditions.len()
    );
    ExitCode::from(0)
}
