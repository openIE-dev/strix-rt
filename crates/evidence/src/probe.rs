//! Phase boundary for later hardware telemetry.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0
//!
//! Cloud runs use [`UnavailableProbe`]: wall time is recorded, joules stay absent.
//! [`SysfsProbe`] wraps `receipt::meter` (sysfs microwatts, trapezoid integral) when a
//! real power file is passed in. It does not invent a reading if the file cannot be sampled.

use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct PhaseObs {
    pub phase: &'static str,
    pub wall_ns: u64,
    pub joules: Option<f64>,
    pub unit: &'static str,
    pub source: String,
    pub provenance: &'static str,
    pub meter_samples: u32,
}

pub trait PhaseProbe {
    fn measure<T>(&mut self, phase: &'static str, work: impl FnOnce() -> T) -> (T, PhaseObs);
}

#[derive(Clone, Debug, Default)]
pub struct UnavailableProbe;

impl PhaseProbe for UnavailableProbe {
    fn measure<T>(&mut self, phase: &'static str, work: impl FnOnce() -> T) -> (T, PhaseObs) {
        let t = Instant::now();
        let value = work();
        (
            value,
            PhaseObs {
                phase,
                wall_ns: t.elapsed().as_nanos() as u64,
                joules: None,
                unit: "J",
                source: "unavailable".into(),
                provenance: "unavailable",
                meter_samples: 0,
            },
        )
    }
}

/// Samples one sysfs power file for the duration of `work`.
/// Provenance matches `receipt::meter`: SMU/hwmon telemetry is `ModelBased`, not a shunt.
pub struct SysfsProbe {
    pub path: String,
}

impl PhaseProbe for SysfsProbe {
    fn measure<T>(&mut self, phase: &'static str, work: impl FnOnce() -> T) -> (T, PhaseObs) {
        let started = receipt::meter::Meter::start(&self.path, std::time::Duration::from_millis(2));
        let t = Instant::now();
        let value = work();
        let wall_ns = t.elapsed().as_nanos() as u64;
        match started {
            Some(m) => {
                let (joules, _, n) = m.stop();
                let joules = if joules.is_finite() {
                    Some(joules)
                } else {
                    None
                };
                (
                    value,
                    PhaseObs {
                        phase,
                        wall_ns,
                        joules,
                        unit: "J",
                        source: format!("sysfs_microwatt_file:{}", self.path),
                        provenance: "ModelBased",
                        meter_samples: n as u32,
                    },
                )
            }
            None => (
                value,
                PhaseObs {
                    phase,
                    wall_ns,
                    joules: None,
                    unit: "J",
                    source: format!("sysfs_microwatt_file_unreadable:{}", self.path),
                    provenance: "unavailable",
                    meter_samples: 0,
                },
            ),
        }
    }
}
