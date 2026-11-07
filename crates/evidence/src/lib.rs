//! Delayed-query evidence conditions on synthetic histories.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0
//!
//! Full history, summary-only, and compact-plus-source are experimental conditions.
//! They are not a standard and this crate does not rank a winner. The diagnostic
//! oracle checks synthetic facts only. Joules are recorded only when a phase probe
//! actually samples a meter; the cloud runner uses [`probe::UnavailableProbe`].

pub mod events;
pub mod probe;
pub mod report;
pub mod scenario;
pub mod store;
pub mod strategy;

use probe::{PhaseObs, PhaseProbe, UnavailableProbe};
use report::{relation, CaseObs, ConditionObs, Report};
use scenario::{faults, world};
use strategy::{Budget, CompactRecoverable, FullHistory, SummaryOnly};

pub fn run_discovery(seed: u64) -> Report {
    run_with(
        seed,
        &mut UnavailableProbe,
        Budget {
            max_reads: 8,
            max_bytes: 1_000_000,
        },
    )
}

pub fn run_with<P: PhaseProbe>(seed: u64, probe: &mut P, budget: Budget) -> Report {
    let mut world = world(seed);
    let events = world.events.clone();
    let (full, full_c) = probe.measure("compact", || FullHistory::compact(&events));
    let (summary, sum_c) = probe.measure("compact", || SummaryOnly::compact(&events));
    let (compact, cmp_c) = probe.measure("compact", || CompactRecoverable::compact(&events));
    faults(&mut world.store);
    let mut full_cases = vec![];
    let mut sum_cases = vec![];
    let mut cmp_cases = vec![];
    for d in &world.diagnostics {
        world.store.begin_case();
        let (obs, retrieve) = probe.measure("retrieve", || {
            full.answer(d.entity, d.slot, &mut world.store, &budget)
        });
        full_cases.push(case(d, obs, retrieve, &world.store));
        world.store.begin_case();
        let (obs, retrieve) =
            probe.measure("retrieve", || summary.answer(d.entity, d.slot, &budget));
        sum_cases.push(case(d, obs, retrieve, &world.store));
        world.store.begin_case();
        let (obs, retrieve) = probe.measure("retrieve", || {
            compact.answer(d.entity, d.slot, &mut world.store, &budget)
        });
        cmp_cases.push(case(d, obs, retrieve, &world.store));
    }
    let auth_blob = world
        .events
        .iter()
        .find(|e| e.note.contains("rm -rf"))
        .map(|e| e.note.clone())
        .unwrap_or_default();
    let decision = receipt::auth::authorize(&[], auth_blob.as_bytes(), None);
    let surface =
        receipt::auth::DecisionsAuthorization::assess(&[], [0; 32], "threadA", [1; 32], None);
    Report {
        seed,
        kind: "synthetic_delayed_query_discovery",
        telemetry_availability: "unavailable",
        conditions: vec![
            ConditionObs {
                name: "full_history",
                compact: full_c,
                cases: full_cases,
            },
            ConditionObs {
                name: "summary_only",
                compact: sum_c,
                cases: sum_cases,
            },
            ConditionObs {
                name: "compact_recoverable",
                compact: cmp_c,
                cases: cmp_cases,
            },
        ],
        auth_unstructured: decision.denial() == Some(receipt::auth::DenyReason::Unstructured),
        auth_executes: surface.to_json().contains("\"executes\":true"),
    }
}

fn case(
    d: &scenario::Diagnostic,
    observation: strategy::Observation,
    retrieve: PhaseObs,
    store: &store::SourceStore,
) -> CaseObs {
    let relation = relation(&d.expect_commit, &observation.answer);
    CaseObs {
        query_id: d.query_id.into(),
        entity: d.entity.into(),
        slot: d.slot.into(),
        diagnostic_note: d.note.into(),
        diagnostic_commit: d.expect_commit.clone(),
        relation,
        observation,
        retrieve,
        retry: PhaseObs {
            phase: "retry",
            wall_ns: store.retry_ns,
            joules: None,
            unit: "J",
            source: "unavailable".into(),
            provenance: "unavailable",
            meter_samples: 0,
        },
        verify: PhaseObs {
            phase: "verify",
            wall_ns: store.verify_ns,
            joules: None,
            unit: "J",
            source: "unavailable".into(),
            provenance: "unavailable",
            meter_samples: 0,
        },
    }
}

pub fn find<'a>(report: &'a Report, condition: &str, query: &str) -> &'a CaseObs {
    report
        .conditions
        .iter()
        .find(|c| c.name == condition)
        .and_then(|c| c.cases.iter().find(|x| x.query_id == query))
        .unwrap_or_else(|| panic!("missing {condition}/{query}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::Answer;
    use std::time::Duration;

    fn rep() -> Report {
        run_discovery(7)
    }

    #[test]
    fn histories_match_and_queries_are_not_inputs_to_compaction() {
        let w = scenario::world(7);
        let a = FullHistory::compact(&w.events);
        let b = FullHistory::compact(&w.events);
        assert_eq!(
            a.answer(
                "T1",
                "serial",
                &mut scenario::world(7).store,
                &Budget {
                    max_reads: 8,
                    max_bytes: 1_000_000
                }
            )
            .bytes_retained,
            b.answer(
                "T1",
                "serial",
                &mut scenario::world(7).store,
                &Budget {
                    max_reads: 8,
                    max_bytes: 1_000_000
                }
            )
            .bytes_retained
        );
        let s = SummaryOnly::compact(&w.events);
        assert!(s
            .answer(
                "T1",
                "serial",
                &Budget {
                    max_reads: 8,
                    max_bytes: 1_000_000
                }
            )
            .lost
            .iter()
            .any(|l| l.contains("serial")));
    }

    #[test]
    fn observations_are_per_case_and_unranked() {
        let r = rep();
        assert_eq!(r.conditions.len(), 3);
        assert!(r.to_json().contains("\"ranking\": null"));
        assert!(r.to_json().contains("\"efficiency_claim\": false"));
        let full = find(&r, "full_history", "t1_serial");
        let summary = find(&r, "summary_only", "t1_serial");
        let compact = find(&r, "compact_recoverable", "t1_serial");
        assert!(matches!(full.observation.answer, Answer::Commit(_)));
        assert!(summary
            .observation
            .lost
            .iter()
            .any(|l| l.contains("serial")));
        assert!(matches!(
            summary.observation.answer,
            Answer::Refuse("no_evidence")
        ));
        assert!(compact
            .observation
            .recovered
            .iter()
            .any(|x| x.contains("verified")));
        assert_eq!(full.relation, "same_commit");
        assert_eq!(summary.relation, "refuse_when_diagnostic_has_fact");
    }

    #[test]
    fn conflict_stale_missing_corrupt_and_distractor() {
        let r = rep();
        assert_eq!(
            find(&r, "full_history", "t1_status").relation,
            "same_refusal"
        );
        assert_eq!(
            find(&r, "compact_recoverable", "t1_status").relation,
            "same_refusal"
        );
        assert_eq!(
            find(&r, "summary_only", "t1_status").relation,
            "commit_when_diagnostic_refuses"
        );
        assert_eq!(
            find(&r, "full_history", "t2_status").relation,
            "same_commit"
        );
        assert!(
            matches!(find(&r, "full_history", "t2_status").observation.answer, Answer::Commit(ref v) if v == "closed")
        );
        assert_eq!(find(&r, "summary_only", "t1_rack").relation, "same_refusal");
        assert!(matches!(
            find(&r, "compact_recoverable", "t3_badge")
                .observation
                .answer,
            Answer::Refuse("missing_source")
        ));
        assert!(matches!(
            find(&r, "compact_recoverable", "t4_lot").observation.answer,
            Answer::Refuse("corrupt_source")
        ));
        assert!(matches!(
            find(&r, "full_history", "t4_lot").observation.answer,
            Answer::Commit(_)
        ));
        let serial = find(&r, "full_history", "t1_serial");
        if let Answer::Commit(v) = &serial.observation.answer {
            assert!(!v.ends_with("4418"), "near-copy serial used: {v}");
        }
    }

    #[test]
    fn tight_budget_can_fail_without_being_a_ranking() {
        let mut probe = UnavailableProbe;
        let r = run_with(
            7,
            &mut probe,
            Budget {
                max_reads: 1,
                max_bytes: 10_000,
            },
        );
        let full = find(&r, "full_history", "t1_serial");
        let compact = find(&r, "compact_recoverable", "t1_serial");
        assert!(
            matches!(full.observation.answer, Answer::Commit(_)),
            "full={:?} reads={:?}",
            full.observation.answer,
            full.observation.reads
        );
        assert!(
            matches!(compact.observation.answer, Answer::Refuse(_)),
            "compact={:?} reads={:?}",
            compact.observation.answer,
            compact.observation.reads
        );
        assert!(compact
            .observation
            .transitions
            .iter()
            .any(|t| t.contains("budget") || compact.observation.retrieval_calls <= 2));
    }

    #[test]
    fn source_text_does_not_grant_and_joules_stay_absent() {
        let r = rep();
        assert!(r.auth_unstructured);
        assert!(!r.auth_executes);
        let json = r.to_json();
        assert!(json.contains("\"joules\":null"));
        assert!(!json.contains("\"joules\":0"));
        for c in &r.conditions {
            assert!(c.compact.joules.is_none());
            assert_eq!(c.compact.provenance, "unavailable");
            assert!(
                c.compact.wall_ns > 0
                    || c.cases
                        .iter()
                        .any(|x| x.retrieve.wall_ns > 0 || x.observation.bytes_retained > 0)
            );
        }
    }

    #[test]
    fn sysfs_probe_labels_a_real_sample_and_is_not_the_discovery_run() {
        let path = std::env::temp_dir().join(format!("evidence-mw-{}", std::process::id()));
        std::fs::write(&path, "20000000\n").unwrap();
        let mut probe = crate::probe::SysfsProbe {
            path: path.to_string_lossy().into(),
        };
        let (_, obs) = probe.measure("compact", || std::thread::sleep(Duration::from_millis(30)));
        let _ = std::fs::remove_file(&path);
        assert_eq!(obs.unit, "J");
        assert_eq!(obs.provenance, "ModelBased");
        assert!(obs.source.starts_with("sysfs_microwatt_file:"));
        assert!(obs.joules.unwrap_or(0.0) > 0.0);
        assert!(obs.meter_samples > 0);
        assert_eq!(rep().telemetry_availability, "unavailable");
    }
}
