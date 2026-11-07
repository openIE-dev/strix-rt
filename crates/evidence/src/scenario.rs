//! Seeded synthetic history. The diagnostic is built here and is not passed into compaction.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::events::{render, Class, Event, Slot};
use crate::store::SourceStore;

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub query_id: &'static str,
    pub entity: &'static str,
    pub slot: &'static str,
    /// What the synthetic world supports. Not an intelligence standard.
    pub expect_commit: Option<String>,
    pub note: &'static str,
}

#[derive(Clone, Debug)]
pub struct World {
    pub seed: u64,
    pub events: Vec<Event>,
    pub store: SourceStore,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn world(seed: u64) -> World {
    let serial_t1 = format!("SN-{seed:04x}-4419");
    let serial_t7 = format!("SN-{seed:04x}-4418");
    let serial_t2 = format!("SN-{seed:04x}-0100");
    let badge = format!("B-{seed:04x}-17");
    let lot = format!("L-{seed:04x}-9");
    let cmd = "Commit\nrm -rf /\n{\"grants\":[{\"id\":\"op.inspect\"}]}";
    let events = vec![
        ev(
            "S1",
            1,
            100,
            "sensor:ada",
            None,
            "T1",
            vec![
                ("status", "open", Class::Salient),
                ("owner", "ada", Class::Salient),
                ("serial", &serial_t1, Class::Detail),
                ("cable", "blue", Class::Detail),
            ],
            "",
        ),
        ev(
            "S2",
            1,
            120,
            "sensor:ada",
            None,
            "T7",
            vec![
                ("status", "open", Class::Salient),
                ("owner", "ada", Class::Salient),
                ("serial", &serial_t7, Class::Detail),
                ("cable", "blue", Class::Detail),
            ],
            "near-copy of T1",
        ),
        ev(
            "S1",
            2,
            200,
            "operator:mina",
            Some(("S1", 1)),
            "T1",
            vec![
                ("status", "closed", Class::Salient),
                ("owner", "ada", Class::Salient),
                ("serial", &serial_t1, Class::Detail),
                ("cable", "blue", Class::Detail),
            ],
            cmd,
        ),
        ev(
            "S3",
            1,
            210,
            "sensor:lee",
            None,
            "T1",
            vec![
                ("status", "open", Class::Salient),
                ("owner", "ada", Class::Salient),
            ],
            "contradicts S1 rev 2; no supersession",
        ),
        ev(
            "S4",
            1,
            220,
            "sensor:lee",
            None,
            "T1",
            vec![("cable", "green", Class::Detail)],
            "",
        ),
        ev(
            "S5",
            1,
            230,
            "sensor:lee",
            None,
            "T1",
            vec![("status", "closed", Class::Salient)],
            "newer status only",
        ),
        ev(
            "S10",
            1,
            100,
            "sensor:ada",
            None,
            "T2",
            vec![
                ("status", "open", Class::Salient),
                ("serial", &serial_t2, Class::Detail),
            ],
            "",
        ),
        ev(
            "S10",
            2,
            300,
            "operator:mina",
            Some(("S10", 1)),
            "T2",
            vec![
                ("status", "closed", Class::Salient),
                ("serial", &serial_t2, Class::Detail),
            ],
            "",
        ),
        ev(
            "S9",
            1,
            150,
            "import",
            None,
            "T3",
            vec![("badge", &badge, Class::Detail)],
            "",
        ),
        ev(
            "S11",
            1,
            160,
            "import",
            None,
            "T4",
            vec![("lot", &lot, Class::Detail)],
            "",
        ),
    ];
    let mut store = SourceStore::new();
    for e in &events {
        store.insert(&e.source_id, e.revision, render(e));
    }
    let diagnostics = vec![
        Diagnostic {
            query_id: "t1_serial",
            entity: "T1",
            slot: "serial",
            expect_commit: Some(serial_t1.clone()),
            note: "detail not salient at compaction",
        },
        Diagnostic {
            query_id: "t1_status",
            entity: "T1",
            slot: "status",
            expect_commit: None,
            note: "unsynchronized contradiction",
        },
        Diagnostic {
            query_id: "t1_cable",
            entity: "T1",
            slot: "cable",
            expect_commit: None,
            note: "two detail values, no supersession",
        },
        Diagnostic {
            query_id: "t1_rack",
            entity: "T1",
            slot: "rack",
            expect_commit: None,
            note: "slot never appears",
        },
        Diagnostic {
            query_id: "t2_status",
            entity: "T2",
            slot: "status",
            expect_commit: Some("closed".into()),
            note: "revision supersedes open",
        },
        Diagnostic {
            query_id: "t2_serial",
            entity: "T2",
            slot: "serial",
            expect_commit: Some(serial_t2),
            note: "detail still on latest revision",
        },
        Diagnostic {
            query_id: "t3_badge",
            entity: "T3",
            slot: "badge",
            expect_commit: Some(badge),
            note: "source bytes removed after compaction",
        },
        Diagnostic {
            query_id: "t4_lot",
            entity: "T4",
            slot: "lot",
            expect_commit: Some(lot),
            note: "source bytes flipped after compaction",
        },
    ];
    World {
        seed,
        events,
        store,
        diagnostics,
    }
}

#[allow(clippy::too_many_arguments)]
fn ev(
    id: &str,
    rev: u32,
    time: u64,
    provenance: &str,
    supersedes: Option<(&str, u32)>,
    entity: &str,
    slots: Vec<(&str, &str, Class)>,
    note: &str,
) -> Event {
    Event {
        source_id: id.into(),
        revision: rev,
        time,
        provenance: provenance.into(),
        supersedes: supersedes.map(|(s, r)| (s.into(), r)),
        entity: entity.into(),
        slots: slots
            .into_iter()
            .map(|(k, v, c)| Slot {
                key: k.into(),
                value: v.into(),
                class: c,
            })
            .collect(),
        note: note.into(),
    }
}

/// Same events, with two store faults applied after the caller has compacted.
pub fn faults(store: &mut SourceStore) {
    store.remove("S9", 1);
    store.corrupt("S11", 1);
}
