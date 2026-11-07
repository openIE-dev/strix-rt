//! Three experimental conditions over one visible history.
//! None of them is a normative baseline. Compaction does not take a query.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::events::{parse_body, preview_salient, render, Class, Event};
use crate::store::{ReadOut, SourceStore};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Budget {
    pub max_reads: u32,
    pub max_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    Commit(String),
    Refuse(&'static str),
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub retained: Vec<String>,
    pub lost: Vec<String>,
    pub recovered: Vec<String>,
    pub transitions: Vec<String>,
    pub reads: Vec<String>,
    pub answer: Answer,
    pub bytes_retained: u64,
    pub bytes_read: u64,
    pub retrieval_calls: u64,
    pub retries: u64,
    pub verify_ns: u64,
}

struct Rec {
    source_id: String,
    revision: u32,
    time: u64,
    entity: String,
    preview: String,
    inline: Option<String>,
    hash: [u8; 32],
}

pub struct FullHistory {
    recs: Vec<Rec>,
    bytes: u64,
    lost: Vec<String>,
    transitions: Vec<String>,
}

pub struct SummaryOnly {
    /// Last salient write per (entity, slot). Details are not stored.
    map: BTreeMap<(String, String), (String, u64)>,
    bytes: u64,
    lost: Vec<String>,
    retained: Vec<String>,
    transitions: Vec<String>,
}

pub struct CompactRecoverable {
    recs: Vec<Rec>,
    bytes: u64,
    lost: Vec<String>,
    retained: Vec<String>,
    transitions: Vec<String>,
}

impl FullHistory {
    pub fn compact(events: &[Event]) -> Self {
        let mut recs = vec![];
        let mut bytes = 0;
        let mut transitions = vec!["compact: query not visible".into()];
        for e in events {
            let body = render(e);
            bytes += body.len() as u64;
            transitions.push(format!(
                "retain full {} rev {} ({} bytes)",
                e.source_id,
                e.revision,
                body.len()
            ));
            recs.push(Rec {
                source_id: e.source_id.clone(),
                revision: e.revision,
                time: e.time,
                entity: e.entity.clone(),
                preview: body.clone(),
                inline: Some(body),
                hash: crate::events::digest(render(e).as_bytes()),
            });
        }
        Self {
            recs,
            bytes,
            lost: vec![],
            transitions,
        }
    }

    pub fn answer(
        &self,
        entity: &str,
        slot: &str,
        store: &mut SourceStore,
        budget: &Budget,
    ) -> Observation {
        let _ = store;
        decide(
            &self.recs,
            entity,
            slot,
            None,
            budget,
            self.bytes,
            &self.lost,
            &self.transitions,
        )
    }
}

impl SummaryOnly {
    pub fn compact(events: &[Event]) -> Self {
        let mut map = BTreeMap::new();
        let mut lost = vec![];
        let mut transitions = vec!["compact: query not visible; detail slots dropped".into()];
        for e in events {
            for s in &e.slots {
                if s.class == Class::Detail {
                    lost.push(format!(
                        "{} {} {} (detail class, not kept)",
                        e.entity, s.key, e.source_id
                    ));
                    transitions.push(format!("drop detail {}.{}", e.entity, s.key));
                    continue;
                }
                let key = (e.entity.clone(), s.key.clone());
                let replace = map.get(&key).map(|(_, t)| e.time >= *t).unwrap_or(true);
                if replace {
                    transitions.push(format!(
                        "last-write {}.{} from {} rev {} at {}",
                        e.entity, s.key, e.source_id, e.revision, e.time
                    ));
                    map.insert(key, (s.value.clone(), e.time));
                }
            }
        }
        let retained: Vec<String> = map
            .iter()
            .map(|((en, sl), (v, t))| format!("{en}.{sl}={v} t={t}"))
            .collect();
        let bytes = retained.iter().map(|s| s.len() as u64).sum();
        Self {
            map,
            bytes,
            lost,
            retained,
            transitions,
        }
    }

    pub fn answer(&self, entity: &str, slot: &str, budget: &Budget) -> Observation {
        let _ = budget;
        let mut transitions = self.transitions.clone();
        transitions.push(format!("lookup {entity}.{slot} in summary"));
        let answer = match self.map.get(&(entity.to_string(), slot.to_string())) {
            Some((v, _)) => Answer::Commit(v.clone()),
            None => Answer::Refuse("no_evidence"),
        };
        if matches!(answer, Answer::Refuse(_)) {
            transitions.push(format!("refuse {entity}.{slot}"));
        }
        Observation {
            retained: self.retained.clone(),
            lost: self.lost.clone(),
            recovered: vec![],
            transitions,
            reads: vec![],
            answer,
            bytes_retained: self.bytes,
            bytes_read: self.bytes,
            retrieval_calls: 0,
            retries: 0,
            verify_ns: 0,
        }
    }
}

impl CompactRecoverable {
    pub fn compact(events: &[Event]) -> Self {
        let mut recs = vec![];
        let mut lost = vec![];
        let mut retained = vec![];
        let mut bytes = 0;
        let mut transitions =
            vec!["compact: query not visible; bodies replaced by hash refs".into()];
        for e in events {
            let body = render(e);
            let hash = crate::events::digest(body.as_bytes());
            let preview = preview_salient(e);
            bytes += preview.len() as u64 + 32;
            for s in e.slots.iter().filter(|s| s.class == Class::Detail) {
                lost.push(format!(
                    "{} {} not inlined ({})",
                    e.entity, s.key, e.source_id
                ));
            }
            let id = format!("{}:{}", e.source_id, e.revision);
            retained.push(format!(
                "ref {id} hash={} preview_bytes={}",
                hex_prefix(&hash),
                preview.len()
            ));
            transitions.push(format!("pointer {id} entity {}", e.entity));
            recs.push(Rec {
                source_id: e.source_id.clone(),
                revision: e.revision,
                time: e.time,
                entity: e.entity.clone(),
                preview,
                inline: None,
                hash,
            });
        }
        Self {
            recs,
            bytes,
            lost,
            retained,
            transitions,
        }
    }

    pub fn answer(
        &self,
        entity: &str,
        slot: &str,
        store: &mut SourceStore,
        budget: &Budget,
    ) -> Observation {
        let mut obs = decide(
            &self.recs,
            entity,
            slot,
            Some(store),
            budget,
            self.bytes,
            &self.lost,
            &self.transitions,
        );
        obs.retained = self.retained.clone();
        obs
    }
}

struct Hit {
    value: String,
    source_id: String,
    revision: u32,
    supersedes: Option<(String, u32)>,
}

#[allow(clippy::too_many_arguments)]
fn decide(
    recs: &[Rec],
    entity: &str,
    slot: &str,
    mut store: Option<&mut SourceStore>,
    budget: &Budget,
    bytes_retained: u64,
    lost: &[String],
    prior: &[String],
) -> Observation {
    let mut order: Vec<usize> = (0..recs.len()).collect();
    order.sort_by(|&a, &b| {
        score(&recs[b], entity, slot)
            .cmp(&score(&recs[a], entity, slot))
            .then(recs[b].time.cmp(&recs[a].time))
            .then(recs[a].source_id.cmp(&recs[b].source_id))
            .then(recs[a].revision.cmp(&recs[b].revision))
    });
    let mut transitions = prior.to_vec();
    transitions.push(format!("rank {} records for {entity}.{slot}", recs.len()));
    let mut reads = vec![];
    let mut recovered = vec![];
    let mut found: Vec<Hit> = vec![];
    let mut saw_missing = false;
    let mut saw_corrupt = false;
    let mut bytes_read = 0u64;
    let mut calls = 0u64;
    let mut n = 0u32;
    for idx in order {
        if n >= budget.max_reads || bytes_read >= budget.max_bytes {
            transitions.push("stop: budget".into());
            break;
        }
        let r = &recs[idx];
        if r.entity != entity && score(r, entity, slot) < 1000 {
            continue;
        }
        let id = format!("{}:{}", r.source_id, r.revision);
        n += 1;
        let body = if let Some(text) = &r.inline {
            calls += 1;
            bytes_read += text.len() as u64;
            reads.push(format!("inline {id}"));
            transitions.push(format!("read retained {id}"));
            Some(text.clone())
        } else if let Some(store) = store.as_mut() {
            let before_calls = store.calls;
            let before_bytes = store.bytes_read;
            let out = store.read(&r.source_id, r.revision, &r.hash);
            calls += store.calls - before_calls;
            bytes_read += store.bytes_read - before_bytes;
            match out {
                ReadOut::Verified(text) => {
                    reads.push(format!("verified {id}"));
                    recovered.push(format!("{id} verified"));
                    transitions.push(format!("recover {id}"));
                    Some(text)
                }
                ReadOut::Missing => {
                    saw_missing = true;
                    reads.push(format!("missing {id}"));
                    transitions.push(format!("missing source {id}"));
                    None
                }
                ReadOut::Corrupt => {
                    saw_corrupt = true;
                    reads.push(format!("corrupt {id}"));
                    transitions.push(format!("reject corrupt {id}"));
                    None
                }
            }
        } else {
            None
        };
        if let Some(text) = body {
            let parsed = parse_body(&text);
            if parsed.entity == entity {
                if let Some((_, v)) = parsed.slots.iter().find(|(k, _)| k == slot) {
                    found.push(Hit {
                        value: v.clone(),
                        source_id: parsed.source_id,
                        revision: parsed.revision,
                        supersedes: parsed.supersedes,
                    });
                    transitions.push(format!(
                        "extract {slot} from {} rev {}",
                        r.source_id, r.revision
                    ));
                }
            }
        }
    }
    let answer = reduce(
        &found,
        saw_missing,
        saw_corrupt,
        n >= budget.max_reads && found.is_empty(),
    );
    if let Answer::Refuse(k) = answer {
        transitions.push(format!("refuse {k}"));
    }
    let (retries, verify_ns) = store
        .as_ref()
        .map(|s| (s.retries, s.verify_ns))
        .unwrap_or((0, 0));
    Observation {
        retained: recs
            .iter()
            .map(|r| {
                format!(
                    "{}:{} {}",
                    r.source_id,
                    r.revision,
                    if r.inline.is_some() {
                        "inline"
                    } else {
                        "pointer"
                    }
                )
            })
            .collect(),
        lost: lost.to_vec(),
        recovered,
        transitions,
        reads,
        answer,
        bytes_retained,
        bytes_read,
        retrieval_calls: calls,
        retries,
        verify_ns,
    }
}

fn reduce(found: &[Hit], missing: bool, corrupt: bool, budget_empty: bool) -> Answer {
    let superseded: Vec<(String, u32)> =
        found.iter().filter_map(|h| h.supersedes.clone()).collect();
    let live: Vec<&str> = found
        .iter()
        .filter(|h| {
            !superseded
                .iter()
                .any(|(s, r)| s == &h.source_id && *r == h.revision)
        })
        .map(|h| h.value.as_str())
        .collect();
    let mut uniq = live.clone();
    uniq.sort();
    uniq.dedup();
    if uniq.len() > 1 {
        Answer::Refuse("conflict")
    } else if uniq.len() == 1 {
        Answer::Commit(uniq[0].to_string())
    } else if corrupt {
        Answer::Refuse("corrupt_source")
    } else if missing {
        Answer::Refuse("missing_source")
    } else if budget_empty {
        Answer::Refuse("budget_exhausted")
    } else {
        Answer::Refuse("no_evidence")
    }
}

fn score(r: &Rec, entity: &str, slot: &str) -> i64 {
    let mut s = 0i64;
    if r.entity == entity {
        s += 1000;
    }
    if r.preview
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|t| t == slot)
    {
        s += 50;
    }
    s + (r.time / 100) as i64
}

fn hex_prefix(h: &[u8; 32]) -> String {
    h.iter().take(4).map(|b| format!("{b:02x}")).collect()
}
