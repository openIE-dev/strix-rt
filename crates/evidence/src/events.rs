//! Visible history. Field class is part of the event, chosen before any query exists.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Salient,
    Detail,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Salient => "salient",
            Class::Detail => "detail",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub key: String,
    pub value: String,
    pub class: Class,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub source_id: String,
    pub revision: u32,
    pub time: u64,
    pub provenance: String,
    pub supersedes: Option<(String, u32)>,
    pub entity: String,
    pub slots: Vec<Slot>,
    /// Opaque note. May contain instruction-like text. It is data, not a grant.
    pub note: String,
}

pub fn render(e: &Event) -> String {
    let sup = e
        .supersedes
        .as_ref()
        .map(|(s, r)| format!("{s}:{r}"))
        .unwrap_or_else(|| "none".into());
    let mut out = format!(
        "source_id={}\nrevision={}\ntime={}\nprovenance={}\nsupersedes={sup}\nentity={}\n",
        e.source_id, e.revision, e.time, e.provenance, e.entity
    );
    for s in &e.slots {
        out.push_str(&format!(
            "{}={} class={}\n",
            s.key,
            s.value,
            s.class.as_str()
        ));
    }
    if !e.note.is_empty() {
        out.push_str("note=");
        out.push_str(&e.note);
        out.push('\n');
    }
    out
}

pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[derive(Clone, Debug)]
pub struct ParsedBody {
    pub source_id: String,
    pub revision: u32,
    pub entity: String,
    pub supersedes: Option<(String, u32)>,
    pub slots: Vec<(String, String)>,
}

pub fn parse_body(body: &str) -> ParsedBody {
    let mut source_id = String::new();
    let mut revision = 0;
    let mut entity = String::new();
    let mut supersedes = None;
    let mut slots = vec![];
    for line in body.lines() {
        if let Some(v) = line.strip_prefix("source_id=") {
            source_id = v.to_string();
        } else if let Some(v) = line.strip_prefix("revision=") {
            revision = v.parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("entity=") {
            entity = v.to_string();
        } else if let Some(v) = line.strip_prefix("supersedes=") {
            supersedes = v
                .split_once(':')
                .and_then(|(s, r)| r.parse().ok().map(|n| (s.to_string(), n)));
        } else if let Some((k, rest)) = line.split_once('=') {
            if k == "note" || k == "time" || k == "provenance" {
                continue;
            }
            let value = rest.split(" class=").next().unwrap_or(rest).to_string();
            slots.push((k.to_string(), value));
        }
    }
    ParsedBody {
        source_id,
        revision,
        entity,
        supersedes,
        slots,
    }
}

pub fn preview_salient(e: &Event) -> String {
    let mut out = format!(
        "source_id={} revision={} time={} entity={}\n",
        e.source_id, e.revision, e.time, e.entity
    );
    for s in e.slots.iter().filter(|s| s.class == Class::Salient) {
        out.push_str(&format!("{}={}\n", s.key, s.value));
    }
    out
}
