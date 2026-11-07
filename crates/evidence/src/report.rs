//! Raw per-case observations. Joules stay absent unless a probe actually sampled them.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::probe::PhaseObs;
use crate::strategy::{Answer, Observation};

#[derive(Clone, Debug)]
pub struct CaseObs {
    pub query_id: String,
    pub entity: String,
    pub slot: String,
    pub diagnostic_note: String,
    pub diagnostic_commit: Option<String>,
    pub relation: &'static str,
    pub observation: Observation,
    pub retrieve: PhaseObs,
    pub retry: PhaseObs,
    pub verify: PhaseObs,
}

#[derive(Clone, Debug)]
pub struct ConditionObs {
    pub name: &'static str,
    pub compact: PhaseObs,
    pub cases: Vec<CaseObs>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub seed: u64,
    pub kind: &'static str,
    pub telemetry_availability: &'static str,
    pub conditions: Vec<ConditionObs>,
    pub auth_unstructured: bool,
    pub auth_executes: bool,
}

impl Report {
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"seed\": {},\n", self.seed));
        s.push_str(&format!("  \"kind\": \"{}\",\n", self.kind));
        s.push_str("  \"ranking\": null,\n");
        s.push_str("  \"efficiency_claim\": false,\n");
        s.push_str(&format!(
            "  \"telemetry_availability\": \"{}\",\n",
            self.telemetry_availability
        ));
        s.push_str(&format!(
            "  \"auth_source_is_unstructured\": {},\n",
            self.auth_unstructured
        ));
        s.push_str(&format!("  \"auth_executes\": {},\n", self.auth_executes));
        s.push_str("  \"conditions\": [\n");
        for (i, c) in self.conditions.iter().enumerate() {
            if i > 0 {
                s.push_str(",\n");
            }
            s.push_str("    {\n");
            s.push_str(&format!("      \"name\": \"{}\",\n", c.name));
            s.push_str("      \"compaction\": ");
            push_phase(&mut s, &c.compact);
            s.push_str(",\n      \"cases\": [\n");
            for (j, case) in c.cases.iter().enumerate() {
                if j > 0 {
                    s.push_str(",\n");
                }
                push_case(&mut s, case);
            }
            s.push_str("\n      ]\n    }");
        }
        s.push_str("\n  ]\n}\n");
        s
    }
}

pub fn relation(diagnostic: &Option<String>, answer: &Answer) -> &'static str {
    match (diagnostic, answer) {
        (Some(d), Answer::Commit(v)) if d == v => "same_commit",
        (Some(_), Answer::Commit(_)) => "different_commit",
        (None, Answer::Refuse(_)) => "same_refusal",
        (Some(_), Answer::Refuse(_)) => "refuse_when_diagnostic_has_fact",
        (None, Answer::Commit(_)) => "commit_when_diagnostic_refuses",
    }
}

fn push_case(s: &mut String, c: &CaseObs) {
    s.push_str("        {\n");
    s.push_str(&format!("          \"query_id\": \"{}\",\n", c.query_id));
    s.push_str(&format!("          \"entity\": \"{}\",\n", esc(&c.entity)));
    s.push_str(&format!("          \"slot\": \"{}\",\n", esc(&c.slot)));
    s.push_str(&format!(
        "          \"diagnostic_note\": \"{}\",\n",
        esc(&c.diagnostic_note)
    ));
    s.push_str("          \"diagnostic_commit\": ");
    push_opt(s, c.diagnostic_commit.as_deref());
    s.push_str(",\n");
    s.push_str(&format!("          \"relation\": \"{}\",\n", c.relation));
    s.push_str("          \"answer\": ");
    match &c.observation.answer {
        Answer::Commit(v) => {
            s.push_str(&format!("{{\"kind\":\"commit\",\"value\":\"{}\"}}", esc(v)))
        }
        Answer::Refuse(k) => s.push_str(&format!("{{\"kind\":\"refuse\",\"reason\":\"{k}\"}}")),
    }
    s.push_str(",\n          \"retained\": ");
    push_arr(s, &c.observation.retained);
    s.push_str(",\n          \"lost\": ");
    push_arr(s, &c.observation.lost);
    s.push_str(",\n          \"recovered\": ");
    push_arr(s, &c.observation.recovered);
    s.push_str(",\n          \"reads\": ");
    push_arr(s, &c.observation.reads);
    s.push_str(",\n          \"transitions\": ");
    push_arr(s, &c.observation.transitions);
    s.push_str(",\n          \"work\": {");
    s.push_str(&format!(
        "\"bytes_retained\":{},\"bytes_read\":{},\"retrieval_calls\":{},\"retries\":{}",
        c.observation.bytes_retained,
        c.observation.bytes_read,
        c.observation.retrieval_calls,
        c.observation.retries
    ));
    s.push_str("},\n          \"telemetry\": [");
    push_phase(s, &c.retrieve);
    s.push(',');
    push_phase(s, &c.retry);
    s.push(',');
    push_phase(s, &c.verify);
    s.push_str("]\n        }");
}

fn push_phase(s: &mut String, p: &PhaseObs) {
    s.push('{');
    s.push_str(&format!(
        "\"phase\":\"{}\",\"wall_ns\":{},",
        p.phase, p.wall_ns
    ));
    s.push_str("\"joules\":");
    match p.joules {
        Some(j) if j.is_finite() => s.push_str(&format!("{j}")),
        _ => s.push_str("null"),
    }
    s.push_str(&format!(
        ",\"unit\":\"{}\",\"source\":\"{}\",\"provenance\":\"{}\",\"meter_samples\":{}",
        p.unit,
        esc(&p.source),
        p.provenance,
        p.meter_samples
    ));
    s.push('}');
}

fn push_arr(s: &mut String, xs: &[String]) {
    s.push('[');
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        s.push_str(&esc(x));
        s.push('"');
    }
    s.push(']');
}

fn push_opt(s: &mut String, v: Option<&str>) {
    match v {
        Some(v) => {
            s.push('"');
            s.push_str(&esc(v));
            s.push('"');
        }
        None => s.push_str("null"),
    }
}

fn esc(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}
