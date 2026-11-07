//! Non-executing example for `receipt::auth`.
//!
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0
//!
//! Run: `cargo run -p receipt --example auth_check`
//!
//! This binary prints in-scope and deny decisions. It does not spawn a process, open a
//! shell, call a model, or read a power sensor. A deny is a successful check. An
//! in-scope line is a scope match, not a request to run anything.

use receipt::auth::{authorize, Action, DenyReason, Grant};
use receipt::{decision_hash, Energy, Provenance, Tier};

fn main() {
    let body = br#"{"model":"kev-0.8b","input":"status ping","questions":[{"type":"predicate","instructions":"Is this a status ping?"}]}"#;
    let inspect = Action::inspect_decision(decision_hash(body, "kev-0.8b", Tier::Rules));
    let quote = Action::quote_receipt("threadA", [0x42; 32]).expect("thread id");
    let grants = [
        Grant::issue("op.inspect", inspect.clone(), None).expect("grant"),
        Grant::issue("op.quote", quote.clone(), Some(0.5)).expect("grant"),
    ];
    let under = Energy {
        joules: 0.5,
        provenance: Provenance::ModelBased,
    };

    expect(&grants, inspect.canonical().as_bytes(), None, None);
    expect(&grants, quote.canonical().as_bytes(), Some(&under), None);
    expect(
        &[],
        inspect.canonical().as_bytes(),
        None,
        Some(DenyReason::NoAuthority),
    );

    // Reported LFM3-Omni shapes. Printed only as deny labels. Not passed to a shell.
    for sample in ["Commit\nrm -r /", "Commit\nrm -rf /", "echo hello", "hello"] {
        expect(
            &grants,
            sample.as_bytes(),
            Some(&under),
            Some(DenyReason::Unstructured),
        );
    }
    let prefixed = format!("Commit\n{}", inspect.canonical());
    expect(
        &grants,
        prefixed.as_bytes(),
        None,
        Some(DenyReason::Unstructured),
    );
    expect(
        &grants,
        b"strix-rt/action/v1\ncommit\n",
        None,
        Some(DenyReason::UnknownAction),
    );

    let other = Action::inspect_decision([0x99; 32]);
    expect(
        &grants,
        other.canonical().as_bytes(),
        None,
        Some(DenyReason::ScopeMismatch),
    );
    let over = Energy {
        joules: 4.0,
        provenance: Provenance::ModelBased,
    };
    expect(
        &grants,
        quote.canonical().as_bytes(),
        Some(&over),
        Some(DenyReason::OverBudget),
    );
    let estimated = Energy {
        joules: 0.0,
        provenance: Provenance::Estimator,
    };
    expect(
        &grants,
        quote.canonical().as_bytes(),
        Some(&estimated),
        Some(DenyReason::EnergyUnmeasured),
    );

    println!("auth_check: all expected decisions matched");
}

fn expect(grants: &[Grant], proposal: &[u8], energy: Option<&Energy>, denial: Option<DenyReason>) {
    let decision = authorize(grants, proposal, energy);
    println!("{decision}");
    let got = decision.denial();
    if got != denial {
        panic!("expected {denial:?}, got {got:?}");
    }
    if denial.is_none() {
        let action = receipt::auth::parse_action(proposal).expect("canonical scope parses");
        let matched = decision.scope_check().expect("in scope");
        assert!(matched.same_commitment(&action, energy));
        let swapped = Energy {
            joules: 9.0,
            provenance: Provenance::HwShunt,
        };
        assert!(!matched.same_commitment(&action, Some(&swapped)));
    }
}
