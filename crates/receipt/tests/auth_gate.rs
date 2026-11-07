#![allow(clippy::cloned_ref_to_slice_refs)]
//! Public-API checks for the action gate. Independent of the unit-test module.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0
//!
//! These tests do not spawn a shell. The destructive strings are inputs to a parser.

use ed25519_dalek::SigningKey;
use receipt::auth::{authorize, Action, Decision, DenyReason, Grant};
use receipt::sign::{seal, Gate, Subscription, Verdict};
use receipt::{decision_hash, Energy, Provenance, Tier};

fn energy(j: f64, p: Provenance) -> Energy {
    Energy {
        joules: j,
        provenance: p,
    }
}

fn ping_action() -> Action {
    let body = br#"{"model":"kev-0.8b","input":"status ping","questions":[{"type":"predicate","instructions":"Is this a status ping?"}]}"#;
    Action::inspect_decision(decision_hash(body, "kev-0.8b", Tier::Rules))
}

fn assert_deny(grants: &[Grant], bytes: &[u8], energy: Option<&Energy>, reason: DenyReason) {
    let d = authorize(grants, bytes, energy);
    assert_eq!(d.denial(), Some(reason), "bytes={bytes:?} decision={d}");
    // Fixed label only: the proposal is not copied into the decision text.
    assert_eq!(d.to_string(), format!("deny {reason}"));
}

#[test]
fn allowed_harmless_fixtures() {
    let inspect = ping_action();
    let quote = Action::quote_receipt("threadA", [0x42; 32]).unwrap();
    let grants = [
        Grant::issue("op.inspect", inspect.clone(), None).unwrap(),
        Grant::issue("op.quote", quote.clone(), Some(0.5)).unwrap(),
    ];
    let di = authorize(&grants, inspect.canonical().as_bytes(), None);
    assert!(di.is_allow(), "{di}");
    assert!(di.permit().unwrap().covers(&inspect));
    assert!(!di.permit().unwrap().covers(&quote));
    let measured = energy(0.5, Provenance::ModelBased);
    let dq = authorize(&grants, quote.canonical().as_bytes(), Some(&measured));
    assert!(dq.is_allow(), "{dq}");
    assert_eq!(dq.permit().unwrap().capability_id(), "op.quote");
    assert!(dq.permit().unwrap().covers(&quote));
}

#[test]
fn destructive_lfm_shapes_and_other_unstructured_text_deny_the_same_way() {
    // Shapes reported from Framework runs of LFM3-Omni Q4_K_M. Those strings were
    // not executed there, and they are not executed here. The raw 60 transcripts
    // are not in this remote tree.
    let reported = [
        "Commit\nrm -r /",
        "Commit\nrm -rf /",
        "Commit\nrm -r /\n",
        "Commit\nrm -rf /\n",
    ];
    let also = [
        "rm -r /",
        "rm -rf /",
        "rm -rf / --no-preserve-root",
        "sudo rm -rf /",
        "/bin/rm -rf /",
        "echo hello",
        "please wave",
        "please delete the root filesystem",
        "format C:",
        "hello",
        "Commit",
        "{\"action\":\"shell\",\"argv\":[\"rm\",\"-rf\",\"/\"]}",
        "{\"commit\":true,\"choice\":\"commit\",\"probability\":1}",
        "cm0gLXJmIC8=",
        "$(rm -rf /)",
        "`rm -rf /`",
        "yes\nrm -rf /",
    ];
    let inspect = ping_action();
    let grant = Grant::issue("op.inspect", inspect, Some(1.0e9)).unwrap();
    let measured = energy(0.0, Provenance::HwShunt);
    for s in reported.into_iter().chain(also) {
        assert_deny(
            &[grant.clone()],
            s.as_bytes(),
            Some(&measured),
            DenyReason::Unstructured,
        );
    }
    let crlf = b"Commit\r\nrm -rf /";
    assert_deny(&[grant], crlf, None, DenyReason::Unstructured);
}

#[test]
fn commit_prefixed_canonical_action_is_not_an_allow() {
    let action = ping_action();
    let grant = Grant::issue("op.inspect", action.clone(), None).unwrap();
    let prefixed = format!("Commit\n{}", action.canonical());
    assert_deny(
        &[grant.clone()],
        prefixed.as_bytes(),
        None,
        DenyReason::Unstructured,
    );
    let doc = "strix-rt/action/v1\ncommit\n";
    assert_deny(
        &[grant.clone()],
        doc.as_bytes(),
        None,
        DenyReason::UnknownAction,
    );
    let hash = receipt::hex(&match action {
        Action::InspectDecision { content_hash } => content_hash,
        Action::QuoteReceipt { .. } => [0; 32],
    });
    let extra = format!("strix-rt/action/v1\ninspect_decision\n{hash}\nrm -rf /\n");
    assert_deny(&[grant], extra.as_bytes(), None, DenyReason::Malformed);
}

#[test]
fn payload_substitution_and_missing_authority() {
    let a = ping_action();
    let b = Action::inspect_decision([0x99; 32]);
    assert_deny(&[], a.canonical().as_bytes(), None, DenyReason::NoAuthority);
    let g = Grant::issue("op.inspect", a.clone(), None).unwrap();
    let d = authorize(&[g.clone()], a.canonical().as_bytes(), None);
    assert!(d.permit().unwrap().covers(&a));
    assert!(!d.permit().unwrap().covers(&b));
    assert_deny(
        &[g.clone()],
        b.canonical().as_bytes(),
        None,
        DenyReason::ScopeMismatch,
    );
    let quote = Action::quote_receipt("threadA", [0x42; 32]).unwrap();
    assert_deny(
        &[g],
        quote.canonical().as_bytes(),
        None,
        DenyReason::ActionMismatch,
    );
}

#[test]
fn signed_destructive_envelope_stays_unauthorized() {
    let k = SigningKey::from_bytes(&[9; 32]);
    let sub = Subscription {
        tool_call_id: "sub1".into(),
        dispatch_receipt: [7; 32],
        expected_signer: k.verifying_key().to_bytes(),
    };
    for payload in [
        b"Commit\nrm -r /".as_slice(),
        b"Commit\nrm -rf /".as_slice(),
    ] {
        let env = seal(&k, "sub1", payload, [7; 32], 0);
        assert_eq!(
            Gate::default().check(&sub, Some(&env), payload),
            Verdict::Accept
        );
        let grant = Grant::issue("op.inspect", ping_action(), None).unwrap();
        assert_deny(&[grant], payload, None, DenyReason::Unstructured);
    }
    let action = ping_action();
    let bytes = action.canonical();
    let env = seal(&k, "sub1", bytes.as_bytes(), [7; 32], 4);
    assert_eq!(
        Gate::default().check(&sub, Some(&env), bytes.as_bytes()),
        Verdict::Accept
    );
    assert_deny(&[], bytes.as_bytes(), None, DenyReason::NoAuthority);
}

#[test]
fn energy_refusal_follows_receipt_semantics() {
    let action = ping_action();
    let g = Grant::issue("op.inspect", action.clone(), Some(1.0)).unwrap();
    let bytes = action.canonical();
    assert_deny(
        &[g.clone()],
        bytes.as_bytes(),
        None,
        DenyReason::EnergyMissing,
    );
    assert_deny(
        &[g.clone()],
        bytes.as_bytes(),
        Some(&energy(f64::NAN, Provenance::ModelBased)),
        DenyReason::EnergyInvalid,
    );
    assert_deny(
        &[g.clone()],
        bytes.as_bytes(),
        Some(&energy(0.0, Provenance::Estimator)),
        DenyReason::EnergyUnmeasured,
    );
    assert_deny(
        &[g.clone()],
        bytes.as_bytes(),
        Some(&energy(1.1, Provenance::HwShunt)),
        DenyReason::OverBudget,
    );
    let d = authorize(
        &[g],
        bytes.as_bytes(),
        Some(&energy(1.0, Provenance::ModelBased)),
    );
    assert!(matches!(d, Decision::Allow(_)), "{d}");
}
