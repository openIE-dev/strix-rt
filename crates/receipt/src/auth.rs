//! Fail-closed action authorization.
//!
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0
//!
//! This module decides whether a proposal is a typed action that an operator grant
//! already allows. It does not spawn processes, open files, read a power meter, or
//! dispatch tools. `sign::Gate` remains a signature check; a verified envelope is
//! not an allow.
//!
//! Grammar (canonical, one byte-string per action):
//! ```text
//! strix-rt/action/v1\n
//! inspect_decision\n
//! <64 lowercase hex content-hash>\n
//! ```
//! ```text
//! strix-rt/action/v1\n
//! quote_receipt\n
//! <thread id>\n
//! <64 lowercase hex receipt-hash>\n
//! ```
//! Thread ids are 1..=64 bytes, `[A-Za-z0-9][A-Za-z0-9_-]*`. Proposals longer than
//! 256 bytes, or that do not start with the magic line, are unstructured. Unknown
//! verbs are denied. There is no command list and no shell action.
//!
//! The parameters of an action are its scope. An in-scope result carries a domain-separated
//! SHA-256 of those canonical bytes, plus the energy commitment that was actually checked.
//! Swapping the action or, when a ceiling applied, the joule figure or its provenance, does
//! not satisfy `ScopeCheck::same_commitment`. The result is not a bearer token and not an
//! instruction to execute. Nothing in this crate consumes it to perform I/O. A signed
//! receipt (`sign`, `ledger`) is a different object and is not an in-scope result.
//!
//! Mesh joule ceilings are not stored on this baseline. When at least one matching
//! grant carries `compute_ceiling_j`, the strictest of those ceilings is applied to
//! the caller-supplied [`Energy`]. `Estimator` and non-finite joules (the receipt
//! sentinel for "not metered") do not satisfy a ceiling. Grants with no ceiling do
//! not invent a number and do not read sysfs. A ceiling-free grant does not erase a
//! stricter grant for the same action.

use crate::{hex, Energy, Hash, Provenance};
use sha2::{Digest, Sha256};

const MAGIC_LINE: &[u8] = b"strix-rt/action/v1\n";
const MAX_PROPOSAL_LEN: usize = 256;
const BINDING_DOMAIN: &[u8] = b"strix-rt/action-binding/v1";

/// Why `authorize` refused. Display text is a fixed label and does not echo the proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenyReason {
    /// Bytes are not a canonical action document.
    Unstructured,
    /// The verb token is well-formed and is not in the closed action set.
    UnknownAction,
    /// The header was recognized but the document is not canonical.
    Malformed,
    /// No grants were supplied.
    NoAuthority,
    /// Grants exist, and none are the same action kind.
    ActionMismatch,
    /// The action kind matched a grant, but the scope did not.
    ScopeMismatch,
    /// A ceiling applies and the caller supplied no energy figure.
    EnergyMissing,
    /// A ceiling applies and joules are negative or not finite.
    EnergyInvalid,
    /// A ceiling applies and provenance is [`Provenance::Estimator`].
    EnergyUnmeasured,
    /// A ceiling applies and joules are above the strictest matching ceiling.
    OverBudget,
    /// `Grant::issue` was given a non-finite or negative ceiling.
    BadCeiling,
    /// `Grant::issue` was given an id or action that cannot be represented.
    BadGrant,
}

impl std::fmt::Display for DenyReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DenyReason::Unstructured => "unstructured",
            DenyReason::UnknownAction => "unknown_action",
            DenyReason::Malformed => "malformed",
            DenyReason::NoAuthority => "no_authority",
            DenyReason::ActionMismatch => "action_mismatch",
            DenyReason::ScopeMismatch => "scope_mismatch",
            DenyReason::EnergyMissing => "energy_missing",
            DenyReason::EnergyInvalid => "energy_invalid",
            DenyReason::EnergyUnmeasured => "energy_unmeasured",
            DenyReason::OverBudget => "over_budget",
            DenyReason::BadCeiling => "bad_ceiling",
            DenyReason::BadGrant => "bad_grant",
        })
    }
}

/// Closed set of scoped references this baseline can name. Both point at objects the
/// runtime has already computed. Neither variant is a tool call or an execution request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Scope: the `/v1/decisions` content hash (`decision_hash` / slice content hash).
    InspectDecision { content_hash: Hash },
    /// Scope: one thread id plus that thread's receipt hash.
    QuoteReceipt {
        thread_id: String,
        receipt_hash: Hash,
    },
}

impl Action {
    pub fn inspect_decision(content_hash: Hash) -> Self {
        Action::InspectDecision { content_hash }
    }

    pub fn quote_receipt(
        thread_id: impl Into<String>,
        receipt_hash: Hash,
    ) -> Result<Self, DenyReason> {
        let thread_id = thread_id.into();
        if !valid_thread(&thread_id) {
            return Err(DenyReason::Malformed);
        }
        Ok(Action::QuoteReceipt {
            thread_id,
            receipt_hash,
        })
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Action::InspectDecision { .. } => "inspect_decision",
            Action::QuoteReceipt { .. } => "quote_receipt",
        }
    }

    /// Canonical proposal bytes. `parse_action` accepts only this encoding.
    pub fn canonical(&self) -> String {
        match self {
            Action::InspectDecision { content_hash } => {
                format!(
                    "strix-rt/action/v1\ninspect_decision\n{}\n",
                    hex(content_hash)
                )
            }
            Action::QuoteReceipt {
                thread_id,
                receipt_hash,
            } => {
                format!(
                    "strix-rt/action/v1\nquote_receipt\n{thread_id}\n{}\n",
                    hex(receipt_hash)
                )
            }
        }
    }

    /// Scope binding of the canonical document. `None` when this value is not one `parse_action` accepts.
    pub fn scope_binding(&self) -> Option<Hash> {
        let text = self.canonical();
        let parsed = parse_action(text.as_bytes()).ok()?;
        if parsed != *self || parsed.canonical() != text {
            return None;
        }
        Some(binding_of(&text))
    }
}

/// Operator-issued capability. Construct it with [`Grant::issue`]. The model cannot mint one.
#[derive(Clone, Debug, PartialEq)]
pub struct Grant {
    capability_id: String,
    action: Action,
    compute_ceiling_j: Option<f64>,
}

impl Grant {
    /// `compute_ceiling_j: None` means this grant does not assert a Mesh ceiling.
    /// It does not mean "unlimited measured joules".
    pub fn issue(
        id: &str,
        action: Action,
        compute_ceiling_j: Option<f64>,
    ) -> Result<Self, DenyReason> {
        if !valid_capability_id(id) {
            return Err(DenyReason::BadGrant);
        }
        let compute_ceiling_j = match compute_ceiling_j {
            None => None,
            Some(c) => Some(canonical_nonneg(c).ok_or(DenyReason::BadCeiling)?),
        };
        let parsed = parse_action(action.canonical().as_bytes())?;
        if parsed != action || action.scope_binding().is_none() {
            return Err(DenyReason::BadGrant);
        }
        Ok(Grant {
            capability_id: id.to_string(),
            action,
            compute_ceiling_j,
        })
    }

    pub fn capability_id(&self) -> &str {
        &self.capability_id
    }
    pub fn action(&self) -> &Action {
        &self.action
    }
    pub fn compute_ceiling_j(&self) -> Option<f64> {
        self.compute_ceiling_j
    }
}

/// Energy fact recorded on an in-scope result. This is not a meter reading taken by `auth`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyCommitment {
    /// No matching grant carried a ceiling. Not an approval of any joule figure.
    NotApplied,
    /// Strictest ceiling, and the exact non-negative measured figure that was within it.
    WithinCeiling {
        ceiling_bits: u64,
        joules_bits: u64,
        provenance: Provenance,
    },
}

/// Record that one proposal matched a grant's exact scope. `_seal` blocks struct literals
/// outside this crate. Holding this value does not run anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeCheck {
    capability_id: String,
    scope_binding: Hash,
    commitment: EnergyCommitment,
    _seal: (),
}

impl ScopeCheck {
    pub fn capability_id(&self) -> &str {
        &self.capability_id
    }
    pub fn scope_binding(&self) -> &Hash {
        &self.scope_binding
    }
    pub fn energy_commitment(&self) -> EnergyCommitment {
        self.commitment
    }
    /// Scope only. A true result does not check energy and is not an execution request.
    pub fn same_scope(&self, action: &Action) -> bool {
        action.scope_binding().as_ref() == Some(&self.scope_binding)
    }
    /// True only for the same canonical scope and the same energy commitment.
    ///
    /// `NotApplied` matches `energy == None` only. A checked ceiling matches only the same
    /// provenance and the same canonical joule bits. This is a comparison of the decision
    /// record, not a second grant and not a call to run the scope.
    pub fn same_commitment(&self, action: &Action, energy: Option<&Energy>) -> bool {
        if !self.same_scope(action) {
            return false;
        }
        match self.commitment {
            EnergyCommitment::NotApplied => energy.is_none(),
            EnergyCommitment::WithinCeiling {
                ceiling_bits,
                joules_bits,
                provenance,
            } => {
                let Some(e) = energy else {
                    return false;
                };
                let Some(ceiling) = canonical_nonneg(f64::from_bits(ceiling_bits)) else {
                    return false;
                };
                let Some(got) = canonical_joule_bits(e.joules) else {
                    return false;
                };
                let stored_ok =
                    canonical_nonneg_bits(f64::from_bits(joules_bits)) == Some(joules_bits);
                e.provenance == provenance
                    && got == joules_bits
                    && stored_ok
                    && f64::from_bits(joules_bits) <= ceiling
            }
        }
    }
}

/// Result of [`authorize`]. [`Decision::InScope`] means the proposal matched a grant.
/// It is not an instruction to execute the named reference.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "an in-scope result must be checked; it does not execute"]
pub enum Decision {
    InScope(ScopeCheck),
    Deny(DenyReason),
}

impl Decision {
    pub fn is_in_scope(&self) -> bool {
        matches!(self, Decision::InScope(_))
    }
    pub fn is_deny(&self) -> bool {
        matches!(self, Decision::Deny(_))
    }
    pub fn scope_check(&self) -> Option<&ScopeCheck> {
        match self {
            Decision::InScope(p) => Some(p),
            Decision::Deny(_) => None,
        }
    }
    pub fn denial(&self) -> Option<DenyReason> {
        match self {
            Decision::InScope(_) => None,
            Decision::Deny(r) => Some(*r),
        }
    }
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decision::InScope(p) => match p.energy_commitment() {
                EnergyCommitment::NotApplied => write!(
                    f,
                    "in_scope capability_id={} scope={} energy=not_applied",
                    p.capability_id(),
                    hex(p.scope_binding())
                ),
                EnergyCommitment::WithinCeiling {
                    ceiling_bits,
                    joules_bits,
                    provenance,
                } => write!(
                    f,
                    "in_scope capability_id={} scope={} energy={} joules={} ceiling_j={}",
                    p.capability_id(),
                    hex(p.scope_binding()),
                    provenance_label(provenance),
                    f64::from_bits(joules_bits),
                    f64::from_bits(ceiling_bits)
                ),
            },
            Decision::Deny(r) => write!(f, "deny {r}"),
        }
    }
}

/// Parse a canonical action. This does not consult grants and does not allow anything.
#[must_use = "parsing does not authorize; call authorize"]
pub fn parse_action(bytes: &[u8]) -> Result<Action, DenyReason> {
    if !bytes.starts_with(MAGIC_LINE) {
        return Err(DenyReason::Unstructured);
    }
    if bytes.len() > MAX_PROPOSAL_LEN || bytes.contains(&0) || bytes.contains(&b'\r') {
        return Err(DenyReason::Malformed);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| DenyReason::Malformed)?;
    if !text.ends_with('\n') {
        return Err(DenyReason::Malformed);
    }
    let lines: Vec<&str> = text[..text.len() - 1].split('\n').collect();
    if lines.first().copied() != Some("strix-rt/action/v1") {
        return Err(DenyReason::Malformed);
    }
    let verb = lines.get(1).copied().unwrap_or("");
    if !verb_token(verb) {
        return Err(DenyReason::Malformed);
    }
    if !known_verb(verb) {
        return Err(DenyReason::UnknownAction);
    }
    match verb {
        "inspect_decision" => {
            if lines.len() != 3 {
                return Err(DenyReason::Malformed);
            }
            let content_hash = parse_hash(lines[2]).ok_or(DenyReason::Malformed)?;
            Ok(Action::InspectDecision { content_hash })
        }
        "quote_receipt" => {
            if lines.len() != 4 {
                return Err(DenyReason::Malformed);
            }
            if !valid_thread(lines[2]) {
                return Err(DenyReason::Malformed);
            }
            let receipt_hash = parse_hash(lines[3]).ok_or(DenyReason::Malformed)?;
            Ok(Action::QuoteReceipt {
                thread_id: lines[2].to_string(),
                receipt_hash,
            })
        }
        _ => Err(DenyReason::UnknownAction),
    }
}

/// Authorize `proposal` against operator `grants`.
///
/// `proposal` is untrusted bytes (model output, a file, a queue message). Grants are
/// a separate argument. There is no commit flag: a model word is not authority.
///
/// On an exact action match, every matching ceiling applies and the strictest one wins.
/// A grant that sets no ceiling does not cancel a grant that sets one. Energy is read
/// only from `energy`; this function does not meter. An in-scope result is not execution.
#[must_use = "an in-scope result must be checked; it does not execute"]
pub fn authorize(grants: &[Grant], proposal: &[u8], energy: Option<&Energy>) -> Decision {
    let action = match parse_action(proposal) {
        Ok(a) => a,
        Err(r) => return Decision::Deny(r),
    };
    if action.canonical().as_bytes() != proposal {
        return Decision::Deny(DenyReason::Malformed);
    }
    let Some(want) = action.scope_binding() else {
        return Decision::Deny(DenyReason::Malformed);
    };
    if grants.is_empty() {
        return Decision::Deny(DenyReason::NoAuthority);
    }

    let mut saw_kind = false;
    let mut strictest: Option<f64> = None;
    let mut chosen_id: Option<&str> = None;
    for g in grants {
        if !same_kind(&g.action, &action) {
            continue;
        }
        saw_kind = true;
        // Exact scope is the canonical binding, not a separately supplied payload.
        if g.action.scope_binding() != Some(want) {
            continue;
        }
        match g.compute_ceiling_j {
            Some(c) => {
                let tighter = strictest.map(|s| c < s).unwrap_or(true);
                if tighter {
                    strictest = Some(c);
                    chosen_id = Some(g.capability_id.as_str());
                }
            }
            None => {
                if chosen_id.is_none() {
                    chosen_id = Some(g.capability_id.as_str());
                }
            }
        }
    }
    let Some(id) = chosen_id else {
        return Decision::Deny(if saw_kind {
            DenyReason::ScopeMismatch
        } else {
            DenyReason::ActionMismatch
        });
    };
    let commitment = if let Some(ceiling) = strictest {
        let Some(e) = energy else {
            return Decision::Deny(DenyReason::EnergyMissing);
        };
        if let Some(reason) = energy_refusal(ceiling, Some(e)) {
            return Decision::Deny(reason);
        }
        let (Some(ceiling_bits), Some(joules_bits)) = (
            canonical_nonneg_bits(ceiling),
            canonical_joule_bits(e.joules),
        ) else {
            return Decision::Deny(DenyReason::EnergyInvalid);
        };
        EnergyCommitment::WithinCeiling {
            ceiling_bits,
            joules_bits,
            provenance: e.provenance,
        }
    } else {
        EnergyCommitment::NotApplied
    };
    Decision::InScope(ScopeCheck {
        capability_id: id.to_string(),
        scope_binding: want,
        commitment,
        _seal: (),
    })
}

fn energy_refusal(ceiling: f64, energy: Option<&Energy>) -> Option<DenyReason> {
    let Some(e) = energy else {
        return Some(DenyReason::EnergyMissing);
    };
    if !e.joules.is_finite() || e.joules < 0.0 {
        return Some(DenyReason::EnergyInvalid);
    }
    match e.provenance {
        Provenance::HwShunt | Provenance::ModelBased => {}
        Provenance::Estimator => return Some(DenyReason::EnergyUnmeasured),
    }
    if e.joules > ceiling {
        return Some(DenyReason::OverBudget);
    }
    None
}

fn binding_of(canonical: &str) -> Hash {
    let mut h = Sha256::new();
    h.update(BINDING_DOMAIN);
    h.update((canonical.len() as u64).to_le_bytes());
    h.update(canonical.as_bytes());
    h.finalize().into()
}

/// `+0.0` for signed zero. `None` when negative or not finite.
fn canonical_nonneg(j: f64) -> Option<f64> {
    if !j.is_finite() || j < 0.0 {
        return None;
    }
    if j.to_bits() == (-0.0f64).to_bits() {
        Some(0.0)
    } else {
        Some(j)
    }
}

fn canonical_nonneg_bits(j: f64) -> Option<u64> {
    canonical_nonneg(j).map(f64::to_bits)
}

fn canonical_joule_bits(j: f64) -> Option<u64> {
    canonical_nonneg_bits(j)
}

/// Two read-only references a `/v1/decisions` response can name: the decision content
/// hash, and the thread plus receipt hash. `grants` is an operator argument. This
/// function does not read an HTTP body, a model probability, or a receipt signature,
/// and it does not execute either reference.
///
/// `grant_source_configured` is false when `grants` is empty. That is the server
/// state when no operator grant store exists. Both references then refuse with
/// `no_authority` if they parse. Empty grants are not replaced with an invented grant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionsAuthorization {
    pub grant_source_configured: bool,
    pub inspect_decision: Decision,
    pub quote_receipt: Decision,
}

impl DecisionsAuthorization {
    pub fn assess(
        grants: &[Grant],
        content_hash: Hash,
        thread_id: &str,
        receipt_hash: Hash,
        energy: Option<&Energy>,
    ) -> Self {
        let inspect = Action::inspect_decision(content_hash);
        let quote_receipt = match Action::quote_receipt(thread_id, receipt_hash) {
            Ok(action) => authorize(grants, action.canonical().as_bytes(), energy),
            Err(reason) => Decision::Deny(reason),
        };
        Self {
            grant_source_configured: !grants.is_empty(),
            inspect_decision: authorize(grants, inspect.canonical().as_bytes(), energy),
            quote_receipt,
        }
    }

    /// Fixed JSON. `executes` is always false, including when a scope matches.
    pub fn to_json(&self) -> String {
        let mut s = String::from(r#"{"v":1,"executes":false,"grant_source":"#);
        if self.grant_source_configured {
            s.push_str(r#""in_process","#);
        } else {
            s.push_str(r#""none","prerequisite":"no operator grant store is configured","#);
        }
        s.push_str(r#""inspect_decision":"#);
        s.push_str(&scope_json(&self.inspect_decision));
        s.push_str(r#","quote_receipt":"#);
        s.push_str(&scope_json(&self.quote_receipt));
        s.push('}');
        s
    }
}

fn scope_json(d: &Decision) -> String {
    match d {
        Decision::Deny(reason) => format!(r#"{{"scope_match":false,"reason":"{reason}"}}"#),
        Decision::InScope(check) => {
            let energy = match check.energy_commitment() {
                EnergyCommitment::NotApplied => "\"not_applied\"".to_string(),
                EnergyCommitment::WithinCeiling {
                    ceiling_bits,
                    joules_bits,
                    provenance,
                } => format!(
                    r#"{{"provenance":"{}","joules":{},"ceiling_j":{}}}"#,
                    provenance_label(provenance),
                    fmt_joules(joules_bits),
                    fmt_joules(ceiling_bits)
                ),
            };
            format!(
                r#"{{"scope_match":true,"capability_id":{},"energy":{}}}"#,
                json_string(check.capability_id()),
                energy
            )
        }
    }
}

fn json_string(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn fmt_joules(bits: u64) -> String {
    let j = f64::from_bits(bits);
    if j.is_finite() {
        format!("{j}")
    } else {
        "null".into()
    }
}

fn provenance_label(p: Provenance) -> &'static str {
    match p {
        Provenance::HwShunt => "hw_shunt",
        Provenance::ModelBased => "model_based",
        Provenance::Estimator => "estimator",
    }
}

fn same_kind(a: &Action, b: &Action) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn known_verb(v: &str) -> bool {
    matches!(v, "inspect_decision" | "quote_receipt")
}

fn verb_token(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 64
        && b[0].is_ascii_lowercase()
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
}

fn valid_thread(s: &str) -> bool {
    let b = s.as_bytes();
    (1..=64).contains(&b.len())
        && b[0].is_ascii_alphanumeric()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || *c == b'_' || *c == b'-')
}

fn valid_capability_id(s: &str) -> bool {
    let b = s.as_bytes();
    (1..=128).contains(&b.len())
        && b[0].is_ascii_alphanumeric()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(*c, b'.' | b'_' | b':' | b'-'))
}

fn parse_hash(s: &str) -> Option<Hash> {
    if s.len() != 64 || !s.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    let mut h = [0u8; 32];
    for i in 0..32 {
        h[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(h)
}

#[cfg(test)]
#[allow(clippy::cloned_ref_to_slice_refs)]
mod tests {
    use super::*;
    use crate::sign::{seal, Gate, Subscription, Verdict};
    use crate::{decision_hash, Tier};
    use ed25519_dalek::SigningKey;

    fn measured(j: f64, p: Provenance) -> Energy {
        Energy {
            joules: j,
            provenance: p,
        }
    }
    fn check_of(d: &Decision) -> &ScopeCheck {
        d.scope_check().expect("in scope")
    }

    #[test]
    fn harmless_fixtures_roundtrip_and_allow() {
        let body = br#"{"model":"kev-0.8b","input":"status ping","questions":[{"type":"predicate","instructions":"Is this a status ping?"}]}"#;
        let hash = decision_hash(body, "kev-0.8b", Tier::Small { batch: 1 });
        let inspect = Action::inspect_decision(hash);
        let quote = Action::quote_receipt("threadA", [0x22; 32]).unwrap();
        for action in [inspect, quote] {
            let bytes = action.canonical();
            assert_eq!(parse_action(bytes.as_bytes()).unwrap(), action);
            assert_eq!(action.canonical().as_bytes(), bytes.as_bytes());
            let bare = authorize(&[], bytes.as_bytes(), None);
            assert_eq!(bare.denial(), Some(DenyReason::NoAuthority));
            let g = Grant::issue("op.ping", action.clone(), None).unwrap();
            assert_eq!(g.compute_ceiling_j(), None);
            let d = authorize(&[g], bytes.as_bytes(), None);
            let p = check_of(&d);
            assert!(p.same_scope(&action));
            assert!(p.same_commitment(&action, None));
            assert!(!p.same_commitment(&action, Some(&measured(0.0, Provenance::ModelBased))));
            assert_eq!(p.capability_id(), "op.ping");
            assert_eq!(p.scope_binding(), &action.scope_binding().unwrap());
        }
    }

    #[test]
    fn model_bytes_do_not_mint_authority() {
        let action = Action::inspect_decision([0x11; 32]);
        let bytes = action.canonical();
        assert_eq!(
            authorize(&[], bytes.as_bytes(), None).denial(),
            Some(DenyReason::NoAuthority)
        );
        let wrapped = format!("Commit\n{bytes}");
        assert_eq!(
            authorize(&[], wrapped.as_bytes(), None).denial(),
            Some(DenyReason::Unstructured)
        );
        let g = Grant::issue("op.ping", action.clone(), None).unwrap();
        assert_eq!(
            authorize(&[g.clone()], wrapped.as_bytes(), None).denial(),
            Some(DenyReason::Unstructured)
        );
        assert!(authorize(&[g], bytes.as_bytes(), None).is_in_scope());
    }

    #[test]
    fn commit_and_other_verbs_are_not_actions() {
        for verb in [
            "commit",
            "shell",
            "exec",
            "run",
            "bash",
            "system",
            "tool",
            "delete_root",
            "echo",
        ] {
            let doc = format!("strix-rt/action/v1\n{verb}\n");
            assert_eq!(
                parse_action(doc.as_bytes()),
                Err(DenyReason::UnknownAction),
                "{verb}"
            );
            let g =
                Grant::issue("op.ping", Action::inspect_decision([1; 32]), Some(1.0e9)).unwrap();
            assert_eq!(
                authorize(&[g], doc.as_bytes(), None).denial(),
                Some(DenyReason::UnknownAction)
            );
        }
        assert_eq!(
            parse_action(b"strix-rt/action/v1\nCommit\n"),
            Err(DenyReason::Malformed)
        );
    }

    #[test]
    fn unstructured_inputs_share_one_reason() {
        let samples: &[&[u8]] = &[
            b"",
            b" ",
            b"\n",
            b"hello",
            b"Commit",
            b"Commit\n",
            b"Commit\nnope",
            b"status ping",
            b"{\"probability\":1.0}",
            b"{\"choice\":\"commit\",\"confidence\":1}",
            b"strix-rt/action/v2\ninspect_decision\n",
            &[0xff, 0xfe],
        ];
        for s in samples {
            assert_eq!(parse_action(s), Err(DenyReason::Unstructured));
        }
        let huge = vec![b'A'; 200_000];
        assert_eq!(parse_action(&huge), Err(DenyReason::Unstructured));
        for b in 0u8..=255 {
            assert_eq!(parse_action(&[b]), Err(DenyReason::Unstructured));
        }
    }

    #[test]
    fn malformed_documents_do_not_allow() {
        let hash = hex(&[0xab; 32]);
        let g = Grant::issue("op.ping", Action::inspect_decision([0xab; 32]), None).unwrap();
        let cases = [
            format!("strix-rt/action/v1\ninspect_decision\n{hash}"),
            format!("strix-rt/action/v1\ninspect_decision\n{hash}\n\n"),
            format!("strix-rt/action/v1\ninspect_decision\n{hash}\nextra\n"),
            format!(
                "strix-rt/action/v1\ninspect_decision\n{}\n",
                hash.to_uppercase()
            ),
            format!("strix-rt/action/v1\ninspect_decision\n{hash} \n"),
            "strix-rt/action/v1\ninspect_decision\n\n".into(),
            "strix-rt/action/v1\n\n".into(),
            format!("strix-rt/action/v1\nquote_receipt\n../x\n{hash}\n"),
            format!("strix-rt/action/v1\nquote_receipt\n-rf\n{hash}\n"),
            format!("strix-rt/action/v1\nquote_receipt\na/b\n{hash}\n"),
            format!("strix-rt/action/v1\nquote_receipt\n\n{hash}\n"),
        ];
        for c in cases {
            let d = authorize(&[g.clone()], c.as_bytes(), None);
            assert_eq!(d.denial(), Some(DenyReason::Malformed), "{c:?}");
        }
        let mut nul = format!("strix-rt/action/v1\ninspect_decision\n{hash}\n").into_bytes();
        nul.push(0);
        assert_eq!(parse_action(&nul), Err(DenyReason::Malformed));
    }

    #[test]
    fn scope_and_kind_mismatch() {
        let inspect = Action::inspect_decision([0x11; 32]);
        let other = Action::inspect_decision([0x22; 32]);
        let quote = Action::quote_receipt("threadA", [0x11; 32]).unwrap();
        let g = Grant::issue("op.ping", inspect.clone(), None).unwrap();
        assert_eq!(
            authorize(&[g.clone()], other.canonical().as_bytes(), None).denial(),
            Some(DenyReason::ScopeMismatch)
        );
        assert_eq!(
            authorize(&[g.clone()], quote.canonical().as_bytes(), None).denial(),
            Some(DenyReason::ActionMismatch)
        );
        let qg = Grant::issue("op.quote", quote.clone(), None).unwrap();
        let moved = Action::quote_receipt("threadB", [0x11; 32]).unwrap();
        let rehash = Action::quote_receipt("threadA", [0x33; 32]).unwrap();
        assert_eq!(
            authorize(&[qg.clone()], moved.canonical().as_bytes(), None).denial(),
            Some(DenyReason::ScopeMismatch)
        );
        assert_eq!(
            authorize(&[qg], rehash.canonical().as_bytes(), None).denial(),
            Some(DenyReason::ScopeMismatch)
        );
        assert_ne!(inspect.scope_binding(), other.scope_binding());
        assert_ne!(inspect.scope_binding(), quote.scope_binding());
    }

    #[test]
    fn permit_does_not_cover_a_substituted_action() {
        let a = Action::inspect_decision([0xab; 32]);
        let b = Action::inspect_decision([0xcd; 32]);
        let g = Grant::issue("op.ping", a.clone(), None).unwrap();
        let d = authorize(&[g], a.canonical().as_bytes(), None);
        let p = check_of(&d);
        assert!(p.same_scope(&a));
        assert!(!p.same_scope(&b));
        assert!(!p.same_commitment(&b, None));
        let mut flipped = a.canonical().into_bytes();
        let last = flipped.len() - 2;
        flipped[last] = if flipped[last] == b'a' { b'b' } else { b'a' };
        assert!(authorize(&[Grant::issue("op.ping", a, None).unwrap()], &flipped, None).is_deny());
    }

    #[test]
    fn every_single_byte_change_of_an_allow_is_denied() {
        let actions = [
            Action::inspect_decision([0xab; 32]),
            Action::quote_receipt("threadA", [0x22; 32]).unwrap(),
        ];
        for action in actions {
            let g = Grant::issue("op.ping", action.clone(), None).unwrap();
            let base = action.canonical().into_bytes();
            assert!(authorize(&[g.clone()], &base, None).is_in_scope());
            for i in 0..base.len() {
                for delta in [1u8, 0x20, 0xff] {
                    let mut tampered = base.clone();
                    tampered[i] ^= delta;
                    if tampered == base {
                        continue;
                    }
                    assert!(
                        authorize(&[g.clone()], &tampered, None).is_deny(),
                        "i={i} d={delta}"
                    );
                }
            }
        }
    }

    #[test]
    fn random_noncanonical_bytes_never_allow() {
        let action = Action::inspect_decision([0x11; 32]);
        let g = Grant::issue("op.ping", action.clone(), None).unwrap();
        let canonical = action.canonical();
        let mut state = 0xdecafbadu64;
        for _ in 0..300 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let n = (state % 80) as usize + 1;
            let mut bytes = vec![0u8; n];
            for b in &mut bytes {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (state >> 33) as u8;
            }
            assert_ne!(bytes.as_slice(), canonical.as_bytes());
            assert!(authorize(&[g.clone()], &bytes, None).is_deny());
        }
    }

    #[test]
    fn binding_is_domain_separated_and_stable() {
        let action = Action::inspect_decision([0x44; 32]);
        let again = action.scope_binding();
        assert_eq!(action.scope_binding(), again);
        let raw: Hash = Sha256::digest(action.canonical().as_bytes()).into();
        assert_ne!(action.scope_binding().unwrap(), raw);
        let text = format!("deny {}", authorize(&[], b"unique-marker-zxq", None));
        assert!(!text.contains("unique-marker"));
    }

    #[test]
    fn energy_ceiling_uses_existing_provenance_and_does_not_invent_a_budget() {
        let action = Action::inspect_decision([0x55; 32]);
        let bytes = action.canonical();
        let open = Grant::issue("op.open", action.clone(), None).unwrap();
        let nan = measured(f64::NAN, Provenance::ModelBased);
        let open_decision = authorize(&[open], bytes.as_bytes(), Some(&nan));
        assert!(open_decision.is_in_scope());
        assert!(check_of(&open_decision).same_commitment(&action, None));
        assert!(!check_of(&open_decision).same_commitment(&action, Some(&nan)));

        let capped = Grant::issue("op.cap", action.clone(), Some(1.0)).unwrap();
        let ok = measured(1.0, Provenance::ModelBased);
        let capped_ok = authorize(&[capped.clone()], bytes.as_bytes(), Some(&ok));
        assert!(capped_ok.is_in_scope());
        assert!(check_of(&capped_ok).same_commitment(&action, Some(&ok)));
        assert!(!check_of(&capped_ok)
            .same_commitment(&action, Some(&measured(1.0, Provenance::HwShunt))));
        assert!(!check_of(&capped_ok).same_commitment(&action, None));
        let shunt = measured(0.25, Provenance::HwShunt);
        assert!(authorize(&[capped.clone()], bytes.as_bytes(), Some(&shunt)).is_in_scope());
        let zero = Grant::issue("op.zero", action.clone(), Some(0.0)).unwrap();
        assert!(authorize(
            &[zero.clone()],
            bytes.as_bytes(),
            Some(&measured(0.0, Provenance::ModelBased))
        )
        .is_in_scope());
        assert_eq!(
            authorize(
                &[zero],
                bytes.as_bytes(),
                Some(&measured(f64::MIN_POSITIVE, Provenance::HwShunt))
            )
            .denial(),
            Some(DenyReason::OverBudget)
        );
        assert_eq!(
            authorize(&[capped.clone()], bytes.as_bytes(), None).denial(),
            Some(DenyReason::EnergyMissing)
        );
        assert_eq!(
            authorize(&[capped.clone()], bytes.as_bytes(), Some(&nan)).denial(),
            Some(DenyReason::EnergyInvalid)
        );
        assert_eq!(
            authorize(
                &[capped.clone()],
                bytes.as_bytes(),
                Some(&measured(f64::INFINITY, Provenance::HwShunt))
            )
            .denial(),
            Some(DenyReason::EnergyInvalid)
        );
        assert_eq!(
            authorize(
                &[capped.clone()],
                bytes.as_bytes(),
                Some(&measured(-0.1, Provenance::ModelBased))
            )
            .denial(),
            Some(DenyReason::EnergyInvalid)
        );
        assert_eq!(
            authorize(
                &[capped.clone()],
                bytes.as_bytes(),
                Some(&measured(0.0, Provenance::Estimator))
            )
            .denial(),
            Some(DenyReason::EnergyUnmeasured)
        );
        assert_eq!(
            authorize(
                &[capped.clone()],
                bytes.as_bytes(),
                Some(&measured(1.0 + f64::EPSILON, Provenance::ModelBased))
            )
            .denial(),
            Some(DenyReason::OverBudget)
        );
        let loose = Grant::issue("op.loose", action.clone(), Some(5.0)).unwrap();
        let mid = measured(2.0, Provenance::ModelBased);
        assert_eq!(
            authorize(
                &[loose.clone(), capped.clone()],
                bytes.as_bytes(),
                Some(&mid)
            )
            .denial(),
            Some(DenyReason::OverBudget)
        );
        assert!(authorize(
            &[loose, capped],
            bytes.as_bytes(),
            Some(&measured(1.0, Provenance::ModelBased))
        )
        .is_in_scope());
        let unbounded = Grant::issue("op.wide", action.clone(), None).unwrap();
        let tight = Grant::issue("op.tight", action, Some(1.0)).unwrap();
        assert_eq!(
            authorize(&[unbounded, tight], bytes.as_bytes(), Some(&mid)).denial(),
            Some(DenyReason::OverBudget)
        );
    }

    #[test]
    fn strictest_ceiling_names_that_grant() {
        let action = Action::inspect_decision([0x66; 32]);
        let bytes = action.canonical();
        let wide = Grant::issue("op.wide", action.clone(), Some(5.0)).unwrap();
        let tight = Grant::issue("op.tight", action, Some(1.0)).unwrap();
        let d = authorize(
            &[wide, tight],
            bytes.as_bytes(),
            Some(&measured(1.0, Provenance::ModelBased)),
        );
        let checked = check_of(&d);
        assert_eq!(checked.capability_id(), "op.tight");
        assert!(matches!(
            checked.energy_commitment(),
            EnergyCommitment::WithinCeiling {
                joules_bits,
                ceiling_bits,
                provenance: Provenance::ModelBased,
            } if joules_bits == 1.0f64.to_bits() && ceiling_bits == 1.0f64.to_bits()
        ));
    }

    #[test]
    fn bad_grants_are_rejected() {
        let action = Action::inspect_decision([1; 32]);
        assert_eq!(
            Grant::issue("", action.clone(), None).unwrap_err(),
            DenyReason::BadGrant
        );
        assert_eq!(
            Grant::issue("has space", action.clone(), None).unwrap_err(),
            DenyReason::BadGrant
        );
        assert_eq!(
            Grant::issue("slash/id", action.clone(), None).unwrap_err(),
            DenyReason::BadGrant
        );
        assert_eq!(
            Grant::issue("op", action.clone(), Some(f64::NAN)).unwrap_err(),
            DenyReason::BadCeiling
        );
        assert_eq!(
            Grant::issue("op", action.clone(), Some(-1.0)).unwrap_err(),
            DenyReason::BadCeiling
        );
        assert_eq!(
            Grant::issue("op", action, Some(f64::INFINITY)).unwrap_err(),
            DenyReason::BadCeiling
        );
        assert_eq!(
            Action::quote_receipt("", [0; 32]).unwrap_err(),
            DenyReason::Malformed
        );
        assert_eq!(
            Action::quote_receipt("-rf", [0; 32]).unwrap_err(),
            DenyReason::Malformed
        );
        assert_eq!(
            Action::quote_receipt("../x", [0; 32]).unwrap_err(),
            DenyReason::Malformed
        );
        let sneaky = Action::QuoteReceipt {
            thread_id: "rm rf".into(),
            receipt_hash: [0; 32],
        };
        assert!(Grant::issue("op", sneaky, None).is_err());
    }

    #[test]
    fn signature_accept_is_not_authorization() {
        let k = SigningKey::from_bytes(&[1; 32]);
        let sub = Subscription {
            tool_call_id: "sub1".into(),
            dispatch_receipt: [7; 32],
            expected_signer: k.verifying_key().to_bytes(),
        };
        let payload = b"Commit\nnope";
        let env = seal(&k, "sub1", payload, [7; 32], 0);
        assert_eq!(
            Gate::default().check(&sub, Some(&env), payload),
            Verdict::Accept
        );
        let g = Grant::issue("op.ping", Action::inspect_decision([0x11; 32]), Some(1.0e9)).unwrap();
        assert_eq!(
            authorize(&[g], payload, Some(&measured(0.0, Provenance::HwShunt))).denial(),
            Some(DenyReason::Unstructured)
        );

        let action = Action::inspect_decision([0x11; 32]);
        let bytes = action.canonical();
        let signed = seal(&k, "sub1", bytes.as_bytes(), [7; 32], 1);
        assert_eq!(
            Gate::default().check(&sub, Some(&signed), bytes.as_bytes()),
            Verdict::Accept
        );
        assert_eq!(
            authorize(&[], bytes.as_bytes(), None).denial(),
            Some(DenyReason::NoAuthority)
        );
        assert_eq!(
            Gate::default().check(&sub, None, bytes.as_bytes()),
            Verdict::Quarantine("unsigned")
        );
        let g = Grant::issue("op.ping", action, None).unwrap();
        assert!(authorize(&[g], bytes.as_bytes(), None).is_in_scope());
    }

    #[test]
    fn noncanonical_struct_is_not_the_same_scope() {
        let good = Action::quote_receipt("threadA", [9; 32]).unwrap();
        let evil = Action::QuoteReceipt {
            thread_id: "threadA\nbad".into(),
            receipt_hash: [9; 32],
        };
        assert!(good.scope_binding().is_some());
        assert!(evil.scope_binding().is_none());
        let g = Grant::issue("op.quote", good.clone(), None).unwrap();
        let d = authorize(&[g], good.canonical().as_bytes(), None);
        assert!(check_of(&d).same_scope(&good));
        assert!(!check_of(&d).same_scope(&evil));
        assert!(!check_of(&d).same_commitment(&evil, None));
    }

    #[test]
    fn negative_zero_ceiling_is_canonical_positive_zero() {
        let action = Action::inspect_decision([0x77; 32]);
        let g = Grant::issue("op.zero", action.clone(), Some(-0.0)).unwrap();
        assert_eq!(g.compute_ceiling_j().unwrap().to_bits(), 0.0f64.to_bits());
        let neg = measured(-0.0, Provenance::ModelBased);
        let pos = measured(0.0, Provenance::ModelBased);
        let d = authorize(&[g], action.canonical().as_bytes(), Some(&neg));
        assert!(check_of(&d).same_commitment(&action, Some(&pos)));
        assert!(!check_of(&d).same_commitment(&action, Some(&measured(0.0, Provenance::HwShunt))));
    }

    #[test]
    fn signed_receipt_json_is_not_an_in_scope_result() {
        use crate::ledger::{self, Ledger};
        let mut ledger = Ledger::open(None, SigningKey::from_bytes(&[3; 32])).unwrap();
        let signed = ledger
            .record(
                "threadA",
                "input-1",
                [1; 32],
                Tier::Rules,
                1,
                1,
                measured(0.0, Provenance::ModelBased),
                measured(0.0, Provenance::Estimator),
                Some("not-a-capability".into()),
            )
            .unwrap();
        let json = ledger::to_json(&signed);
        let action = Action::inspect_decision([1; 32]);
        let g = Grant::issue("op.ping", action, Some(1.0)).unwrap();
        assert_eq!(
            authorize(
                &[g],
                json.as_bytes(),
                Some(&measured(0.0, Provenance::ModelBased))
            )
            .denial(),
            Some(DenyReason::Unstructured)
        );
        assert!(!json.contains("strix-rt/action/v1"));
    }

    #[test]
    fn decisions_surface_default_denies_without_inventing_a_grant() {
        let model = br#"{"object":"decision","answers":[{"type":"choice","choice":"commit","probability":1}]}"#;
        assert_eq!(parse_action(model), Err(DenyReason::Unstructured));
        let content = [0x11; 32];
        let receipt = [0x22; 32];
        let surface = DecisionsAuthorization::assess(&[], content, "default", receipt, None);
        assert!(!surface.grant_source_configured);
        assert_eq!(
            surface.inspect_decision.denial(),
            Some(DenyReason::NoAuthority)
        );
        assert_eq!(
            surface.quote_receipt.denial(),
            Some(DenyReason::NoAuthority)
        );
        let json = surface.to_json();
        assert!(json.contains(r#""executes":false"#), "{json}");
        assert!(json.contains(r#""grant_source":"none"#), "{json}");
        assert!(
            json.contains("no operator grant store is configured"),
            "{json}"
        );
        assert!(
            json.contains(r#""scope_match":false,"reason":"no_authority""#),
            "{json}"
        );
        assert!(!json.contains("commit"));
        let bad_thread = DecisionsAuthorization::assess(&[], content, "bad id", receipt, None);
        assert_eq!(
            bad_thread.quote_receipt.denial(),
            Some(DenyReason::Malformed)
        );
        let grant = Grant::issue("op.inspect", Action::inspect_decision(content), None).unwrap();
        let matched = DecisionsAuthorization::assess(
            &[grant],
            content,
            "default",
            receipt,
            Some(&measured(0.0, Provenance::ModelBased)),
        );
        assert!(matched.grant_source_configured);
        assert!(matched.inspect_decision.is_in_scope());
        assert_eq!(
            matched.quote_receipt.denial(),
            Some(DenyReason::ActionMismatch)
        );
        let matched_json = matched.to_json();
        assert!(
            matched_json.contains(r#""executes":false"#),
            "{matched_json}"
        );
        assert!(
            matched_json.contains(r#""grant_source":"in_process""#),
            "{matched_json}"
        );
        assert!(
            matched_json.contains(r#""scope_match":true"#),
            "{matched_json}"
        );
        assert!(!matched_json.contains("prerequisite"));
    }
}
