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
//! The parameters of an action are its scope. The allow binding is a domain-separated
//! SHA-256 of the canonical bytes, so a later payload swap does not match `Permit::covers`.
//! A `Permit` is the record of one `authorize` call. It is not a bearer token; nothing
//! in this crate consumes a permit to perform I/O.
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

/// Closed set of actions this baseline can name. Both are references to objects the
/// runtime has already computed. Neither variant runs a tool.
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

    /// Domain-separated hash of [`Self::canonical`]. This is the scope binding.
    pub fn binding(&self) -> Hash {
        binding_of(&self.canonical())
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
        if let Some(c) = compute_ceiling_j {
            if !c.is_finite() || c < 0.0 {
                return Err(DenyReason::BadCeiling);
            }
        }
        let parsed = parse_action(action.canonical().as_bytes())?;
        if parsed != action {
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

/// Record of an allow. `_seal` stops other crates from building a permit by struct literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permit {
    capability_id: String,
    binding: Hash,
    _seal: (),
}

impl Permit {
    pub fn capability_id(&self) -> &str {
        &self.capability_id
    }
    pub fn binding(&self) -> &Hash {
        &self.binding
    }
    /// True only when `action` is the exact action this allow was computed for.
    pub fn covers(&self, action: &Action) -> bool {
        self.binding == action.binding()
    }
}

/// Result of [`authorize`]. An allow is not an instruction to execute anything.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "an authorization decision must be checked; allow does not execute"]
pub enum Decision {
    Allow(Permit),
    Deny(DenyReason),
}

impl Decision {
    pub fn is_allow(&self) -> bool {
        matches!(self, Decision::Allow(_))
    }
    pub fn is_deny(&self) -> bool {
        matches!(self, Decision::Deny(_))
    }
    pub fn permit(&self) -> Option<&Permit> {
        match self {
            Decision::Allow(p) => Some(p),
            Decision::Deny(_) => None,
        }
    }
    pub fn denial(&self) -> Option<DenyReason> {
        match self {
            Decision::Allow(_) => None,
            Decision::Deny(r) => Some(*r),
        }
    }
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decision::Allow(p) => write!(
                f,
                "allow capability_id={} binding={}",
                p.capability_id(),
                hex(p.binding())
            ),
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
/// only from `energy`; this function does not meter.
#[must_use = "an authorization decision must be checked; allow does not execute"]
pub fn authorize(grants: &[Grant], proposal: &[u8], energy: Option<&Energy>) -> Decision {
    let action = match parse_action(proposal) {
        Ok(a) => a,
        Err(r) => return Decision::Deny(r),
    };
    if action.canonical().as_bytes() != proposal {
        return Decision::Deny(DenyReason::Malformed);
    }
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
        if g.action != action {
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
    if let Some(ceiling) = strictest {
        if let Some(reason) = energy_refusal(ceiling, energy) {
            return Decision::Deny(reason);
        }
    }
    Decision::Allow(Permit {
        capability_id: id.to_string(),
        binding: action.binding(),
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
    fn allow_of(d: &Decision) -> &Permit {
        d.permit().expect("allow")
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
            let p = allow_of(&d);
            assert!(p.covers(&action));
            assert_eq!(p.capability_id(), "op.ping");
            assert_eq!(p.binding(), &action.binding());
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
        assert!(authorize(&[g], bytes.as_bytes(), None).is_allow());
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
        assert_ne!(inspect.binding(), other.binding());
        assert_ne!(inspect.binding(), quote.binding());
    }

    #[test]
    fn permit_does_not_cover_a_substituted_action() {
        let a = Action::inspect_decision([0xab; 32]);
        let b = Action::inspect_decision([0xcd; 32]);
        let g = Grant::issue("op.ping", a.clone(), None).unwrap();
        let d = authorize(&[g], a.canonical().as_bytes(), None);
        let p = allow_of(&d);
        assert!(p.covers(&a));
        assert!(!p.covers(&b));
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
            assert!(authorize(&[g.clone()], &base, None).is_allow());
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
        let again = action.binding();
        assert_eq!(action.binding(), again);
        let raw: Hash = Sha256::digest(action.canonical().as_bytes()).into();
        assert_ne!(action.binding(), raw);
        let text = format!("deny {}", authorize(&[], b"unique-marker-zxq", None));
        assert!(!text.contains("unique-marker"));
    }

    #[test]
    fn energy_ceiling_uses_existing_provenance_and_does_not_invent_a_budget() {
        let action = Action::inspect_decision([0x55; 32]);
        let bytes = action.canonical();
        let open = Grant::issue("op.open", action.clone(), None).unwrap();
        let nan = measured(f64::NAN, Provenance::ModelBased);
        assert!(authorize(&[open], bytes.as_bytes(), Some(&nan)).is_allow());

        let capped = Grant::issue("op.cap", action.clone(), Some(1.0)).unwrap();
        let ok = measured(1.0, Provenance::ModelBased);
        assert!(authorize(&[capped.clone()], bytes.as_bytes(), Some(&ok)).is_allow());
        let shunt = measured(0.25, Provenance::HwShunt);
        assert!(authorize(&[capped.clone()], bytes.as_bytes(), Some(&shunt)).is_allow());
        let zero = Grant::issue("op.zero", action.clone(), Some(0.0)).unwrap();
        assert!(authorize(
            &[zero.clone()],
            bytes.as_bytes(),
            Some(&measured(0.0, Provenance::ModelBased))
        )
        .is_allow());
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
        .is_allow());
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
        assert_eq!(allow_of(&d).capability_id(), "op.tight");
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
        assert!(authorize(&[g], bytes.as_bytes(), None).is_allow());
    }
}
