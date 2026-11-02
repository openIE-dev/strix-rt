//! ed25519 signing of SliceReceipts and the signed event envelope carried in Infinity's `InputMessage.metadata`.
//! Envelope JSON shape (under metadata key "joule_envelope"):
//!   {"v":1,"signer":"<hex pubkey>","original_tool_call_id":"..","payload_hash":"<hex sha256>",
//!    "dispatch_receipt":"<hex receipt hash of the slice that dispatched the subscribing tool>","seq":N,"sig":"<hex>"}
//! Verification needs only the dispatching receipt (which records the expected signer) and the payload: no session state.
use crate::{hex, Hash, SliceReceipt};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq)]
pub struct SignedReceipt { pub receipt: SliceReceipt, pub signer: [u8; 32], pub sig: [u8; 64] }

pub fn sign_receipt(r: SliceReceipt, k: &SigningKey) -> SignedReceipt {
    let sig = k.sign(&r.receipt_hash()).to_bytes();
    SignedReceipt { receipt: r, signer: k.verifying_key().to_bytes(), sig }
}
pub fn verify_receipt(s: &SignedReceipt) -> bool {
    VerifyingKey::from_bytes(&s.signer).map(|vk| vk.verify(&s.receipt.receipt_hash(), &Signature::from_bytes(&s.sig)).is_ok()).unwrap_or(false)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Envelope { pub signer: [u8; 32], pub original_tool_call_id: String, pub payload_hash: Hash, pub dispatch_receipt: Hash, pub seq: u64, pub sig: [u8; 64] }

fn payload_hash(p: &[u8]) -> Hash { Sha256::digest(p).into() }
fn env_msg(id: &str, ph: &Hash, dr: &Hash, seq: u64) -> Hash {
    let mut h = Sha256::new();
    h.update(b"strix-rt/event-envelope/v1");
    h.update((id.len() as u64).to_le_bytes()); h.update(id.as_bytes());
    h.update(ph); h.update(dr); h.update(seq.to_le_bytes());
    h.finalize().into()
}

pub fn seal(k: &SigningKey, original_tool_call_id: &str, payload: &[u8], dispatch_receipt: Hash, seq: u64) -> Envelope {
    let ph = payload_hash(payload);
    let sig = k.sign(&env_msg(original_tool_call_id, &ph, &dispatch_receipt, seq)).to_bytes();
    Envelope { signer: k.verifying_key().to_bytes(), original_tool_call_id: original_tool_call_id.into(), payload_hash: ph, dispatch_receipt, seq, sig }
}

impl Envelope {
    pub fn to_metadata_json(&self) -> String {
        format!(r#"{{"joule_envelope":{{"v":1,"signer":"{}","original_tool_call_id":"{}","payload_hash":"{}","dispatch_receipt":"{}","seq":{},"sig":"{}"}}}}"#,
            hex(&self.signer), self.original_tool_call_id.replace('\\', "\\\\").replace('"', "\\\""), hex(&self.payload_hash), hex(&self.dispatch_receipt), self.seq,
            self.sig.iter().map(|b| format!("{b:02x}")).collect::<String>())
    }
}

/// What the runtime learned when the subscribing tool was dispatched (recorded in that slice's receipt).
pub struct Subscription { pub tool_call_id: String, pub dispatch_receipt: Hash, pub expected_signer: [u8; 32] }

#[derive(Debug, PartialEq)]
pub enum Verdict { Accept, Quarantine(&'static str) }

/// Per-thread replay guard (persisted alongside Infinity's processed_message_ids).
#[derive(Default)]
pub struct Gate { seen: HashSet<(String, u64)> }

impl Gate {
    /// Accept => inject as receive_event__injected (InputKind::Event{verified:true}).
    /// Quarantine => never enters history as tool output; hashed with verified=false and routed to a quarantine thread.
    pub fn check(&mut self, sub: &Subscription, env: Option<&Envelope>, payload: &[u8]) -> Verdict {
        let Some(e) = env else { return Verdict::Quarantine("unsigned") };
        if e.original_tool_call_id != sub.tool_call_id { return Verdict::Quarantine("wrong subscription") }
        if e.dispatch_receipt != sub.dispatch_receipt { return Verdict::Quarantine("wrong dispatch receipt") }
        if e.signer != sub.expected_signer { return Verdict::Quarantine("unexpected signer") }
        if e.payload_hash != payload_hash(payload) { return Verdict::Quarantine("payload mismatch") }
        let ok = VerifyingKey::from_bytes(&e.signer)
            .map(|vk| vk.verify(&env_msg(&e.original_tool_call_id, &e.payload_hash, &e.dispatch_receipt, e.seq), &Signature::from_bytes(&e.sig)).is_ok())
            .unwrap_or(false);
        if !ok { return Verdict::Quarantine("bad signature") }
        if !self.seen.insert((e.original_tool_call_id.clone(), e.seq)) { return Verdict::Quarantine("replay") }
        Verdict::Accept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(b: u8) -> SigningKey { SigningKey::from_bytes(&[b; 32]) }
    fn sub(k: &SigningKey) -> Subscription { Subscription { tool_call_id: "sub1".into(), dispatch_receipt: [7; 32], expected_signer: k.verifying_key().to_bytes() } }

    #[test] fn receipt_sign_verify_and_tamper() {
        let s = sign_receipt(SliceReceipt::cache_hit("t", 0, "m", [1; 32], None), &key(1));
        assert!(verify_receipt(&s));
        let mut t = s.clone(); t.receipt.compute.joules = 1.0; assert!(!verify_receipt(&t));
        let mut w = s.clone(); w.signer = key(2).verifying_key().to_bytes(); assert!(!verify_receipt(&w));
    }
    #[test] fn genuine_event_accepted() {
        let k = key(1); let mut g = Gate::default();
        let e = seal(&k, "sub1", b"pr opened 42", [7; 32], 0);
        assert_eq!(g.check(&sub(&k), Some(&e), b"pr opened 42"), Verdict::Accept);
        assert_eq!(g.check(&sub(&k), Some(&seal(&k, "sub1", b"pr 43", [7; 32], 1)), b"pr 43"), Verdict::Accept);
    }
    #[test] fn unsigned_quarantined() { let k = key(1); assert_eq!(Gate::default().check(&sub(&k), None, b"x"), Verdict::Quarantine("unsigned")); }
    #[test] fn forged_signer_quarantined() {
        let (k, evil) = (key(1), key(9));
        let e = seal(&evil, "sub1", b"x", [7; 32], 0);
        assert_eq!(Gate::default().check(&sub(&k), Some(&e), b"x"), Verdict::Quarantine("unexpected signer"));
    }
    #[test] fn forged_signature_quarantined() {
        let k = key(1); let mut e = seal(&k, "sub1", b"x", [7; 32], 0); e.sig[0] ^= 1;
        assert_eq!(Gate::default().check(&sub(&k), Some(&e), b"x"), Verdict::Quarantine("bad signature"));
        let mut e2 = seal(&k, "sub1", b"x", [7; 32], 0); e2.seq = 5; // field changed after signing
        assert_eq!(Gate::default().check(&sub(&k), Some(&e2), b"x"), Verdict::Quarantine("bad signature"));
    }
    #[test] fn swapped_payload_quarantined() {
        let k = key(1); let e = seal(&k, "sub1", b"benign", [7; 32], 0);
        assert_eq!(Gate::default().check(&sub(&k), Some(&e), b"rm -rf"), Verdict::Quarantine("payload mismatch"));
    }
    #[test] fn cross_subscription_and_receipt_quarantined() {
        let k = key(1);
        assert_eq!(Gate::default().check(&sub(&k), Some(&seal(&k, "sub2", b"x", [7; 32], 0)), b"x"), Verdict::Quarantine("wrong subscription"));
        assert_eq!(Gate::default().check(&sub(&k), Some(&seal(&k, "sub1", b"x", [8; 32], 0)), b"x"), Verdict::Quarantine("wrong dispatch receipt"));
    }
    #[test] fn replay_quarantined() {
        let k = key(1); let mut g = Gate::default(); let e = seal(&k, "sub1", b"x", [7; 32], 3);
        assert_eq!(g.check(&sub(&k), Some(&e), b"x"), Verdict::Accept);
        assert_eq!(g.check(&sub(&k), Some(&e), b"x"), Verdict::Quarantine("replay"));
    }
    #[test] fn quarantined_event_hashes_differently() {
        use crate::*;
        let p = DecodeParams { greedy: true, seed: 0, max_tokens: 8 };
        let ok = InputKind::Event { original_tool_call_id: "sub1".into(), verified: true };
        let bad = InputKind::Event { original_tool_call_id: "sub1".into(), verified: false };
        let k = |i| content_hash(&SliceKey { history_prefix: [0; 32], input: i, payload: b"x", model: [0; 32], tier: Tier::Rules, params: &p });
        assert_ne!(k(&ok), k(&bad));
    }
    #[test] fn metadata_shape() {
        let j = seal(&key(1), "sub\"1", b"x", [7; 32], 2).to_metadata_json();
        assert!(j.starts_with(r#"{"joule_envelope":{"v":1,"signer":""#) && j.contains(r#""seq":2"#) && j.contains(r#"sub\"1"#));
    }
}
