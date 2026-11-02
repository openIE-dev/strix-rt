//! JouleContract-style per-slice receipts for an Infinity-style suspend/resume layer in front of /v1/decisions
//! (docs/infinity-suspend-resume.md). Types + content hash only: no GPU, no I/O, no signing yet.
//! A slice = one completion + persist + at most one tool dispatch. Two keys, deliberately separate:
//!   - `input_id`: delivery dedup (Infinity's processed_message_ids). Opaque, NOT hashed.
//!   - `content_hash`: what was computed (history prefix, input payload, model, tier, decode params). Doubles as the
//!     JouleClaw L0 cache key. Must not depend on run-time randomness (e.g. Infinity's uuid synthetic tool-call ids).
pub mod sign;
pub mod ledger;
pub mod meter;
use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

/// Which cascade tier answered the slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier { L0Cache, Rules, Small { batch: u16 }, Kev9b { batch: u16 } }

/// JouleClaw provenance of an energy figure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance { HwShunt, ModelBased, Estimator }

/// How an input reached the slice; synthetic events carry whether their envelope verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputKind { User, ToolResult { tool_call_id: String }, Event { original_tool_call_id: String, verified: bool } }

/// Deterministic decode settings that affect output (part of the content hash).
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeParams { pub greedy: bool, pub seed: u64, pub max_tokens: u32 }

/// Everything that determines a slice's output. Hashing this gives the content hash.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceKey<'a> {
    pub history_prefix: Hash,
    pub input: &'a InputKind,
    pub payload: &'a [u8],
    pub model: Hash,
    pub tier: Tier,
    pub params: &'a DecodeParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Energy { pub joules: f64, pub provenance: Provenance }

#[derive(Clone, Debug, PartialEq)]
pub struct SliceReceipt {
    pub thread_id: String,
    pub slice_seq: u64,
    pub input_id: String,
    pub content_hash: Hash,
    pub prev_receipt: Option<Hash>,
    pub tier: Tier,
    pub tokens_in: u32,
    pub tokens_out: u32,
    /// GPU/CPU energy of this slice (its share when batched).
    pub compute: Energy,
    /// Marginal energy attributed while suspended before this slice (box idle floor is reported separately).
    pub wait: Energy,
    pub dispatched_tool: Option<String>,
}

struct H(Sha256);
impl H {
    fn new(domain: &str) -> Self { let mut h = H(Sha256::new()); h.bytes(domain.as_bytes()); h }
    fn bytes(&mut self, b: &[u8]) { self.0.update((b.len() as u64).to_le_bytes()); self.0.update(b); }
    fn u64(&mut self, v: u64) { self.0.update(v.to_le_bytes()); }
    fn f64(&mut self, v: f64) { self.0.update(v.to_bits().to_le_bytes()); }
    fn tier(&mut self, t: Tier) {
        match t { Tier::L0Cache => self.u64(0), Tier::Rules => self.u64(1),
            Tier::Small { batch } => { self.u64(2); self.u64(batch as u64) }, Tier::Kev9b { batch } => { self.u64(3); self.u64(batch as u64) } }
    }
    fn done(self) -> Hash { self.0.finalize().into() }
}

/// Tier identity without batch size: batch composition must not change what a slice *means*.
fn tier_class(t: Tier) -> Tier { match t { Tier::Small { .. } => Tier::Small { batch: 0 }, Tier::Kev9b { .. } => Tier::Kev9b { batch: 0 }, x => x } }

pub fn content_hash(k: &SliceKey) -> Hash {
    let mut h = H::new("strix-rt/slice-content/v1");
    h.bytes(&k.history_prefix);
    match k.input {
        InputKind::User => h.u64(0),
        // tool_call_id is a stable model/runtime id for real calls; kept so a result binds to its call.
        InputKind::ToolResult { tool_call_id } => { h.u64(1); h.bytes(tool_call_id.as_bytes()) }
        // Events bind to the ORIGINAL subscription call, never the per-injection uuid.
        InputKind::Event { original_tool_call_id, verified } => { h.u64(2); h.bytes(original_tool_call_id.as_bytes()); h.u64(*verified as u64) }
    }
    h.bytes(k.payload);
    h.bytes(&k.model);
    h.tier(tier_class(k.tier));
    h.u64(k.params.greedy as u64); h.u64(k.params.seed); h.u64(k.params.max_tokens as u64);
    h.done()
}

/// Extend a history-prefix hash with one appended item (rolling, so resume never rehashes the whole history).
pub fn extend_prefix(prev: &Hash, item: &[u8]) -> Hash { let mut h = H::new("strix-rt/history/v1"); h.bytes(prev); h.bytes(item); h.done() }

impl SliceReceipt {
    /// Hash of the full receipt (chained via prev_receipt; what a signature will cover).
    pub fn receipt_hash(&self) -> Hash {
        let mut h = H::new("strix-rt/slice-receipt/v1");
        h.bytes(self.thread_id.as_bytes()); h.u64(self.slice_seq); h.bytes(self.input_id.as_bytes());
        h.bytes(&self.content_hash);
        match &self.prev_receipt { Some(p) => { h.u64(1); h.bytes(p) } None => h.u64(0) }
        h.tier(self.tier); h.u64(self.tokens_in as u64); h.u64(self.tokens_out as u64);
        for e in [&self.compute, &self.wait] { h.f64(e.joules); h.u64(e.provenance as u64) }
        match &self.dispatched_tool { Some(t) => { h.u64(1); h.bytes(t.as_bytes()) } None => h.u64(0) }
        h.done()
    }
    /// Receipt for a slice answered from the L0 content-hash cache: no compute, estimator-tagged.
    pub fn cache_hit(thread_id: &str, slice_seq: u64, input_id: &str, content_hash: Hash, prev: Option<Hash>) -> Self {
        SliceReceipt { thread_id: thread_id.into(), slice_seq, input_id: input_id.into(), content_hash, prev_receipt: prev,
            tier: Tier::L0Cache, tokens_in: 0, tokens_out: 0,
            compute: Energy { joules: 0.0, provenance: Provenance::Estimator },
            wait: Energy { joules: 0.0, provenance: Provenance::Estimator }, dispatched_tool: None }
    }
}

/// Split one batch's measured energy across its member slices by token share (ModelBased provenance).
pub fn apportion(batch_joules: f64, tokens: &[u32]) -> Vec<Energy> {
    let total: u64 = tokens.iter().map(|&t| t as u64).sum();
    tokens.iter().map(|&t| Energy {
        joules: if total == 0 { batch_joules / tokens.len().max(1) as f64 } else { batch_joules * t as f64 / total as f64 },
        provenance: Provenance::ModelBased }).collect()
}

pub fn hex(h: &Hash) -> String { h.iter().map(|b| format!("{b:02x}")).collect() }

#[cfg(test)]
mod tests {
    use super::*;
    fn p() -> DecodeParams { DecodeParams { greedy: true, seed: 0, max_tokens: 64 } }
    fn key<'a>(i: &'a InputKind, pl: &'a [u8], pr: &'a DecodeParams, t: Tier) -> SliceKey<'a> {
        SliceKey { history_prefix: [1; 32], input: i, payload: pl, model: [2; 32], tier: t, params: pr }
    }

    #[test] fn deterministic() {
        let (i, pr) = (InputKind::User, p());
        assert_eq!(content_hash(&key(&i, b"q", &pr, Tier::Small { batch: 8 })), content_hash(&key(&i, b"q", &pr, Tier::Small { batch: 8 })));
    }
    #[test] fn batch_size_does_not_change_meaning() {
        let (i, pr) = (InputKind::User, p());
        assert_eq!(content_hash(&key(&i, b"q", &pr, Tier::Small { batch: 1 })), content_hash(&key(&i, b"q", &pr, Tier::Small { batch: 8 })));
        assert_ne!(content_hash(&key(&i, b"q", &pr, Tier::Small { batch: 8 })), content_hash(&key(&i, b"q", &pr, Tier::Kev9b { batch: 8 })));
    }
    #[test] fn inputs_and_params_matter() {
        let (u, pr) = (InputKind::User, p());
        let base = content_hash(&key(&u, b"q", &pr, Tier::Rules));
        assert_ne!(base, content_hash(&key(&u, b"r", &pr, Tier::Rules)));
        let pr2 = DecodeParams { seed: 1, ..p() };
        assert_ne!(base, content_hash(&key(&u, b"q", &pr2, Tier::Rules)));
        let tr = InputKind::ToolResult { tool_call_id: "c1".into() };
        assert_ne!(base, content_hash(&key(&tr, b"q", &pr, Tier::Rules)));
    }
    #[test] fn length_prefix_prevents_concat_collisions() {
        let pr = p();
        let a = InputKind::ToolResult { tool_call_id: "ab".into() }; let b = InputKind::ToolResult { tool_call_id: "a".into() };
        assert_ne!(content_hash(&key(&a, b"c", &pr, Tier::Rules)), content_hash(&key(&b, b"bc", &pr, Tier::Rules)));
    }
    #[test] fn unverified_event_differs_from_verified() {
        let pr = p();
        let v = InputKind::Event { original_tool_call_id: "sub1".into(), verified: true };
        let u = InputKind::Event { original_tool_call_id: "sub1".into(), verified: false };
        assert_ne!(content_hash(&key(&v, b"e", &pr, Tier::Rules)), content_hash(&key(&u, b"e", &pr, Tier::Rules)));
    }
    #[test] fn input_id_not_in_content_hash_but_in_receipt() {
        let ch = [9; 32];
        let a = SliceReceipt::cache_hit("t", 0, "msg-1", ch, None); let b = SliceReceipt::cache_hit("t", 0, "msg-2", ch, None);
        assert_eq!(a.content_hash, b.content_hash);
        assert_ne!(a.receipt_hash(), b.receipt_hash());
        assert_eq!(a.compute.joules, 0.0); assert_eq!(a.compute.provenance, Provenance::Estimator);
    }
    #[test] fn chain_links() {
        let r0 = SliceReceipt::cache_hit("t", 0, "m0", [3; 32], None);
        let r1 = SliceReceipt::cache_hit("t", 1, "m1", [3; 32], Some(r0.receipt_hash()));
        let r1b = SliceReceipt::cache_hit("t", 1, "m1", [3; 32], Some([0; 32]));
        assert_ne!(r1.receipt_hash(), r1b.receipt_hash());
    }
    #[test] fn prefix_rolls() {
        let a = extend_prefix(&[0; 32], b"x"); assert_ne!(a, extend_prefix(&[0; 32], b"y"));
        assert_ne!(extend_prefix(&a, b"y"), extend_prefix(&extend_prefix(&[0; 32], b"y"), b"x"));
        assert_eq!(hex(&a).len(), 64);
    }
    #[test] fn apportion_conserves() {
        let e = apportion(8.0, &[1, 3, 4]);
        assert!((e.iter().map(|x| x.joules).sum::<f64>() - 8.0).abs() < 1e-12);
        assert!(e.iter().all(|x| x.provenance == Provenance::ModelBased));
        assert_eq!(apportion(6.0, &[0, 0, 0])[1].joules, 2.0);
    }
}

/// Content hash of one stateless /v1/decisions slice. `body` is the request with transport ids (thread_id, input_id)
/// already removed by the caller; greedy pointer-head scoring, so decode params are fixed.
pub fn decision_hash(body: &[u8], model: &str, tier: Tier) -> Hash {
    let model_h: Hash = Sha256::digest(model.as_bytes()).into();
    let p = DecodeParams { greedy: true, seed: 0, max_tokens: 0 };
    content_hash(&SliceKey { history_prefix: [0; 32], input: &InputKind::User, payload: body, model: model_h, tier, params: &p })
}

#[cfg(test)]
mod decision_tests {
    use super::*;
    #[test] fn decision_hash_stable_and_model_bound() {
        assert_eq!(decision_hash(b"{}", "kev-0.8b", Tier::Small { batch: 1 }), decision_hash(b"{}", "kev-0.8b", Tier::Small { batch: 8 }));
        assert_ne!(decision_hash(b"{}", "kev-0.8b", Tier::Small { batch: 1 }), decision_hash(b"{}", "kev-9b", Tier::Small { batch: 1 }));
    }
}
