//! Per-thread receipt chains for kev_serve: assigns slice_seq, links prev_receipt, signs, persists the seq record,
//! and renders the receipt as JSON for the /v1/decisions response. No GPU, no JSON crate.
//! Ledger file (append-only TSV, one line per slice): thread_id \t slice_seq \t receipt_hash_hex \t content_hash_hex
//! On open the last line per thread restores (next seq, prev hash), so chains survive a restart.
use crate::sign::{sign_receipt, SignedReceipt};
use crate::{hex, Energy, Hash, Provenance, SliceReceipt, Tier};
use ed25519_dalek::SigningKey;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

pub struct Ledger { path: Option<PathBuf>, heads: HashMap<String, (u64, Hash)>, key: SigningKey }

fn unhex(s: &str) -> Option<Hash> {
    if s.len() != 64 { return None }
    let mut h = [0u8; 32];
    for i in 0..32 { h[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()? }
    Some(h)
}

/// Load a 32-byte ed25519 seed (raw or 64 hex chars). The key file is never written by the server and never committed.
pub fn load_key(path: &str) -> std::io::Result<SigningKey> {
    let b = std::fs::read(path)?;
    let seed = if b.len() == 32 { b.try_into().unwrap() } else {
        unhex(std::str::from_utf8(&b).unwrap_or("").trim()).ok_or_else(|| std::io::Error::other("key must be 32 raw bytes or 64 hex chars"))? };
    Ok(SigningKey::from_bytes(&seed))
}

/// Ephemeral key from the OS RNG (receipts verify against the pubkey reported in each receipt, but it changes per run).
pub fn ephemeral_key() -> std::io::Result<SigningKey> {
    use std::io::Read; let mut s = [0u8; 32]; std::fs::File::open("/dev/urandom")?.read_exact(&mut s)?; Ok(SigningKey::from_bytes(&s))
}

impl Ledger {
    pub fn open(path: Option<PathBuf>, key: SigningKey) -> std::io::Result<Self> {
        let mut heads = HashMap::new();
        if let Some(p) = &path { if let Ok(t) = std::fs::read_to_string(p) { for l in t.lines() {
            let f: Vec<&str> = l.split('\t').collect();
            if let [th, seq, rh, _] = f[..] { if let (Ok(s), Some(h)) = (seq.parse::<u64>(), unhex(rh)) { heads.insert(th.to_string(), (s + 1, h)); } }
        } } }
        Ok(Ledger { path, heads, key })
    }
    pub fn pubkey_hex(&self) -> String { hex(&self.key.verifying_key().to_bytes()) }
    pub fn next_seq(&self, thread: &str) -> u64 { self.heads.get(thread).map_or(0, |h| h.0) }

    /// Build, sign, and persist the next receipt of `thread`. Thread ids must not contain tabs or newlines.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, thread: &str, input_id: &str, content_hash: Hash, tier: Tier, tokens_in: u32, tokens_out: u32,
                  compute: Energy, wait: Energy, dispatched_tool: Option<String>) -> std::io::Result<SignedReceipt> {
        if thread.contains(['\t', '\n']) || input_id.contains(['\t', '\n']) { return Err(std::io::Error::other("tab/newline in thread or input id")) }
        let (seq, prev) = match self.heads.get(thread) { Some(&(s, h)) => (s, Some(h)), None => (0, None) };
        let r = SliceReceipt { thread_id: thread.into(), slice_seq: seq, input_id: input_id.into(), content_hash, prev_receipt: prev,
            tier, tokens_in, tokens_out, compute, wait, dispatched_tool };
        let s = sign_receipt(r, &self.key); let rh = s.receipt.receipt_hash();
        if let Some(p) = &self.path {
            if let Some(d) = p.parent() { std::fs::create_dir_all(d)? }
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p)?;
            writeln!(f, "{thread}\t{seq}\t{}\t{}", hex(&rh), hex(&content_hash))?; f.sync_data()?;
        }
        self.heads.insert(thread.into(), (seq + 1, rh));
        Ok(s)
    }
}

fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() { match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) } }
    o.push('"'); o
}
fn num(x: f64) -> String { if x.is_finite() { format!("{x}") } else { "null".into() } }
fn prov(p: Provenance) -> &'static str { match p { Provenance::HwShunt => "HwShunt", Provenance::ModelBased => "ModelBased", Provenance::Estimator => "Estimator" } }
fn tier(t: Tier) -> (String, u16) { match t { Tier::L0Cache => ("l0_cache".into(), 0), Tier::Rules => ("rules".into(), 0), Tier::Small { batch } => ("small".into(), batch), Tier::Kev9b { batch } => ("kev9b".into(), batch) } }

/// JouleContract-style JSON. NaN joules (unmetered) render as null.
pub fn to_json(s: &SignedReceipt) -> String {
    let r = &s.receipt; let (tn, tb) = tier(r.tier);
    format!(concat!(r#"{{"v":1,"thread_id":{},"slice_seq":{},"input_id":{},"content_hash":"{}","prev_receipt":{},"receipt_hash":"{}","#,
        r#""tier":"{}","batch":{},"tokens_in":{},"tokens_out":{},"compute_j":{},"compute_provenance":"{}","wait_j":{},"wait_provenance":"{}","#,
        r#""dispatched_tool":{},"signer":"{}","sig":"{}"}}"#),
        js(&r.thread_id), r.slice_seq, js(&r.input_id), hex(&r.content_hash), r.prev_receipt.map_or("null".into(), |h| format!("\"{}\"", hex(&h))), hex(&r.receipt_hash()),
        tn, tb, r.tokens_in, r.tokens_out, num(r.compute.joules), prov(r.compute.provenance), num(r.wait.joules), prov(r.wait.provenance),
        r.dispatched_tool.as_deref().map_or("null".into(), js), hex(&s.signer), s.sig.iter().map(|b| format!("{b:02x}")).collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*; use crate::sign::verify_receipt;
    fn e(j: f64) -> Energy { Energy { joules: j, provenance: Provenance::ModelBased } }
    fn tmp(n: &str) -> PathBuf { let p = std::env::temp_dir().join(format!("receipt-ledger-{n}-{}.tsv", std::process::id())); let _ = std::fs::remove_file(&p); p }

    #[test] fn chain_and_persist_across_restart() {
        let p = tmp("restart"); let k = SigningKey::from_bytes(&[4; 32]);
        let mut l = Ledger::open(Some(p.clone()), k.clone()).unwrap();
        let a = l.record("t1", "m0", [1; 32], Tier::Small { batch: 8 }, 10, 2, e(1.5), e(0.0), None).unwrap();
        let b = l.record("t1", "m1", [2; 32], Tier::Small { batch: 8 }, 10, 2, e(1.5), e(0.0), None).unwrap();
        let c = l.record("t2", "m0", [3; 32], Tier::Rules, 0, 0, e(0.0), e(0.0), None).unwrap();
        assert_eq!((a.receipt.slice_seq, b.receipt.slice_seq, c.receipt.slice_seq), (0, 1, 0));
        assert_eq!(b.receipt.prev_receipt, Some(a.receipt.receipt_hash())); assert_eq!(c.receipt.prev_receipt, None);
        assert!(verify_receipt(&a) && verify_receipt(&b));
        let mut l2 = Ledger::open(Some(p.clone()), k).unwrap();
        assert_eq!(l2.next_seq("t1"), 2);
        let d = l2.record("t1", "m2", [4; 32], Tier::Kev9b { batch: 1 }, 1, 1, e(2.0), e(0.0), None).unwrap();
        assert_eq!(d.receipt.prev_receipt, Some(b.receipt.receipt_hash()));
        let _ = std::fs::remove_file(p);
    }
    #[test] fn rejects_tabs() { let mut l = Ledger::open(None, SigningKey::from_bytes(&[1; 32])).unwrap();
        assert!(l.record("a\tb", "m", [0; 32], Tier::Rules, 0, 0, e(0.0), e(0.0), None).is_err()); }
    #[test] fn json_shape() {
        let mut l = Ledger::open(None, SigningKey::from_bytes(&[1; 32])).unwrap();
        let s = l.record("t\"x", "m", [0; 32], Tier::Small { batch: 8 }, 3, 1, Energy { joules: f64::NAN, provenance: Provenance::Estimator }, e(0.0), None).unwrap();
        let j = to_json(&s);
        assert!(j.contains(r#""thread_id":"t\"x""#) && j.contains(r#""compute_j":null"#) && j.contains(r#""tier":"small","batch":8"#) && j.contains(r#""prev_receipt":null"#), "{j}");
        assert!(j.contains(&format!(r#""signer":"{}""#, l.pubkey_hex())));
    }
    #[test] fn key_file_hex_and_raw() {
        let p = tmp("key"); std::fs::write(&p, [7u8; 32]).unwrap(); let a = load_key(p.to_str().unwrap()).unwrap();
        std::fs::write(&p, "07".repeat(32) + "\n").unwrap(); let b = load_key(p.to_str().unwrap()).unwrap();
        assert_eq!(a.to_bytes(), b.to_bytes()); std::fs::write(&p, "nope").unwrap(); assert!(load_key(p.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(p);
    }
}
