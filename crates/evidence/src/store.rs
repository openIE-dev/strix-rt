//! External source bytes. Reads are counted. A hash mismatch is not treated as success.
//! Copyright 2026 Open Interface Engineering, Inc.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::events::digest;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct SourceStore {
    bodies: BTreeMap<(String, u32), Vec<u8>>,
    pub calls: u64,
    pub bytes_read: u64,
    pub retries: u64,
    pub retry_ns: u64,
    pub verify_ns: u64,
}

#[derive(Debug)]
pub enum ReadOut {
    Verified(String),
    Missing,
    Corrupt,
}

impl Default for SourceStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceStore {
    pub fn new() -> Self {
        Self {
            bodies: BTreeMap::new(),
            calls: 0,
            bytes_read: 0,
            retries: 0,
            retry_ns: 0,
            verify_ns: 0,
        }
    }

    pub fn insert(&mut self, id: &str, rev: u32, body: String) {
        self.bodies.insert((id.to_string(), rev), body.into_bytes());
    }

    pub fn remove(&mut self, id: &str, rev: u32) {
        self.bodies.remove(&(id.to_string(), rev));
    }

    pub fn corrupt(&mut self, id: &str, rev: u32) {
        if let Some(b) = self.bodies.get_mut(&(id.to_string(), rev)) {
            if let Some(last) = b.last_mut() {
                *last ^= 0x5a;
            }
        }
    }

    pub fn begin_case(&mut self) {
        self.calls = 0;
        self.bytes_read = 0;
        self.retries = 0;
        self.retry_ns = 0;
        self.verify_ns = 0;
    }

    pub fn read(&mut self, id: &str, rev: u32, expect: &[u8; 32]) -> ReadOut {
        self.calls += 1;
        let Some(bytes) = self.bodies.get(&(id.to_string(), rev)).cloned() else {
            return ReadOut::Missing;
        };
        self.bytes_read += bytes.len() as u64;
        if !self.matches(expect, &bytes) {
            let t = Instant::now();
            self.retries += 1;
            self.calls += 1;
            self.bytes_read += bytes.len() as u64;
            let still_bad = !self.matches(expect, &bytes);
            self.retry_ns += t.elapsed().as_nanos() as u64;
            if still_bad {
                return ReadOut::Corrupt;
            }
        }
        ReadOut::Verified(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn matches(&mut self, expect: &[u8; 32], bytes: &[u8]) -> bool {
        let t = Instant::now();
        let ok = digest(bytes) == *expect;
        self.verify_ns += t.elapsed().as_nanos() as u64;
        ok
    }
}
