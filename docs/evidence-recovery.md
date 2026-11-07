# Delayed-query evidence conditions

Copyright 2026 Open Interface Engineering, Inc.
SPDX-License-Identifier: MIT OR Apache-2.0

This is a small synthetic harness (`crates/evidence`). It is not a production
integration and it does not sit on `/v1/decisions`.

Three conditions see the same seeded event list:

- `full_history` keeps every rendered source.
- `summary_only` keeps the latest salient slot per entity and drops detail slots.
- `compact_recoverable` keeps salient previews plus a content hash for each source,
  and may read the source back if the hash matches.

They are experimental conditions. The report sets `"ranking": null` and
`"efficiency_claim": false`. A diagnostic list says what the synthetic world
supports. Matching or missing that list is an observation about this plumbing,
not a standard of intelligence and not a winner.

Compaction functions take events only. The diagnostic is applied after answers
exist. A bounded ranker chooses what to read from previews it actually holds.
It does not receive the diagnostic's preferred source. The bound can stop a
read early.

## What a run records

For each condition and each query: retained ids, dropped slots, recovered
reads, state transitions, the answer or refusal, bytes retained, bytes read,
retrieval calls, retries, and per-phase wall time. Joules are `null` unless a
phase probe sampled a real meter.

## Telemetry boundary

`UnavailableProbe` is the cloud default. Wall time is filled in. Joules are
not. `SysfsProbe` is the optional hook for a later Framework run: it wraps
`receipt::meter` around one phase, reading a sysfs file of microwatts
(`power1_input` style, or whatever path you pass) and integrating with the
trapezoid already in that crate. Provenance is `ModelBased` when a sample
exists, because that is what `receipt::meter` reports for SMU/hwmon telemetry.
There is no shunt in that path. If the file cannot be read, joules stay absent.

A later instrumented run should sample compaction, retrieval, retries, and
verification separately, keep coverage of recovered sources next to any energy
figure, and report joules per verified decision only from those samples. Byte
counts and wall time are not joules. This harness does not claim an efficiency
gain.

Pass `--power <sysfs file>` to `evidence_run` only when that file is a real
sensor. The default command does not.

## Authorization

Source notes may contain `Commit` and an `rm` string. Those bytes are data.
`receipt::auth` still refuses them when no operator grant exists, and the
discovery report records that refusal. The harness does not spawn a process.

## Later model-backed runner

Not in this phase. A later runner can take the same visible preview a condition
already holds and return a commit or a refusal. It must not receive the
diagnostic, and it must not be wired to credentials here. No model is
downloaded and no API is called.

## Commands

```sh
cargo test -p evidence --locked
cargo clippy -p evidence --all-targets --locked -- -D warnings
cargo run -p evidence --locked --bin evidence_run -- --seed 7 --out evidence-run.json
```

## Limits

The extractor is a line parser, not a reader that understands prose. Salient
versus detail is a field on the event, not a judgment the conditions learn.
Last-write in `summary_only` will commit through a contradiction; that is the
condition, not a defect to be tuned until it wins. Hash failure and a missing
source are refusals for the pointer condition even when another condition still
holds the original text. No service is started. No GPU or NPU work runs.
