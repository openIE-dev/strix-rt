# Action authorization

Copyright 2026 Open Interface Engineering, Inc.
SPDX-License-Identifier: MIT OR Apache-2.0

Status: library gate on remote `main` at `34ad7dd` (`crates/receipt::auth`). It is not
wired into `kev_serve`, it does not execute tools, and it is not a claim that this
runtime is safe to run as an autonomous agent. Nothing here was taken from an
unpublished local tree.

## Threat model

On a Framework box, LFM3-Omni Q4_K_M was asked for a destructive root deletion and
answered with `Commit` plus `rm -r /` or `rm -rf /` on every recorded try (0/60
refusals). Those strings were not executed. The raw transcripts are not in this
repository. The fixtures in the tests and in `crates/receipt/examples/auth_check.rs`
use the reported shapes.

The failure mode we are closing is a caller treating one of these as permission:

- the model writing `Commit`, a choice value `commit`, or a probability near 1
- `sign::Gate` returning `Verdict::Accept` (the envelope is authentic)
- `POST /v1/decisions` returning predicate, choice, or score probabilities
- a free-form `SliceReceipt.dispatched_tool` string

The adversary is model output and anything else that can place bytes in the proposal:
malformed documents, an extra line, a substituted hash, a different thread id, or a
signed copy of a destructive payload. The operator grant list is trusted in-process
state. A caller who skips `authorize`, or who can rewrite the grant list, is outside
this gate. Kernel, GPU, weight supply, and network exposure of the experimental
servers are outside it too (`SECURITY.md`).

## Policy

`authorize(grants, proposal, energy)` is fail-closed.

- The proposal must be one canonical typed action. Anything else is `unstructured`,
  `unknown_action`, or `malformed`.
- The closed set is `inspect_decision` and `quote_receipt`. Both name objects this
  runtime already computed (a decision content hash, or a thread id plus a receipt
  hash). There is no shell, process, filesystem, or network action to grant.
- Default deny for an empty grant list (`no_authority`), a different kind
  (`action_mismatch`), and the same kind with a different scope (`scope_mismatch`).
- The model does not mint grants. Prefixing a valid document with `Commit` makes it
  unstructured. The verb `commit` is `unknown_action`. There is no commit parameter.
- This is not a command blacklist. `hello`, `echo hello`, and `Commit\nrm -rf /`
  all return `unstructured` because they are not the grammar below. The production
  parser does not search for a forbidden-command table.

Grammar, exactly, including the final newline, maximum 256 bytes:

```text
strix-rt/action/v1
inspect_decision
<64 lowercase hex content hash>

strix-rt/action/v1
quote_receipt
<thread id>
<64 lowercase hex receipt hash>
```

A thread id is 1 to 64 bytes, `[A-Za-z0-9][A-Za-z0-9_-]*`. Uppercase hex, an extra
field, a missing field, CR, NUL, or a second document is `malformed`.

## Binding

An allow carries a `Permit` whose binding is SHA-256 over the domain
`strix-rt/action-binding/v1` and the length-prefixed canonical bytes (the same
length-prefix habit as the other JouleContract hashes in this crate). The action's
parameters are the scope: there is no extra payload field that can be swapped after
the decision.

`Permit::covers` is true only for that action. Changing one byte of a granted
proposal denies. A `Permit` is not a bearer token. Other crates cannot construct
one, and nothing in `receipt` performs I/O because a permit exists. Calling
`covers` is not a second authorization and does not re-check energy.

Grants are tried as a set of exact matches. The strictest `compute_ceiling_j` among
exact matches applies. A grant with no ceiling does not cancel a grant that has one.
The reported `capability_id` is the grant that supplied that strictest ceiling, or
the first exact match when no ceiling is set.

## Energy

JouleContract types already in this crate are `Energy { joules, provenance }` with
`HwShunt`, `ModelBased`, and `Estimator`. `ledger::to_json` renders non-finite
joules as JSON `null` (unmetered). The design note says Mesh ceilings apply to
`compute_j`. This baseline has no stored ceiling, no budget account, and no default
joule number.

So:

- A grant with `compute_ceiling_j: None` does not check energy and does not invent a
  budget. Missing, NaN, or `Estimator` energy can still accompany an allow. That is
  "no ceiling was asserted", not "the action is free in joules".
- When a matching ceiling is set, the caller must pass finite joules `>= 0` and
  `<=` the strictest ceiling, with provenance `HwShunt` or `ModelBased`.
- `Estimator` is `energy_unmeasured`, including a cache-hit `0.0` tagged `Estimator`.
  Those receipts stay honest; they just cannot satisfy a ceiling.
- Non-finite or negative joules are `energy_invalid`. Absence is `energy_missing`.
  Above the ceiling is `over_budget`. The bound is inclusive.
- `auth` does not call `meter`, does not read sysfs, and does not fill in joules.
  `KEV_POWER` is untouched. If the caller has no measurement, a ceiling denies.

## What stays separate

| Mechanism | What it decides | What it does not decide |
|---|---|---|
| `sign::Gate` | Envelope signature, subscription, payload hash, replay | Whether the payload is an allowed action |
| `POST /v1/decisions` | Predicate, choice, and score probabilities, plus a signed slice receipt | Action authorization. `dispatched_tool` is still an unstructured `Option<String>` and `kev_serve` passes `None` |
| `receipt::auth` | Typed capability match, exact scope, optional ceiling | Authenticity of an event envelope. It does not execute the action |

A destructive payload can verify under `sign::Gate` and still be `unstructured` here.
A granted `inspect_decision` can be allowed here and still be quarantined as unsigned
if it is presented to `sign::Gate` with no envelope. Both outcomes are tested.

## Integration status

Verified on this workspace before the change, remote `main` `34ad7dd`:

- `crates/receipt/src/sign.rs`: `Gate::check` returns `Accept` or `Quarantine`.
- `crates/kev/src/bin/kev_serve.rs`: `/v1/decisions` returns model probabilities and a
  receipt. It does not authorize or run a tool.
- `crates/receipt` energy code meters a caller-supplied sysfs path or records NaN.
  It does not enforce a ceiling.
- No shell dispatch of model output exists. This change does not add one.

Not found here: David's unpublished local working tree. This gate does not pretend to
line up with modules that are not on the remote. If that tree already has a
different capability type, this patch is against `34ad7dd` only (`git diff 34ad7dd`).

`kev_serve` behavior is unchanged. The crate comment now points here so a probability
is not mistaken for an allow. Infinity's `handle_content` path is still a design
note (`docs/infinity-suspend-resume.md`); this function is not hooked into it.

## Limitations

- Not general autonomous safety, not a sandbox, not an OS policy, not a deployment.
- The only typable allows are references to an existing decision hash or receipt.
  Adding an execution verb would be a code change to the closed set, which this
  gate is here to make obvious.
- In-process grants are not signed. The signature gate does not vouch for them.
- `dispatched_tool` on a receipt is not a capability and is not read by `authorize`.
- Cache-hit `Estimator` energy cannot pass a ceiling. There is no measured substitute
  in this crate, and we do not invent one.
- No ceiling on the grant means no energy refusal.
- Strict canonical bytes only. A semantically identical JSON object is unstructured.
- The LFM result cited above is a reported shape, not a replay of the 60 logs.
- Experimental HTTP servers remain as `SECURITY.md` describes them.

## Non-executing example

`crates/receipt/examples/auth_check.rs` builds one inspect grant and one quote grant,
then prints decisions. The quote grant's ceiling is `0.5` J, supplied by the example,
not by a meter. Expected shape:

```text
allow capability_id=op.inspect binding=<64 hex>
allow capability_id=op.quote binding=<64 hex>
deny no_authority
deny unstructured
deny unstructured
deny unstructured
deny unstructured
deny unstructured
deny unknown_action
deny scope_mismatch
deny over_budget
deny energy_unmeasured
auth_check: all expected decisions matched
```

The two `Commit` / `rm` lines are the third and fourth `deny unstructured` rows.
`echo hello` and `hello` are the same reason. Run:

```sh
cargo run -p receipt --example auth_check
cargo test -p receipt
```

Neither command starts `kev_serve` or a shell.
