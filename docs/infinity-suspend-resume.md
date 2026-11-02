# Infinity-style suspend/resume in front of `/v1/decisions`

Status: design note (Nov 2, 2026). Nothing here is implemented yet. Source: AWS/Berkeley Hydro "Infinity" (Apache-2.0, Rust),
https://aws.amazon.com/blogs/opensource/introducing-infinity-a-runtime-for-extreme-agent-concurrency/ (Oct 23, 2026).
Spec we build against: Transaction Science (transaction.science): JouleClaw, JouleContract, Mesh.
Metric: energy to run. Open, no moat.

## What Infinity does
- Agent = stackless coroutine. Tool call = `await`. The conversation history is the continuation.
- **Slice** = load history -> one model completion -> persist the turn -> dispatch at most one tool call -> end. A slice never blocks.
- Everything that resumes an agent (user text, tool results, timers, child reports) is a message on a per-thread input queue. In-order per `thread_id`, parallel across threads.
- At-least-once delivery: every input carries a stable ID; the state store records processed IDs and skips duplicates. A turn is persisted *before* its tool is dispatched.
- Async events are delivered as **synthetic tool calls** (`receive_event__injected{original_tool_call_id,...}`), threaded (child thread per event) or associative.
- Local mode (tokio tasks) and step mode (SQS FIFO + Lambda, nothing resident while waiting).
- Their numbers (~100 KB per idle agent, ~75k agents in 8 GB) come from a benchmark with a **simulated model**: they measure runtime overhead only, nothing about model-call cost or energy. They list event authenticity as an open question.

## Mapping onto strix-rt
### 1. Slice = metering unit
A slice is bounded and has one completion, which makes it the natural JouleContract unit. Per slice receipt:
- `thread_id`, `slice_seq`, `input_id`, cascade tier that answered (L0 cache / rules / small model batch / Kev-9B), model hash, tokens in/out, batch size and the slice's share of the batch.
- `compute_j`: SMU PPT differential over the slice's GPU window, apportioned per member when batched. Provenance tag `ModelBased` for apportioned shares; `HwShunt` only when a shunt is present (not today); `Estimator` for cache/rules hits.
- `wait_j`: energy attributed while suspended. With suspension the per-agent marginal is ~0 (history bytes at rest); the box idle floor is shared and reported separately, not charged per agent. A blocking harness would have to charge held process/KV memory for the whole wait; that is the comparison worth measuring.
- `prev_receipt_hash` to chain slices into a per-thread transaction.

### 2. Compute joules vs wait joules
Report both, never a blend. Idle floor comes from `tools/cascade/idle_loop.sh` (still collecting clean windows; no number quoted here). Wait accounting = (idle floor x wall time) / resident threads, plus a storage term. Mesh joule ceilings apply to `compute_j`; fidelity floor applies to the answering tier.

### 3. Idempotent input IDs vs content-hashed receipts
Two different keys, keep both:
- **Input ID** (Infinity): "have I processed this delivery?" Dedup of redelivery. Cheap, opaque.
- **Content hash** (JouleContract): hash of (history-prefix hash, input payload, model hash, tier, params). Answers "has anyone computed this exact slice?" That is a cache key (JouleClaw L0) as well as a receipt ID. Duplicate input ID -> drop. New input ID with known content hash -> return cached result, emit receipt with `compute_j`≈0, tag `Estimator`/cache. Deterministic decoding (greedy, fixed seeds) is required for the content hash to mean reuse.

### 4. Synthetic events and authenticity
Their open question maps directly onto JouleClaw provenance plus signed receipts:
- Every message entering an input queue carries a signed envelope: source identity, `original_tool_call_id`, payload hash, and (for tool results produced by our runtime) the receipt hash of the dispatching slice.
- The runtime only injects `receive_event__injected` if the envelope verifies against a key registered for that subscription at dispatch time (key recorded in the dispatching slice's receipt). Unverified events go to a quarantine thread, never into history as tool output.
- The provenance tag is rendered into the synthetic call args (`provenance: verified|unverified, signer: ...`) so the model can see it, but policy enforcement is in the runtime, not the model.
- Stateless-friendly: verification needs only the persisted receipt chain and a public key; no session state. Keys and receipt formats published openly (no registry moat).

### 5. Composition with the cascade
Slice -> `/v1/decisions` request. Cascade order unchanged:
1. L0 content-hash cache and rules (no GPU, no batch wait).
2. Small model batched at >=8 questions/request. Measured on this box: kev-0.8b 0.66 J/decision at 8/request vs 4.18 J at 1/request (docs/decision_models.md). Batching is the biggest lever, and suspension is what supplies it: many suspended threads produce many ready slices.
3. Kev-9B on low confidence (Kev-27B dropped as default).

Scheduler: a ready-slice queue per tier. Collect slices across *different* threads into a batch for a bounded window (deadline-driven, user-facing threads first: their "scheduling policies" open question). Never two slices of the same thread in one batch (preserves per-thread ordering). Pad to prerecorded batch-size ladder (B=1,2,4,8) from docs/flare-comparison.md. A slice escalated to Kev-9B is re-queued to the Kev-9B tier as a new sub-slice with its own receipt linked by hash.

On the 128 GB Strix Halo: histories live in RAM/disk store; KV is not kept for suspended threads (current KV is a fixed max_ctx buffer per model). Resume cost = prefill of the history; prefix caching of shared system prompt/decision schema is the next lever. Whether keeping KV warm for short waits beats re-prefill is an energy question to measure, not assume.

## Adopt
- Slice model, per-thread FIFO input queue, persist-before-dispatch, input-ID dedup.
- Non-blocking tools that post results to the queue.
- Synthetic tool calls for events, plus our signed-envelope gate.
- Local mode only (tokio, in-process) on the Framework.

## Skip (for now)
- Step mode / SQS / Lambda / Aurora DSQL / CDK: cloud plumbing, irrelevant to a single box.
- Threaded child-per-event by default: each child re-prefills parent context, which costs joules; prefer associative for decisions, threaded only when the context saving is measured to pay.
- Taking a dependency on the crate before reading it; the design is small enough to reimplement in kev_serve, and we can align wire formats instead.

## Next steps
1. ~~Read the Infinity repo source~~ done (see Source findings).
2. Define `SliceReceipt` (JouleContract-compatible JSON, ed25519-signed) in `crates/kev`; emit it from the existing `/v1/decisions` path even before suspension exists.
3. Add a cross-thread slice batcher in front of `/v1/decisions` (deadline window, ladder padding); measure J/decision vs arrival rate with `driver.sh` + gpu_lock, after current queued runs finish.
4. Measure resume cost: re-prefill vs warm KV over wait durations; publish the crossover.
5. Implement the signed event envelope and quarantine path; test with forged events.
6. Publish compute_j/wait_j split vs a blocking harness on the same workload (real models, not simulated).

## Source findings (github.com/hydro-project/infinity @ 1abc6fc, 2026-10-27)
Read: `crates/infinity-agent-core/src/{traits.rs,stores.rs,message.rs,event_processor.rs,system/builder.rs}`, `crates/infinity-agent-lambda/src/tools/sqs_sender.rs`.

- **Stores** (`traits.rs`): `ConversationStore` (ensure_root_thread, thread_exists, load_history_up_to, append_messages(thread, Vec<(InfinityMessage, message_id)>), compaction summaries indexed by `up_to_order`). `StateStore`: get/add_processed_message_ids, metadata, subscriptions, pending user choices; snapshot type `ThreadState { processed_message_ids: HashSet<String>, metadata, subscriptions: HashSet<ToolCallId>, pending_user_choices }`. `InputSender::send_to_input_queue(InputMessage, dedup_id)`.
- **Messages** (`message.rs`): `InputMessage { content, group_id: ThreadId, metadata: Option<serde_json::Value>, synthetic: Option<SyntheticKind>, display_as, ... }`. History items are `InfinityMessage` (User, Assistant, ToolCall, ToolResult, SubscriptionEvent { result, tool_call_id, child_thread_id, invocation }). `SyntheticKind::Tagged` = `subscription_event { tool_call_id, associative, final }` | `thread_report` | parent message.
- **Slice pipeline** (`event_processor.rs`, `step()` in `system/builder.rs`): inputs enter as *unvalidated* items, become *pending* (known-safe) once a completion shows the context did not overflow, then `sync()` writes them with `append_messages`. A tool call ends the turn: `flush_turn()` commits the buffered assistant content, and `sync()` persists it before the tool's `execute` (fire-and-forget) result can arrive. Streamed text chunks are coalesced into one row.
- **Dedup**: `handle_content` skips any `message_id` already in `processed_message_ids`. Only **user text and subscription events** get durable dedup IDs; **tool results are deduped against the history tail** (`match_tool_result` walks back: Unanswered / AlreadyAnswered / NoPendingCall). Completions are skipped by `completion_id`. On SQS the FIFO dedup id is `format!("{dedup_id}-{now}")`, so queue-level dedup is time-salted and the state store is the real guard.
- **Synthetic events**: for thread reports, associative and parent events the runtime mints `receive_event__injected` with a **fresh `uuid::v4` call id** and args `{original_tool_name, original_tool_call_id, original_args}`; `final` removes the subscription. If the model calls `receive_event__injected` itself, it gets an error tool result. The only authenticity check is that `original_tool_call_id` names an active subscription; `InputMessage.metadata` is free-form JSON and nothing verifies a sender.

### Where the source changes the design
1. Dedup is per-message-kind, not uniform. Our receipts keep `input_id` for user/event inputs and use the tool_call_id binding for tool results, mirroring theirs.
2. Synthetic call ids are random, so the content hash must bind events to `original_tool_call_id`, never to the injected id (done in `crates/receipt`). Otherwise L0 reuse never fires for replayed events.
3. Persistence happens at `sync()` after validation, not per chunk: the receipt should be written in the same `append_messages` batch as the turn (atomic with history), not after.
4. The signed envelope has a natural home: `InputMessage.metadata` (no wire change needed). Verify in `handle_content` before `append_unvalidated`; unverified events still get a receipt with `verified=false` in their content hash, and go to quarantine.
5. Their two-phase unvalidated -> known-safe split means a context-overflow retry re-runs the completion: count those joules in the receipt of the slice that finally commits (do not drop them).

## Prototype
`crates/receipt` (no GPU deps; sha2 only): `SliceReceipt`, `SliceKey`, `Tier`, `Provenance`, `InputKind`, `content_hash` (domain-separated, length-prefixed SHA-256; batch size excluded so batching does not change meaning), rolling `extend_prefix`, `receipt_hash` (chain), `cache_hit`, `apportion` (batch joules by token share, ModelBased). 9 unit tests. Signing (ed25519) and wiring into kev_serve are next.
