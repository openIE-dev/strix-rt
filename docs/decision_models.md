# Decision ("System One") models on strix-rt — 2026-10-14

All four names are from the Sept–Oct 2026 "System One" decision-model wave: a model reads a state (text/JSON) plus typed
questions (choice / noul yes-no / score) and returns calibrated probabilities over the options in one forward pass, no text
generation. They are not JEPA / world models.

| name | what | weights | status here |
|---|---|---|---|
| Jev (TypeSafe) | the original hosted System One model, `jev-1.13.0`, `POST api.typesafe.ai/v1/systemone`, released 2026-09-15 | closed, size unpublished | no port possible; Kev and autotrust/JEV are open re-implementations of its interface |
| Kev (Jared Palmer) | open re-implementation of Jev: Qwen3.5 / Qwen3.8 backbone + pointer head, TypeSafe API-compatible; 0.8B/4B/9B (LoRA) + 27B (full bf16) | Apache-2.0, HF `jaredpalmer/kev-*` | **Kev-0.8B ported** (crates/kev), parity below |
| Laya (Convai Innovations) | ModernBERT-large encoder (395M) + 2-layer post-norm-first transformer head + [MASK]-marker scorer + act/escalate head, 421M | Apache-2.0, HF `convaiinnovations/laya` (+ multilingual mmBERT 322M, typed-decisions) | weights in $MODELS/decision/laya, PyTorch golden in crates/laya/ref; Vulkan port = design below |
| ThinkThen (Ian Maurer) | not a model: Rust engine + CLI + 24 bindings wrapping System One backends in 10 functions (decide, choose, tag, score, filter, rank, find, recognize, relate, annotate); default backend Jev | open source (thinkthen.dev), alpha 0.1 | integration target: expose a System One endpoint from strix-rt so ThinkThen can point at it |

## Kev on strix-rt (done)
- Backbone: `ref/merge_lora.py` folds the r=16/alpha=32 LoRA into Qwen3.5-0.8B-Base in fp32, `convert_hf_to_gguf.py --outtype bf16`
  -> `$MODELS/decision/kev-0.8b-bf16.gguf`; runs on the existing `prefill::qwen35` batched path (one chunk per question row).
- Head: final RMSNorm (output_norm) on CPU over the rows <decide> and each </opt>, then `(W_k h_opt + b_k)·(W_q h_dec + b_q)/16/T`, T = 2.351.
- Parity vs upstream `kev` PyTorch fp32 CPU (4 questions, 2–5 options): argmax 4/4, worst max|Δprob| 0.0051, worst hidden cosine 0.999996.
- Next: shared state prefix (run the state once, branch rows from the DeltaNet/KV snapshot, like kev.serve), GPU head, System One HTTP
  server; Kev-4B/9B same recipe; Kev-27B = Qwen3.8-27B full fine-tune -> quantize to the Q4/Q8 formats the qwen38 path already runs.

## Laya port design (next)
- Encoder: ModernBERT-large — 28 layers, d=1024, 16 heads, GeGLU MLP (2624), pre-LayerNorm without bias, RoPE (global θ=160k every
  3rd layer, local θ=10k sliding window 128 elsewhere), bidirectional attention, final norm. New kernels: bidirectional + sliding-window
  attention mask, LayerNorm, GeGLU; GEMMs reuse prefill's coop-matrix bf16 path.
- Head: + type_emb[qtype]; 2× torch `TransformerEncoderLayer(norm_first, d=1024, 16 heads, ffn 4096, GELU)` with padding mask;
  gather at [MASK] marker positions -> LayerNorm -> Linear(1024,1024) -> GELU -> Linear(1024,1) -> softmax over a question's options /
  per-type temperature (choice by option count, score, noul); act head on [CLS] + 4 confidence features.
- Input: one row per question: `[CLS] <type> instr [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`, ≤512 tokens.
- Parity: golden.json holds input_ids, marker_pos, qtype, encoder CLS/mean, logits, act logits for 2 states × 3 questions.

## Round 2 status (2026-10-14 evening)
Parity = Rust/Vulkan vs upstream PyTorch on CPU (Kev: kev package, fp32; Kev-27B: bf16, 108 GB fp32 does not fit) with the same token rows.
Latency = idle GPU (logs/gpu_idle_run.sh, no other GPU job), shared state prefix + GPU pointer head, BENCH_STATE=2048, 8 questions of 23 tokens.

| model | weights on strix-rt | argmax | max abs dprob | min hidden cos | 2048-tok state | per question |
|---|---|---|---|---|---|---|
| Kev-0.8B | bf16 GGUF 1.5 GB | 4/4 | 0.0051 | 0.999996 | 358 ms | 26 ms |
| Kev-4B | bf16 GGUF 8.3 GB | 4/4 | 0.0006 | 0.999998 | 1.48 s | 85 ms |
| Kev-9B | Q8_0 GGUF 9.3 GB | 4/4 | 0.0125 | 0.999907 | 1.53 s | 73 ms |
| Kev-9B | bf16 GGUF 17.5 GB | 4/4 | 0.0009 | 0.999997 | 13-16 s (slow path, see below) | 0.6-0.8 s |
| Kev-27B | Q8_0 GGUF 27.2 GB (from the full bf16 fine-tune) | 4/4 | 0.0043 | 0.999904 | 4.6 s | 0.26-0.43 s |

Laya (laya crate, F16 weights, one row per question, raw scorer logits before temperature): argmax 9/9, max abs dprob 0.0006,
including ~480-token rows that exercise the +-64 sliding window; ~15 ms GPU per 50-token row, 52 ms at L=512.

Serving: `kev_serve <gguf> <head.bin> [addr] [model]` speaks TypeSafe System One (POST /v1/systemone). ThinkThen 0.1.2
`thinkthen check --url http://127.0.0.1:8009/v1 --model kev-0.8b`: 8 ok, 0 critical, 0 warning; decide/choose work.

Open items: bf16 9B is ~10x slower than its Q8_0 (bf16 4B and Q8_0 27B are fine), root cause not found yet; branches run one at a time
(batching question rows would amortize the weight stream, the main per-question cost); Laya not yet behind kev_serve.

## Round 3 (2026-10-14)
- One-pass batched branches (Kev): every question of a document is in one forward over the shared state snapshot (kb_* shaders). Logits are bit-identical to the one-at-a-time path (max|dlogit| 0). 8 questions after a 2048-token state: 0.8B 210 -> 57 ms; 9B Q8_0 ~710 -> ~290 ms; 9B bf16 4.65 s -> 1.31 s. 27B runs overlapped other GPU jobs and need an idle rerun.
- Tokenizer: generated Unicode L / L|M / N tables (ref/gen_ucat.py, Unicode 16) plus NFC. Kev's tokenizer.json uses \p{L}+, the Qwen3.5 base uses [\p{L}\p{M}]+ (KEV_TOK=qwen35). Both match HF on 20/20 mixed-script strings.
- Laya on kev_serve (LAYA=laya-f16.gguf): HF-json BPE tokenizer (27/27 vs HF), build_head/build_sequence port, temperatures. laya_check vs Agent.system_one on 3 states: identical usage, max |d| 0.0022.
- Laya 0.96 cosine: a single token (row 2, token 37, id 13), at massive-activation dim 379 (-19.7 vs -17.2). Every other token is >= 0.9996 and the head output is >= 0.9998: an f16-sensitive outlier token, not a layout bug.
- bf16 9B: GPU 100% busy and GTT steady (no thrash), so the cost is in compute: the bf16 GEMM is slow at 9B shapes (d 4096 / ffn 12288). 4B bf16 is fine. Next step: profile the per-GEMM tile config.

## Round 4 (2026-10-14)
- bf16 9B fix: BF16 rows of 8 KiB (K=4096) put all 64 lanes of a wave (one row each) on the same memory channel (pf_gemm: 4096x4096 7.4 TFLOPS, 8192x4096 6.6, vs 4B bf16 16-17). rtcore now stores such blk.* matrices with rows padded by 256 B (private qtype 13, RT_PAD16=1, on by default in kev only). Decode GEMVs don't know 13. 9B bf16 2048-token document: 15.5 s -> 1.63 s; 8-question pass 1.76 s -> 0.21 s. Parity unchanged (4/4, 0.0009).
- Laya: the residual stream x is already f32 (GEMMs accumulate into f32 x; LayerNorm reads f32). The outlier token comes from f16 GEMM inputs, so there is no cheap fp32 change to make.
- Live ThinkThen check against Laya-only kev_serve: 8 ok, 0 critical, 0 warnings.

## Round 5: energy per decision (2026-10-17, in progress)
Harness: `dm_energy` bin + `tools/comp/dm_energy.sh` (energy recorder, SMU socket + gfx, differential (J(NHI)-J(NLO))/(NHI-NLO) per iteration; NHI sized for a >=40 s window),
sessions via `logs/dm_session.sh` under `logs/gpu_lock.sh` + `perf_high.sh`, 3 repeats. Raw: logs/dm/summary.log, logs/dm/energy/*.json.
Iteration = "state" (set_state of a 2048-token random state) or "full" (state + 8 questions of 23 tokens, one pass); Laya iteration = 1 request (3 questions, golden states).
Median of 3 (J/iteration, socket / gfx):
| model | state only | state + 8 questions | per question (full-state)/8 |
|---|---|---|---|
| Kev-0.8B bf16 | 34.0 / 9.2 | 42.6 / 11.2 | ~1.1 socket |
| Kev-4B bf16 | 187 / 18.3 | 184.6 / 43.0 | not resolvable (noise > difference) |
| Kev-9B Q8_0 | 257 / 27.1 | 292 / 30.8 (2 runs) | ~4 socket |
| Kev-9B bf16 | pending | 314 / 31.4 | pending |
| Kev-27B Q8_0 | pending | pending | pending |
| Laya F16 | - | 16.0 / 1.9 per 3-question request (~5.3 J/question, includes tokenization + CPU work) | - |
Caveats: the box load average was 19-30 (other workers compiling), so socket J includes unrelated CPU power; run-to-run spread is large (Kev-0.8B full 39-58 J, Kev-4B 172-242 J)
and gfx J/iter for Kev-4B full varied 19-47. Treat socket numbers as upper bounds; the state prefix dominates; per-question cost is small relative to noise.
No llama.cpp / PyTorch energy reference exists for these models (no equivalent System One runtime), so there is no competitor row; docs/comparisons.md has no decision-model section yet.
Jev is closed/hosted: no port or measurement possible. ThinkThen is a client; works against kev_serve (see round 2/4).

### Round 5 final numbers (all runs complete; median of 3, J/iteration socket / gfx)
| model | state only (2048 tok) | state + 8 questions | per question |
|---|---|---|---|
| Kev-9B bf16 | 276 / 26.3 | 314 / 31.4 | ~4.7 socket, ~0.6 gfx |
| Kev-27B Q8_0 | 688 / 90.1 | 788 / 104.1 | ~12.5 socket, ~1.7 gfx (spread across 3 runs only ~3%, the cleanest set) |
The 27B runs (wall-clock 40+ s each) were tight; smaller models were noisier because other workers loaded the CPU. Not measured: Kev-0.8B/4B/9B/27B parity re-run this round (last parity: round 2/4 above, unchanged code except GPU kernels since; a re-run is still owed).
