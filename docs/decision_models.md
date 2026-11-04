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

## Round 6: parity re-run after kernel changes (2026-10-17)
Same goldens and scripts (`logs/dm_parity.sh`, logs in logs/dm/parity/), GPU lock, fresh `cargo build --release -p kev -p laya`. No regression, no code change.
| model | argmax (was) | argmax (now) | worst max abs dprob (was -> now) | min hidden cos (now) |
|---|---|---|---|---|
| Kev-0.8B bf16 | 4/4 | 4/4 | 0.0051 -> 0.0052 | 0.999996 |
| Kev-4B bf16 | 4/4 | 4/4 | 0.0006 -> 0.0009 | 0.999998 |
| Kev-9B Q8_0 | 4/4 | 4/4 | 0.0125 -> 0.0126 | 0.999907 |
| Kev-9B bf16 | 4/4 | 4/4 | 0.0009 -> 0.0007 | 0.999997 |
| Kev-27B Q8_0 | 4/4 | 4/4 | 0.0043 -> 0.0043 | 0.999903 |
| Laya F16 (laya_parity, 9 rows) | 9/9 | 9/9 | 0.0006 -> 0.0006 | 0.961 (same single outlier token as before) |
| Laya via laya_check vs Agent.system_one (3 states) | max|d| 0.0022 | max|d| 0.0022, no mismatches | | |
The 4B figure moved 0.0006 -> 0.0009 (0.0009 is also what the older round-4 bf16-9B note lists); all changes are at the 1e-4 level and far below argmax-relevant size.

## Round 7 — cascade/router harness, measured energy (Oct 2026)

Harness: `tools/cascade/` (eval.jsonl = 1000 items: ag_news, sst5, mnli, boolq; dev = first 50/dataset for temperature refit, test = 800). Tier 1 = System One (kev_serve), tier 2 = Qwen3.8-27B Q4_K_M zero-shot (first-token letter logprobs). Energy = SMU socket differential (two pass counts, same 100 items), quiet gate + gpu_lock, 3 repeats, median; windows with >1.5 avg busy cores dropped. Full output: `logs/cascade/analysis.md`.

| model | test acc | ECE (refit) | J/decision gross (net) | wall s/dec |
|---|---|---|---|---|
| Laya | 0.750 | 0.037 | 1.92 (1.66) | – |
| Kev-0.8B | 0.743 | 0.039 | 4.90 (3.61) | ~0.14 |
| Kev-4B | 0.804 | 0.020 | 18.90 (14.76) | ~0.45 |
| Kev-9B Q8_0 | 0.828 | 0.038 | 26.39 (20.34) | – |
| Kev-27B (tier-2 alt) | 0.833 | 0.029 | 80.43 (66.39); all 4 datasets, 3 reps run but only 1 clean rep per dataset (3 windows dropped, 11-16 busy cores) | ~1.4 |
| Qwen3.8-27B (tier 2) | 0.782 | 0.114 | 65.28 (59.94) | ~0.5–0.8 |

Cascade (gross J, test n=800; always-large = Qwen alone 0.782 acc, 65.3 J/dec, 83.4 J/correct):

| tier1 → Qwen | thr | escalated | acc | J/dec | J/correct vs large | break-even r* |
|---|---|---|---|---|---|---|
| Laya | 0.9 | 0.713 | 0.790 | 48.5 | −26.5% | 0.971 |
| Kev-0.8B | 0.8 | 0.593 | 0.792 | 41.8 | −36.8% | 0.925 |
| Kev-4B | 0.8 | 0.417 | 0.809 | 43.9 | −35.0% | 0.711 |
| Kev-9B Q8_0 | 0.8 | 0.380 | 0.815 | 46.5 | −31.6% | 0.624 |
| Kev-9B Q8_0 | 0 (never escalate) | 0 | 0.828 | 24.5 | −64.5% | – |
| oracle Laya→Qwen | – | 0.250 | 0.896 | 17.4 | −76.7% | – |

Findings: (1) On this in-domain eval the zero-shot 27B LLM is a weaker tier 2 than the fine-tuned System One models, so escalating to it lowers accuracy for Kev-4B/9B; always-small Kev-9B dominates every cascade on both accuracy and J. (2) Cascades beat always-large in J/correct up to thr≈0.9–0.97 (break-even escalation 0.62–0.97). (3) The oracle gap (0.89 acc at ~17–35 J) shows routing quality, not tier cost, is the bottleneck — max-prob confidence is a weak router. (4) Laya is the cheapest tier 1 (1.9 J/dec, ~34x below Qwen). Kev-27B as tier 2 (Nov 2): 0.833 acc at 80.43 J/dec gross, so Kev-9B alone (0.828, 24.51) costs 3.3x less for -0.005 acc.
Caveats: eval in-domain for Kev; per-decision cost includes per-request state prefill (one question per request); gross includes idle; Kev-9B and Laya maxed 5.6/4.4 busy cores in some windows (kept, below drop rule only partially — see analysis.md); cascade energy is predicted from per-tier costs (live val windows not run); router overhead assumed 0.

## Round: Base Labs, /v1/decisions floor, router probe (Oct 30-Nov 2)
Script: `tools/baselabs/bl_analyze.py`. Socket differential, quiet gate + gpu_lock, median of 3.
| Model / setup | J/dec gross | J/dec net | ms/dec | parity |
|---|---|---|---|---|
| joey-1 bf16 (block-bidirectional) | 9.33 [9.11-9.75] | 7.82 | 154 | 100/100, max dlogit 0.0016 |
| elephant-1 bf16 | 42.26 [41.29-61.13; r1 at 102 W, likely contaminated] | 36.20 | 672 | 16/16, max dlogit 0.0018 |
| kev-0.8b /v1/decisions, 8 questions/request | 0.66 | 0.52 | 16 | - |
| kev-0.8b /v1/decisions, 1 question/request | 4.18 | 2.73 | 115 | - |
Hidden-state probe vs confidence (AUROC predicting tier-1 wrong; confidence / probe dev / probe oof): kev08 0.801/0.563/0.742, kev4b 0.817/0.785/0.739, kev9q8 0.810/0.635/0.758. Confidence routing is as good as the probe.
Idle power (server loaded vs none, quiet box, 3 windows each, Nov 3 2026): socket median 7.88 W with no server (7.61/9.40/7.88) and 7.53 W with kev_serve loaded and idle (7.41/7.53/8.24). An idle kev_serve adds ~0 W, so gross and net J/decision differ only by the box floor. One contaminated Nov 2 window (22.7 W) is kept in logs/cascade/floor/contaminated_1102. Logs: logs/cascade/floor/idle_*.out, idle_clean.txt.

## Round: open-weight rival decision models (Nov 2-3, 2026; done)

Same harness as Round 7 (tools/cascade, 1000 items = 250 each of ag_news/sst5/mnli/boolq; test = 800, dev = first 50/ds for the temperature refit). Rivals are served by upstream llama-server `/v1/systemone` on Vulkan (~/llama-latest e36df6047), not kev_serve.

Accuracy (all 1000 items; analyze.py test-split numbers in parentheses):

| model | file | acc all | ag_news | sst5 | mnli | boolq | ECE refit (test) |
|---|---|---|---|---|---|---|---|
| vLLM-SR d3-edge | ggml-org/vllm-d3-edge-GGUF BF16 | 0.690 (0.688) | 0.840 | 0.400 | 0.804 | 0.716 | 0.042 |
| vLLM-SR d3-lite | ggml-org/vllm-d3-lite-GGUF BF16 | 0.742 (0.738) | 0.852 | 0.452 | 0.840 | 0.824 | 0.028 |
| vLLM-SR d3-flash | ggml-org/vllm-d3-flash-GGUF Q8_0 | 0.819 (0.819) | 0.884 | 0.564 | 0.892 | 0.936 | 0.024 |
| Kev-0.8B (ref) | | 0.749 (0.743) | 0.904 | 0.512 | 0.780 | 0.800 | 0.039 |
| Kev-9B Q8 (ref) | | 0.822 (0.828) | 0.920 | 0.588 | 0.876 | 0.904 | 0.038 |

Energy finished Nov 3 07:54 UTC (rivals_chain3 done). Method: e1 = batch 1, 3 reps, differential socket J (hi pass minus lo pass), windows with > 1.5 foreign busy cores dropped; b8 = 8 questions per request (K=8) vs 1 (K=1), 2 reps, 320-decision difference. Full tables: logs/cascade/analysis.md; ML.ENERGY export: logs/mlenergy/.

Test split (800 items), tools/cascade/analyze.py, gross socket J/decision (mean over the 4 datasets; net = idle baseline removed):

| model | acc (test) | ag_news | sst5 | mnli | boolq | ECE refit | J/dec gross | J/dec net | gfx J (ag_news) | clean reps |
|---|---|---|---|---|---|---|---|---|---|---|
| vLLM-SR d3-edge BF16 | 0.688 | 0.840 | 0.405 | 0.815 | 0.690 | 0.042 | 3.14 | 2.72 | 1.31 | 3 |
| vLLM-SR d3-lite BF16 | 0.738 | 0.845 | 0.450 | 0.850 | 0.805 | 0.028 | 6.29 | 5.45 | 2.73 | 3 |
| vLLM-SR d3-flash Q8_0 | 0.819 | 0.880 | 0.565 | 0.895 | 0.935 | 0.024 | 19.82 | 18.05 | 7.38 | 2 (2 dropped) |
| pplx-decider-flash v1.2 Q8_0 (reconverted) | 0.802 | 0.850 | 0.555 | 0.900 | 0.905 | 0.039 | 10.64 | 9.64 | 4.07 | 2 (1 dropped) |
| pplx-decider v1.2 27B Q8_0 (converted) | 0.821 | 0.870 | 0.575 | 0.910 | 0.930 | 0.023 | 62.60 | 57.56 | 25.16 | 2 (1 dropped) |
| Drex Max v1 Q8_0 (nace-ai fork) | 0.839 | 0.935 | 0.595 | 0.895 | 0.930 | 0.027 | 64.72 | 59.18 | 25.61 | 2 (2 dropped) |
| Laya F16 (ref) | 0.750 | 0.915 | 0.495 | 0.840 | 0.750 | 0.037 | 1.92 | 1.65 | 0.82 | 2 |
| Kev-0.8B (ref) | 0.743 | 0.895 | 0.505 | 0.785 | 0.785 | 0.039 | 4.90 | 3.61 | 2.13 | 3 |
| Kev-4B (ref) | 0.804 | 0.910 | 0.560 | 0.845 | 0.900 | 0.020 | 18.90 | 14.76 | 6.34 | 3 |
| Kev-9B Q8_0 (ref) | 0.828 | 0.915 | 0.605 | 0.880 | 0.910 | 0.038 | 24.51 | 20.40 | 8.28 | 1 |
| Kev-27B Q8_0 (ref) | 0.833 | 0.890 | 0.635 | 0.895 | 0.910 | 0.029 | 80.43 | 66.39 | 23.80 | 1 |
| Qwen3.8-27B Q4_K_M zero-shot (t2) | 0.782 | 0.845 | 0.565 | 0.830 | 0.890 | 0.114 (raw) | 65.29 | 59.86 | 29.58 | 3 |

Batch-8 floor protocol (logs/cascade/b8, /tmp/b8an.py; J/decision = (J(K=8 run) - J(K=1 run)) / 320, socket gross / net / gfx, median of clean reps):

| model | K | J/dec gross [range] | net | gfx | clean reps (dropped) |
|---|---|---|---|---|---|
| d3-edge | 1 / 8 | 2.76 [2.74-2.78] / 2.74 [2.72-2.76] | 2.37 / 2.36 | 1.19 / 1.18 | 2 (0) |
| d3-lite | 1 / 8 | 5.63 [5.59-5.67] / 5.75 [5.50-6.00] | 4.87 / 5.02 | 2.41 / 2.37 | 2 (0) |
| d3-flash | 1 / 8 | 16.48 [16.13-16.83] / 16.07 [16.05-16.09] | 14.94 / 14.61 | 6.18 / 6.09 | 2 (0) |
| pplx-flash | 1 / 8 | 9.04 [8.61-9.48] / 8.60 [8.58-8.62] | 8.15 / 7.76 | 3.31 / 3.27 | 2 (0) |
| pplx-27B | 1 / 8 | 51.68 / 51.71 | 47.77 / 47.55 | 20.74 / 20.63 | 1 (1) |
| Drex Max | 1 / 8 | 52.86 [52.68-53.04] / 52.87 [52.51-53.23] | 49.58 / 48.46 | 21.49 / 21.17 | 2 (0) |
| Kev-0.8B (kev_serve) | 1 / 8 | 4.35 [4.12-4.57] / 0.67 [0.64-0.69] | 3.02 / 0.47 | 2.00 / 0.30 | 2 (0) |
| Kev-9B Q8_0 (kev_serve) | 1 / 8 | 22.26 [21.87-22.64] / 3.72 | 17.75 / 3.02 | 7.61 / 1.23 | 2 (0) / 1 (1) |

Reading: on upstream llama-server /v1/systemone, packing 8 questions per request does not lower J/decision for any rival (K=8 within 0-5% of K=1), while kev_serve batching cuts Kev-0.8B 6.5x (4.35 -> 0.67 J) and Kev-9B 6.0x (22.26 -> 3.72 J). Best accuracy is Drex Max (0.839) at 64.7 J/dec; Kev-9B Q8_0 reaches 0.828 at 24.5 J (batch 1) or 3.7 J (K=8). Kev-0.8B at K=8 (0.67 J) is 4.1x below the cheapest rival (d3-edge 2.74 J) with +5.5 pts accuracy.

Notes:
- pplx-decider-flash v1.2: the published GGUF (perplexity-ai/pplx-decider-flash-gguf-v1.2) has only `pplx_decider.*` keys and no `qwen35.decision.type`, so llama-server returns 501 on /v1/systemone. Re-converted from perplexity-ai/pplx-decider-flash-v1.2 safetensors with llama.cpp master (3f8af15a7) convert_hf_to_gguf --outtype q8_0 -> pplx-flash-v1.2-conv-Q8_0.gguf (chain2).
- pplx-decider v1.2 27B: no GGUF published; converted the same way (chain3) after download.
- Drex Max v1: nace-ai/drex-max-v1-Q8_0, served with the nace-ai/llama.cpp fork branch drex-max-v1 (8f6f9cc) built for Vulkan at ~/llama-drex (chain3).
- AWS Strands Decider 2B (StrandsAgents/strands-decider-2B-hobson-v21): LoRA + head on Qwen3.5-2B, served only by the torch `strands-decider` package; no llama.cpp arch or GGUF, so not runnable on the Vulkan backend. Not measured.
- PolicyLM-1.7B (musubilabs/policylm-1.7b): BidirLM torch-only guardrail classifier, not a System One decision model; no GGUF/Vulkan path. Not measured.
