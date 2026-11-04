# Reflection Beam-1.0-501B-A23B on strix-rt (disk-streamed experts)

Status: **in progress** (see `crates/beam/PROGRESS.md` for the running log). Local only, nothing pushed.

## Model (config.json + llama.cpp `src/models/beam.cpp`, convert `conversion/beam.py`)
- 501.3B total / 23.4B active, Apache-2.0, 52 layers = 3 dense + 49 MoE, hidden 5120, vocab 151,936, 1M ctx, softcap 30 on logits.
- Attention: gated GQA, 48 q heads / 8 kv heads / head_dim 128. Per-head RMSNorm on q and k (eps 1e-6). Output gate:
  `ao *= sigmoid(gate_proj(x) + gate_bias[head])` (gate_proj is 6144x5120, bias per head) before `o_proj`.
  43 sliding layers (window 512, NeoX RoPE theta 10000, full rotary 128) and 9 full-attention layers (5,11,17,23,29,35,41,47,51) with **no RoPE**
  and a query temperature `q *= (1 + (pos+1)/2048)^0.1`.
- Sandwich norms (input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm) and a **scaled residual**:
  `x = (1/sqrt(i)) * post_norm(branch) + sqrt((i-1)/i) * x` with `i = 2*(layer+1)` after attention and `i+1` after the FFN.
- Dense layers 0-2: SwiGLU, intermediate 20480. MoE layers 3-51: 256 routed experts (intermediate 2560), top-8, **no shared expert**.
  Router: sigmoid(W x) (fp32 [256,5120]); selection on `sigmoid + e_score_correction_bias`; weights = sigmoid (unbiased) renormalised over the 8, times 2.5.
  llama.cpp does not apply the config n_group/topk_group (8/4); `BEAM_GROUPS=1` enables DeepSeek-V3 style group limiting for comparison.

## NVFP4 checkpoint (reflection/Beam-1.0-501B-A23B-NVFP4, 273 GB) layout
- `model-layer-000NN.safetensors` (one per layer) + `model-tail.safetensors` (lm_head, embed_tokens, final norm; BF16, 1.56 GB each) + index.
- Layers 0-2 are fully BF16 (0.84 GB each). Layers 3-51 (5.88 GB each): attention BF16 (210 MB), router fp32, norms BF16, and
  per expert 3 x {weight U8 [N,K/2] (two e2m1 per byte, low nibble = even k), weight_scale F8_E4M3 [N,K/16], weight_scale_2 f32, input_scale f32}.
  Per expert: 19.66 MB packed weights + 2.46 MB scales = **22.1 MB**, stored contiguously (down, gate, up) so one pread each for weights and scales.
- Expert bytes dominate: 49 x 256 x 22.1 MB = 277 GB. Resident (non-expert) set is only ~16 GB.
- Per token the router touches 8 experts x 49 layers = 392 experts = **8.7 GB** of weights.
- Quantization is W4A4 in the checkpoint (static `input_scale`); strix-rt dequantises weights and keeps activations in f32 (W4A16, as llama.cpp does).

## Design
- Resident in UMA GPU-visible memory: BF16 attention (q, gate, k, v, o) per layer, dense MLPs, lm_head, router matrices, fp32 norms, KV cache f32.
- Embedding rows are pread from the tail shard per token.
- Per MoE layer: one submit `[experts(L-1) -> attention(L) -> router logits(L)]`, host reads 256 router logits, picks top-8,
  8 threads pread the experts' weight/scale blocks into 8 slot buffers (host-visible UMA, 22.1 MB each), the next submit runs the NVFP4 kernels.
- Kernels (crates/beam/shaders): `bm_gemv` (BF16, one WG per row), `bm_fp4gu` (gate+up fused SwiGLU, wave32 per row, fp4/fp8 LUTs in shared memory),
  `bm_fp4dn` (down, summed over experts with routing weights), `bm_qkv`, `bm_attn`, `bm_rmsres`, `bm_rlog`.
- `tools/beam_index.py DIR` writes `DIR/beam.idx` (tensor name, dtype, shard, absolute file offset, bytes, dims).

## Parity reference
llama.cpp (b11701+, built at `~/llama.cpp-beam`) converts the same NVFP4 safetensors to GGUF NVFP4 and supports Beam; used as the logprob reference.

## Results (2026-10-18, Framework 128 GB, NVFP4 shards on NVMe, 120 GB page cache)
- **Correct output.** Raw prompt "The capital of France is" -> " Paris" (logprob -0.22). With the Harmony template ("What is the capital of France?")
  the model answers "<|channel|>Paris.<|end|>"; continuing past it (our loop does not stop at EOS) it asks and answers a follow-up coherently. Logs: logs/beam/first.log, chat.log.
- Per token: 392 experts x 22.1 MB = 8.7 GB streamed (+ ~16 GB resident set read once at load, ~1 min).
- Speed (greedy, ctx<=512, prompt processed token by token, no batching): cold ~1.0-1.2 s/token (0.8-1.0 tok/s; NVMe CT4000T710 measured 5.8 GB/s single stream, ~7 GB/s parallel);
  once the page cache holds the hot experts ~0.45 s/token (~2.2 tok/s). GPU time is only ~0.13 s/token, so decode is disk/page-cache bound.
- Net energy (SMU socket minus idle, differential 50-step vs 12-step windows, quiet-box gate, **warm cache**): **21-22 J/token** (2 reps: 22.2, 20.6).
  Cold-cache energy not yet measured; expect more joules per token (longer wall time at ~35-40 W net).
- llama.cpp parity: convert_hf_to_gguf.py converted the same NVFP4 checkpoint to a 293.6 GB GGUF (needs --use-temp-file; it OOM-kills without), but llama.cpp b11718 CPU (-ngl 0 --device none)
  generated garbage ("GGGGGGGG") for the templated prompt at 4.1 tok/s, and the Vulkan path ran out of memory at load. So llama.cpp is NOT a usable reference here; our correctness evidence is the semantic output only.
  No numeric logprob parity vs a reference was obtained yet.

## Caveats / next
- No batching of prompt tokens (each prompt token streams its own experts); prefill of long prompts would be very slow. Union-of-experts prefill is the obvious fix.
- Speculative prefetch (router of layer L+1 applied to layer L hidden state), O_DIRECT, and overlapping expert reads with GPU compute are not done.
- ctx limited to 4096 by the attention kernel (scores in shared memory); no EOS stop; greedy only.
- Cold-cache and sustained-throughput numbers need a drop-caches-free methodology (reading a large unrelated file) before quoting as disk-bound figures.

## Prefetch / overlap / prefill update (2026-10-18, later)
Reader: 32 threads x 1 MB chunks (was 8 threads x whole experts); optional O_DIRECT (BEAM_DIRECT=1); cold start via fadvise DONTNEED (BEAM_EVICT=1). Output unchanged (same top-5, logprob -0.1229 at the chat prompt; " Paris" prompt unchanged).
- Cold first token: 1.77 s -> 1.55 s (32x1 MB reads). Sustained cold (O_DIRECT, no page cache): **1.37-1.43 s/token (0.7 tok/s), 6.9 GB/s = the NVMe limit** (8.7 GB/token). Warm cache: 0.4-0.5 s/token (variable with neighbour load).
- Speculative prefetch (BEAM_PF_DEPTH, next-layer router applied to the current residual, readahead of top-12): predicts the actual top-8 well (87% at 1 layer ahead, 79% at 2, 84%/78% for top-16 at depth 2/3) but gives **no speedup** (cold 23.4 s vs 23.4-25 s for 34 steps; slightly slower with O_DIRECT because prefetch competes for the same disk). Reason: the disk is already saturated during the demand read; only ~10% of a layer (GPU+host sync) is overlappable. Off by default.
- Batched prefill (BEAM_PREFILL=1, 16 tokens per chunk, union of experts streamed once, per-token expert kernels): identical logits. Cold 11-token prompt 9.5 s vs 10.8 s token-by-token; 37-token prompt 18.7 s vs 24.2 s (1.3x). Gain is small because 16 tokens still touch most experts in each layer.
- Net energy (SMU socket minus idle, differential 50- vs 12-step windows, gate passed at start but the cascade-harness worker ran jobs on the box, load avg ~7, so treat as upper bounds): warm cache 26.7-32.7 J/token (earlier quiet run: 20.6-22.2); sustained cold O_DIRECT 31-53 J/token at 1.4 s/token. Raw JSON/logs: logs/beam/energy2.
- Honest conclusion: Beam on one 128 GB Strix Halo is NVMe-bound at about 0.7 tok/s sustained-cold and 2-2.5 tok/s once the 120 GB page cache holds the hot experts. Prefetch cannot beat disk bandwidth; only more bytes/s (RAID/second drive), fewer bytes (lower-bit experts) or better caching (expert-popularity pinning) help.

## Phase 3: lower-bit experts and a managed expert cache (measured 2026-10-18)

All speed numbers: O_DIRECT (no page cache), 16 reader threads x 4 MB, quiet-box gate passed (<0.8 busy cores), GPU lock held, prompt "capital of ..." chat (11 tokens), 39 decoded tokens, 2 reps (wall identical to 0.1 s). Raw: logs/beam/bench, summary logs/beam/bench_summary.txt, tools/beam_bench.py.

**Lower-bit experts** (tools/beam_requant.py: symmetric group-32 round-to-nearest, fp16 scale, from the NVFP4 checkpoint; GLSL kernels bm_qgu/bm_qdn with generic bitstream decode; `BEAM_QDIR=$MODELS/new/beam/q2|q3`). Quality = teacher-forced on the NVFP4 greedy continuations of 12 prompts (Paris raw/chat, math, 2 code, prose, science, history, German, SQL, logic, JSON), 564 positions, vs NVFP4 on the same positions (tools/beam_evalq.py, logs/beam/evalq_q*.txt):

| format | bytes/expert | s/token (cold, direct) | tok/s | top-1 agreement | mean dlogp of chosen token | KL(top-20) | p(chosen) |
|---|---|---|---|---|---|---|---|
| NVFP4 | 22.1 MB | 1.397 | 0.72 | (ref) | 0 | 0 | 0.871 |
| 3-bit | 17.2 MB (0.78x) | 1.122 | 0.89 | **95.7%** | -0.031 | 0.027 | 0.859 |
| 2-bit | 12.3 MB (0.56x) | 0.840 | 1.19 | 73.8% | -1.60 | 0.83 | 0.631 |

Honest read: 3-bit RTN is nearly lossless (some prompts still diverge after a few tokens: 89-100% top-1 per prompt) and buys 1.25x; naive 2-bit RTN is too lossy for general use (prose 40% top-1) and would need GPTQ/AWQ-style calibration or mixed precision. Speed is still disk-bound (6-7 GB/s) so gain tracks bytes. 3-bit output on Paris prompts still starts correctly; free-running q2 text degrades into repetition on some prompts (logs/beam/evalq/*.q2.free.log).

**Managed GPU-visible expert cache** (`BEAM_CACHE=K`: per-layer LRU, K slots/layer in device-visible UMA buffers, kernels read the cached slot directly, misses are read into the evicted slot; per-layer buffers stay under the 4 GB binding limit so no buffer-device-address needed). Output tokens identical to the uncached run. Cache warmed on the other 11 prompts (16 tokens each, held-out), then decode of the measured prompt:

| config | cache RAM | decode hit rate | s/token | tok/s | disk bytes/39 tok |
|---|---|---|---|---|---|
| NVFP4 + K=64 | 69 GB | 76.6% | 0.432 | 2.31 | 132 GB (vs 391) |
| 3-bit + K=96 | 81 GB | 76.5% | 0.372 | 2.69 | 103 GB |
| 2-bit + K=128 | 77 GB | 88.0% | 0.209 | 4.78 | 53 GB |

Popularity analysis over the 12 traces (logs/beam/prof): experts are skewed but not hugely: static pin of the held-out-popular top-K per layer hits 21/33/49/61/71% for K=16/32/64/96/128 (uniform would be 6/12/25/38/50%); an LRU hits 53/70/79/85% for K=32/64/96/128 per layer. Caveat: short prompts, and the measured prompt's greedy text is repetitive, which flatters temporal locality; expect lower hit rates on long diverse generations. The cache gives 3.2x on NVFP4 (0.72 -> 2.31 tok/s) with no quality loss; combined with 3-bit it is 3.8x.

**Energy**: uncached direct runs, differential net SMU socket J/token (2 reps): NVFP4 15.7 and 47.7 (inconsistent between reps; whole-run net 24.8-54 J per token incl. load), 3-bit 38, 2-bit 29; average socket power is flat at ~39-40 W so J/token tracks s/token. Cached runs: the 90-115 s warmup sits inside the telemetry window so the differential is noise (-35..+27 J); not reported as a measurement. Estimated from flat-power scaling (power x s/token, ~40-54 W total during cached decode): roughly 17-20 J/token total at 0.43 s/token for NVFP4+cache, 11 J at 2-bit+cache, but this is an estimate, not a measured differential. TODO: separate warmup from the window.

## Phase 4: cached-run energy (warm-up outside the differential) and mixed precision (2026-10-18)

**Cached J/token, re-measured.** The cache warm-up (11 held-out prompts x 16 tokens, ~100 s) is identical in the long and short telemetry windows, so it cancels in the difference: net SMU-socket J/token = (net J of N=200 run - net J of N=40 run) / 160 tokens (uncached NVFP4: N=30 vs 6). Direct IO, quiet-box gate (<0.8 busy cores), GPU lock, 5 s idle baseline, 2 reps each (tools/beam_ce.py, logs/beam/cachee):

| config | s/token (marginal) | marginal decode hit rate | net J/token (rep1/rep2) |
|---|---|---|---|
| NVFP4 uncached, cold direct | 1.40 | - | 45.8 (46.3 / 45.3) |
| NVFP4 + 64 slots/layer cache | 0.53 | 69.8% | 30.8 (38.8 / 22.7, noisy) |
| 3-bit + 96 slots/layer cache | 0.25 | 88.2% | **13.9** (13.8 / 14.1) |
| 2-bit + 128 slots/layer cache | 0.147 | 95.4% (very repetitive text) | **10.0** (10.1 / 9.95) |

So the cache cuts energy per token about 1.5x for NVFP4 and about 3.3x vs uncached NVFP4 when combined with 3-bit (45.8 -> 13.9), at 5.6x the speed. The long 200-token generations are repetitive, so hit rates (70-88%) are optimistic for diverse text. NVFP4+cache reps disagree by 16 J (ordinary telemetry noise on a ~0.5 s/token loop); the 3-bit reps agree to 2%.

**Mixed precision (hot experts 3-bit, rest 2-bit).** Hot sets chosen from routing traces of 6 prompts (A), quality measured on the other 6 (B: science, history, German, SQL, logic, JSON; 282 teacher-forced positions), so there is no leakage (logs/beam/evalB_*.txt):

| format (B set) | top-1 vs NVFP4 | mean dlogp | KL(top-20) | bytes streamed / 39 tok |
|---|---|---|---|---|
| 2-bit | 81.9% | -0.95 | 0.433 | 218 GB |
| mixed 32 hot/layer @3-bit | 91.1% | -0.12 | 0.074 | - |
| mixed 64 hot/layer @3-bit | 91.8% | -0.065 | 0.063 | 272 GB |
| 3-bit | 94.3% | -0.041 | 0.031 | 304 GB |

Mixed precision recovers most of the 2-bit quality loss (KL 0.43 -> 0.06-0.07) by upgrading only 12-25% of experts, but since the popular experts are only about 40-50% of routed picks, bytes drop just 11% vs all-3-bit; uncached it is a poor trade vs plain 3-bit. Its value is with the cache, where hot experts stay resident anyway. Implementation: per-expert bits in the slot meta, runtime-bits decode in the shaders (`BEAM_QDIR=q2 BEAM_QHI=q3 BEAM_QHOT=hotfile`).

**Activation-aware 2-bit** (tools/beam_requant.py --act): group scale chosen by an 11-point clip search plus least-squares refinement, with the error weighted per input channel by mean-square activations (diagonal-Hessian weights, shrunk 20% to the mean) collected from a calibration run over prompts A (BEAM_ACTOUT dump: FFN input for gate/up, pooled down input). Evaluated on the disjoint set B (282 positions): top-1 **85.1%** (data-free 2-bit 81.9%), mean dlogp **-0.58** (-0.95), KL(top-20) **0.237** (0.433); 3-bit is 94.3% / -0.04 / 0.031. Same bytes and speed as plain 2-bit (0.84 s/token cold; 0.147 s/token cached). It halves the KL but 2-bit is still clearly worse than 3-bit; a GPTQ-style full-Hessian pass with a larger calibration set (our 360 calibration tokens are too few for that) or mixed precision on top would be the next step.

## Add-before-multiply (ABM) 3-bit expert GEMV (2026-10-30)
Idea (requested by David): with 3-bit codes q and per-group-32 scale s, y = s * sum_i (q_i - C) x_i. Instead of one multiply per weight, first add activations into bins, then multiply each bin once.
Prior art: arXiv 2610.31913 (add-before-multiply GEMV); LUT-GEMM (Park et al., arXiv 2206.09557: bit-plane lookup tables of partial sums); T-MAC (Wei et al., arXiv 2407.00088: bit-serial table lookup for low-bit LLMs on CPU).
Implementation: spec constant `ABM` in `bm_qgu.comp`/`bm_qdn.comp` via `qcommon.glsl` `qdot32`, selected at load by env `BEAM_ABM`:
- 0 = MAC (default, unchanged)
- 1 = bit-plane (T-MAC/LUT-GEMM form): p_k = sum of x where bit k of q is set, dot = sum_k 2^k p_k - C * sum x
- 2 = code-bin (the 8-bin scatter form): b[q] += x, dot = sum_b (b - C) * bin[b]

Exactness (full q3 model, teacher-forced top-20 logprobs on science_chat + sql, 47 positions each; tools/abm_cmp.py, logs/beam/abm/exact.txt): both variants match MAC on top-1 47/47 and top-20 tokens and order 47/47 on both prompts; max |dlogprob| 0.0001 (the log's last printed digit). It is mathematically identical; only the float summation order differs, so it is not bit-exact on every row (31-34/47 rows identical to 4 d.p.).

Kernel time (BEAM_NL=8, 224-slot expert cache, GPU-bound, GPU seconds of N=200 minus N=40 runs, 2 interleaved reps; logs/beam/abm/partB.txt):

| variant | GPU ms/step (rep1, rep2) | median |
|---|---|---|
| MAC (current) | 28.75, 31.31 | 30.0 |
| ABM bit-plane | 28.56, 31.94 | 30.3 |
| ABM code-bin | 52.88, 42.31 | 47.6 |

Verdict: no win on gfx1151. The GPU GEMV is memory-bound, so FMA count is not the limit. Bit-plane ties MAC. Code-bin's per-thread 8-entry bin array is dynamically indexed, so it spills to scratch and runs 1.6-1.8x slower. MAC stays the default. Quiet rerun (evening 2026-10-30, logs/beam/abm/partB_quiet.txt; daytime noisy windows archived in logs/noisy_1030): GPU ms/step MAC 25.8, bit-plane 25.7, code-bin 33.7; net J/token (socket, N=200 minus N=40, idle subtracted) MAC 1.94, bit-plane 2.12, code-bin 2.85 (medians of 2 reps; per-rep spread about +-0.5 J, so MAC and bit-plane are a tie). ABM's natural home is CPU/NPU integer pipelines (T-MAC), not this GPU.

**Default quant path: per-layer codebook on all layers (cbBall).** `BEAM_QDIR=cb` (alias `default`) selects `~/models/new/beam/cbBall/q3`, a per-layer 3-bit Lloyd codebook on every expert layer, and loads the `lut.txt` stored in that directory as the LUT (`BEAM_CBLUT` still overrides). Measured against nvfp4 (282 tokens, 6 prompts, teacher-forced): top-1 0.950, KL 0.024, compared with q3 at 0.943/0.031. Check: `BEAM_QDIR=cb` on sql matches the explicit cbBall + LUT run bit for bit (47/47 forced positions). If `BEAM_QDIR` is unset, beam uses the full-precision nvfp4 experts as before.
