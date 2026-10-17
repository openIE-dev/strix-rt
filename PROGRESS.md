# strix-rt progress

## Milestone 1 (2026-10-14)
Workspace: crates/gguf (GGUF v3 parser, mmap, CPU dequant Q4_K/Q6_K/Q8_0/F16/F32),
crates/vkrt (ash Vulkan layer: picks RADV, reports props/subgroup/coop-matrix/heaps, HV|DL|HC buffers,
pipelines w/ spec consts + required subgroup size, timestamp-timed submits; glslc in build.rs),
crates/bench (Q4_K GEMV correctness vs CPU f64 reference + bandwidth on real q7 tensors).

Device: Radeon 8060S (RADV STRIX_HALO), Mesa 26.0.8, Vulkan 1.4.335, subgroup 32..64 (default 64).
Heaps: heap0 40.2 GiB (host), heap1 80.3 GiB DEVICE_LOCAL; types 3/4/9 are DL|HV|HC (used for all buffers).
Coop matrix: 16x16x16 subgroup, int8->int32 and f16->f16/f32.

Bench (logs/bench1.log, ITERS=100, weights cycled across copies >=256MiB to defeat the 32MB MALL; GB/s = (W+x+y)/time, 1e9):
best config per shape (gemv_q4k.comp, ls/rows-per-wg/subgroup):
- 18944x3584 ffn_gate/up: 32/1/32 216.4 GB/s (84.5%)
- 3584x18944 ffn_down:    32/1/32 200.4 GB/s (78.3%)
- 3584x3584  attn_o:      32/1/32 169.1 GB/s (66.1%)
- 152064x3584 token_embd: 64/4/64 205.6 GB/s (80.3%)
llama.cpp Vulkan tg128 47.4 tok/s * 4.36GiB ~= 222 GB/s effective.
Correctness: max_abs <= 1.9e-6, max_abs/max|ref| <= 3.5e-7 on all shapes/configs.

## Next
- Q6_K GEMV (output.weight, attn_v, half of ffn_down are Q6_K), fused gate+up GEMV, small-N tuning (more rows/WG, wave32)
- Qwen2 decode forward pass (RMSNorm, RoPE, KV-cache attention, SwiGLU) + compare vs llama.cpp
- Zero-copy GGUF import via VK_EXT_external_memory_host

## Milestone 2: Qwen2 decode forward pass (2026-10-14)
- crates/qwen2: model loader (quant weights in 3 DL|HV arenas <=2GiB, whole layers per arena; F32 norms/biases in one buffer;
  token_embd dequantized per token on host from the mmap), whole-token graph recorded ONCE into one command buffer and
  resubmitted per token (pos read from a params buffer). Kernels: gemv.comp (Q4K/Q6K/Q4K+SwiGLU fused gate+up, bias, residual-accumulate,
  one row per subgroup), rmsnorm, rope_kv (NeoX RoPE + KV-cache store, f32 cache), attn (GQA, 1 WG/head, ctx<=4096).
- bin `strix gen <ids> [n]` and `strix kbench` (SWEEP=1, SHARED_X=1, GEMV_LS/GEMV_SG env).
- Decode q7, 128 tokens after 12-token prompt (logs/gen128.log, 3 runs): 44.6-44.9 tok/s wall, GPU-only 21.27 ms/tok = 47.0 tok/s.
  (llama.cpp Vulkan tg128 47.4). Fix that mattered: logits in HOST_CACHED memory (CPU reads of DL|HV memory are uncached: 14.6 -> 45 tok/s).
- Output: 64 greedy tokens identical to llama.cpp CPU backend (-ngl 0). llama.cpp Vulkan differs at token 19 (" as" vs our " becoming");
  our logits there: becoming 19.5154 vs as 19.5119 (near tie), and the sequences re-converge. llama.cpp HIP diverges from token 1.
- kbench (logs/kbench_sweep_noshared.log, ls64/sg32, cycling all layers): attn_q/o 192 GB/s, ffn_gate 194, gate+up fused 230,
  ffn_down Q4K 198 / Q6K 222, output Q6K 231. attn_k / attn_v sets total <=34MB and are MALL-resident (attn_v >256 GB/s: not DRAM-bound).
- Shared-memory x staging made 3584-K GEMVs much slower (90 vs 192 GB/s at ls64/sg32); disabled by default (SHARED_X=1 to enable).
- Not done: zero-copy VK_EXT_external_memory_host (file 4.68GB > 4GB maxStorageBufferRange per binding; needs import + split buffers).
## Next
- GPU embedding + argmax to remove ~1.2ms/token host overhead (wall 44.8 vs GPU 47.0); double-buffer submits.
- Fuse rmsnorm into following GEMV, q/k/v into one dispatch; f16 KV cache; split-K attention for long ctx.
- Coop-matrix prefill (pp) path.

## Memory tier investigation (2026-10-14), see docs/memory-tier.md
- vkrt: VK_EXT_external_memory_host enabled, `Ctx::import_host`; qwen2 `ZEROCOPY=1` (per-layer imports of mmap'd GGUF), `ZC_CHUNK_GB`.
- Stock RADV rejects file-backed mmaps (libdrm passes ANONONLY); kernel accepts READONLY file userptr. tools/libuserptr_shim.so proves
  zero-copy works: bit-identical GEMV, same decode text. Imported GEMV 169-194 vs copied 197-208 GB/s; per-submit tax ~1.1-1.4 ms per GiB imported;
  zero-copy decode 35.1 tok/s wall / 43.7 GPU vs copied 44.6 / 47.0.
- NVMe: Gen4 x4 link (drive Gen5), ~7.2 GB/s in all cold read patterns, 1.71M IOPS 4k. O_DIRECT into Vulkan-mapped memory = EFAULT;
  buffered pread into DL|HV 6.6 GB/s cold.

## Zero-copy at copied speed (2026-10-14), see docs/memory-tier.md section 4
- Per-submit cost confirmed with bpftrace: hmm_range_fault on every userptr BO on every CS (5908 calls / 211 submits / 28 BOs).
- Slowdown cause: 4 KiB vs 2 MiB GPU fragments (snooping ~2%). Fix: 2 MiB-aligned file ranges + VA alignment.
- kmod/strixpc: pins page-cache folios and exports a dma-buf; ZEROCOPY=dmabuf decode 44.81-45.06 tok/s wall / 47.05-47.06 GPU vs copied 44.51-44.59 / 47.00-47.02; identical tokens. No shim needed.

## Turn 5 checkpoint: GPU-driven fast path (2026-10-14)
- GPU embed lookup + argmax, SIMULTANEOUS_USE per-token cmdbuf, chunked submits (16/submit, fence on last), fused rmsnorm (xw producer + sumsq atomics), fused qkv, f16 KV cache, per-op LPR tuning, attn16 P.V split over 4 waves with f16vec4 loads.
- q7, ZEROCOPY=dmabuf, 12-token prompt + 128 decode, 3 runs: 49.60 tok/s wall, 20.15 ms/tok GPU (llama.cpp Vulkan ref 47.4). Before attn change: 48.65-48.67 wall.
- 128 decoded ids identical to `strix gen` (old path); text matches llama.cpp CPU reference opening.
- Open: (d) dma-buf per-kernel gap (~1.7% vs copied), (e) DKMS + pin LRU, >50 tok/s.

## LFM2.5-24B-A2B (lfm2moe) Q4_K_M (2026-10-14)
Model: $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf (14,526,258,464 B, from HF).
Arch from GGUF: 40 layers (30 shortconv L=3 + 10 GQA at 2,6,..,38; 32 heads, 8 kv, head_dim 64, q/k RMSNorm), 2 leading dense SwiGLU (ffn 11776), 38 MoE layers (64 experts, top-4, ff_exp 1536, sigmoid gating + selection bias, normalised weights), tied Q6_K token_embd/lm_head, final norm = token_embd_norm, rope NeoX theta 1e7.

Baseline llama.cpp 10da8c463, llama-bench -p 512 -n 128 -r 3 -ngl 99 (logs/lfm_llama_{vulkan,hip}.log):
- Vulkan: tg128 118.89 +- 0.46 tok/s, pp512 2280.40
- ROCm/HIP: tg128 101.80 +- 0.74 tok/s, pp512 1698.03

strix-rt (crates/lfm2 + crates/rtcore, bench bin `lfm`), 13-token prompt + 128 decode, logs/lfm_final_bench.log:
- ZEROCOPY=dmabuf: 123.98-124.89 tok/s wall (5 runs), GPU 8.00-8.06 ms/tok
- copied: GPU 7.87 ms/tok = 127.06 tok/s; wall 108.7-126.9 (noisy; not investigated)
- Profile (dmabuf, before the MoE gate/up LPR change): moe_gu 2805 us, moe_dn 1677, in_proj 1133, lm_head 941, out_proj 419, router 315, dense 440, attn path ~450.
Correctness: free-run greedy ids identical to llama.cpp Vulkan and HIP for the first 10 tokens (those two diverge from each other at the same index); llama.cpp CPU diverges from all GPU backends at token 1.
Teacher-forced top-1 agreement over 128 generated positions (FORCE=, last position is a retokenisation artifact): vs CPU 117/129, vs Vulkan 121/129, vs HIP 121/129. Q8EMU=1 (q8_K activation rounding as on CPU) does not raise CPU agreement (116/129).
Key fix: the serial router (one workgroup + serial top-k) was 45% of the step (7.0 ms); now a wave-per-expert logits kernel + top-k selected redundantly per subgroup inside the expert GEMVs -> 15.3 -> 8.0 ms/tok.
strixpc: DKMS package (kmod/strixpc/dkms.conf, dkms-install.sh), installed for 7.0.0-38-generic, loaded via modprobe (no boot autoload). Pin LRU not done.

## Qwen3.8-27B (GGUF arch qwen35), crate `qwen38`: decode works (2026-10-14)
- Model: unsloth Qwen3.8-27B-UD-Q4_K_M (15.38 GB of trunk+head weights read per token; mixed Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS/IQ4_NL/Q3_K/IQ3_S).
- Pipeline: 48 Gated DeltaNet layers (gdn_conv conv1d+SiLU+L2norm, gdn_step recurrent state + gated RMSNorm) + 16 gated-attention
  layers (rope_qkn q/k norm + partial NeoX RoPE 64/256, split-K attn_split/attn_combine with sigmoid gate), SwiGLU FFN, all GEMVs via gemvq.
- Correctness, teacher-forced top-1 over 129 generated positions (refs: logs/q38_ref_{cpu,vk,hip}_ids.txt, llama-simple greedy 128 tok):
  vs CPU 126/129, vs Vulkan 126/129, vs HIP 127/129 (the last position is a tokenization-boundary artifact in all three).
  Free-running greedy output matches llama.cpp Vulkan text for the first 31 tokens checked.
- Speed (logs/q38_final_bench.log, prompt 12 tok, decode 127 tok, ternary worker idle):
  dma-buf zero-copy 13.57 tok/s (73.7 ms/tok) steady; copied 13.90 tok/s. llama.cpp: Vulkan tg128 12.94 +- 0.56, HIP 12.80.
- gemvq fixes on the way: (1) IQ3_S grid / IQ4_NL table as GLSL const arrays were lowered to scratch (256 VGPRs, 128 KiB scratch)
  -> ~100x slower; tables now live in f32b[0..528) (rtcore::TBL_FLOATS) and are staged in shared memory. (2) per-format pipeline
  specialization (rtcore::GkQ; GEMVQ_GENERIC=1 disables). (3) uvec4/uvec2 loads for Q4_K/Q5_K/IQ4_XS, u16 loads for 2-aligned
  formats. (4) two items per loop iteration. Per-category bandwidth now ~200-215 GB/s.
- DBG_LAYERS=n truncates the trunk (debug). Gotcha: `pkill -f release/q38` over ssh kills the ssh session itself.
- Not done yet: batched prompt processing (prefill is token-by-token: 12 tok in 0.88 s), MTP speculative decoding, Bonsai 2 PQ2_0 wiring.

## Qwen3.8 MTP speculative decoding (blk.64 nextn head)
- qwen38: `record_batch(m)` verify pass (weights streamed once for m token columns via gemvq NB mode; per-column GDN conv/step,
  RoPE/attn/combine; GDN state snapshot after column 0 for rollback), `record_mtp(hcol, tcol, head)` (embed -> mtp_prep ->
  eh_proj -> blk.64 gated attention with its own KV -> SwiGLU -> shared_head_norm -> output.weight -> argmax_b), `generate_spec`
  (greedy, 1 draft/cycle). Bench: `MTP=1 q38 ...`. MTP KV positions are shifted by -1 (pair (h_p, t_{p+1}) stored at p; RoPE relative).
- Qwen3.8-27B-UD-Q4_K_M, ZEROCOPY=dmabuf, 12-token prompt + 128 tokens (logs/q38_mtp_bench.log):
  - acceptance 60/68 drafts = 88.2%, 1.868 tokens/cycle; output identical to plain greedy for 128/128 tokens
  - 20.48-20.50 tok/s wall (runs 2-3; run 1 15.29 overlapped another worker's pf_q38 job) vs plain decode 13.56 tok/s same session = 1.51x
  - GPU: verify (2 columns) 80.7-80.9 ms/cycle vs 73.7 ms single-token step; MTP head 7.9-8.4 ms per cycle
- Earlier logs/q38_mtp_first.log (11.70 tok/s) ran during GPU contention; same tokens/acceptance.

## Bonsai 2 (PQ2_0 / PTQ1_0) in qwen38
- gemvq types 8 PQ2_0 (ggml 142) / 9 PTQ1_0 (143) / 10 BF16 / 11 F16; codebooks in f32b staged to shared mem; PQ2 whole-block dword reads.
- Prism Hadamard: ternary FWHT before every rotated GEMV (401 weights), ssm_out grouped perm (128,16,3), inverse transform after embed_any lookup.
- Decode, 12-token prompt + 127 tokens (logs/bonsai_final_bench.log; fork logs/bonsai_fork_*.log):
  - PQ2_0: 27.20 tok/s dma-buf, 27.79 copied | fork tg128 Vulkan 21.97 +- 0.90, HIP 24.06 +- 0.19
  - PTQ1_0: 30.0 tok/s dma-buf and copied | fork tg128 Vulkan 26.80 +- 0.59, HIP 27.86 +- 0.34
- Teacher-forced agreement vs fork CPU: 112/113 both formats (only miss: EOS vs re-tokenized newline at end of reference); PQ2_0 vs fork Vulkan 127/129.

## Speculative decoding round 2: Bonsai 2 MTP graft, chained MTP, int8 ternary path
- Chained MTP: k drafts per cycle (MTP head on trunk h, then k-1 MTP steps on its own shared_head_norm output), one (k+1)-column verify,
  per-column GDN state snapshots for rollback. `MTP=1,2,3 q38 ...`.
- Option A (graft): `MTP_GGUF=Qwen3.8-27B-UD-Q4_K_M.gguf` loads blk.64 from the unsloth file onto Bonsai 2 (embedding via Bonsai's
  rotated token_embd + inverse FWHT, draft lm_head = Bonsai's rotated output.weight). The residual stream is compatible: 62.8% acceptance at k=1.
- Option B: old Ternary-Bonsai 1.7B/4B/8B (local copy) are qwen3 with a 151669-token vocab; only 281 of 151669 ids map to the same string
  as Bonsai 2's 248320-token vocab -> not usable as drafts without re-tokenization; not copied. (The drive also has a
  Ternary-Bonsai-27B-dspark-Q4_1 block-diffusion drafter for Bonsai v1: arch dspark, 6 layers, target_layers 1/16/31/46/61, not implemented.)
- Ternary gemvq int8-activation path (TERN_Q8, default on): xq8.comp quantizes x per 128 block, trits unpacked to u8 and dotted with
  dotPacked4x8 (minus block sum). Verify m=2 63 ms -> ~34 ms; NR=1 default. Fork-CPU agreement 112/113 (unchanged).
- Deterministic norm sums (gemvq flag 64: last workgroup reduces partials in order) + NB path uses single-path summation order,
  so spec output is bitwise identical to greedy (before: near-tie flips at token 15 in some runs).
- Results (12-token prompt + 128, ZEROCOPY=dmabuf, GPU idle; logs/spec_bonsai.log, logs/spec_final3.log):
  - Bonsai 2 PTQ1_0 plain 33.23 tok/s; MTP k=1 40.75-40.89 tok/s (62.8% acc, 1.628 tok/cycle), k=2 39.62-39.67 (44.1%), k=3 37.70-37.81 (36.6%); 128/128 = greedy
  - Bonsai 2 PQ2_0 plain 27.48; k=1 34.63-34.69, k=2 34.35, k=3 32.87-32.93; 128/128 = greedy
  - Qwen3.8 Q4_K_M plain 13.62; k=1 18.78-18.84 (88.2%), k=2 17.37-17.40 (67.6%, 2.352 tok/cycle), k=3 15.12-15.17 (57.4%, 2.702); best k=1.
    Before the determinism change (logs/spec_final.log, contended) k=1 21.05, k=2 21.97-22.04: the ordered reduction / paired summation
    costs Q4_K verify time; open item.

## Qwen3.8 MTP final (6979677)
- k=2 chained MTP: 21.96-22.07 tok/s (plain 13.55), 67.6% acceptance, 128/128 equal to greedy (deterministic flag-64 reductions, sequential item order, Q4_K/Q6_K decode-once columns). MTP_VOCAB trim gave no clear gain (left off).
- dspark drafter for Bonsai 27B: skipped (not quick; grafted MTP already 40.8 tok/s).

## Kitten TTS 2 (speech LM on GPU)
- TQ2_1 (ggml 43) added as gemvq type 12 (int8-activation ternary path) + gguf/rtcore/q_any.
- crates/kitten: Qwen3 1.7B speech LM (speaker vector at pos 0, F16 tied head, fork penalties on host), KV prefix reuse for the per-voice reference block. bench bin kt.
- Fork (kitten-tts-2-cpp, CPU, 16 threads), Bruno, greedy, 306-token prompt: 125 tokens, LM 2.26 s, decoder 1.60 s, audio 5.06 s, RTF 0.763.
- strix-rt LM (pf_q38 worker on the GPU concurrently): decode 6.2-6.6 ms/token GPU (109-145 tok/s wall); cold prefill token-by-token 2.9-6.5 s (306 steps); with cached reference prefix 0.007 s.
- Tokens: first 90 identical to fork, 104/125 position-equal; divergence at step 90 is a near-tie (fork raw logits 72.093 vs 72.056 for our pick).
- Decoder: still the fork TorchScript (CPU) via --decode-tokens. ours.wav vs fork.wav log-mel corr 0.968 (frame cosine 0.953); fork re-decode is bit-identical. Waveform corr is low (0.06): the flow decoder is non-causal, so the different tail changes the whole waveform.
- Next: batched prefill; port the S3 flow decoder + vocoder.

## Kitten TTS 2: all-Rust pipeline (Oct 14)
- Recovered an interrupted run's uncommitted work (key-split flash sdpa, DEC_GPROF GPU timestamp profiler, table-twiddle stft/istft) -> 7c40fa5.
- Rust tokenizer (crates/kitten/src/tok.rs): GPT-2 byte BPE + qwen2 pre-tokenizer + parse_special, from the GGUF vocab. 14/14 test strings
  (contractions, numbers, unicode, CJK, newlines, specials) identical to llama-tokenize (logs/kt/tokcmp.sh). The 306-token Bruno prompt built in Rust is identical to the fork's.
- `ktts` bin: text -> tokenizer -> fork prompt layout -> speech LM (GPU) -> S3 flow decoder + HiFT vocoder (GPU, Vulkan interpreter of the exported graph). No C++ / TorchScript at runtime.
  Text normalizer (kitten_text_processing: numbers, abbreviations) is NOT ported: whitespace collapse only, warns on digits. Chunking (>380 chars) not ported.
- Batched prompt (KT_BATCH, default on): whole prompt recorded in one command buffer (vkCmdUpdateBuffer pos per position). Tokens identical; cold 306-token prefill
  0.90-0.93 s vs 0.97 s per-token submit: still gemv-bound (~3 ms/position). Real speedup needs multi-column (NB<=4 gemvq path / GEMM) prefill: open.
- Coop-matrix f16 decoder GEMM (DEC_COOP=1): no gain at these sizes (M=750, N=256..1024; latency/occupancy bound), corr 0.9998; default off.
- Results (logs/kt/kt_e2e.log, perf level high, Bruno, greedy, 129 tokens -> 5.22 s audio):
  - warm (reference prefix + same text cached), median of reps 2-6, GPU idle: LM 0.466 s + decoder 0.358 s = 0.829 s, RTF 0.159 (best 0.149). Second run with another worker on the GPU: RTF 0.21.
  - cold (rep 1): prefill 0.93-1.07 s, RTF 0.40.
  - fork (CPU, 16 threads): LM 2.26 s + decoder 1.60 s, RTF 0.763 -> ours ~4.8x faster warm.
- Quality: Rust decoder on the fork's tokens vs fork.wav: waveform corr 1.0000, max|diff| 3.1e-4. Full Rust e2e wav vs the fork-TorchScript decode of our tokens (ours.wav):
  corr 1.0000, max|diff| 1.5e-4. vs fork.wav: log-mel corr 0.968 (LM tokens diverge at step 90 on a near-tie, 72.093 vs 72.056; non-causal flow decoder spreads it).
- Gaps to RTF < 0.1 (need ~0.52 s for 5.2 s audio): LM decode 3.0-3.6 ms/token (0.39-0.47 s) alone is ~0.08 RTF; decoder 0.33-0.36 s is ~3200 dispatches
  with full barriers (sdpa ~0.10 s GPU, 0.1 s host/CPU-fallback ops). Next: fuse decoder elementwise ops / batch CFG branches, coopmat flash attention,
  move remaining CPU ops (bitwise_not, cat, div paths, istft) to GPU, multi-column prefill for new text, cheaper per-step submit (no per-step query pool).

## Kitten TTS 2: round 2 toward RTF < 0.1 (Oct 14)
- Decoder (fork tokens, 5.06 s audio): 0.33-0.45 s -> 0.22-0.23 s.
  - coop-matrix f16 attention (sdpa_cm): f16-typed shared arrays made coopMatLoad ~6x slower than uvec4-typed ones (same lesson as pf_gemm). 1.10 -> 0.58 ms/call.
  - coop-matrix gemm for linear/matmul (uvec4 tiles); convs stay f32 (f16 convs: max|diff| 0.044). 
  - conv_transpose1d, reflect pad, stft, istft on GPU (misc.comp) -> 0.355 -> 0.240 s (the CPU ops + their forced syncs were ~0.1 s).
  - fused linear epilogues (div-by-const + residual add, gelu, residual add): 3053 -> 2605 dispatches.
  - arena: flush+retry on exhaustion (deferred frees no longer drained by CPU-fallback flushes).
  - Guidance branches: none to batch. The exported decoder runs the estimator twice (2 flow steps, 56 attention calls each), batch 1, no CFG pair.
  - The CPU fallbacks named earlier (bitwise_not, cat, div) were profiling artifacts; the real ones were conv_transpose/pad/stft/istft/randn.
    Left on CPU (9 small syncs/decode): embedding, upsample_nearest1d, sum, repeat_interleave, scalar*tensor mul, cumsum, complex, randn_like.
- LM: KT_LMPROF per-barrier GPU profile (2.80 ms/token at pos 366): ternary gemvs at ~200 GB/s (swiglu 0.90, down 0.53, qkv 0.36, out 0.26 ms),
  attention+combine 0.51, head 0.36 -> 0.16 (GEMV_HEAD 32,256), fused attention+combine (kt_attn.comp) -> 2.71 ms. Persistent-fence step submit
  (no per-step query pool / command buffers): no measurable change. ATTN_NC / GEMV_TERN / GEMVQ_NR sweeps: defaults best, NC changes tokens.
- Text: Rust port of the fork's text.cpp (normalize rules, math symbols, tags, 380-char split, join) + ktts chunking: 17/17 identical to
  `kitten-tts --frontend` with normalize=false (logs/kt/textcmp.py). The FST backend (numbers/dates/currency: "3" -> "three") is not ported.
- gpu_idle_run.sh / kt_e2e.sh treat logs/r11/pfq as a GPU job. Other workers kept the GPU busy for >30 min, so kt_e2e.sh now waits at most MAXWAIT.
- Results (logs/kt/kt_e2e_r2.log, perf level high, Bruno, greedy, 129 tokens -> 5.22 s; another worker on the GPU throughout):
  warm median LM 0.458 s + decoder 0.226 s = 0.685 s, RTF 0.131 (best 0.119); cold RTF 0.29-0.39. Fork CPU RTF 0.763.
  Idle-GPU estimate from GPU timestamps: LM 2.71 ms x 129 = 0.35 s + decoder 0.22 s = 0.57 s -> RTF ~0.11.
- Quality: decoder on fork tokens vs fork.wav corr 1.0000 (max|diff| 0.025, log-mel 0.009 dB); e2e vs fork-decoder of our tokens corr 1.0000
  (max|diff| 0.018); vs fork.wav log-mel corr 0.968 (LM near-tie at step 90, unchanged). Prompt identical to fork; tokens identical to round 1.
- Remaining to < 0.1 (~0.52 s): LM is bandwidth-bound (363 MB ternary + 13-27 MB head per token, floor ~1.9 ms); options: speculative/multi-token
  decoding, int8 head; decoder: fuse layer_norm/add/mul chains, QKV concat, coop flash attention single-pass, remaining 9 CPU syncs.

## Kitten TTS 2: round 3 toward RTF < 0.1 (Oct 14)
- Decoder (fork tokens, 5.06 s audio, idle GPU): 0.215 s -> 0.197-0.200 s. Commits:
  - 85594aa sdpa writes [b,l,h,e] so the transpose+reshape after it is a view; rownorm uses subgroup reductions.
  - d6d9b63 single-pass attention (online softmax; each lane keeps 32 output columns in registers): 0.48 -> 0.35 ms per call.
  - 8d59882 embedding, upsample_nearest1d, sum, cumsum (two-level chunked scan), repeat_interleave, and complex run on the GPU: 10 -> 4 flushes per decode. The 3 that remain (mul/max/Int on scalars) feed data-dependent shapes, so they need the value on the host.
  - a358b21 q/k/v merged into one gemm (concatenated weight cached, outputs are column views; last-dim split of a strided view stays a view); a contiguous copy is skipped when its only consumer reads strided input.
  - 2facb38 layer_norm -> transpose -> mish -> mask [-> +bias -> mask] fused into one rownorm dispatch. 2504 -> 2195 dispatches per decode.
  - 1abcd50 profile labels for fused ops; fixed rownorm/gemm_cm push-constant sizes; split-K gemm behind DEC_SPLITK=1 (no gain, off by default).
- Profile now (per decode): attention 39 ms; linears 46 ms (FFN 1024->256 is 128 us/call at ~3 TFLOPS; the gemm is bound by f32->f16 conversion of the tiles, not by the 48-workgroup grid, since split-K didn't help); convs ~45 ms; elementwise ~20 ms.
- Speech model: n-gram/prompt-lookup drafts accept only 7/129 (1-gram), 2/129 (2-gram), and 1/129 (3-gram) of greedy tokens, and there is no smaller draft model, so speculative decoding can't pay for its verify pass. Not implemented.
  int8 head not done: the speech-only head is 0.16 ms of ~3.4 ms per token (<5%). Keeping greedy output exact would also need an exact rescoring pass over the top candidates.
- e2e (logs/kt/kt_e2e_r3.log, GPU idle, perf high): warm median LM 0.444 s + decoder 0.196 s = 0.640 s, RTF 0.122-0.123 (round 2: 0.131). Cold RTF 0.27-0.28. Fork CPU RTF 0.763.
- Quality: decoder on fork tokens vs fork_redec.wav: corr 1.0000, log-mel 0.010 dB. e2e vs the fork decoder of our tokens (ours.wav): corr 1.0000, 0.009 dB. vs fork.wav: log-mel corr 0.968 (same LM near-tie). Tokens are identical to rounds 1-2.
- Gap: LM decode (3.4 ms/token, 0.44 s) alone is RTF 0.085. Reaching < 0.1 needs the decoder at about 0.08 s, or a faster LM step. Options: f16 pre-converted weights for the decoder gemms/convs (halves loads, no per-tile conversion), coopmat implicit-GEMM convs at acceptable precision, and LM step-latency work (it is under bandwidth: 975 MiB at 3.4 ms ~ 290 GB/s only for full weights, so check per-layer dispatch gaps).

## Kitten TTS 2: round 4, warm RTF under 0.1 (Oct 14)
- e2e (logs/kt/r7.log, GPU idle, perf high, 2 runs x 8 reps): warm LM 0.356-0.363 s + decoder 0.150-0.153 s = 0.506-0.514 s, **RTF 0.097-0.098** (round 3: 0.122). Cold RTF 0.253. Fork CPU 0.763.
- Greedy tokens are identical to rounds 1-3 (rs_e2e tokens unchanged; second text, 197 tokens, identical to host sampling).
- Quality: e2e vs fork decoder on our tokens: corr 1.0000, 0.009 dB. Decoder changes are bit-identical except the earlier ones (0.010 dB vs fork_redec). vs fork.wav: log-mel corr 0.968 (same LM near-tie).
- Decoder 0.196 -> 0.151 s:
  - 2b7297f: linear weights pre-packed to f16 once (RTNE), loaded straight into coopmat tiles; bit-identical; linears 46 -> 30 ms.
  - 8bc6f16: snake fused. 610b0f7: conv residual add in conv epilogue. 5e2278c: zero pad folded into conv. Dispatches/decode 2195 -> 1731.
  - 5694a3b: attention key-mask tile in shared memory (was a global load per score): 0.35 -> ~0.24 ms/call.
  - Coopmat convs (f16 inputs, f32 accumulate; push-size bug fixed in a later commit): -5 ms/decode at 0.05 dB log-mel / waveform corr 0.9998 vs 0.010 dB. Left off (DEC_COOP_CONV=1).
  - Tried and dropped: split-K gemm, 32-row attention tiles, f16-prepacked K/V for attention (no gain).
- LM 0.444 -> 0.358 s:
  - 2845811/cfac598: GPU greedy sampler (fork penalties, correctly-rounded division, first-max argmax), 8 tokens per submit (KT_MT), no host round trip per token.
  - d0d227b: steps after the stop token are skipped with VK_EXT_conditional_rendering (vkrt enables it when present): -20 ms.
  - 1c7198a: q/k norm + RoPE + KV store fused into attention (28 fewer dispatches/token); timing neutral, kept (KT_ROPEFUSE=0 disables).
- Remaining: LM step ~2.7 ms is near weight bandwidth for the FFN gemvs; decoder attention (~27 ms), scalar convs (~40 ms) and linears (~30 ms) are the next targets.

## Music generation: YuE2-3B port (crates/yue2), in progress, Oct 14 2026
- Survey: MiniMax Music 3 (2026-08-13) has open weights: Qwen3-8B global LM + 0.6B local LM + 2.4B flow-matching DiT + 123M Flow-VAE, about 23 GB bf16. YuE2-3B (m-a-p, CC BY-NC 4.0): a 3B AR+NAR MoT, 7.3 GB, plus an Oobleck VAE; it is newer and benchmarks above MiniMax Music 3 on its own WildSongBench table. ACE-Step 1.5 (Feb 2026; 2B/4B DiT + Qwen3 LM) is older. Chosen: YuE2-3B.
- Reference: ~/yue_ref (uv venv, CPU torch 2.10, yue2_infer 0.1.5), seed 1234, cot=full, 30.0 s of audio. abc 545 tok 148.9 s, semantic 750 tok 131.4 s, NAR 156.4 s, VAE 8.5 s: 0.067 s of audio per s. Dumps are in ~/yue_ref/out_ref.
- Port: weights in 256-byte-aligned PAD_BF16 arenas (safetensors offsets are only 2-byte aligned, which broke the zero-copy GEMM); prefill coop-matrix GEMMs; qimage norm/qk/ew; ye_flash = qi_flash plus a causal mask; ye_misc (NAR input, snake+im2col, polyphase convT im2col); host sampler; midpoint ODE; Oobleck VAE as im2col + GEMM.
- Parity so far: VAE decode of the reference latents: cos 1.000000, rel 6.9e-4, GPU 1.06 s for 30 s (the CPU reference takes 8.5 s). Backbone layers 1-2 at 300 tokens: cos 1.000000, rel 2.8e-4 vs fp32. Full-depth logits are still off (cos 0.77), and NAR v0 cos 0.88; per-layer bisection is in progress (yue2 dbg, YE_NL / YE_DN).
- Since about 6 PM the GPU has had ring timeouts/resets and another worker was holding the GPU at 100%, so all timings after 6 PM are suspect (one run ended in device lost).

### YuE2 2026-10-14 (late): parity fixed + gen numbers (all under gpu_lock.sh)
- Bug: prefix.npy already ends with abc+ABC_END+MUSIC_START; the bin appended abc again. Fixed (1546ff0).
- Parity: AR logits cos 0.999984 (codec argmax 30/31); v0 cos 0.999939; 32-step latents cos 0.999842; VAE cos 1.000000; end-to-end audio cos 0.998670.
- gen, 30 s audio, perf=high, seeds 1234/1235/1236: wall 65.6/60.6/62.8 s, median 0.478 audio-s/s (CPU ref 0.067, so 7.1x). Breakdown: abc ~28.5 tok/s, semantic ~26.8 tok/s (28 s), NAR 14.1 s, VAE 0.42 s.
- Sample: logs/ye/gen_1234.wav. pf_busy.sh is now lock-based (0fcd545), so no yue2 pattern is needed.

## Video generation: Kandinsky 6.0 Pro-distill (crates/k6), 2026-10-14

Model choice: Kandinsky 6.0 Pro-distill (Oct 6, MIT, 30B DiT, ~76 GB): newest open model that fits ~120 GB.
MiniMax-H3 rejected (license excludes the US); LTX-2.5 (Aug, 22B) older (fallback).
Test clip: 320x576, 49 frames (2.04 s @ 24 fps) + 44.1 kHz audio, seed 1234, 10 PiFlow steps, prompt "A red fox trotting through fresh snow...".

Ported (Vulkan, strix-rt): 60-block video/audio DiT + text blocks (qimage st zero-copy weights, prefill GEMM, qimage flash attention),
PiFlow scheduler, HunyuanVideo causal 3D VAE decoder (f16 channel-last activations, chunked im2col GEMM convs, GN, causal frame
attention; width-tiled at full res because the 256-ch full-res tensor exceeds the 4 GB buffer range), MMAudio VAE decoder + BigVGAN-v2
vocoder (1D im2col GEMM, convT col2im, fused anti-aliased SnakeBeta). Not ported: Qwen2.5-VL-7B + CLIP-L text encoders (reference embeddings used).

Parity (fixed seed, diffusers ROCm bf16 reference):
| stage | cos | rel |
|---|---|---|
| DiT step 0 video / audio grid | 0.999890 / 0.999676 | 1.5e-2 |
| DiT teacher-forced steps 1/5/9 (video) | 0.99981 / 0.99997 / 0.99993 | <=1.9e-2 |
| scheduler, all 9 transitions | 0.999999 | 1.66e-3 (bf16 rounding) |
| free-running 10 steps: video / audio latents | 0.976 / (see note) | 0.22 |
| VAE decoder crop 4x16x16 latent (fp32 CPU ref), every stage | >= 0.99999 | <= 8.6e-4 |
| audio VAE mel / vocoder wav (fp32 CPU ref) | 1.000000 / 0.99728 | 6.9e-4 / 7.4e-2 |
The free-running drift grows ~1.7x per step from bf16-level per-step differences (cos 0.9999 after step 3, 0.982 after step 8);
pixel PSNR of our 10-step clip vs the reference latents (both through our VAE) is 20.0 dB: same scene/composition, different fine detail.
Earlier "audio latents cos 0.036" was a reader bug (reference .npy is fortran_order), fixed in c75c448/0bca0d1.

Timing per clip (s, not perf-level-pinned, taken during other workers' jobs after 6 PM: treat as indicative):
| stage | strix-rt | ROCm reference |
|---|---|---|
| DiT 10 steps | 610.9 (61.4/step median) | 1169.6 (117.0/step) |
| video VAE decode | 20.9 (tiled) | > 1100 (diffusers tiled decode, still running at 18 min; 40 min timeout) |
| audio decode | 0.50 | pending |
| text encoders | (reference, 2.1) | 2.1 |

### K6 update 2026-10-15: text encoders + perf-pinned timings
Text encoders ported: Qwen2.5-VL-7B text path (GPU; qimage norm/qk/small/ew kernels + prefill GEMM, zero-copy BF16 from a
16-byte-repacked copy text_encoder_al, one submission per layer) and CLIP-L (CPU, own BPE tokenizer). Tokenizer ids match the
reference exactly (154 Qwen / 26 CLIP). Parity: Qwen prompt_embeds cos 0.999928 (rel 1.2e-2), CLIP pooled cos 0.999946.
The whole pipeline (prompt -> mp4) now runs on strix-rt with no reference components.

Perf level pinned high, medians (logs/k6/bench.log, logs/k6/refbench.log):
| stage | strix-rt | ROCm reference |
|---|---|---|
| text encoders | 0.22 (Qwen, GPU) + 0.074 (CLIP, CPU) | 1.41 |
| DiT, 10 steps | 456.0 (45.6/step) | 1070 (107.0/step, 3-step median x10) |
| video VAE decode | 19.7 (width-tiled) | 17.56 warm, tiled (not pinned); first call 2372 s of MIOpen kernel search |
| audio decode | 0.25 | not measured separately |
| total per clip (excl. load) | ~476 | ~1089 (+ audio) |
DiT 2.35x faster; whole clip ~2.3x. The reference VAE is ~1.1x faster than ours once warm.
