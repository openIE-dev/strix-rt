# Batched prefill (`crates/prefill`)

Batched prompt processing for strix-rt on the Radeon 8060S (gfx1151, RADV). It replaces the token-by-token decode loop
for the prompt. Status: **qwen38 (qwen35 arch) works and matches the decode path. lfm2 is not done yet.**

## Layout
- `shaders/pf_gemm.comp`: quantized-weight x f16-activation GEMM using KHR_cooperative_matrix (f16 x f16 -> f32,
  16x16x16, wave32). The weight format is a spec constant and covers every rtcore::qtype: Q4_K, Q6_K, Q5_K, Q8_0,
  IQ4_NL, IQ4_XS, Q3_K and IQ3_S, decoded by `pf_deq.glsl` (adapted from vkrt `q_any`).
  - Workgroup tile is 128 (m) x 128 (n) x 32 (k), with 8 waves each owning a 32x64 tile (2x4 accumulators).
  - Shared A/B tiles are double-buffered. Global A loads and the B dequant for step k+1 go into registers before the
    MMAs of step k, so there is one barrier per k-step.
  - m-tiles are scheduled fastest, so the 4 workgroups that share a weight row-tile at M=512 hit L2 together.
  - Residual accumulation is fused (flags bit0). y_col writes into a column window of a wider output, so
    [qkv|z|alpha|beta] and [gate|up] land side by side without copies.
  - Edge tiles use a guarded store staged through shared memory.
- `GemmOp` / `Pf::gemm` is the GEMM interface. One call covers the row range [row0, row0+rows) of a weight, which lets an
  NPU (crates/npu) take part of N while the GPU does the rest into the same Y via y_col.
- `pf_rmsnorm`: f32 residual to f16 normalized activations.
- `pf_gdn_conv` / `pf_gdn_convst`: batched causal conv1d + SiLU + q/k L2 norm, reading tokens before the chunk from the
  conv state. A separate pass then writes the final state.
- `pf_gdn_scan`: Gated DeltaNet recurrence. One workgroup per value head, each thread holds one state row in
  registers, and tokens are staged 16 at a time in shared memory.
- `pf_gnorm`: gated RMSNorm.
- `pf_rope`: per-head q/k norm + partial NeoX RoPE + f16 KV write at pos0+t.
- `pf_attn`: causal GQA flash-style attention over the f16 cache, 16 tokens x 1 head per workgroup, with the sigmoid
  output gate applied.
- `pf_swiglu`.
- `qwen35::prefill(pf, model, scratch, prompt)` is the function the model calls. It resets the recurrent state and runs
  prompt[..L-1] batched (chunks of `PF_M`, default 512, at any pos0). It writes kc/vc, the conv state and the
  DeltaNet state exactly where decode would. The last prompt token then goes through the model's own decode step, so
  logits, argmax and toks[L] come from the decode code and generation continues with `ctx.submit_repeated(step, n)`.
  Hadamard-rotated (Bonsai) GGUFs are refused for now.
- Embedding is done on the CPU with `gguf::dequant` (512 rows, ~ms).

## Verification (`pf_q38 <gguf> L n_gen`)
Random token prompt, compared with `Model::generate` (token-by-token) on Qwen3.8-27B-UD-Q4_K_M:

| L | logits max abs diff (max abs logit) | cosine | top-10 overlap | greedy continuation agreement |
|---|---|---|---|---|
| 4 | 0.0057 (11.02) | 1.000000 | 10/10 | 4/4 |
| 32 | 0.0043 (13.07) | 1.000000 | 10/10 | 4/4 |
| 256 | 0.0058 (12.01) | 1.000000 | 10/10 | - |
| 512 | 0.0034 (11.96) | 1.000000 | 10/10 | 16/16 |

The logits move by 2e-5 to 6e-5 between repeated prefills. That variation comes from the decode step's split-K
combine, not from the prefill.

## Performance (pp512, wall time including CPU embed, 8 submits, and the decode step for token 512)
See logs/pf_q38_512_final.log. The machine is shared with other workers' GPU jobs, so take the best of several reps.
- qwen38 Q4_K_M, final kernels, GPU otherwise idle: best 1571.7 ms -> **325.8 tok/s**. Another run gave 1585.4 ms -> 322.9.
  An earlier variant once hit 1485 ms (344.7) but did not reproduce.
- The final verified run (logs/pf_q38_512_final.log) overlapped another worker's q38 job: 2053 ms -> 249.4 tok/s.
- llama.cpp pp512 for comparison: 445.06 (Vulkan) / 331.59 (HIP).
- GEMM on real tensors with warm L2 (`pf_gemm`, M=512): 20-22.7 TFLOPS for the large layer weights.
- Time split from `PF_SKIP` runs: GEMMs ~1.25-1.35 s; DeltaNet scan ~55 ms; attention ~12 ms; conv, norms, rope and
  swiglu ~170 ms; fixed overhead (CPU embed, submits, decode of the last token) ~100 ms.

What's left to beat llama.cpp Vulkan:
- The GEMM (prefetch distance 2, or a larger wave tile without spills; it is at 216 VGPRs now).
- Fuse swiglu into the gate/up GEMM epilogue. The [M][2*ffn] f32 round trip is ~70 MB per layer.
- A dedicated small-N kernel for ssm_alpha/beta (N=48: ~0.26 ms each x 96).
- Batch the last token and use an M=1 lm_head instead of the decode step.

## Ternary activation quant (crates/ternary tern_quantx)
Now one 256-thread workgroup per token, with each wave owning whole 128-groups. The group amax is a single
subgroupMax, the input is read as vec4 and the output stored as packed int8x4. There is no barrier and no shared memory.
- K=17408, M=512: 276 -> 239 us.
- K=5120: 16.6 us.

At K=17408 the 35.6 MB f32 input no longer fits in the cache, so the kernel now runs at DRAM bandwidth (~186 GB/s).
The remaining cost can only be removed by having the producer (the swiglu epilogue) emit int8 + scales directly.

## Round 2 (Oct 14)
Benchmarks were run through `logs/pf_idle_run.sh`, which waits for the other workers' q38/npu jobs to finish.
Results are in logs/pf_round2_bench.log, all pp512 in tok/s. The run was idle at both start and end.

| model | strix-rt prefill | llama.cpp Vulkan, same session |
|---|---|---|
| Qwen3.8-27B Q4_K_M | 349.70 (best of 4: 2821/1464/1963/1828 ms) | 403.54 ± 16.93 |
| LFM2.5-24B-A2B Q4_K_M | 2068.21 (247.6 ms, stable) | 2258.78 ± 4.09 |
| Bonsai-2-27B PQ2_0 | 264.62 | n/a |
| Bonsai-2-27B PTQ1_0 | 256.20 | n/a |

Qwen timing varies from run to run (1.46-2.8 s for the same binary at idle). llama.cpp also varied in other
sessions (352.95 ± 73.81 in logs/pf_vs_llama_now.log), so compare the two only within one session.

Changes:
- int8 coop-matrix: `pf_peak` measures 52.3 TFLOPS f16 and 54.8 TOPS s8 for 16x16x16 KHR coop-matrix on gfx1151/RADV.
  int8 is not 2x f16 here, so a Q4_K->int8 rewrite was not done.
- The fused SwiGLU epilogue (`Pf::gemm_swiglu`, PF_SWI=1) is correct. Its gain was not consistent across sessions.
- Small-M GEMM tile variants: 16x256, 32x256 and 64x128. They are chosen automatically for M <= 64.
- PF_HEAD=1 (default): all L tokens are batched, followed by an M=1 lm_head and a CPU argmax that hands toks[L] to decode.
  Verified: L=32 cosine 1.000000, greedy 4/4.
- Bonsai (PQ2_0 / PTQ1_0, Hadamard-rotated ternary), `qwen35::Bon`:
  - Each ternary linear runs f16 -> f32 (`pf_h2f`), then the FWHT (with signs and, for ssm_out, the GDN v-group
    permutation), then the ternary per-token int8 quant, then `Pf::tgemm` (int8 coop-matrix ternary GEMM from
    crates/ternary).
  - The inverse FWHT is applied to the embeddings.
  - Verified against token-by-token: PQ2_0 L=32 cosine 0.999995, 4/4; PQ2_0 L=512 cosine 0.999995, 8/8; PTQ1_0
    L=32 0.999995, 4/4.
  - `Pf::tgemm` keeps the row-range contract (row0, rows, y_col), so npu::split::SplitLinear can take rows [0, nn).
    NPU_SPLIT=1 itself is NOT wired yet. SplitLinear::forward blocks and does its own rotation and quant, so each
    split linear needs a submit boundary.
- LFM2-MoE prefill (`lfm2moe.rs`, bin `pf_lfm`):
  - Batched shortconv as a parallel FIR plus a state kernel. hd=64 rope/attention.
  - The router runs as an f16 GEMM (PF_RF32=1 keeps the f32-weight kernel), followed by top-k selection.
  - Experts are grouped on the GPU (`pf_moesort`: counts, scatter, and tile tables), then: gather, a grouped GEMM
    (spec constant GRP, one dispatch per weight, 64-row tiles), SwiGLU, grouped down GEMM, weighted combine.
    There is no CPU sync per layer.
  - Correctness against token-by-token decode:
    - L=1: cosine 1.000000. L=2: 0.999995. L=8: 0.99984.
    - L=32: 0.995-0.997, greedy 4/4 and 8/8.
    - L=512: 0.997, top5 overlap 4/5, argmax differed on the random-token prompt.
    - The model is very sensitive on random prompts. Rounding the decode path's own activations to q8 (Q8EMU=1) moves
      it to cosine 0.98 against the same prefill. Running prefill one token per chunk (PF_M=1) gives the same result
      as fully batched.
  - PF_SKIP split at L=512: GEMMs ~210 ms, the rest ~35 ms.

## Round 3 (Oct 14): GEMM, variance, lfm2 real text

### GEMM
Measured with `pf_gemm` at M=512 on real Qwen3.8 tensors, warm L2. Some of these ran while another worker's q38
was running.

PF_PROBE=1 skips global A loads and B dequant. It shows the kernel structure's own ceiling.
- 128x128 tiles, A and B staged through shared memory: 28-30 TFLOPS with PF_PROBE=1, 17.5-22 TFLOPS real.
- Shared-memory row stride: LS4=5 (80 B) is best. LS4=6 gives 22.7 and LS4=4 gives 14.9 (PF_PROBE=1).
- wave64 (PF_SG=64, 4 waves) was worse with every tile shape: 22-23 with PF_PROBE=1, 14-16 real.
- MI=4 x NJ=4 per wave collapses to 8 TFLOPS (register pressure).
- A loaded directly from the f16 activation buffer into coop matrices (ADIR=1, now the default): structure ceiling
  29.8 -> 39.4 TFLOPS. Real ffn pair 17.5 -> 25.9.
  - Activation buffers are padded by 256 rows because whole A tiles are read.
- Tile 256(m) x 128(n), 8 waves stacked in m, each wave 32x128 (MI=2, NJ=8, WR=8; now the default):
  - All-tensor aggregate: 21.2 -> 26.6 TFLOPS (logs/pf_gemm_adir.log). The 128x128 tile with ADIR gave 21.2.
  - Q4_K attn_qkv 34.0, Q4_K ffn_gate 30.7 (contended), IQ4_XS ffn_down 24.2, Q6_K output 26.6.
  - Fused SwiGLU keeps 128x128 tiles: its gate/up pairing needs 2+ wave columns.
- IQ4_NL dequant now uses 5 aligned dword loads plus funnel shifts instead of 9 u16 loads.
- `pf_gemm` frees and reallocates buffers per tensor. That can produce a spurious NONDET (stale cached descriptor
  sets). The same tensor run on its own is deterministic, and the prefill pipeline is unaffected (L=32 cosine 1.000000).
- >35 TFLOPS was NOT reached on the large weights. What remains is B dequant cost. Each weight is decoded once per
  256-row m-tile, by one thread per row, with uncoalesced row reads.

### lfm2 on a real-text prompt (logs/pf_lfm_realtext.log)
The prompt is ~/llama.cpp/docs/build.md tokenized with llama-tokenize: 1702 tokens, truncated to L. `PF_TOKS=<ids file>`
selects it in `pf_lfm`.

| L | logits cosine | top10 overlap | argmax | max abs diff (max ref) | greedy agreement (16 new tokens) |
|---|---|---|---|---|---|
| 32 | 0.999711 | 10/10 | equal | 0.53 (20.7) | 15/16 |
| 128 | 0.999458 | 10/10 | equal | 0.45 (15.8) | 3/16 |
| 512 | 0.999612 | 10/10 | equal | 0.33 (22.7) | 4/16 |

On real text the last-token logits match closely. The random-prompt run showed cosine 0.997 and a flipped argmax;
that was prompt sensitivity, not a bug.

The greedy continuations later split at near-tied tokens. Decode then continues from a KV cache and conv state written
from f16 batched activations instead of the f32 per-token path.

### Variance diagnosis and one-session pp512 (logs/pf_session.log, logs/pf_mon_*.csv)
`logs/pf_session.sh` runs llama.cpp Vulkan and strix-rt prefill back to back at perf level auto, then high.
`logs/pf_gpumon.sh` samples sclk (hwmon freq1), the fclk/mclk DPM level, power and temperature every 100 ms. The
script restores `power_dpm_force_performance_level=auto` at the end, and it was confirmed as auto afterwards. The GPU
was idle at start and end; no other GPU processes ran.

pp512 in tok/s, with sclk avg/min (MHz) and average power during each run:

| run | auto | high |
|---|---|---|
| llama.cpp qwen38 | 357.59 ± 22.14 (2263/1184, 108 W) | 407.90 ± 2.41 (2663/2548, 113 W) |
| strix-rt qwen38 | 449.69, runs 1669/1138/1157/1168/1161/1161 ms (2016/773) | 452.53, runs 1131-1158 ms (2748/2573) |
| llama.cpp lfm2 | 2175.04 ± 28.99 (2189/1561) | 2252.28 ± 22.18 (2818/2703) |
| strix-rt lfm2 | 1950.69 (1389/713, 75 W) | 2000.95 (2856/2797, 71 W) |
| strix-rt Bonsai PQ2_0 | 316.77 | 318.59 |
| strix-rt Bonsai PTQ1_0 | 265.33 | 290.23 |

Findings:
- In auto the clock drops to 600-800 MHz between jobs. It takes the first run(s) to ramp back up, which explains the
  slow first rep (1669 ms).
- In auto, fclk often reads no active level and mclk sometimes too. In high, fclk is pinned at 2000 MHz and mclk at 1000 MHz.
- With high, both stacks tighten:
  - llama.cpp qwen38: ±22 -> ±2.4.
  - strix-rt qwen38: 1131-1158 ms across 6 runs.
- The GPU runs at 99-104 C and 100-140 W in every mode. sclk sags to ~2.5-2.7 GHz under the heaviest GEMMs, so
  thermal and power limits remain a source of noise.
- The 1.9-2.1 s qwen runs earlier in the day overlapped other workers' q38/npu jobs, which ran almost continuously.
  The idle filter now also catches `npu/target_*` and `pf_*` binaries.
- lfm2 prefill only averages 1389 MHz in auto. Its short, bursty dispatches keep DPM from ramping. In high it gets
  2856 MHz, but the gain is small (1951 -> 2001), so lfm2 is latency/dispatch bound, not clock bound.

Result: Qwen3.8 pp512 452.53 (high) / 449.69 (auto) vs llama.cpp 407.90 / 357.59 in the same session, which beats
llama.cpp. lfm2: 2000.95 vs 2252.28 (89%).

## Round 4 (Oct 14)
All timings below were taken at perf level high and restored to auto afterwards, unless marked otherwise.

- T8 repack (`repack_t8`, `Pf::gemm_t8`, `T8=1 pf_gemm`): weights converted at load time into 128-row x 32-k int8
  chunks with one f16 scale per (row, 32-k block). Each tile's k-slab is then a single contiguous, coalesced 4 KB read.
  - Memory: 34/32 B per weight, a copy of about 1.9x the IQ4/Q4_K size (94.7 MB vs 47-50 MB per ffn tensor).
  - Repack time: 0.02-0.03 s per tensor (all threads). Error vs original format: 1.7e-3 to 7e-3 relative.
  - Speed (logs/pf_gemm_t8.log, auto): ffn_down 17.8, ffn_gate IQ4_XS 31.5, ffn_gate Q4_K 27.2 TFLOPS. The
    original-format kernel does 24-26 / 30-31 / 27-30. No gain, so it is not wired into prefill.
  - Conclusion: uncoalesced quantized reads are not the cap. The int8 copy doubles weight traffic, and the per-tile
    convert-to-f16-and-store-to-shared-memory work remains.
- Narrow-output heuristic (128x128 tiles when the 256x128 grid is under 160 workgroups) made ffn_down slower (24.5 vs 26).
  Kept off (PF_MINWG=0).
- Ternary GEMM decode now writes packed int8x4 words to shared memory instead of single bytes (`pf_tgemm`).
  - Correctness at L=32: cosine 0.999992, greedy 4/4 for both PTQ1_0 and PQ2_0.
  - pp512 at high (logs/pf_r4_b.log): PQ2_0 318.59 -> 328.36; PTQ1_0 290.23 -> 291.77 (no real change).
  - The PTQ1_0 5-trit unpack is not the limiter at this tile size.
- Same run: Qwen3.8 pp512 470.75 tok/s (runs 1169/1094/1087/1093 ms).
- lfm2 at high: 254.8 ms (2009.44 tok/s).
  - GEMMs: about 207 ms.
  - Routing, gather, swiglu and combine: about 14 ms.
  - Attention plus route-select: about 10 ms.
  - Everything else: about 24 ms.

### Round 4 (part B): lfm2 dispatch reduction (perf level high, best of 4, pp512)
- Grouped-MoE tile rows sweep (logs/pf_r4_lfm_tiles.log): GBM=64 (old default) 253.7 ms / 2018 tok/s; GBM=32/32 244.4 ms / 2095; GBM=32 + MSWI=1 242.6 ms / 2110; GBM=16 274.3 ms; PF_MINWG=160 252.2 ms (≈noise). GBM=128 was mis-mapped to the 256-row tile (fixed: 128 -> 0x2000).
- New default GBM=GBM2=32.
- New pf_combnorm kernel: fuses MoE combine + residual add + next-layer attn RMSNorm (one dispatch instead of 2, PF_COMBNORM=1 default). L=32 greedy agreement 1/1 with ref for both settings.
- A/B (logs/pf_r4_comb.log, pf_r4_comb_ab.log; another GPU job was active during part of these runs, so outliers dropped): COMBNORM=1 240.8–241.7 ms (2118–2126 tok/s) vs COMBNORM=0 244.9 ms (2091 tok/s).
- lfm2 now 2126 tok/s best vs 2009 at start of round 4; llama.cpp 2252 (round 3, high). Gap ~6%.

## Round 5 (Oct 14): cooperative K-slab dequant, fused expert select, lfm2 tiles
Perf level high for all timings, restored to auto afterwards. Idle gating uses logs/pf_busy.sh, which anchors on argv[0] so waiting wrappers dont deadlock each other.
- **Cooperative K-slab (PF_KS, spec constant 11).** Each k-step is now BK = 32*KS. All 256 lanes dequantize the BN x BK weight slab into shared memory once, and every subgroup reuses it. Lane t decodes sub-block t/BN of row t%BN.
  - Before this, 128-wide tiles left half the lanes idle during decode.
  - KS=2 also halves the barriers per K. It is the default for BN=128 tiles.
  - Accumulation order is unchanged, so logits are bitwise equal to KS=1.
  - pf_gemm M=512 (logs/pf_r5_ks.log): q38 aggregate 31.5 -> 33.6 TFLOPS. Best per tensor is 37.1 (ffn_gate IQ4_XS); ffn_down goes 28.1 -> 35.9. lfm2 dense aggregate 23.9 -> 25.5.
- **pf_moesel (PF_MSEL=1).** lfm2 top-k expert selection is fused into the grouping/sort pass: one 1024-thread dispatch replaces rsel + moesort. Tokens and logits are identical. Speed is unchanged (rsel was cheap).
- **lfm2 tiles.**
  - Gate/up grouped GEMMs use 32x128 tiles (PF_G32N, so KS=2 applies), with fused SwiGLU on, down at 64 rows, and 24 layers per submit.
  - When there are >= 64 rows per expert (PF_WIDE_RPE, e.g. pp2048), 32x256 tiles are used for gate/up and down instead.
  - PF_HYB (mixed 64/32-row tiles) was slower and is left off.
- **Note:** PF_SKIP probes that leave buffers stale run faster partly because the data is zero, which means lower power and higher clocks on a power-capped GPU. Don't read them as kernel cost (logs/pf_r5_brk.log).
- lfm2 greedy continuation is identical before and after at L=512 and L=2048. At L=2048 the random-token prompt disagrees with token-by-token decode both before and after; that predates this round.

Results (logs/pf_r5_final.log, pf_r5_wide.log, pf_r5_s2.log, pf_r5_session.log):

| model | llama.cpp pp512 / pp2048 | strix before | strix after |
|---|---|---|---|
| lfm2 | 2271 / 2276 | 2061 / 2382 | 2293-2328 / 2662 |
| qwen3.8 | 392 / 393 (contended; 449 / 436 in pf_r5_session) | 473 / 376 | 533-539 / 459-483 |
| Bonsai PQ2_0 (prism fork) | 305 / 296 | 282 / 282 (median, noisy) | 279 / 309 (no real change: ternary tgemm path) |
| Bonsai PTQ1_0 | 205 / 205 | 288 / 286 | 292 / 284 |

## Round 6 (Oct 14): L=2048 correctness fix, ternary through the f16 coop-matrix GEMM
- **Bug:** batched prefill at L=2048 (one chunk) disagreed with token-by-token decode (argmax 816 vs 207).
  - It was correct with PF_M <= 1536, or with 2048 split into chunks (logs/pf_r6_l.log).
  - Cause: vkrt `Ctx::dispatch` split more than 65535 workgroups into a (gx, gy) grid. That grid launched gx*gy >= n workgroups.
  - Shaders that read only gl_WorkGroupID.x ran some workgroups twice. In-place pf_rope64 then rotated q/k twice: n*(heads+kv) = 2048*40 > 65535.
- **Fix:** dispatch now issues exactly n workgroups as 1-D vkCmdDispatchBase chunks (pipelines are created with DISPATCH_BASE).
- **Verified, L=2048 and 4096 in one chunk** (logs/pf_r6_chk.log): lfm2, qwen3.8, Bonsai PQ2_0 and PTQ1_0 all match argmax and greedy 8/8.
  - Cosine: q38 1.000000, Bonsai 0.99999, lfm2 0.9992 / 0.9973.
- **Ternary (PF_TF16=1, default without NPU_SPLIT):** PQ2_0 / PTQ1_0 weights are decoded to f16 inside the cooperative K-slab GEMM (pf_gemm; new deq32 decoder for PTQ1_0).
  - Rotated layers feed fwht output through pf_f2h; unrotated layers read s.a directly.
  - There is no int8 activation quantization, so it is more accurate: L=512 max|diff| 0.0685 -> 0.0618, greedy 8/8.
  - The int8 pf_tgemm is kept for NPU_SPLIT and the lm_head.

Final table (logs/pf_r6_table.log, one session, perf level high, best of 6; llama.cpp Vulkan, Bonsai via prism fork):

| model | llama.cpp pp512 / pp2048 | strix-rt pp512 / pp2048 |
|---|---|---|
| qwen3.8 Q4_K_M | 446.7 / 431.1 | 528.1 / 474.4 |
| lfm2 Q4_K_M | 2247.3 / 2205.0 | 2286.0 / 2604.3 |
| Bonsai PQ2_0 | 309.7 / 304.2 | 513.5 / 437.6 |
| Bonsai PTQ1_0 | 208.8 / 208.1 | 396.2 / 350.5 |

## Round 6 (Oct 15): LFM2.5 expert GEMM after PF_HYB=1 default
Alternated A/B runs with 15 s cool-downs (logs/lfmx/spd*.log). Baseline: pp512 ~2320-2340, pp2048 ~2411.
- No existing knob beats the defaults:
  - PF_MSWI=0: 512 -1%, 2048 -1%.
  - PF_WIDE_RPE=16: 512 -3%. PF_WIDE_RPE=1000: 2048 no change.
  - PF_KS=1: 512 -12%. PF_KS=2: same as the default.
  - PF_HYBM=128 (new switch, 1ee70c2; 128-row full tiles under hyb): 512 -0.8%, 2048 -0.1%.
  - PF_SG=64 panics at lib.rs:102 (sg 64 is not a supported tile config).
- llama.cpp ROCm MMQ, Q4_K on RDNA3.5 (mmq-config-rdna3-5.cuh): 128 weight rows x J tokens per tile, with J = 16..128 in steps of 16 picked per launch. 256 threads, 2 workgroups/CU. One k-step = 256 weights (a whole Q4_K superblock), so one barrier and one scale decode per superblock. Activations are q8_1 and weights int8 in shared memory, using int8 WMMA.
- Our kernel stages f16 weights in shared memory (2 B/weight, double-buffered). For BN=128, BK is capped at 64 (KS=2) by the 64 KB LDS limit: 2*BN*(4*KS+1)*16 B; KS=4 would need 69 KB. So it does 4x more barriers and sub-block scale decodes per K than MMQ.
  - I prototyped a multi-sub-block-per-lane decode (KR) and dropped it: deeper K-slabs do not fit in LDS with f16 staging.
- Next lever: stage packed weights (int8 or raw q4 + scales) in LDS so a 256-deep k-step fits. MMA rate is no help (s8 54.8 vs f16 52.3 TOPS), so the gain has to come from fewer barriers and less decode/LDS traffic per FLOP. The T8 experiment (Round 4) changed only the global-memory layout, not the LDS depth.
- Caveat: the PF_SKIP=1 expert-GEMM share (~80%) overstates the GEMMs' cost, because skipped (stale/zero) data raises clocks (Round 5 note).

## Nemotron-H MoE (crates/nemotron) and the shared packed-LDS expert kernel (Oct 15)
- Nemotron 3 Nano 30B-A3B prefill (`nemotron::prefill`) uses `Pf::gemm_grouped` for its experts: 128 experts, top-6,
  no gate (up -> squared ReLU -> down). up_exps Q5_0 [K=2688 -> 1856], down_exps Q8_0 or Q5_0 per layer [K=1856 -> 2688].
  32-row tiles from its own grouping pass (`nh_psort`, same gt/g layout as pf_moesort, up to 256 experts) and pf_gather / pf_moecomb.
- Q5_0 (qtype 15) was added to pf_deq.glsl for these weights; a packed-LDS kernel needs Q5_0 and Q8_0 at K = 1856 / 2688 (K % 64 == 0, not % 256).
- Baseline (one session, Oct 15): pp511 1267 tok/s vs llama.cpp Vulkan pp512 1676.
- Plan: Nemotron switches to the shared packed-LDS expert kernel once it lands (call sites: the two gemm_grouped calls in
  crates/nemotron/src/prefill.rs, Kind::Moe). Until then only tile-size tuning on the Nemotron side; no edits to pf_gemm.comp.
- Oct 16 tile sweep (logs/nh/tile30.log, pp511, 30B): PF_M 512 x PFG_M 16/32/64 = 1175/1282/1267 tok/s; PF_M 256 = 922/993/912. Defaults (512, 32) kept; continuation ids identical in all six. llama.cpp pp512 = 1673.
- The landed shared kernel `pf_qgemmh` (d6d56a7) is Q4_K-only; Nemotron 30B experts are Q5_0 (up) / Q8_0 (down) at K = 2688 / 1856, so it does not apply yet. Need Q5_0/Q8_0 packed-LDS variants (K % 64) before switching the two gemm_grouped call sites.
- Oct 16 (Nemotron worker): pf_qgemmh.comp has uncommitted TY=3 (Q8_0) / TY=15 (Q5_0) support in the working tree (packed-LDS worker, in progress).
  To avoid collisions the Nemotron worker does NOT edit pf_qgemmh.comp / qmm.rs / pf_gemm.rs. Once TY 3/15 is committed with a grouped entry point
  (Pf::qgemmh_grouped taking the weight type), Nemotron switches its two gemm_grouped call sites behind PF_QMM=1, checks parity (pp511 continuation ids
  and teacher-forced agreement vs the f16 path), then benchmarks. Ask: please note here when TY 3/15 is committed and its K/mrows constraints (K=1856 is not % 256).

## Gemma 4 26B-A4B grouped MoE on pf_qgemmh (crates/gemma4, Oct 16)
- Call sites: crates/gemma4/src/pf.rs grouped-MoE block (two qgemmh_grouped calls: gate_up, Q4_K, K = 2816 = 11 x 256) and the down GEMM on the prefill-only Q8_0 repack
  of down_exps (Q5_1 in the gguf; K = 704, 704 % 32 == 0). Both go through `Pf::qgemmh_grouped`. The tile rows come from `Pf::qgemm_rows()`
  (g4_msort builds the 128-expert grouping with that bm). Gemma 4 only reads the shared kernel; it does not edit pf_qgemmh.comp / qmm.rs / pf_gemm.rs.
- G4_QMMH defaults to on (0 = previous `gemm_grouped` f16 path). One session, perf high: pp512 1450 -> 1536 and pp2048 1390 -> 1477 tok/s
  (llama.cpp 1717 / 1627). Teacher-forced 63/64 vs llama.cpp Vulkan, identical to the f16 path (the miss is the retokenized tail).
- To the packed-LDS / Nemotron workers: if the qgemmh_grouped signature or the qgemm_rows() contract changes, gemma4 pf.rs needs
  the same edit. Please note it here. A native Q5_1 TY would let gemma4 drop the ~8 GB Q8_0 down repack.
- Oct 16 (Nemotron worker): Nemotron prefill now calls Pf::qgemmh_grouped (aba8338, TY 15 up / TY 3|15 down) behind PF_QMM=1, tiles = qgemm_rows() (64).
  30B results (logs/nh/qmm30.log, perf_high): pp511 f16 1285 vs qgemmh 1186 tok/s; 829-tok prompt 1158 vs 1035. qgemmh is ~8-11% SLOWER here.
  Parity: 829-tok continuation 48/48 identical; p512 continuation diverges at token 7 (98473 vs 54570), likely a near-tie (kernel not bitwise, 5.8e-4 rel).
  Default stays f16 gemm_grouped. Probable cause: K=1856 (down) / 2688 (up) are not % 256, so every row ends in a masked partial 256-step, and N=1856 is not % 128
  (last N tile 50% idle). For the packed-LDS worker: a 64- or 128-deep k-step variant for K % 256 != 0 and a BN=64 tile would likely fit these shapes.
- Oct 16 (packed-LDS worker): pf_qgemmh owner notes.
  - Spec variants (f0409b5): PF_QH_NJ (BN 64/128), PF_QH_KD (k-step 64/128/256) for Q8_0/Q5_0; PF_QH4_NJ for Q4_K. d2a170a adds PF_QH_WR=1 (1x8 waves, B unpacked once). Defaults are unchanged (NJ 2, KD 8, WR 2); qgemm_rows() and the qgemmh_grouped signature are unchanged.
  - **Bug window:** builds of d2a170a before b5de625 ran the shader with the wrong wave layout. Q5_0/Q8_0 results were wrong (6.6e-1 rel) and Q4_K was likely affected too. Rebuild, and discard any pf_qgemmh numbers taken in that window, including the logs/qmm/t9/t10 NJ/KD sweep (another worker wrote it into logs/qmm, which this worker uses; please use your own log directory).
  - Single-tensor results (M=512, f0409b5, logs/qmm/t8.log): no NJ/KD variant consistently beats the f16 pf_gemm on the Nemotron experts. Q8_0 down 12.2-14.6 vs 13.3-14.9 TF (KD=2 best); Q5_0 up 12.1-14.3 vs 15.7-19.6 TF.
  - Diagnosis: with WR=2, every B fragment is unpacked (int8 -> f32 -> scale -> f16) by 2 waves for only 2 MMAs each, so the kernel is VALU-bound. That is why PF_QH_WR=1 was added; it is being measured.

## pf_qgemmh on Nemotron 30B expert shapes (Oct 16, logs/qmm/t8-t10)
`pf_gemm` QMMH=1, blk.1 experts, M=512, one dense expert slice (Q5_0 up K=2688, Q8_0 down K=1856), each repeat its own
locked session. Medians (us), f16 pf_gemm vs packed pf_qgemmh:
- t9 (5 repeats, baseline kernel): sweep over (NJ,KD) = 2/8, 2/2, 1/8, 1/4, 1/2 in logs/qmm/t9.log.
- t10 (A/B, same session, old vs. Q8_0/Q5_0 no-offset shader): up f16 237-241 vs qgemmh 330-426 (never faster);
  down f16 290-312 vs qgemmh 311-348 (NJ=1 KD=4 closest, 324-328 vs 290-312; still 4-12% slower).
- No (NJ,KD) beats f16 on either shape. The f16 path stays the Nemotron default (prefill.rs PF_QMM is opt-in, default 0).
- Skipping the `ms` offset load/subtract/store for Q8_0/Q5_0 (no dmin) is bitwise identical (same differing-count and max
  error) and gave no measurable speed change (down 317 vs 318, up 396-426 vs 402-419).
- Caveat: NJ=2 KD=2 reports max error 0.73 in pf_gemm: that config is numerically wrong and must not be used; the Q5_0
  `up` check also printed 6.6e-1 in this sweep (earlier t8 run: 4.1e-4), unresolved (suspect pf_gemm per-tensor buffer
  reuse, see NONDET note above), so only the timings are used here.
- Cause: with MI=2 each wave converts int8 -> f32 -> *scale -> f16 for every B fragment for only 2 MMAs, and the dense f16
  kernel reaches 21 TF on these shapes. A real win needs conversion amortised over more tokens or done once per k-step
  into LDS shared by all wave rows; not done.

## Nemotron 3 Nano 30B-A3B prefill (Oct 16, crates/nemotron, logs/nh/)
pp511 on p512.ids, wall GPU time of `nh` with `PF=1`, best-of-5 reps per session, Q4_K_M. llama.cpp Vulkan reference 1687 tok/s.
- Start of day: 1122 tok/s. Per-op profile (`PFPROF=n`, timestamp after every barrier; it sums to less than the PF=1 wall time
  and shifts the per-op shares, so use it for ranking only): expert up 76 ms, Mamba scan 68 ms, expert down 56 ms,
  router+shared-up 43 ms, shared-down+relu2 28 ms, ssm_in 24 ms, ssm_out 16 ms, attention total ~17 ms.
- Mamba scan was 64 workgroups (one per head) with 3 barriers per token. `NH_PSPLIT` (default 8) splits each head's state rows
  over several workgroups with identical per-element arithmetic: scan 67.7 -> 22.7 ms, pp511 1122 -> 1327 tok/s (commit 01938b3).
- Router (`nh_psel`): each workgroup re-read the 128 x 2688 f32 gate weights per token. `NH_PSEL_TB` (default 4) handles 4 tokens per
  workgroup with the same summation order: pp511 1213 -> 1356 tok/s in one session (TB=2 1313, TB=8 1218) (commit a6393d0).
- Parity for both: generated ids (48 tokens, PF=1) identical for every setting on 6 prompts (p512, long, p30, div30, 200- and 90-token
  prefixes of p512).
- Grouped expert tile height (`PFG_M`): 32 ~1352, 64 ~1362 (within noise), 16 ~1060, 128 ~1165, 32x256 (0x8020) ~1295 tok/s. 32 stays
  default; ids identical for 32/64/128/0x8020 on the same 6 prompts. A first run of 0x8020 reported 1510 tok/s: that was wrong, the
  tile table was built with the raw 0x8020 height and dropped tiles (ids changed). Fixed (`gm & 0xff` for the sort and `maxt`);
  the real number is the slower 1295.
- Now ~1355 tok/s, 80% of llama.cpp. Remaining: expert up/down GEMMs, shared expert GEMMs, ssm_in/out. The packed pf_qgemmh does not help (see above).
- Q5_0/Q8_0 check: `pf_gemm` vs CPU dequant, Nemotron blk.1: f16 path 1.8e-4 (Q5_0 up), packed 1.1e-4; the earlier 0.64 "error" was the harness
  reading wrong rows for stacked expert tensors (fixed, 0654c00). The earlier NONDET on ffn_up_shexp does not reproduce when the tensor is run alone.

## Gemma 4 26B-A4B prefill re-bench and router-logits (Oct 16, logs/g4/rebench_*.log, rlog*.log)
- Same method as the comparison table (`g4_run G4_QMMH=1 G4_PF=1 rand:L 2`, perf level high, 5 reps): pp512 1535-1548 (table 1536.2), pp2048 1471-1473 (table 1477.0).
  llama.cpp Vulkan same session: pp512 1718.98 +- 6.22, pp2048 1631.3 +- 3.85. The table stands; gap is -10%.
- `G4_PFPROF=1` (per-label timestamp profile, first chunk): sum 306 ms = run() GPU time. Expert up 96.8 ms (31.6%), expert down 63.3 (20.6%),
  router segment (router logits + top-k) 29.2 (9.5%), qkv 25.5, attn_o 23.2, SWA attention 19.5, dense ffn gate/up 12.4 + down 9.9.
- Router logits with TB columns per workgroup (`G4_RLOG_TB`, g4_rlog.comp): generated ids identical to the 1-column kernel for TB 1-5 on 6 prompts, but no gain:
  segment 30.5 ms (TB 2), 42 (3), 37.5 (4), 35 (5) vs 30.2 (1); pp512 1541 / 1490 / 1511 / 1524 vs 1541. Not weight-read bound; default stays 1.
  (An earlier TB=2 "1632 tok/s" result was a host bug that dispatched the 1-column kernel on a TB-sized grid; it changed the ids.)

- Nemotron dense GEMM tile (`PF_MINWG` = use 128x128 tiles when 256x128 gives fewer workgroups than the threshold; narrow outputs N=2688/3712 give only 42/58 workgroups at M=511):
  pp511 1337-1375 tok/s for thresholds 0/64/100/128/200/400 (2 rounds x 5 reps), no difference beyond noise; ids identical on 6 prompts for 0/128/400. Default stays 0. logs/nh/minwg.log.

## Nemotron prefill under perf level high, and what the expert GEMMs cost (Oct 16, logs/nh/high.log, skiph.log, bw.log)
- Under `tools/comp/perf_high.sh` (same condition as the llama.cpp numbers) the same binary gives pp511 1469 tok/s (6 reps, +-1), not ~1355: the earlier
  sessions ran at the auto power level. Earlier per-session comparisons stay valid (same level within a session), the absolute figure was low.
- Timestamp profile (`PFPROF`) sums to 233 ms against 348 ms wall GPU time, so per-op shares from it under-count. Skip-mask differences are polluted too
  (skipping rmsnorm "saves" 112 ms because the GEMMs then read zero activations and run faster), so only skips that keep real data flowing are trusted:
  skipping the expert GEMMs (`NH_SKIP=32`) gives 3825 tok/s = 133 ms, so experts cost 215 ms of 348 (62%).
- Expert cost versus prompt length (experts = wall minus skip-32 wall): 89 tok 121 ms, 199 tok 149 ms, 511 tok 215 ms. Weights read per prefill: Q5_0 up 10.1 GB +
  Q8_0 down 15.6 GB = 25.7 GB; at ~212 GB/s that is ~121 ms, exactly the 89-token cost. So at short prompts the expert GEMMs sit on the DRAM floor; at 511 tokens
  there are ~94 ms of compute/dequant above the floor (padded 32-row tiles: ~3066 routed rows over 128 experts = ~24 rows/expert; padded MMA work at peak ~76 ms,
  dequant and imperfect overlap on top). Closing that is the way to pass llama.cpp (1687 tok/s needs the experts at ~130 ms). Tile heights 16/64/128 and 32x256 were
  measured (see above) and none helps; the packed LDS kernel is slower. What is left is a different kernel structure (not tried): bigger per-expert M via
  expert-sorted batching across layers is impossible (layers are sequential), so the lever is dequant/MMA overlap inside the 32-row tile.

## Expert GEMM probe: dequant is not the bottleneck (2026-10-16, perf_high, 5 reps, pf_gemm blk.1 slices, 512 rows dense)
PF_PROBE bit0 = no global loads/dequant, bit1 = no MMA. Q5_0 up / Q8_0 down:
- full kernel: up 217-221 us (~23 TF), down 263-281 us (~19 TF)
- MMA only (PROBE=1): up 192-237 us, down 251-272 us  (same as full)
- load+dequant only (PROBE=2): up 62-92 us, down 65-109 us
- neither (PROBE=3): ~38-48 us / ~54-57 us (launch/epilogue floor)
Dequant+loads already overlap fully under the MMA via register prefetch; a cheaper-dequant (packed-f16 magic-number) path cannot help, so it was not built. The limit is the coop-matrix MMA issue/LDS-fragment-load rate (~23 TF dense). Remaining Nemotron expert cost is MMA tile efficiency/padding (24 rows/expert), not dequant.

## Expert GEMM register blocking / LDS layout sweep (2026-10-16, perf_high, 5 reps, dense 512 rows, blk.1 slice, us up/down)
Default MI2 NJ8 WR8: 204-234 / 262-280. MI4 NJ4 WR4: 397-415 / 508-519; MI4 NJ8: ~910 / ~800; MI2 NJ16: ~1300 / ~1170 (register spills); MI1 NJ8: ~300 / ~290; MI2 NJ4 WR4: 263-280 / 246-269; MI4 NJ2: 248-267 / 231-292. LDS row stride (uvec4) 9/11/13: same as default; 16 (bank conflicts): 3x slower; ADIR=0: slower (267-313 / 357-376); KS=1: 227-251 / 286-317. The shipped tiling/layout is already the optimum of the explored space; larger blocking only spills. Dense single-slice microbench has 30 (up) / 42 (down) workgroups on 40 CUs, so it understates the grouped kernel. No change made.

Note: MTP verify cannot use the batched prefill path (109/125/132/135 ms for 1-4 tokens); the decode-style 2-column verify is used instead (see comparisons.md, Super MTP section).

## Running Nemotron 3.5 Super (batched prefill + MTP decode)
`ZEROCOPY=0 target-n35/release/n35 <super.gguf> "<comma ids>" 128` : batched prefill (PF=1, `PF_M` chunk, default 512; 436 tok/s at pp512) then MTP speculative decode (~26 tok/s vs 17.7 plain, output identical). `MTP=0` for plain decode, `PF=0` for token-by-token prefill. See docs/comparisons.md (Super MTP sections) for details and measurements.

## Nemotron Nano expert GEMM: hybrid 32/16-row tile table (2026-10-17, logs/nh/hyb.log, hyb_one.sh)
Idea: remove the tail padding of the padded 32-row tiles. nh_psort writes two tile tables: full tiles of 32 rows, and a second table with one 16-row tile
for each expert whose remaining rows are <= HYB (PFG_HYB=16 or 8; default 0 = off, old behaviour). The 16-row tiles run as a second grouped dispatch
(0x200 variant) with no barrier between the two. Result under perf_high, pp511, 3 rounds x 5 reps, interleaved: HYB=0 1468.6-1469.6, HYB=16 1466.3-1468.4,
HYB=8 1471.8-1474.5 tok/s (+0.3%, i.e. noise). Generated ids identical to HYB=0 on p512 (48 tokens) and p30. Kept as an opt-in switch, default unchanged.
Reason it cannot help: padding only changes for rows 1-16 above a multiple of 32; the extra dispatch and duplicated weight dequant for the 16-row tiles cancel the saved MMA work. The gap to
llama.cpp (1687 tok/s, 87%) is unchanged; with dequant, loads and tile sizes all ruled out (sections above), the remaining lever is a different MMA/LDS structure, not tiling.

## Gemma 4 26B-A4B prefill: pf_qgemmh wave layout / k-step tuning (Oct 17, logs/g4/pp_sweep*.log, pp_final.log, pp_par.log, ppe/)
- Sweep (perf_high, one locked session, 5 reps pp512, same binary, env only): base 1546 | f16 gemm_grouped 1465 | PF_QH_NJ=1 1579 | PF_QH4_NJ=1 1549 | PF_QH_WR=1 1580 | PF_QH_KD=4 1597 |
  KD4+NJ1 1610 | **KD4+WR1 1621** | KD4+WR1+NJ1 1623 | KD2+NJ1 1611 | KD4+NJ1+Q4K-NJ1 1609. (First rep of each session is a cold outlier and is dropped.)
- Shipped: gemma4 pf.rs sets PF_QH_WR=1 and PF_QH_KD=4 as process-local defaults when unset (07134e2). Shared defaults in qmm.rs are unchanged (Nemotron/LFM2 unaffected);
  PF_QH_WR=2 PF_QH_KD=8 reproduces the old config.
- Final A/B (6 reps each): pp512 1543.9 -> **1621.5** tok/s (+5.0%), pp2048 1472.4 -> **1539.9** (+4.6%). llama.cpp Vulkan 1719 / 1631 (earlier same-method run): gap now -5.7% / -5.6% (was -10% / -10%).
- Parity: teacher-forced vs llama.cpp on 26bc prompt 62/64 with identical mismatch list (positions 31 near-tie margin 0.064, 63) and identical greedy ids for base, WR1+KD4 and WR1+KD4+NJ1. (The 26b prompt file in logs/g4 is a stale/garbage pair, 66/78 in all configs incl. base: not used as a gate.)
- Prefill energy (new, logs/g4/pp_energy.sh: differential of 20 x pp2048 vs 20 x pp512 processes, energy recorder 20 Hz, perf high, 30720 tokens; load cancels): gfx 0.0406 -> **0.0342 J/token (-16%)**, socket 0.0878 -> 0.0852 J/token (-3%). Single run per config; no llama.cpp prefill energy measured yet.
- Oct 17 follow-up (logs/g4/pp_sweep3.log, pp_mi1.log): PF_QMM_MI=1 (32-row expert tiles) on top of WR1/KD4: pp512 1621 -> **1637**, pp2048 1540 -> **1551** tok/s (4-5 reps); with WR=2 MI=1 is worse (1576); KD=2 equals KD=4. Greedy ids md5 identical and forced parity 62/64 with the same mismatch list. Now the gemma4 process default (MI=1, WR=1, KD=4). Gap to llama.cpp Vulkan: -4.8% pp512 (1719), -4.9% pp2048 (1631).

## Expert GEMM wave-tile shape for the 32-row grouped tile (2026-10-17, logs/nh/g32c.log)
Hypothesis: the 32x128 grouped tile (MI1 NJ2 WR2, 16x32 per wave, 1.5 LDS fragment loads per MMA, B decoded redundantly by two wave rows) is LDS-load bound, so a wave
tile of 32 rows (WR=1, MI=2) that shares each B fragment over two MMAs and decodes B once per tile should help. Opt-in PF_G32C: 1 = MI2 NJ2 WR1 (32x256), 2 = MI2 NJ1 WR1 (32x128, 32x16 per wave).
Nano 30B pp511, perf_high, 2 rounds x 5 reps (box was slower than the 1469 session: baseline 1425-1426): PF_G32C=1 1409-1413 (-1%), PF_G32C=2 1433-1439 (+0.6..+0.9%, within what I would call noise-plus).
Default unchanged. Not run on Gemma (the gain is too small to matter). The persistent queue, gate+up fusion and split-K ideas were not built: the earlier probes show the kernel
is bound by coop-matrix issue/LDS fragment loads, not by launch, dequant or load balance, and the grouped experts already run ~1000 workgroups per GEMM, so a persistent queue has no idle CUs to recover.

## Expert-GEMM rewrites, Oct 17 (logs/g4/pers.log, logs/g4/prof_act.sh)
1. Fused gate+up sharing the activation tile: not built, bounded. Nemotron Nano experts have no gate (up only, relu2). Gemma 26B already computes gate|up in one GEMM pass over a fused tensor;
   the only thing fusion could remove is the f32 yg round trip + GeGLU kernel, which G4_PFPROF measures at moe_act 2.85 ms of 286.8 ms (1.0%) at pp512, upper bound; the existing
   fused SwiGLU epilogue was already shown 12-15% slower on Q4_K. Expected gain < 1%.
2. Dequant once and reuse across M-tiles: already shipped for Gemma as pf_qgemmh WR=1 (each B fragment unpacked once per tile, +4..5%). Reuse over several M-tiles per expert does not exist:
   ~24 (Nemotron) / ~32 (Gemma) routed rows per expert fit one tile. Nemotron pf_gemm dequant is fully hidden (PF_PROBE, above).
3. Persistent work queue (PF_QH_PERS=N workgroups, pf_qgemmh grouped, loops over the gt[0] x n-tile pairs): Gemma pp512, perf_high, 5 reps, interleaved with baseline:
   base 1639-1648; PERS=80 1635-1636; 160 1632-1637; 240 1640-1644; 480 1641-1648 tok/s. No gain (equal or -0.5%). Kept as opt-in (default off); output equal at 16 tokens (md5 of g4_run output).
   Tiles are uniform and the non-persistent grid already exits excess slots instantly, so there is no tail to recover.
4. Split-K for small experts: not built; stop rule (two prototypes in a row < 1%). Grouped GEMMs already run >1000 workgroups on 40 CUs, so split-K would only add atomics.
Nemotron 30B (1469 vs 1687) and Gemma 26B (1637-1645 vs 1719) gaps are unchanged; no default changed, no energy measurement (no winner).
