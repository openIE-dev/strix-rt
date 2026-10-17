# MiniCPM-V 4.7 on strix-rt (crate `mcv`)

Model: openbmb MiniCPM-V-4.7-35B-A3B (newest open MiniCPM-V), GGUF Q4_K_M + mmproj f16,
in `$MODELS/minicpmv47` on the Framework. Vision: SigLIP 27L + 4.7 window-attention merger + DownsampleMLP (`vision.rs`, own safetensors
converted from mmproj). LLM: qwen35moe (qwen3.5 hybrid gated-delta-net/attention trunk, 256-expert top-8 MoE + shared expert, interleaved M-RoPE)
(`llm.rs` stepwise decode, `prefill.rs` batched prefill).

Reference: llama.cpp Vulkan mtmd (`crates/mcv/ref/mcvdump.cpp`, `run_ref.sh`, `refs.sh`): greedy tokens + image embeddings.
Harness: `mcv_ref <gguf> <refdir> [n]` with `OWN_VIS=<ppm>` (our vision), `FORCED=1` (teacher-forced top-1 agreement), `PREFILL=1 REPS=n`.
All runs are under `logs/gpu_lock.sh`; scripts and logs are in `logs/mcv/` (`e3.sh` gives the table below).

## Parity (own vision + own LLM vs mtmd, prompt "Describe this image in one paragraph.")

| image | tiles / rows | emb cos mean (min) | teacher-forced top-1 | greedy identical prefix (stepwise / batched prefill) | ref-emb forced |
|---|---|---|---|---|---|
| fox 576x320 | 1 / 67 | 0.99991 (0.9981) | 125/126 | 83/100 / 83/100 | 126/126 |
| gold 512x512 | 3 / 202 | 0.99996 (0.9996) | 110/111 | 75/110 / 111/111 | 110/111 |
| chess 1440x900 | 7 / 458 | 0.99973 (0.925) | 124/128 | 21/120 / 21/120 | 125/128 |

Every teacher-forced miss is a near tie (top-2 margin 0.03-0.35 logits). Greedy runs only split at those ties; after a split
the text stays coherent. Text-only: forced 115/117. The batched prefill gives top-5 logits within about 0.3 of stepwise.

Fix this session: `prefill.rs` had no barrier between `moecomb` and `mcv_shadd`, and both read-modify-write x. That race made image
greedy runs diverge after one token (1/116). With the barrier, batched prefill matches stepwise.

## Performance (Radeon 8060S, RADV; llama.cpp build 10da8c463 Vulkan, fa=1)

| | strix-rt | llama.cpp |
|---|---|---|
| image encode fox / gold / chess (warm, ms) | 58-65 / 186 / 308-362 | 168 / 478 / 889 (mtmd-cli; 144 / 407 / 819 in mcvdump) |
| prefill pp91 / pp226 / pp482 (tok/s) | 383 / 906 / 1438 | pp91 724, pp512 1518 (llama-bench) |
| decode (tok/s) | 80-83 | tg128 75.9 |

The llama.cpp encode times are single-shot, so they may include some warm-up. Next steps: the small-M prefill is about 2x behind
llama.cpp (pp91), so per-layer launch overhead dominates there. Chess has a low-cosine embedding row (min 0.925; mean is fine), so check the slice-separator / resize path.

## Session 2 (chess embedding row, short-prompt prefill)

**Chess row 94 (cos 0.925): this is reference noise, not a bug.** With `mcvdump` using `MTMD_CPU=1` (the mmproj on the CPU backend),
llama.cpp CPU vs llama.cpp Vulkan gives cos **0.915** on that same row. Every other row is >= 0.993 in both comparisons. The token is
numerically ill-conditioned, so a 0.92 cosine is within the reference's own spread. Resize (Pillow bicubic, 22-bit fixed point),
slice layout and separators all match mtmd. Parity on chess is still 124/128 teacher-forced, and greedy is 100/100 identical with batched prefill.
Tools: `crates/mcv/ref/cpuref.sh`, `crates/mcv/ref/embcmp.py`; mcv_ref prints the worst rows with their norms.

**Prefill profile** (`PF_TS=1`: per-phase GPU timestamps; also `PF_PROF`, `PF_SKIPK`, `PF_XBAR`), pp91 before this session:
host argmax over 1 MB of uncached logits 35 ms (!), GDN mixer 31 ms (mostly GEMMs with too few workgroups at M=91), router+shared 10.5 ms,
grouped MoE 92 ms (19 GB of expert weights at ~210 GB/s, close to peak), comb 1.5 ms. A dispatch + barrier costs ~4 us, so with
everything already in one command buffer, fusing small kernels buys little.

Changes: GPU argmax (argmax_w, key at aux u64[64]); recurrent-state reset on the GPU (vkCmdFillBuffer);
grouped MoE tiles 16 (gate|up) / 32 (down); 64-row GEMM tiles for n <= 160 (`PF_SMT`, `PF_SMT_M`) on the mixer, router and shared
expert (`prefill::gemm_swiglu_tile`). A split-row GDN scan (`mcv_gscan`, `PF_GSCAN_R`) was tried: no gain, off by default.

| prompt | strix-rt before | strix-rt now | llama.cpp (llama-bench, same session) |
|---|---|---|---|
| pp23 (text) | 204 | **295** | 288 |
| pp91 (fox) | 383 | **694** | 726 |
| pp226 (gold) | 906 | **1108** | 1215 |
| pp482 (chess) | 1438 | **1435** | 1526 |

strix-rt times are wall time and include input staging and argmax; llama-bench excludes sampling. Greedy parity is unchanged
(83/100, 75/100, 100/100, 44/100, i.e. the same near-tie splits as stepwise). Still to do to pass llama.cpp at 91-482: the GDN GEMMs
run at ~47 GB/s at M=91 (0.7 GB in 15 ms). Split-K or a fused qkv|gate|alpha|beta GEMM should give back ~10 ms. Grouped gate|up
runs at 195 GB/s vs 238 for down.

## Session 3 (closing the prefill gap)

| prompt | strix-rt | llama.cpp | |
|---|---|---|---|
| pp23 | **335** | 288 | +16% |
| pp91 | **763** | 725 | +5% |
| pp226 | 1109 | 1218 | -9% |
| pp482 | 1468 | 1527 | -4% |

Parity is unchanged (83/100, 75/100, 100/100, 44/100 greedy).

- **Multi-row GEMV for MoE gate|up** (`mcv_gmv_swiglu`, n <= `PF_GMV_M`=160, 4-row tiles `PF_GMV_T`): gate|up 58 -> 42 ms at pp91.
  Weights are read once per tile with uvec4 loads, the activations sit in bank-skewed LDS (unskewed it was 32-way bank-conflicted:
  226 -> 160 ms), and 4-row tiles keep occupancy high (8 rows: 163 ms, 4: 42, 3: 43, 2: 48). Above ~160 rows it loses to the coopmat GEMM.
- **mcv_psel**: parallel prefix sums and per-expert tile writes replace single-thread loops (52 us -> ~25 us per layer).
- **Split-K in pf_gemm** (`pc.pad` slices, atomic f32 adds; `PF_SPLITK`): correct, but **no gain**. The GDN input GEMMs are not
  occupancy-bound. At pp91 they spend 10.5 ms on 0.55 GB (the coopmat kernel is dequant-bound at M=91); at pp482 they are compute-bound (~28 TFLOPS).
- **4-lane GDN scan** (`mcv_gscan4`, `PF_GSCAN_R=4`): only -3.5 ms at pp482, off by default. It flips chess's known near-tie at
  token 14 (forced gap 0.27), the same flip any change of rounding gives.
- pp482 breakdown (ms): GDN input GEMMs 26, scan 25, conv/norm/out 20, attention 17, router/shared 15, gate|up 135, down 70, comb 10.

Still behind at 226-482. The obvious next steps are a chunked (WY / matmul) gated-delta-rule scan and a multi-row GEMV
down-projection (Q6_K) for small n. A GEMV that holds 8+ rows in registers would cover gate|up up to ~256 rows.

## Session 4: ahead of llama.cpp at every measured length

| prompt | strix-rt | llama.cpp | |
|---|---|---|---|
| pp23 | **342** | 289 | +18% |
| pp91 | **800** | 724 | +10% |
| pp226 | **1242** | 1219 | +2% |
| pp482 | **1670** | 1533 | +9% |

Parity with own vision (e3.sh) matches session 1: teacher-forced 125/126, 110/111, 124/128; greedy 83 / 75 (stepwise) /
111 (batched) / 21 (chess). Batched-prefill logits on chess are now closer to the stepwise path than before
(top-1 gap 0.035 vs 0.105). The chess token-14 greedy flip that shows up under batched prefill is the known near-tie.

- **Row-per-thread gate|up GEMV** (`mcv_gmv2_swiglu`): each thread computes the gate and up row for 8 tokens, the activations come
  from LDS as broadcast uvec4 loads, and there are no cross-lane reductions. Used for 160 < n <= 600 (`PF_GMV2_M`, `PF_GMV_M`, `PF_GMV2_T`);
  n <= 160 keeps the 4-token lane-split kernel. gate|up at pp482: 135 -> 114 ms; pp226: 87.7 -> 75.7 ms.
- **Dense GEMM tiles**: 64 rows for n <= 160, 128 rows for 160 < n <= 320 (`PF_SMT2_M`), default above. pp226 1200 -> 1241.
- **MoE kernel choice is per layer**: the GGUF mixes Q4_K and Q6_K `ffn_down_exps` across layers, and the GEMV kernels are gated on quant type and K.
- Tried and left off: a Q6_K multi-row down GEMV (`mcv_gmv_down`, `PF_GMVD_M`). It is correct but 1.6x slower than the coopmat
  grouped GEMM: small unaligned loads plus per-row reductions. The 4-lane / larger-chunk GDN scans (`PF_GSCAN_R=4`,
  `PF_GSCAN_CH`) did not move the scan (~25 ms at pp482), so the chunked delta-rule (WY) formulation remains the next step there.

## GDN scan profile (chess, 482 tokens)
- Extra-scan timing (PF_XSCAN): the scan costs 10.7 ms at 226 tokens and 23.7 ms at 482 tokens (30 layers), linear in tokens, about 1.6 us per token per layer.
- With the recurrence math removed (PF_GSCAN_PM=1), gdn drops 63.5 to 46.1 ms. Removing per-token LDS reads of k/q (PM=2) changes nothing. Splitting rows (R=64/32) is slower.
- Conclusion: the scan is bound by its serial per-token FMA dependency chain at low occupancy, not by LDS or bandwidth. The chunked WY/UT form is the right fix, but it can recover at most about 17 to 24 ms (about 5 to 7 percent of pp482).

## Chunked GDN scan (shared, crates/prefill) — work in progress
- Entry point: Pf::gdn(ctx, bufs, push, cb), used by mcv and prefill/qwen35. Push and bindings are the same as pf_gdn_scan.
- PF_GDN_CHUNK: 0 = sequential pf_gdn_scan (the default), 1 = chunked, 2 = chunked when M >= PF_GDN_CHUNK_MIN (default 64).
- Chunked path: pf_gdn_chunkp pre-pass (per chunk and head: cumulative log decay, beta, A, P into a grow-only scratch buffer). Then pf_gdn_chunkcm (PF_GDN_CM=1, the default), a cooperative-matrix WY/UT scan with C=16 and the state in f32 accumulators, split hi+lo f16 for the products. PF_GDN_CM=0 uses the scalar pf_gdn_chunk (spec PF_GDN_R, PF_GDN_C), which is slower.
- Status: the cm path keeps greedy parity on gold (226). Timing is about equal to sequential so far; chess and 91-token runs are pending.
- Other workers touching GDN (e.g. for pp8192): please go through Pf::gdn and these knobs rather than forking the kernel.
- Results (07:50): parity with PF_GDN_CHUNK=1 (cm path) matches the sequential path exactly: gold 100/100 greedy, chess 14/100 (same as sequential; the chess divergence is the known near-tie).
- Timing: gdn time is gold 36.4 vs 35.0 ms and chess 71.5 vs 68.3 ms, so the chunked path is about 1-3 ms slower. WG=32 (row split) is much slower (chess gdn 141 ms), because per-workgroup staging of K, Q and V dominates. PF_GDN_CHUNK stays 0 by default.
- Next ideas: vectorized loads of K, Q and V; C=32 to halve the per-chunk barriers and serial work; fuse the pre-pass into gdn_conv. A long prompt (pp8192) may favour the chunked path, so measure that with PF_GDN_CHUNK=2.
