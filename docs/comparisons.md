# strix-rt next to reference engines on Strix Halo (Ryzen AI Max+ 395, Radeon 8060S gfx1151, XDNA2, 128 GB)

> Note for readers: this is a raw lab notebook, written while the work happened, so it keeps the original wording
> ("competitor", ✅/❌, "where strix-rt loses"). We do not see llama.cpp, vLLM or FastFlowLM as competitors: llama.cpp is the
> quality and energy bar we measure against, and most of what we learned came from reading it. Everything here is meant to be
> copied. Absolute paths were replaced with `$MODELS` / `~`. Some runs were taken on a busy machine; each section says so.


Status: **partial, in progress (2026-10-15)**. Every number below comes from a log in `logs/comp/` (llama-bench JSON in
`logs/comp/json/`). Cells marked `pending` have not been measured yet. Nothing is estimated.

## Setup and versions
| component | version / build |
|---|---|
| OS / kernel / Mesa | Ubuntu 26.04.1, 7.0.0-38-generic, Mesa RADV 26.0.8 |
| llama.cpp Vulkan | `e36df6047` (2026-10-15), `~/llama-latest/build-vulkan` (`-DGGML_VULKAN=ON`) |
| llama.cpp ROCm | `e36df6047`, `~/llama-latest/build-rocm713`: **ROCm 7.13.0** (TheRock gfx1151 wheels, `~/venvs/rocm713`), `-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_ROCWMMA_FATTN=OFF`, run with `ROCBLAS_USE_HIPBLASLT=1` |
| Bonsai 2 reference | PrismML llama.cpp fork `e3de715` (mainline has no PQ2_0/PTQ1_0), Vulkan + ROCm builds |
| vLLM | kyuz0 `vllm-therock-gfx1151:latest` = vLLM 0.31.0+strix, torch 2.11.0+rocm10.0.0, Triton 3.8.0 |
| FastFlowLM | FLM v1.0.8 (`/usr/bin/flm`), amdxdna 0.7, NPU FW 1.1.2.65 |
| strix-rt | this repo, bins built in `target-comp` (commit db3fa84+), `ZEROCOPY=dmabuf` |

**ROCm 7.14 not usable:** the only 7.14 builds are TheRock nightlies (latest `7.14.0a20260612`); `rocminfo` and every HIP
program segfault in `rocr::AMD::GpuAgent::InitDma()` during `hsa_init` on kernel 7.0 (also with `HSA_ENABLE_SDMA=0`).
ROCm 7.13.0 (latest release) works and is used instead. (The Bonsai HIP column uses a prism build compiled against the 7.14
headers but run with the 7.13 runtime libraries.)

## Method
- Every GPU/NPU run: `logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/perf_high.sh ...` (`tools/comp/run.sh`), perf level high, restored to auto.
- llama.cpp: `llama-bench -ngl 999 -fa on -b 2048 -ub 512 -p 512,2048,8192 -n 128 -r 3 -o json` and
  `-p 0 -n 128 -d 8192 -r 3` (decode at depth 8192) (`tools/comp/lb.sh`), same flags as howquant.dev.
- strix-rt: `pf_q38`/`pf_lfm NOREF=1 REPS=3 <model> L 1` (best of 3, random-id prompt), `DEPTH_TG=1 ... 8192 129` (128 decode steps after an
  8192-token batched prefill), `q38`/`lfm <model> <12-token prompt> 128` RUNS=3; MiniCPM via `mcv_ref PREFILL=1 REPS=3` (median) (`tools/comp/srt.sh`).
- TTFT single-stream ~= prefill time of the prompt; for strix-rt it is pp512 time (512 / pp512 tok/s, includes the first decode step).
- Concurrency: `tools/comp/conc_bench.py`, Soothill method (unique ~600-token prompts, 64 output tokens with ignore_eos, temperature 0,
  seed 12345, streaming, 2 warm-up + 10 measured waves per C), `tools/comp/serve_bench.sh`.
- Energy: `tools/energy/record.py record`, SMU socket energy, differential (J(N=320) - J(N=64)) / 256 decode tokens (`tools/comp/energy.sh`).

## Single-stream tables (tok/s, higher is better; **bold** = best competitor in each column, mark ✅ where strix-rt beats it, ❌ where it loses)

### Qwen3.8-27B UD-Q4_K_M (dense hybrid GDN)
| engine | pp512 | pp2048 | pp8192 | tg128 | tg @ depth 8192 |
|---|---|---|---|---|---|
| llama.cpp Vulkan | 416.7 | 306.8 | 251.1 | **13.09** | **12.78** |
| llama.cpp ROCm 7.13 | **425.4** | **426.0** | **404.4** | 12.9 | 12.3 |
| strix-rt | ✅ 517.1 | ✅ 498.6 | ❌ 325.1 | ✅ 13.47 | ❌ 12.48 |

Speculative (MTP, 12-token prompt, greedy, 128 tokens): llama.cpp Vulkan `--spec-type draft-mtp`: n-max 1 20.8-21.2, n-max 2 22.8-23.9,
**n-max 3 24.5-25.2 tok/s** (accept 80-81/137-140); strix-rt MTP k=1 19.3-19.5, k=2 20.1-20.2, k=3 19.4 tok/s (output = greedy 128/128). ❌
After multi-column GDN conv/step (c094fce) and decode-once multi-column GEMV for Q5_K/Q8_0/IQ4 (89d44b2): k=1 21.0, **k=2 22.9, k=3 22.1** (greedy 128/128); verify m=4 101 -> 96.3 ms (m=1 73.3). Still ❌ vs 24.5-25.2 and k=3 < k=2: remaining cost is multi-column GEMV (+19 ms at m=4 vs single-column, VONE test) and 7.6 ms per chained draft (full Q6_K lm_head). MTP_VOCAB truncation lowers acceptance and is slower.
llama.cpp ROCm 7.13 (the earlier exit was the 7.14 runtime crash): base 12.75-12.79, MTP n-max 1 20.6-20.7, n-max 2 20.2-23.7, n-max 3 18.7-23.6 tok/s (runs drift down; GPU shared).

### LFM2.5-24B-A2B Q4_K_M
| engine | pp512 | pp2048 | pp8192 | tg128 | tg @ 8192 |
|---|---|---|---|---|---|
| llama.cpp Vulkan | 2280.4 | 2228.6 | 1661.7 | **116.4** | **107.1** |
| llama.cpp ROCm 7.13 | **2626.7** | **2707.0** | **2625.8** | 100.7 | 95.9 |
| strix-rt | ❌ 2310.2 (-12%) | ❌ 2354.5 (-13%) | ❌ 1957.1 (-25%) | ✅ 127.0 | ✅ 112.1 |

### MiniCPM-V 4.7 (35B-A3B, text path) Q4_K_M
| engine | pp512 | pp2048 | pp8192 | tg128 | tg @ 8192 |
|---|---|---|---|---|---|
| llama.cpp Vulkan | **1547.0** | 1523.5 | 1067.0 | **71.3** | 59.3 |
| llama.cpp ROCm 7.13 | 1530.9 | **1543.4** | **1425.5** | 64.0 | **63.5** |
| strix-rt | ✅ 1809.8 | ✅ 1702.9 | ❌ 1321.2 | ✅ 83.2 | ✅ 71.6 |

### Bonsai 2 27B PQ2_0 (prism fork as reference)
| engine | pp512 | pp2048 | pp8192 | tg128 | tg @ 8192 |
|---|---|---|---|---|---|
| prism Vulkan | 305.6 | 238.6 | 215.6 | 24.1 | 22.2 |
| prism ROCm | **380.9** | **315.6** | **250.3** | **24.8** | **23.5** |
| strix-rt | ✅ 528.4 | ✅ 396.2 | ✅ 295.9 | ✅ 27.65 | ✅ 23.73 |
strix-rt MTP graft (blk.64 from the Qwen3.8 GGUF) k=1 34.95 tok/s; llama.cpp cannot graft a draft head (no comparable spec path).

### Bonsai 2 27B PTQ1_0
| engine | pp512 | pp2048 | pp8192 | tg128 | tg @ 8192 |
|---|---|---|---|---|---|
| prism Vulkan | 207.4 | 207.7 | 200.8 | **27.1** | **25.2** |
| prism ROCm 7.13 | **323.7** | **325.1** | **274.9** | 24.2 | 23.1 |
| strix-rt | ✅ 399.9 | ❌ 288.0 (-11%) | ❌ 241.1 (-12%) | ✅ 31.76 | ✅ 26.97 |
strix-rt MTP graft k=1 42.4-42.7 tok/s.

### Qwen3.6-35B-A3B Q8_0
| engine | pp512 | pp2048 | pp8192 | tg128 | tg @ 8192 |
|---|---|---|---|---|---|
| llama.cpp Vulkan | **1491.0** | **1484.2** | 1264.9 | **55.2** | **51.2** |
| llama.cpp ROCm 7.13 | 1078.1 | 1440.9 | **1270.6** | 48.6 | 47.4 |
| strix-rt (mcv, Q8_0 experts, 89d44b2; logs/q8moe/bench2.log) | 1467.1 (✅ vs ROCm 1078; ❌ -2% vs Vulkan 1491) | 1429.9 (❌ -4% vs Vulkan 1484, -1% vs ROCm 1441) | 1232.1 (❌ -3% vs Vulkan 1265, -3% vs ROCm 1271) | ✅ 56.5 | 50.5 (❌ -1% vs Vulkan 51.2, ✅ vs ROCm 47.4) |
Q8_0 parity (12-word text prompt, greedy vs llama.cpp Vulkan dump `~/mcv_ref/ref_q36txt`): first 15/110 tokens identical, then diverges; teacher-forced check pending (`logs/q8moe/`).

### gpt-oss-20b MXFP4
| engine | pp512 | pp2048 | pp8192 | tg128 | tg @ 8192 |
|---|---|---|---|---|---|
| llama.cpp Vulkan | **2362.2** | **2292.4** | **1915.4** | **74.2** | **69.8** |
| llama.cpp ROCm 7.13 | 2092.3 | 2075.7 | 1423.2 | 72.0 | 65.4 |
| strix-rt (crate `gptoss`, decode only, bb42d8a) | gap (no batched prefill yet) | | | ❌ 52.9 | pending |
gpt-oss parity (20-token raw prompt, greedy, `logs/gptoss/`): first 32/128 tokens identical to llama.cpp Vulkan, then a plausible alternative continuation (likely near-tie; teacher-forced check pending).

### Gemma 4 E4B-it Q8_0 (crates/gemma4, `G4_PF=1`, chunk 512)
| engine | pp512 | pp2048 | tg128 | J/token (socket / gfx) |
|---|---|---|---|---|
| llama.cpp Vulkan (`-fa on`) | **2462.1** | 2401.4 | 40.19 | 2.326 / **0.643** |
| strix-rt | ✅ 2524.7 (+3%) | ✅ 2613.1 (+9%) | ✅ 41.64 (+4%) | ✅ 1.953 / ❌ 0.989 (llama 2.343 / 0.610, 2026-10-16 re-run) |

Parity: greedy 64/64 vs llama.cpp Vulkan; teacher-forced 64/65 (the miss is a retokenization tail artifact), 64/64 after 512-chunk prefill. Energy from tools/comp/energy.sh (N=320 minus N=64, over 256 tokens) at perf level high. Logs: logs/g4/bench_e4b_{ours,llama,energy}.log.

### Gemma 4 26B-A4B-it UD-Q4_K_M (crates/gemma4, `G4_PF=1`, chunk 512)
| engine | pp512 | pp2048 | tg128 | J/token (socket / gfx) |
|---|---|---|---|---|
| llama.cpp Vulkan (`-fa on`) | **1716.9** | **1626.7** | 52.06 | 1.758 / **0.539** |
| strix-rt | ❌ 1637 (-5%) | ❌ 1551 (-5%) | ✅ 55.34 (+6%) | ✅ 1.375 / ❌ 0.685 (llama 1.777 / 0.569 that run) |

Parity (after the router top-k softmax fix c6feb6a), against the llama.cpp chat-template reference (Vulkan and CPU produce identical ids): greedy 63/64, teacher-forced 63/64 both token-by-token and with 512-chunk prefill. The one miss is the retokenized tail. Updated 2026-10-16. MoE prefill uses prefill::gemm_grouped, with down_exps requantized Q5_1 -> Q8_0 for prefill only. Attention prefill uses coopmat flash attention (hd 256 SWA and hd 512 global). Decode router logits run over 16 workgroups with a subgroup top-k (it was a single-workgroup F32 GEMV plus a serial top-k: 31 -> 55 tok/s). Parity is unchanged (63/64). The remaining pp gap is the grouped expert GEMMs; the shared packed-LDS expert kernel slots in there. Energy was re-measured on 2026-10-16 (llama 1.758 that run). 2026-10-17: pp via pf_qgemmh WR=1/KD=4 (gemma4 process default WR1/KD4/MI1; 1536->1637, 1477->1551, forced parity unchanged; docs/prefill.md). Prefill energy ours 0.034 J/token gfx (was 0.041). Logs: logs/g4/bench_26b_{ours,llama,energy}.log, logs/g4/job_26bpar2.log. 2026-10-16 update: the expert GEMMs use the shared packed-LDS `pf_qgemmh` (G4_QMMH, default on; pp from logs/g4/job_spec.log, forced parity unchanged at 63/64). Energy (logs/g4/job_epw.log, job_idot.log, bench_26b3_energy.log): GPU clocks are pinned at 2900 MHz and gfx activity is about 92-93% for both engines, so neither idle-spin nor clock residency explains it. The gap is power per busy cycle (ours 44 W vs llama 30.5 W during decode). Skipping the experts removes about 9 W. llama.cpp runs Q4_K at K >= 2048 through MMVQ (q8_1 activations plus integer dot). Ours now does the same for the Q4_K expert gate_up (G4_IDOT, q8 x per workgroup): 41.2 -> 38.2 W at the same 54 tok/s, GPU 0.804 -> 0.685 J/token. Forced parity is 62/64: one extra 0.025-logit near-tie flip. The remaining gap is the float-dequant dense/attention GEMVs (shared gemvq) and the Q5_1 down. For E4B (Q8_0, which llama also runs through float GEMVs on AMD) the GPU gap is not explained yet.

### Muse Glimmer 30B Q4_K_M (dense, crates/glimmer, chunk 512; details docs/glimmer.md)
| engine | pp512 | pp2048 | tg128 | J/token (socket / gfx) |
|---|---|---|---|---|
| llama.cpp Vulkan | 451.1 | 440.5 | **13.14** | **6.98 / 2.27** |
| llama.cpp ROCm (hip 7.14) | **465.8** | **461.2** | 12.77 | 8.12 / 3.24 |
| strix-rt | ✅ 540 (+16% vs ROCm) | ✅ 468 (+1.5% vs ROCm) | ✅ 13.34 (+1.5%) | ❌ 7.92 / 4.9 (+13.5% socket vs Vulkan; was 8.44 / +21%) |
Greedy ids vs llama.cpp Vulkan: 765/768 teacher-forced over 6 text prompts x 128 (misses are near ties), long prompts 48/48; vision prompt 126/128 forced with our ViT embeddings (128/128 with llama.cpp embeddings).
Energy gap: same clocks and activity as llama.cpp; the float-dequant Q4_K/Q6_K GEMVs draw 67-77 W gfx at the DRAM wall. An int-dot (MMVQ-style) GEMV (`GLM_IDOT=1`) was tried and failed (4x slower, opt-in only).

## Concurrency (total completion tok/s, Soothill method)
| model / engine | C=1 | C=2 | C=4 | C=8 | TTFT C=1 (ms) |
|---|---|---|---|---|---|
| gpt-oss-20b, vLLM 0.31 (HF MXFP4) | 14.36 | 19.74 | 27.35 | 111.37 | 230.5 |
| gpt-oss-20b, FastFlowLM NPU | 8.01 | 8.01 | 8.01 | pending | 4613.6 |
| Qwen3.8-27B Q4_K, strix-serve (8 slots) | 10.42 | 11.98 | 14.04 | 14.08 | 1430.9 |
| Bonsai 2 27B PQ2_0, strix-serve (8 slots) | 17.09 | 22.52 | 21.76 | 22.00 | 1430.1 |
| Qwen3.8-27B Q4_K, strix-serve chunked prefill (r3, SERVE_CHUNK=256) | 9.64 | 11.74 | 13.77 | 13.81 | 1770.2 |
| Qwen3.8-27B Q4_K, strix-serve packed cross-prompt prefill (r4, PF_M/SERVE_PACK=1024) | 10.33 | 15.17 | 14.55 | 14.84 | 12405 (C=8) |
| Qwen3.8-27B Q4_K, strix-serve packed prefill (r5, PF_M/SERVE_PACK=512) | 10.41 | 12.14 | 14.17 | 14.19 | 11254 (C=8); J/tok 12.24/9.89/8.08/7.97 |
| Qwen3.8-27B Q4_K, strix-serve r6b (gemvq 8 columns per weight read, eq 8/8) | 9.40 | 12.05 | 14.26 | **15.36** | 11029 (C=8); J/tok 13.664/9.911/8.001/7.182 |
| Qwen3.8-27B Q4_K, strix-serve r7b (r6b + SERVE_PF_PASSES=2) | 8.82 | 12.11 | 14.31 | **15.47** | 10618 (C=8); J/tok 14.194/9.851/7.905/7.119; GPU hot from the preceding runs, C=1 depressed |
| Qwen3.8-27B Q4_K, strix-serve r8 (r6b settings, after the fan curve f862ce7: ~2850 MHz at 85 C; eq 8/8) | 10.55 | 16.71 | 21.64 | **23.62** | TTFT ms 1365.75/2531.63/4117.44/7442.52; J/tok 11.084/7.79/6.432/5.22 |
| Qwen3.8-27B UD-Q4_K_M, llama-server Vulkan (-np 8), rerun after the fan curve | 9.97 | 15.12 | 20.16 | 22.28 | TTFT ms 1593.98/3236.24/6539.07/13638.26; J/tok 10.581/7.717/6.257/5.767 |
Thermal fix (f862ce7, custom fan curve: GPU ~2850 MHz at 85 C under load instead of ~2000 MHz at 95-100 C). Same-conditions comparison: strix-serve r8 vs llama.cpp Vulkan = 10.55/16.71/21.64/23.62 vs 9.97/15.12/20.16/22.28 tok/s at C=1/2/4/8; strix-serve now leads in throughput and TTFT at every concurrency; J/token is lower only at C=8 (5.22 vs 5.77) and 1-5% higher at C=1/2/4 (11.08/7.79/6.43 vs 10.58/7.72/6.26). Rows above r8 (and the 17.08 llama Vulkan row) were measured on the throttling fan curve and are not comparable with r8.
| Qwen3.8-27B UD-Q4_K_M, llama-server Vulkan (-np 8) | 9.87 | 13.91 | 15.07 | **17.08** | 1653.0 |
| Qwen3.8-27B UD-Q4_K_M, llama-server ROCm 7.13 (-np 8) | 9.48 | 11.72 | 13.77 | 14.43 | 1786.5 |
strix-serve (`crates/serve`, `tools/comp/serve_bench_strix.sh`, logs `logs/comp/serve/strix_*`): socket J per output token over all 12 waves incl. prefill:
Qwen3.8 12.22 / 10.10 / 8.15 / 8.04 (C=1/2/4/8), Bonsai 6.72 / 5.44 / 5.28 / 5.19. Greedy batched output == single-stream output 8/8 prompts
(`tools/comp/serve_eq.py`, logs/serve/eq_*.log, both models). Decode scales (`serve_steps`, q38 at depth 600: m=1 73 ms, m=2 78, m=4 101, m=8 195 ms/step
= 13.6 -> 41 tok/s aggregate) but the Soothill waves are prefill-bound: each ~600-token prompt takes ~2.1 s of prefill that blocks the decode steps
(admission is one prompt per step, prompts are not batched together), so C=8 TTFT ~10 s and throughput flattens at 14 / 22 tok/s.
Prefill analysis (`serve_pf`, logs/serve/pf_q38.log): the prefill GEMM is compute-bound and works in 256-token tile rows, so a 600-token
prompt costs 3 tile rows (~1.4-2.1 s, noisy under other workers) vs 2 for pp512 (1.07 s); the slot/cache copies are only 4 ms. That is the 2.1 vs 1.2 s gap.
Chunked prefill (r3: 256-token chunks, one per decode step, fixed boundaries so greedy output stays 8/8 equal) keeps decode running between chunks
but cannot raise throughput: at ~480 prefill tok/s each 600-token request costs ~1.25 s of GPU, which caps this workload near 50 output tok/s
even with free decode. C=8 TTFT 11.3 s (strix r3) vs 20.1 s (llama Vulkan); llama Vulkan C=8 throughput is 17.08 vs 13.81-14.08 here.
Cross-prompt prefill batching is now implemented (`crates/prefill/src/q35multi.rs`, packed path in strix_serve): row-wise GEMMs/norms run once over rows packed from several prompts, and the per-sequence GDN/attention runs per segment on that sequence slot. Root cause of the 2.1 s prefill (logs/serve/pf2_q38.log): a 600-token prompt needs 3 tile rows (512+88 tail) and passes above 512 rows are superlinear (1024-row pass ~4.1 s vs ~1.73 s for 512, about 2.4x); some early measurements also ran without the perf-level pin. Host copies are ~4 ms. Defaults are now PF_M=512, SERVE_PACK=512.
Prefill under load (logs/serve/pf_load_q38.{log,csv}, tools/comp/pf_load.sh: 24 back-to-back 600-token prefills, no decode, nothing else on the GPU, perf level high, hwmon sampled every 0.25 s): 1331 ms per prompt for the first ~12 s at 2890 MHz / ~150 W, then the GPU hits ~95-100 C and the clock falls to ~2000 MHz (power 115 W) and the same prefill takes 1834 ms; time tracks 1/clock (1331 x 2890/2000 = 1923). The 2.2 s server prefill under load is therefore thermal clock throttling, not interleaving overhead (r3 GPU-path time excludes the decode steps and still rises from 1.76 to 2.2 s within the C=1 wave) and not memory contention (submissions are serial, other GPU jobs are excluded by gpu_lock). The earlier "passes above 512 rows are superlinear" reading (pf2) was taken on a hot GPU in the same way; 1x512 vs 2x256 vs 4x128 packed passes differ by <15% and that difference also rises with temperature. Sustained throughput is set by the thermal limit, so the lever is fewer FLOPs/bytes per token, not scheduling.
Cooling (logs/serve/thermal_fan.log): during the benchmarks the cros_ec fan target stays at ~926 rpm (fan reads ~980 rpm) while the CPU/APU sits at ~93 C and the mainboard ambient at ~66 C, with platform_profile already performance. The EC fan curve is the likely limiter; a more aggressive fan curve (framework_tool/ectool, not changed here) should recover much of the 2890 -> 2000 MHz drop.
LFM2.5-24B-A2B and MiniCPM-V 4.7 are served by strix-serve1 (crates/serve/src/bin/strix_serve1.rs): one request at a time (no batched decode yet), batched prefill (prefill::lfm2moe / mcv::prefill), SSE, sampling, and for MiniCPM-V image_url content parts (data: URLs, png/jpeg/webp) through the own vision tower. Smoke tests (tools/comp/serve1_test.py, logs/serve/s1_lfm.log, s1_mcv.log): greedy runs repeat exactly; LFM2.5 48 tokens in 0.49 s (TTFT 93 ms); MiniCPM-V text 38 tokens in 0.56 s, fox image -> correct one-paragraph description, 96 tokens in 1.47 s with TTFT 199 ms incl. vision encode. Image M-RoPE positions follow llama.cpp mtmd and are verified for single-tile images only; usage.prompt_tokens counts each image as one token.
Competitor Qwen3.8 rows (llama-server Vulkan/ROCm, vLLM FP8) are queued in tools/comp/batch3.sh; vLLM gpt-oss C=8 111.4 above.
Gaps: LFM2.5 (lfm2moe) and MiniCPM-V (qwen35moe, image input) are not served yet (their decoders have no multi-column/slot path);
KV is one contiguous max_ctx block per slot (not paged).
vLLM gpt-oss: C=1..4 are far below llama.cpp single-stream (74 tok/s); C=8 jumps to 111 (likely a CUDA-graph/batch-size capture effect); not investigated yet.
FLM processes one request at a time (C=1/2/4 identical throughput, TTFT grows linearly).

## Energy (J per decoded token, SMU socket, differential (J(N=320)-J(N=64))/256; two sessions where available)
| model | llama.cpp Vulkan | llama.cpp ROCm 7.13 | strix-rt |
|---|---|---|---|
| Qwen3.8-27B | 8.90 / 7.94 | 9.48 | 11.12 (overlapped downloads) / 9.02 ❌ vs Vulkan (+1-14%) |
| LFM2.5 | 0.641 / 0.634 | 0.885 | 0.836 ❌ (+31%) |
| MiniCPM-V 4.7 | 1.350 / 1.158 | 1.462 | ✅ 1.065 (-8% vs best; copied weights; dma-buf mode lost the device: "Not enough memory for command submission") |
| Qwen3.6-35B-A3B Q8_0 | 2.065 / 1.759 | 1.851 | gap |
| gpt-oss-20b | 1.113 | 1.272 | gap |
| Bonsai PQ2_0 (prism) | 4.208 | 4.796 | ✅ 3.399 (-19%) |
| Bonsai PTQ1_0 (prism) | 4.033 | 4.670 | ✅ 3.222 (-20%) |
| Muse Glimmer 30B Q4_K_M | 6.98 | 8.12 | ❌ 7.92 (+13.5% vs Vulkan, -2.5% vs ROCm; gfx 5.52 vs 2.27; cause = float-dequant GEMV power, see docs/glimmer.md) |
Session-to-session spread on the same engine is up to ~12% (other workers share the SoC), so differences under ~10% are not significant.

## Where strix-rt loses (so far)
1. **No serving**: no OpenAI server, no batched decode -> no concurrency numbers at all (vLLM 111 tok/s at C=8 on gpt-oss).
2. **Model coverage**: Qwen3.6-35B-A3B Q8_0 (Q8_0 MoE experts) and gpt-oss-20b (arch + MXFP4) do not run.
3. **Speculative decoding on Qwen3.8**: llama.cpp Vulkan MTP n-max 3 = 24.5-25.2 tok/s vs strix-rt best 20.2 (k=2): -20%.
   strix-rt chained k=3 is slower than k=2 (verify cost); llama.cpp scales to 3 drafts.
4. **Long-context prefill on dense/GDN models**: Qwen3.8 pp8192 325 vs ROCm 404 (-20%); MiniCPM pp8192 1321 vs ROCm 1425 (-7%).
5. **Long-context decode on Qwen3.8**: 12.48 vs Vulkan 12.78 @ 8192 (-2%).
6. **Energy per token** (Vulkan comparison): LFM2.5 +30%, Qwen3.8 +25% more J/token (to be re-verified); Glimmer +13.5% (7.92 vs 6.98).

## pp8192 profile notes (2026-10-15)
MiniCPM pp8192 (`logs/pp8k/prof1.log`, PF_GDN_CHUNK=0, 1129 tok/s in that run): per-chunk attention grows linearly to 237 ms in the last chunk
while GDN stays ~90-110 ms; attention (scalar `pf_attn.comp`, 16-token x 32-key tiles, no coopmat) is the long-context bottleneck.
Next step: coopmat flash attention for prefill; chunked GDN is owned by the other worker (`PF_GDN_CHUNK`).

## Update 2026-10-15 (later)
- Qwen3.6-35B-A3B Q8_0 teacher-forced vs Q8 reference: 106/110 agree (`logs/q8moe/forced.log`; MiniCPM Q4 baseline 115/117).
- gpt-oss-20b: faster MXFP4 load path (target-q8b) 52.9 -> **61.0 tok/s** tg, same greedy parity (first 32/128 vs llama.cpp Vulkan); teacher-forced vs llama.cpp 116/128 (`logs/gptoss/ab2.log`, `forced.log`). Still below llama.cpp 74. Sweep (`sweep1.log`): LM head ~2.7 ms/tok, ~0.67 ms/layer; GEMV/ATTN_NC knobs within noise.
- gpt-oss batched prefill (GO_PF=1, default off, commit 8e91ac7): QKV/out-proj via gemvq nb columns, rest per column. 20-token prompt 60 -> 79 tok/s at NB=4, identical greedy output (`logs/gptoss/pf1.log`). MoE still per column -> next step is grouped experts.
- Qwen3.8 MTP (`logs/mtp/run5.log`, IQ3_S/Q3_K decode-once cols + IQ4_NL draft head + GEMVQ_NBNR_Q=2): k=3 **25.15 tok/s**, k=2 24.20, greedy exact 128/128. A repeat of the same config gave k=3 21.19 (verify 106.9 vs 88.9 ms/cycle: GPU contention), so 25+ is not yet stable. NBNR_Q=2 cuts verify m=4 99.8 -> 88.8 ms.
- Coopmat flash-attention prefill `pf_attnfa` (PF_FA=1, **default OFF**, commits 4554103/a1f440b) (`logs/pp8k/fa1.log`): MiniCPM pp8192 1210 -> 1489 tok/s (attn per chunk 228 -> 133 ms), Qwen3.8 pp8192 281 -> 325 (FA=0 run slower than earlier 325 baseline, so contention likely). Qwen3.8 L512 parity 16/16, but **MiniCPM L512 first pred differs (1091 vs 286)**: kernel not verified; stays off until fixed.

## PF_FA window check
PF_FA (coopmat flash attention, wrong on MiniCPM at 512 tokens) was on by default between 4554103 and a1f440b (2026-10-15 11:20-11:21 UTC).
All strix-rt single-stream, MTP and energy numbers in this file come from `target-comp` binaries built at 08:30 UTC from db3fa84,
which predates 4554103, so they are **not affected**.

## vLLM gpt-oss-20b: the 4 -> 8 client jump (logs/comp/vllmj.log, serve/vllmj_*.json)
Total completion tok/s by concurrency, three server instances:
| run | C=1 | C=3 | C=4 | C=5 | C=6 | C=7 | C=8 |
|---|---|---|---|---|---|---|---|
| original order 1,2,4,8 | 14.36 | | 27.35 | | | | 111.37 |
| reversed order 8,1,4,6,8 | 14.81 | | 29.01 | | 100.34 | | 98.53 (first) / 138.92 (last) |
| `--enforce-eager` 1,8 | 13.98 | | | | | | 95.90 |
| threshold scan 3,5,6,7 | | 18.52 | | 71.90 | 94.72 | 96.56 | |
Findings:
- Not warm-up: running C=8 first still gives ~99 tok/s, and C=1/C=4 stay slow afterwards.
- Not CUDA/HIP graphs: `--enforce-eager` reproduces it (14.0 at C=1, 95.9 at C=8).
- It is a step-function between 4 and 5 running sequences: per-sequence decode is ~5-7 tok/s at <=4 sequences and ~14-17 at >=5,
  i.e. each decode step gets *cheaper* when the batch grows past 4. That points to a batch-size-dependent kernel choice, not scheduling.
- Candidate code paths in this build with a `<= 4` token threshold: `fused_moe/router/rocm_fp32_router_gemm.py` (launch config for
  num_tokens <= 4), `fused_moe.py` (num_stages 4 if M <= 4). Root cause is **not confirmed**; next step is a torch profiler trace at C=4 vs C=5.
- Practical consequence: vLLM single-stream (14 tok/s) is ~5x slower than llama.cpp Vulkan (74 tok/s, llama-bench) on gpt-oss here; only at >=5 clients does it pass llama-server (~70-80 tok/s total).

## Concurrency, all engines (Soothill method, total completion tok/s at C=1/2/4/8), 2026-10-15 13:50 UTC

Unique ~600-token prompts, 64 out tokens, ignore_eos, T=0, seed 12345, 2 warm-up + 10 measured waves. llama-server: `-ngl 999 -fa on -np 8 -c 16384`
(Vulkan build in ~/llama-latest/build-vulkan, ROCm 7.13 build in build-rocm713 with ROCBLAS_USE_HIPBLASLT=1). Logs: logs/comp/serve.log, logs/comp/serve/.
**Bold** = best per model and column.

| Model | Engine | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|---|
| gpt-oss-20b | vLLM 0.31.0 ROCm | 14.4 | 19.7 | 27.4 | **111.4** |
| gpt-oss-20b | llama.cpp Vulkan | **53.6** | **70.5** | 72.9 | 70.7 |
| gpt-oss-20b | llama.cpp ROCm 7.13 | 50.9 | 66.6 | **80.4** | 73.4 |
| gpt-oss-20b | FLM 1.0.8 (NPU) | 8.0 | 8.0 | 8.0 | – |
| gpt-oss-20b | strix-rt | gap (not served) | | | |
| Qwen3.6-35B-A3B | llama.cpp Vulkan | – | **51.0** | **64.4** | 69.3 |
| Qwen3.6-35B-A3B | llama.cpp ROCm 7.13 | **34.4** | 48.0 | 62.2 | **69.5** |
| Qwen3.6-35B-A3B | FLM (NPU) | 6.2 | 6.2 | – | – |
| Qwen3.6-35B-A3B | vLLM FP8 | server died at startup (diagnosis in progress) | | | |
| LFM2.5-24B-A2B | llama.cpp Vulkan | **68.1** | 83.4 | 96.8 | 86.4 |
| LFM2.5-24B-A2B | llama.cpp ROCm 7.13 | 68.0 | **88.6** | **106.6** | **99.4** |
| LFM2.5-24B-A2B | vLLM FP8 | server died at startup | | | |
| LFM2.5-24B-A2B | strix-rt | gap (no multi-slot decoder) | | | |
| MiniCPM-V 4.7 (text) | llama.cpp Vulkan | **46.9** | **62.4** | 73.9 | 66.6 |
| MiniCPM-V 4.7 (text) | llama.cpp ROCm 7.13 | 43.9 | 61.0 | **74.7** | **67.3** |
| MiniCPM-V 4.7 (text) | strix-rt | gap (no multi-slot decoder) | | | |
| Qwen3.8-27B Q4_K | llama.cpp Vulkan | **9.87** | **13.91** | **15.07** | **17.08** |
| Qwen3.8-27B Q4_K | llama.cpp ROCm 7.13 | 9.48 | 11.72 | 13.77 | 14.43 |
| Qwen3.8-27B Q4_K | vLLM FP8 | server died at startup | | | |
| Qwen3.8-27B Q4_K | FLM qwen3.8-mtp (NPU) | 1.29 | 1.28 | – | – |
| Qwen3.8-27B Q4_K | strix-serve r2 | 9.59 | 12.13 | 14.36 | 14.47 |
| Qwen3.8-27B Q4_K | strix-serve r3 (chunked prefill) | 9.64 | 11.74 | 13.77 | 13.81 |
| Bonsai 2 27B PQ2_0 | strix-serve r3 | 16.75 | 19.68 | 20.96 | 21.20 |

strix-serve rows come from the other worker (target-serve). strix_q38 run 1 (10.4/12.0/14.0/14.1) straddles a rebuild inside the PF_FA window, so it is excluded; r2/r3 are used.
The vLLM FP8 containers exited before /v1/models answered, and `--rm` discarded their logs; `tools/comp/vllm_diag.sh` reruns one without `--rm` to capture the error.

## EAGLE3 speculative decoding, gpt-oss-20b (llama.cpp Vulkan, logs/comp/spec5.log)
`tools/comp/spec_gptoss.sh`: llama-server `-ngl 999 -fa on -c 4096 -np 1`, draft `ggml-org eagle3-gpt-oss-20b-Q8_0.gguf` with `--spec-type draft-eagle3 -md ... -ngld 999 --spec-draft-n-max K`; greedy 128 tokens, 4 runs.

| Config | Decode tok/s (4 runs) | Draft accepted / drafted |
|---|---|---|
| no draft | 76.73 / 76.51 / 76.45 / 76.30 | – |
| EAGLE3 n-max 2 | 63.99 / 60.49 / 59.58 / 59.65 | 49/153, 45/161, 43/166, 43/166 (~28%) |
| EAGLE3 n-max 4 | 45.94 / 46.55 / 46.41 / 46.79 | 52/296 to 52/295 (~18%) |

EAGLE3 is slower than plain decode on this prompt (acceptance only 18–28%), so it is not a competitor threat for gpt-oss here.
strix-rt gpt-oss decode is 75.5 tok/s (dd5640a, see the afternoon update), which is about level with this 76.5 llama-server run and above the llama-bench tg128 74.2.

## Where strix-rt loses (updated 2026-10-15 13:50 UTC; supersedes the list above)
1. **Serving**: Qwen3.8 at C=8 is 14.47 (strix-serve r2) vs 17.08 for llama.cpp Vulkan (−15%). LFM2.5, MiniCPM-V and gpt-oss are not served at all; llama.cpp gets 66–106 tok/s at C=4–8 on them and vLLM 111 tok/s on gpt-oss at C=8.
2. **Model coverage**: Qwen3.6-35B-A3B Q8_0 does not run ("MoE gate/up must be Q4_K in one arena"). gpt-oss-20b decodes (75.5 tok/s) but has no default batched prefill, so no pp numbers.
3. **Prefill vs llama.cpp ROCm 7.13**: LFM2.5 pp512 2310 vs 2627 (−12%), pp2048 2354 vs 2707 (−13%), pp8192 1957 vs 2626 (−25%); Qwen3.8 pp8192 325 vs 404 (−20%); MiniCPM pp8192 1321 vs 1425 (−7%); Bonsai PTQ1_0 pp2048 288 vs 325 (−11%), pp8192 241 vs 275 (−12%).
4. **Long-context decode on Qwen3.8**: 12.48 vs 12.78 for Vulkan at depth 8192 (−2%).
5. **MTP on Qwen3.8**: strix-rt now reaches 24.2 (k=2) to 25.1 (k=3) in a quiet GPU (run6), level with llama.cpp Vulkan n-max 3 at 25.2, but not robustly under contention. llama.cpp ROCm MTP (20.2–23.7) is below it.
6. **Energy per token vs Vulkan**: LFM2.5 0.836 vs 0.634 J/token (+32%); Qwen3.8 9.02 vs 7.94–8.90 (+1–14%).

## Update 2026-10-15 (afternoon)
- **gpt-oss-20b tg: 75.5 tok/s** (default build; 77.0 with ATTN_NC=2), beating llama.cpp Vulkan 74.2; greedy parity unchanged (first 32/128 vs llama.cpp). Steps: MoE LPR=32 + GEMV_PROJ/OUT=32,128 (60.9 -> 65.9), ATTN_NC=4 (66.4), fused post-norm+router kernel go_normrouter (75.5). `logs/gptoss/tg2-4.log`.
- gpt-oss batched prefill (GO_PF=1, still default off): norm/router/MoE gate-up/down batched over columns (workgroup y = column) -> 20-token prompt 60 -> 107 tok/s at NB=4, identical greedy (`pf2.log`).
- Qwen3.8 MTP rerun (`logs/mtp/run6.log`, MTP_HEAD_Q=1 GEMVQ_NBNR_Q=2): k=3 25.14 / 24.88 / 21.68 (last rep GPU-contended, verify 104 vs 89 ms), k=2 24.16 x3; greedy exact 128/128. 25+ reached in a quiet GPU, not robustly.
- pf_attnfa (PF_FA, **still default OFF**): now Q split into f16 hi+lo; fa_check max err 4.9e-4 (f16 output ulp). PF_FA_CMP=1 on MiniCPM L512 (same inputs to both kernels, every layer): max |pf_attn - pf_attnfa| 0.0002-0.0039 per call, no NaN. Still, full FA runs flip MiniCPM L512 first token (286 -> 1091; PF_M=128/384 and L2048/L8192 match), i.e. a near-tie amplified by ~1-ulp attention differences, not a kernel bug. Kept off per parity rule. pp8192 MiniCPM FA=1 1295 (hi/lo Q) vs 1489 (f16 Q) vs 1210 FA=0, contended runs.

## gpt-oss-20b decode, fair re-measure (logs/comp/gofair.log, tools/comp/gptoss_fair.sh), 2026-10-15 14:3x UTC
Same lock session, same raw prompt ("The history of the Roman Empire can be summarized as follows:", 12 tokens, no BOS), greedy, 128 new tokens,
alternating engines twice. llama.cpp = llama-server Vulkan no-draft (`timings.predicted_per_second`); strix-rt = `go_run` at 329abd4 (includes dd5640a),
default env, wall-clock decode (GPU-timestamp decode is within 0.1%).

| Round | llama.cpp Vulkan tok/s (4 runs) | strix-rt tok/s wall (4 runs) |
|---|---|---|
| 1 | 76.71 / 76.63 / 76.67 / 76.06 | 76.90 / 76.92 / 76.81 / 76.93 |
| 2 | 73.81 / 74.67 / 74.47 / 73.47 | 74.17 / 75.71 / 75.73 / 74.22 |

Result: **parity** (strix-rt +0.3% in round 1, +1% in round 2; both engines dropped ~3% in round 2, so the round-to-round drift is larger than the gap).
The earlier "75.5 vs 74.2" compared go_run against llama-bench tg128 from a different session; on matched conditions it is a tie, not a win.

## FA prefill decision (2026-10-15)
- MiniCPM L512 top-2 logit margin with the old kernel (pf_attn): **0.0030** (286 9.0477 vs 1091 9.0448) -> near-tie (`logs/pp8k/margin.log`).
- Teacher-forced agreement after batched prefill, FA=0 vs FA=1 (`logs/pp8k/fa8*.log`): MiniCPM ref_chess/ref_gold/ref_txt/L512/L2048 = 48/48, 48/48, 48/48, 48 vs 47 (only the near-tie first token), 13/13; Qwen3.8 seeds 1-5 at L256/512/512/768/1024 = 32/32 for both kernels. All within 1 token.
- Clean-GPU pp8192 (wait_quiet, alternating, `fa9.log`): MiniCPM FA=0 1224 / 1306 vs FA=1 1401 / 1389 tok/s (+7-14%). Qwen3.8 noisy (FA=0 294 / 264, FA=1 401 / 284; per-rep times 20-35 s), no slowdown, median about +3%.
- **Decision: PF_FA default ON** (commit 5a6efbf). PF_FA=0 restores pf_attn.
- gpt-oss GO_PF=1 on a 762-token prompt (`logs/gptoss/pf3-5.log`): prompt 76 -> 135 tok/s, but greedy diverges at token 54/64 for NB=1, 2 and 4. Both paths are deterministic run to run. **GO_PF stays default OFF** until the column path is made bitwise identical to decode.

## vLLM gpt-oss 4 -> 5 client step: root cause (confirmed), 2026-10-15
Traces: `tools/comp/vllm_prof.sh` (torch profiler, `--profiler-config {"profiler":"torch",...}`), analysis `tools/comp/vprof_an*.py`, traces in logs/comp/vprof/c4, c5 (not committed, 5-9 MB each).
Under the profiler the step reproduces (C=4 28.6, C=5 82.5 tok/s). Both runs execute the same 3216 model steps; total GPU kernel time is 118.0 s at C=4 vs 51.2 s at C=5.
The dominant kernel is the triton_kernels MoE matmul `_matmul_NNT_bf16xbf16xmxfp4_32x256x128x1(_swiglu)` (same tile name in both): median 1890 us at C=4 vs 677 us at C=5 (2.8x slower with fewer tokens).
Cause: `vllm/third_party/triton_kernels/matmul_details/opt_flags.py` (AMD path) sets `num_warps = 2 if m <= 16 else 8`, where m = tokens x top-k.
gpt-oss uses top-4, so C<=4 gives m<=16 -> 2 warps per 32x256 tile, which starves the gfx1151 CUs; C=5 gives m=20 -> 8 warps.
Confirmation (`tools/comp/vllm_nw8.sh`, bind-mounting `tools/comp/vpatch/opt_flags_nw8.py` with `num_warps = 8` always; logs/comp/vnw8.log):

| C | stock vLLM | num_warps=8 patch |
|---|---|---|
| 1 | 14.4 | 30.6 |
| 2 | 19.7 | 46.3 |
| 4 | 27.4 | 72.2 |
| 5 | 71.9 | 81.6 |
| 8 | 111.4 | 106.5 |

The step disappears, and C=1-4 throughput rises 1.7-2.6x. So the stock vLLM C=1-4 gpt-oss numbers are a vLLM tuning bug on RDNA3.5, not a hardware limit.
Even patched, vLLM C=1 (30.6) stays far below llama.cpp (53.6) and strix-rt single-stream (~76). At C=8 vLLM (106-111) remains the best gpt-oss server measured.

## Update 2026-10-15 (evening)
- **gpt-oss GO_PF default ON** (a5f62a8): margin at the divergent token 54 = 0.019 (pred 1 vs 192031) -> near-tie. Teacher-forced after prefill, GO_PF=0 vs 1, 5 prompts (762-1112 tokens: benchmarking.md, comparisons.md, gptoss lib.rs, pf_attnfa.comp, mcv llm.rs): 64/64 vs 63/64, 64/64 vs 64/64, 64/64 vs 64/64, 64/64 vs 63/64 (gap 0.008), 64/64 vs 64/64 (`logs/gptoss/pf6.log`). Bisect not needed. Prefill is still small-batch (4 tokens/pass, ~135 tok/s), far from llama.cpp pp.
- **Qwen3.6-35B-A3B Q8_0 now runs end to end** (the "gate/up must be Q4_K" item above is stale): pp512 1467 / pp2048 1430 / pp8192 1232 tok/s with FA (`logs/q8moe/bench2.log`) vs llama.cpp Vulkan 1491 / 1484 / 1265 (-2 / -4 / -3%) and ROCm 1078 / 1441 / 1271. Decode tg128 56.5 (vs 55.2); decode at depth (63 steps after the prompt; output truncated at EOS only for display): 55.8 @512, 54.1 @2048, **50.5 @8192 vs Vulkan 51.2 (-1%)**.

## vLLM FP8 failure and AWQ substitute, 2026-10-15
Diagnosis (`tools/comp/vllm_diag.sh`, logs/comp/serve/diag_LFM2.5-24B-A2B-FP8.log): `NotImplementedError: No FP8 MoE backend supports the deployment configuration`.
In this vLLM build `RocmPlatform.supports_fp8()` is `on_cdna() or on_rdna4()`, so gfx1151 (RDNA3.5) has no FP8 kernels at all. No flag fixes this; vLLM has no FP8 path for this GPU.
Substitute: AWQ INT4 checkpoints (cyankiwi/Qwen3.8-27B-AWQ-INT4, QuantTrio/Qwen3.6-35B-A3B-AWQ, cyankiwi/LFM2.5-24B-A2B-AWQ-INT4), `tools/comp/batch5.sh`.
serve_bench.sh now keeps the container until its logs are saved (the earlier `--rm` lost the FP8 errors).

| Model | Engine | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | vLLM AWQ-INT4 | 21.9 | 30.5 | 49.3 | 63.6 |
| LFM2.5-24B-A2B | vLLM AWQ-INT4 | 31.7 | 26.4 | 33.8 | 44.2 |
| Qwen3.8-27B | vLLM AWQ-INT4 (r3, --max-num-seqs 64) | 8.46 | 10.93 | 13.65 | 10.06 |

Qwen3.6 AWQ on vLLM trails llama.cpp Q8_0 (Vulkan 51.0/64.4/69.3 at C=2/4/8; ROCm 34.4 at C=1) at every C.
Qwen3.8 AWQ: the first start died with the log lost; the second failed with `max_num_seqs (1024) exceeds available Mamba cache blocks (1007)`; r3 with `--max-num-seqs 64` ran.
LFM2.5 AWQ on vLLM is far behind llama.cpp Q4_K_M (Vulkan 68.1/83.4/96.8/86.4) at every C; C=2 is below C=1 as measured.
Qwen3.8 AWQ trails llama.cpp Vulkan (9.87/13.91/15.07/17.08) and strix-serve r2 (9.59/12.13/14.36/14.47) at every C; its C=8 (10.06, TTFT 26 s) is below its C=4 as measured, not explained.
All vLLM rows except gpt-oss use the stock opt_flags (the num_warps finding affects only the MXFP4 MoE path).

### 2026-10-15 prefill FA follow-ups
- gpt-oss GO_PF (batched prefill, default on) now uses fused nrouter: bitwise IDENTICAL to decode on 5 prompts (0c9be12, logs pf8.log).
- PF_FA_QLO default 0 (single f16 Q, no hi/lo split): Qwen3.6-35B-A3B pp2048 ~1424->1449 tok/s, pp8192 1212->1287 tok/s (Vulkan 1265: now ahead). Agreement (teacher-forced): MiniCPM 48,48,48,47,13/13; Qwen3.8 5 seeds 32/32 -- all within 1 token of the FA=0 baseline. logs/q8moe/qlo.log, logs/pp8k/qlo_*.log.
- PF_FA64 (LFM head-dim-64 FA): pp8192 1951->2175 tok/s (+11%), pp512 ~+1%. Kept OFF by default: teacher-forced on 5 LFM text prompts (L768, 32 tok) FA64=0 vs 1 = 32/32, 31 vs 29, 31 vs 32, 32/32, 32/32 -- p2 differs by 2 tokens, failing the within-1 rule. logs/pp8k/fa64b.log.
- FA64 follow-up (logs/pp8k/fa64c.log, fa64d1/2.log): forced agreement is not deterministic run-to-run (p2 FA64=0 gives 32/32 or 31/32 on repeats), and the old p2 FA64 mismatches at 17/25 did not reproduce (a new one at 20 has gap 0.023). FA64+QLO=0 on p1-p5 was within 1. On p6-p10: FA64=0 vs 1 = 29/30, 31/31, 31/30, 32/29, 32/32. p9 fails (3 mismatches, gaps 0.030, 0.126, 0.153 -- two are not near-ties), so FA64 stays OFF.
- LFM2.5 expert GEMMs are ~80% of prefill (PF_SKIP=1: 2230 -> ~11000 tok/s). PF_HYB=1 (mixed 64/32-row expert tiles) is now the default (56283f8): pp512 2230->2338 (+5%), pp2048 2327->2417 (+4%), alternated runs with cool-down; teacher-forced agreement on 10 prompts all within 1 token (logs/lfmx/hyb1/2.log). Gap to ROCm is now about -11% at pp512 (2627) and -11% at pp2048 (2707).

### strix-serve1 multi-sequence (LFM2.5, MiniCPM-V), 2026-10-15

Every sequence gets its own state slot: KV, conv/DeltaNet state, token ring and aux words, plus its own recorded step. `with_st` swaps the slot into the model, so prefill and decode run unchanged per sequence. Steps run round robin. In r2, the steps of all greedy slots go to the GPU in one submission, separated by barrier command buffers. This is concurrency without weight sharing: each token still reads its own weights, so aggregate throughput stays flat.

- Equality: MiniCPM-V passed 8/8. LFM2.5 passed only 7/8 at first: a near-tie flipped late (',' vs ';') because of float-atomic norm sums in gemv3. The fix is gemv3 flag 128, which sums the norms deterministically. LFM2 turns it on with `LFM_DET=1` (default off, so other users of the crate are unaffected; strix-serve1 sets it). After the fix LFM2.5 passed 8/8.
- Results, completion tok/s at C=1/2/4/8 (with mean TTFT):
  - LFM2.5 r2: 71.7 / 73.4 / 74.6 / 75.2 (TTFT 333 / 506 / 858 / 1591 ms)
  - LFM2.5 r1 (one submit per slot): 73.9 / 73.6 / 72.2 / 72.5
  - MiniCPM-V r2: 54.7 / 55.5 / 56.1 / 56.6
  - llama.cpp Vulkan, MiniCPM-V: 46.9 / 62.4 / 73.9 / 66.6
- strix leads at C=1 and falls behind from C=2. Beating llama at C≥2 needs real weight-shared batched decode: NB-column gemv for the dense parts, with per-column attention, DeltaNet/conv and MoE. That is not done yet.

### strix-serve r9 (Qwen3.8): GEMV lanes/2 with 2x workgroup rows, 2026-10-15

Same settings as r8, plus `GEMV_PROJ=8,128 GEMV_OUT=16,256 GEMV_SWIGLU=16,256 GEMV_DOWN=8,128`. These came from decode power sweeps (`serve_steps SPOW`, logs/serve/dec_power*_q38.log). In decode-only runs this config used about 8% less energy per token at batch sizes 1 to 4. Equality check passed 8/8.

| C | tok/s | TTFT ms | J/out-tok r9 | r8 | llama.cpp Vulkan (fan) |
|---|---|---|---|---|---|
| 1 | 10.26 | 1372 | 11.09 | 11.08 | **10.58** |
| 2 | 16.33 | 2554 | 7.73 | 7.79 | **7.72** |
| 4 | 22.28 | 4125 | **6.25** | 6.43 | 6.26 |
| 8 | **25.19** | 7334 | **5.16** | 5.22 | 5.77 |

- C=8 rose from 23.62 to 25.19 tok/s.
- J/token at C=2 and C=4 now matches llama.cpp, within noise.
- C=1 is still about 5% worse.
- The bench J includes prefill, and package power sits at its ~140 W cap at batch sizes of 3 and up.
- The remaining gap at C=1 is GPU-core power: 68 W against llama's 40 W.

### strix-serve1 batched decode (record_multi), LFM2.5 and MiniCPM-V, 2026-10-15

Each step now decodes all active sequences together: every dense projection is one gemvq over the m token columns (up to 8 per weight read). Attention, conv/DeltaNet and the routed MoE run per column on that sequence's state slot, using small per-column scratch buffers. LFM2's fixed-layout kernels (embed, shortconv, rope, router, gemv3 MoE) get their inputs and outputs copied in and out of the scratch. m=1 runs through the same path and is the single-stream reference. Equality: 8/8 for both models (logs/serve/eq1_*_m1.log). Each benchmark concurrency level ran in its own gpu_lock session (`serve_split.sh`).

Decode step alone (`s1_steps`, fresh slots, pos 600), tok/s at m=1/2/4/8:
- MiniCPM-V: 80.7 / 128.7 / 177.5 / 221.4
- LFM2.5: 116.2 / 194.3 / 281.6 / 366.2

Server benchmark, completion tok/s, TTFT and J/out-tok at C=1/2/4/8:

| Model | Server | tok/s | TTFT ms | J/out-tok |
|---|---|---|---|---|
| MiniCPM-V | strix m1 | **53.4 / 71.0 / 85.9 / 95.1** | **370 / 556 / 933 / 1703** | **2.08 / 1.64 / 1.46 / 1.39** |
| MiniCPM-V | llama Vulkan | 46.9 / 62.4 / 73.9 / 66.6 | 490 / 939 / 1895 / 4580 | 2.47 / 1.93 / 1.69 / 1.89 |
| LFM2.5 | strix m1 | **71.3** / 86.3 / 98.5 / **107.6** | 316 / **484 / 816 / 1489** | **1.40 / 1.20 / 1.13 / 1.09** |
| LFM2.5 | llama Vulkan | 68.1 / 83.4 / 96.8 / 86.4 | 353 / 631 / 1250 / 3420 | 1.63 / 1.40 / 1.28 / 1.44 |
| LFM2.5 | llama HIP | 68.0 / **88.6 / 106.6** / 99.4 | **278** / 518 / 1003 / 2364 | 1.59 / 1.39 / 1.28 / 1.34 |

The llama rows are from earlier runs, before the fan curve.

- strix-serve1 now beats llama.cpp Vulkan on throughput, TTFT and J/token at every concurrency for both models.
- llama.cpp HIP still has higher throughput on LFM2.5 at C=2 and C=4.
- The server numbers sit well below the step bench because prefill is serialized: one prompt is admitted per loop iteration. The next lever is batching prefill across prompts, as strix-serve already does for Qwen3.8.

## Qwen3.8 J/token at C=1: row-tiling experiment (dec_power3)

Decode step power (serve_steps SPOW), m=1/m=2. Config = rows-per-WG,WG-size for the 4 dense projection groups:

| Config | m=1 ms/step | m=1 W | m=1 J/tok | m=2 J/tok |
|---|---|---|---|---|
| A r9 (8,128 16,256 16,256 8,128) | 76.1 | 103.9 / 106.3 | 7.91 / 8.09 | 4.66 / 4.73 |
| B (8,256 16,512 16,512 8,256) | 77.0 | 99.0 | 7.62 | 4.67 |
| C (4,128 8,256 8,256 4,128) | 80.4 | 97.9 | 7.87 | 4.70 |
| D (8,256 x4) | 78.1 | 104.5 | 8.16 | 4.70 |

Run-to-run spread for A alone is ~0.2 J (2.4 W). B is the best by ~0.3 J/token, which is close to that noise, and it costs about 1 ms/step. So the L2-reread hypothesis gives no real win. C=1 decode is bandwidth-bound at ~100 W GPU-package draw, and llama.cpp draws less power at a similar step time. The remaining options are clock/power-state limits (sysfs power_dpm levels, not tested because they would affect the other workers) or less work per token (MTP).

## Nemotron 3.5 Super VL 120B-A12B Q4_0 (perf high, 2026-10-16)
| Engine | pp512 tok/s | tg128 tok/s | J/tok socket | J/tok gfx |
|---|---|---|---|---|
| llama.cpp Vulkan f3363593f (fa on) | 451.07 | 18.66 | 5.286 | 1.476 |
| strix-rt n35 | n/a (no batched prefill) | 17.65 | 5.120 | 2.032 |
Parity: teacher-forced 285/288 (misses <0.09 nats margin); vision cosine 0.999.

### strix-serve1 batched prefill rerun on a fresh build, 2026-10-16 (tools/comp/batch_day2b.sh; cargo build -p serve, target-serve)
Same conditions as the pfb rows, perf level via run.sh, 20-80 requests per level, success 100%. completion tok/s (mean TTFT ms):
| model | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| LFM2.5-24B-A2B Q4_K_M, strix-serve1 (pfb2) | 70.8 (320) [pfb run] | 94.0 (497) | 116.3 (873) | 131.8 (1464) |
| MiniCPM-V 4.7 35B-A3B Q4_K_M, strix-serve1 (pfb) | 52.1 (400) | 71.6 (703) | 91.3 (1311) | 100.7 (2308) |
MiniCPM-V greedy batched == single-stream 8/8 (logs/serve/eq1_mcv_pfb.log); text smoke deterministic (logs/serve/s1test_mcv_pfb.log; that log has no image line, the image check was not re-run).
Two-model unified server (LFM + MiniCPM-V in one strix-serve process, `--aliases lfm,mcv`, logs/serve/merge2.log): /v1/models lists both, routing by "model" works, concurrent requests to both models return, unknown model gives 404. The C=1 LFM pfb2 value is from the earlier pfb run (no C=1 in this batch).
Not done: llama.cpp ROCm reruns and the remaining table rows.

### llama.cpp ROCm 7.13 rerun, 2026-10-16 (tools/comp/rocm_rerun.sh, logs/comp/rocm_rerun.log; lock held, perf level high, -r 3)
LFM2.5-24B-A2B Q4_K_M: pp512 2677.3 / pp2048 2723.0 / tg128 103.0 (table above: 2626.7 / 2707.0 / 100.7). Qwen3.8-27B UD-Q4_K_M: pp512 424.4 / pp2048 426.4 / tg128 12.85 (table: 425.4 / 426.0 / 12.9). The earlier ROCm rows reproduce within 2%. The strix-rt Q8_0 MiniCPM row above was filled from logs/q8moe/bench2.log (it was stale pending).
Still pending: gpt-oss strix-rt tg @ 8192, FastFlowLM gpt-oss tg @ 8192.

## Nemotron 3 Nano 30B-A3B Q4_K_M prefill (2026-10-16)
pp511: strix-rt 1122 -> ~1355 tok/s after the Mamba scan head-dim split and the 4-token router (generated ids identical on 6 prompts); llama.cpp Vulkan 1687. Still behind (-20%). Details and the per-op profile: docs/prefill.md.

Gemma 4 26B decode energy attribution (2026-10-16, logs/g4/eskip.log, energy-recorder snapshots at perf high, `G4_SKIP` masks, ~35 samples each, +-1.5 W run-to-run):
full 37.0 W gfx at 54.7 tok/s = 0.677 J/token gfx (llama.cpp 0.569 J = ~29.6 W at 52 tok/s, same ~92% gfx activity and 2900 MHz). Per-kernel-group power from
differences (time per token / power): routed experts 4.08 ms at ~48 W (29% of gfx energy), qkv GEMVs 3.25 ms ~39 W, lm_head 3.44 ms ~37 W, dense FFN 2.44 ms ~34 W, attention 3.5 ms ~26 W,
remaining small kernels 5.86 ms ~37 W. The expert GEMVs are the hottest per busy cycle; attention is the coolest. lm_head alone is 19% of decode time and 19% of gfx energy.

### Gemma 4 26B decode: Q5_1 expert-down cheaper dequant (G4_DQ2=1, opt-in) 2026-10-16
Byte-packed unpack (qh spread via multiply, hoisted x-sum) in g4_moe_dn_v2.comp. perf_high, 5 alternating pairs, 512-token decode: 54.67 -> 54.82 tok/s (+0.3%); gfx power 37.3 -> 36.5 W mean (paired diff -0.5..-1.0 W each pair, DQ2=1 always run second/warmer); gfx J/token 0.682 -> 0.666 (-2.3%). Parity (FORCED, vs llama.cpp Vulkan): 26bc 62/64 -> 63/64 (baseline's mismatch was a 0.007 near-tie); degenerate repeat prompt 26b 62/65 -> 61/65 (mismatch gaps ~1 either way, both runs already disagree with the reference there). lm_head (Q8_0 token_embd 784 MB, 3.44 ms = 228 GB/s) and experts (~0.9 GB/token in 4.08 ms = ~220 GB/s) are already at the DRAM wall, so only power-per-byte (ALU) is left to cut; lm_head not changed.
G4_DQ2 parity, 7 more chat prompts (teacher-forced vs llama.cpp Vulkan, 64 tok; agree DQ2=0 -> 1): 62->63, 64->64, 61->61, 63->64, 64->64, 60->60, 63->63 (total 437 -> 439/453); every mismatch is a <=0.3-gap near-tie or the final end-of-turn token; per-prompt never worse than baseline by >1. G4_DQ2 is now the default (G4_DQ2=0 restores the old kernel).

Gemma 4 26B decode bandwidth check per group (bytes from the GGUF, time from the G4_SKIP attribution, 2026-10-16): lm_head Q8_0 784 MB / 3.44 ms = 228 GB/s; routed experts (8/128 of Q4_K gate_up 8.56 GB + Q5_1 down 5.52 GB, ~0.9 GB/token) / 4.08 ms = ~220 GB/s; dense FFN Q8_0 (3 x 190 MB x 8/... per-token all layers 569 MB) / 2.44 ms = 233 GB/s; attention q/k/v/o Q8_0 (~1.18 GB total weights, read once per token) / 3.25+3.5 ms. Every weight-streaming group already runs at the ~220-230 GB/s DRAM wall, so wider loads / cheaper dequant cannot shorten decode; only watts per byte can move (G4_DQ2 gave -0.8 W on the Q5_1 down). Dense FFN/qkv/gate_up energy items therefore not pursued as kernel rewrites.

Nemotron 3.5 Super VL status (another agent's commits 1db238e..6ed9da8 in crates/nemotron35, tree clean): 88-layer load works (68139 MiB in 41 arenas, ZEROCOPY=0), text parity teacher-forced 285/288 vs llama Vulkan, vision cosine 0.999, decode 17.65 tok/s (llama 18.66). MTP (blk.88 nextn head: eh_proj 8192->4096 + attn + LatentMoE) not implemented; expectation (not measured): 512 experts top-22 means two verified tokens share ~0.95 of 22 experts, so a 2-token verify reads ~2x the expert bytes and MTP is near break-even at best for decode throughput.

### Nemotron 3.5 Super VL batched prefill (crates/nemotron35/src/prefill.rs, n35 PF=1), 2026-10-16, perf_high, 3 runs
pp511 (p512 ids, 511 tokens prefilled + decode step): 436.2 tok/s GPU (1.17 s) vs llama.cpp Vulkan pp512 451.07 (-3%); token-by-token before: 17.6 tok/s (25x). pp199: 279 tok/s. Decode unchanged 17.7 tok/s. Per-op (logs/n35/pf_prof1.log + later profile, 852 ms chunk): down_exps 229, up_exps 197, ssm_in 92, scan 91, shexp down 69, ssm_out 55, shexp up 34, router 15, latent down 11. First version used the nh_psel-style f32 router (10 ms/layer = 399 ms of a 1245 ms chunk); router logits as an f16 GEMM (BF16 router weights converted exactly to f16 once) + subgroup top-k took it to 15 ms. New: Q4_0 (ty 16) in pf_deq.glsl, n35_psel/psort/pcomb (512 experts, top-22), latent gather/GEMM path.
Parity vs ref_llama (6 prompts x 48): greedy 264/288, teacher-forced 285/288 (same as the decode path; misses at llama margin <0.09 nats; teacher-forced uses the decode path). Prefill vs token-by-token ids: p30/p90/p200 32/32 identical; the 512-token p512 prompt diverges at generated token 9 (near-tie suspected, margin not measured).
MTP feasibility on the Nemotron 3.5 Super path (measured, not implemented): verify cost through the batched prefill path (PFPROF, PF_N=n, perf_high, timestamp sum) is 109 ms for 1 token, 125 ms for 2, 132 ms for 3, 135 ms for 4, versus 56.6 ms for the GPU-driven decode step. Even at 100% draft acceptance a 2-token verify through this path yields 2/(125+draft+state snapshot) = <16 tok/s < 17.7 tok/s baseline, so MTP cannot win on the batched kernels (M=1-4 GEMMs run ~3x off the DRAM floor: ssm_in 0.62 ms/layer vs 0.21 ms of weight bytes). A win needs a dedicated multi-column decode path (2-column GEMV sharing weight reads, expert union for 44 of 512 experts, 2-step conv/scan with state snapshot); estimated step ~75 ms for 2 tokens => break-even acceptance ~0.33, +28% at acceptance 0.7 (estimate, unmeasured).

## Nemotron 3.5 Super VL: MTP speculative decode (opt-in, `MTP=n` in n35 bin, needs `PF=1`, `N35_DET=1`)
Dedicated 2-column verify (shared weight reads, per-column experts, alt conv/scan state, flip on accept) = 68.5-69.5 ms per 2-token cycle vs 56.6 ms single decode. blk.88 head (MTP GGUF) k=1 draft, 2.15 ms/call, fed the post-norm hidden (H_MODE=1; pre-norm gave only 42% acceptance, 19.5 tok/s).
Measured at perf_high, 128 tokens, greedy output identical to plain N35_DET=1 decode (128/128) on all 5 prompts:
| prompt | accept | tok/cycle | MTP tok/s | plain tok/s |
|---|---|---|---|---|
| p30 | 86.8% | 1.868 | 25.73 | 17.74 |
| p90 | 88.2% | 1.868 | 25.39 | 17.76 |
| div30 | 81.4% | 1.814 | 24.81 | 17.68 |
| p200 | 66.2% | 1.649 | 22.65 | 17.69 |
| long_prompt | 81.4% | 1.814 | 24.77 | 17.70 |
Head KV starts empty (not seeded from the prompt); seeding and k>1 drafts are untried. Log: logs/n35/mtp2.log.

### Super MTP round 2 (SEED=1, CHEAP=1 defaults; still opt-in)
- Head KV seeded from the prompt (prefill captures output-normed final hidden per position, one KV-fill head call per prompt position) and a KV-fill-only head call on the accept path (skips q/attn/MoE/lm_head: 2.15 -> ~1.25 ms/call average).
- p200 acceptance 66.2% -> 75.3% (SEED=0/CHEAP=1 gave 66.2%, 23.13 tok/s; SEED=1 24.36 tok/s). Other prompts rose too.
- 6 prompts x 128 tokens, perf_high, greedy output identical to plain N35_DET=1 (128/128 on all): p30 86.8% 25.75 tok/s (plain 17.30); p90 95.4% 27.02 (17.32); div30 92.4% 26.48 (17.18); p200 75.3% 24.38 (17.74); long_prompt 88.2% 25.30 (17.02); p512 71.6% 23.44 (17.26). Log: logs/n35/mtp3.log.
- Energy (tools/energy/record.py snapshot sampling, p30, 256 tokens, socket W over gfx_activity>50 samples, 2 runs each; socket W is noisy, +-10 W between runs): plain 101.9/104.2 W at 17.49/17.59 tok/s = ~5.9 J/token; MTP 115.9/97.1 W at 26.06/26.00 tok/s = ~4.1 J/token (-30%), because the extra column reuses the weight stream. logs/n35/mtp_energy.log.
- k>1 drafting NOT implemented: needs a 3-column verify (every kernel, conv/scan state chain, AOFF for col 2). Scaling estimate from measured 1 col 56.6 ms -> 2 col ~69 ms (+12.4 ms/col) gives ~81 ms for 3 columns; at ~0.85 first-draft acceptance and a lower second-draft rate the projected gain is small (a few %), so it was not worth the rewrite yet. Unmeasured projection.

### Nemotron 3.5 Super: MTP is now the default (how to run)
`target-n35/release/n35 <super.gguf> "<ids>" [n_new]` runs MTP speculative decode by default (PF=1 batched prefill and N35_DET=1 exact summation order are defaults too; the draft head is read from `$MTP_GGUF`, default `$MODELS/new/nemotron35vl/mtp-...-Q4_0.gguf`, falling back to plain decode if missing). Use `ZEROCOPY=0`. Off-switch: `MTP=0` (plain greedy). `MTP_CMP=n` runs plain then MTP n times and prints the greedy match; `SEED=0` / `CHEAP=0` / `H_MODE=0` disable head-KV seeding / the KV-fill-only accept call / the post-norm hidden input. Output ids are identical to MTP=0.
Exactness vs `MTP=0`, same binary, perf_high: EXACT on p30, p90, p200, p512, div30 (840-token prompt), long_prompt (830) at 128 tokens, and p30 and p200 at 512 generated tokens. MTP decode 24.0-27.3 tok/s (128 tok), 26.4 tok/s (512 tok, 87.5-88.6% accepted) vs plain 17.7. Head seeding cost (GPU): 0.7 ms (7 positions), 7.2 ms (89), 15.6 ms (199), 40.4 ms (511), 68 ms (830-840), about 0.08 ms per prompt position. Logs: logs/n35/def2.log, def3.log. k>1 drafting not done (needs a 3-column verify).

## Meta Muse Glimmer 30B (dense, arch `muse-glimmer`): llama.cpp baseline
- Model: huggingface.co/meta-models/Muse-Glimmer-30B-GGUF (not gated; Apache 2.0). Downloaded to $MODELS/new/glimmer: `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (16.76 GB, 15.59 GiB, 27.85 B params as loaded), `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` (1.40 GB), `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` (1.63 GB drafter). Sizes match HF content-length. Script: logs/glimmer/dl.sh.
- llama.cpp: no new build needed; the existing ~/llama-latest (b11654, e36df6047) already contains `src/models/muse-glimmer.cpp` (>= b10353) with build-vulkan and build-rocm714.
- llama-bench, perf_high, -ngl 999 -fa 1, r=3 (logs/glimmer/bench_llama.log):
| backend | pp512 | pp2048 | tg128 |
|---|---|---|---|
| Vulkan | 451.06 +- 0.67 | 440.47 +- 1.06 | 13.14 +- 0.01 |
| ROCm (hip 7.14) | 465.79 +- 3.05 | 461.22 +- 0.23 | 12.77 +- 0.05 |
- Decode energy (tools/comp/energy.sh differential N=320 minus N=64, energy recorder socket): Vulkan 6.98 J/token socket (2.27 gfx), ROCm 8.12 J/token socket (3.24 gfx) (logs/glimmer/energy_llama.log, logs/comp/energy/glimmer_llama_*).
- Reference outputs (llama-server Vulkan, greedy, 128 tokens, top-2 margins): logs/glimmer/ref_llama.json (6 text prompts, same set as logs/n35/prompts.json; prompt 5 'In 1969, the Apollo 11 mission' has a min top-2 margin of 0.007 nats, a near-tie) and logs/glimmer/ref_llama_img.json (image prompt: logs/ test image, 368 prompt tokens with mmproj, media marker from /props, output begins ' a red fox sitting in a snowy forest...'). Scripts: sess_ref.sh, ref_llama.py, ref_img.py.
- Decode is DRAM-bound: 15.6 GiB weights at 13.1 tok/s is about 220 GB/s, the same wall as Gemma.

### Nemotron 3.5 Super VL: real-image input (2026-10-17)
strix-rt now takes real PNG/JPEG images end to end (`n35vl`): Rust decode, mtmd-identical dynamic-resolution preprocessing (smart_resize to a multiple of 32 within 262144..3407872 pixels, Pillow bicubic, CLIP mean/std), ViT on GPU, embeddings injected into prefill. Preprocessed pixel tensors are bit-exact with llama.cpp mtmd for PNG and JPEG (6/6 images; JPEG via a faithful Rust port of stb_image's decoder, also byte-exact on a 34-file corpus of baseline/progressive/subsampling/CMYK/gray variants), and greedy answers match llama.cpp mtmd-cli token for token to EOS on 6 of 6 images. ViT encode vs llama.cpp mtmd (single runs): 256 tokens 156 vs 328 ms, 1024 tokens 1073 vs 1516 ms, 1457 tokens ~1.6-1.9 s vs 2460 ms, 3290 tokens 8.4 vs 8.26 s (equal). Strix-rt encode energy (socket, differential, windows >= 30 s): 13.6 J (256 tok), 88 J (1024 tok), ~159 J (1457 tok, noisy); llama.cpp image-encode energy not measured. Details: docs/new-models.md.

## Decision ("System One") models: Kev, Laya (2026-10-17)
No competitor engine exists for these (no llama.cpp/vLLM path for the pointer head / Laya scorer), so rows are strix-rt only; parity is against upstream PyTorch (see docs/decision_models.md).
Energy: energy recorder differential, perf high, GPU lock, 3 repeats, windows >=30 s, medians. Socket J includes unrelated CPU load on a busy box (load avg 19-30), so smaller-model socket numbers are upper bounds.
| model | parity (argmax / max abs dprob) | 2048-tok state J (socket/gfx) | 8-question doc J | Laya per 3-question request J |
|---|---|---|---|---|
| Kev-0.8B bf16 | 4/4, 0.0051 | 34.0 / 9.2 | 42.6 / 11.2 | |
| Kev-4B bf16 | 4/4, 0.0006 | 187 / 18 | 185 / 43 (noise-dominated) | |
| Kev-9B Q8_0 | 4/4, 0.0125 | 257 / 27 | 292 / 31 | |
| Kev-9B bf16 | 4/4, 0.0009 | 276 / 26 | 314 / 31 | |
| Kev-27B Q8_0 | 4/4, 0.0043 | 688 / 90 | 788 / 104 | |
| Laya F16 | 9/9, 0.0006 | | | 16.0 / 1.9 |
Jev: hosted closed model, not portable. ThinkThen: client; `thinkthen check` 8 ok against kev_serve (Kev and Laya).
