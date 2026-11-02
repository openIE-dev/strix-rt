# beam progress (interruption notes)
- Download: $MODELS/new/beam/nvfp4 (resumable: re-run $MODELS/new/beam/nvfp4/dl2.sh; log dl.log ends with ALLDONE). config/tokenizer there.
- Build: `PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=target-beam cargo build --release -p beam --offline`; bin: target-beam/release/beam DIR ids_csv [n_new]
- Index: python3 tools/beam_index.py $MODELS/new/beam/nvfp4 (after download completes)
- llama.cpp ref: ~/llama.cpp-beam (build-vk built). Convert NVFP4->GGUF with its convert_hf_to_gguf.py once the download finishes.
- Code compiled; not yet run (waiting for all shards).
- Download complete + verified (53 shards). beam.idx built. 7-layer smoke test ran, no NaN. NVMe: 5.8 GB/s single stream, ~7 GB/s parallel.
- Next: full-model run vs llama.cpp (convert NVFP4->GGUF), then perf/energy.
- [07:30] FULL MODEL RUNS. "The capital of France is" -> " Paris" then ".\nThe capital" (logprob -0.22). ~1.0-1.2 s/token cold-ish, stream 7.8 GB/s, 8.7 GB/token. log logs/beam/first.log
- Layers 10-13 shards had been corrupted by concurrent curl; re-downloaded (fix.sh). Verify sizes with header check before trusting.
- Next: llama.cpp GGUF-NVFP4 parity (convert in $MODELS/new/beam/gguf, log convert.log), energy, docs.
- [08:06 UTC] GGUF convert: first try OOM-killed (137, no temp file). Rerun detached: $MODELS/new/beam/gguf/conv.sh (--use-temp-file, TMPDIR there), log convert.log ends with "EXIT n"; output beam-nvfp4.gguf. Then: llama-cli/perplexity parity on "The capital of France is" via ~/llama.cpp-beam/build-vk/bin.
- [08:50 UTC] results written to docs/beam.md, comparisons.md, new-models.md. llama.cpp NVFP4 GGUF reference garbage on CPU.
- [11:25 UTC] Added: chunked parallel reader (BEAM_RD_THREADS=32, BEAM_RD_CH=1MB default), BEAM_DIRECT (O_DIRECT), BEAM_EVICT (fadvise DONTNEED = cold), speculative prefetch (BEAM_PF_DEPTH/TOP; hit-rate d1 .87 d2 .79 but NO speed gain: disk-saturated), batched prefill (BEAM_PREFILL=1, union experts, PF_T=16, 128 slots; meta stride 8).
  Cold results (logs/beam/rd): first token 1.77s (old) -> 1.55s (32x1MB); O_DIRECT steady 1.37 s/token (6.9 GB/s = NVMe limit). Prefill bench queued behind another agent GPU lock: logs/beam/pre_bench.sh -> logs/beam/pre/.
- [12:05 UTC] energy2 + docs done; prefill verified identical; prefetch no gain. Remaining ideas: expert pinning, second drive.
- [13:00 UTC] Task: pinning + lower-bit experts. Added: tools/beam_prompts.py (12 prompts -> logs/beam/prompts.tsv), logs/beam/prof.sh (baseline gen + routing trace, queued behind cascade GPU lock -> logs/beam/prof/), tools/beam_requant.py BITS OUTDIR [l0 l1] [--device cuda] (NVFP4->2/3-bit g32 fp16 scale; rel-MSE layer3 2-bit .118, 3-bit .032; record 12.3MB/17.2MB vs 22.1MB), shaders bm_qgu/bm_qdn + BEAM_QDIR=$MODELS/new/beam/q2|q3 in lib.rs, BEAM_FORCE=FROM teacher-forced top-20 dump in beam bin.
- Next: run requant on cuda for both (under gpu lock), eval drift vs baseline, then expert cache/pinning from prof traces.
- Phase3 status: q2 requant done (EXIT 0), q3 running. phase3.sh (queued on gpu_lock) = wait q3 -> logs/beam/evalq.sh (teacher-forced base/q2/q3 + free runs; analyze with tools/beam_evalq.py) -> logs/beam/bench.sh (speed+energy direct IO, nvfp4/q2/q3, +cache variants BEAM_CACHE=K BEAM_WARMSEQ). Cache implemented (per-layer LRU, GPU-visible, BEAM_CACHE). Popularity analysis (12 prompts, traces in logs/beam/prof): heldout static pin top-K/layer hit: 16:.21 32:.33 64:.49 96:.61 128:.71 (uniform=K/256); LRU over sequence: 32:.53 64:.70 96:.79 128:.85.
- q2 quality (564 teacher-forced positions, 12 prompts): top1 agree .738, mean dlogp(chosen) -1.60, KL(top20) .83; p(chosen) .871->.631. See logs/beam/evalq_q2.txt
- q3 quality: top1 .957, dlogp -0.031, KL .027, p(chosen) .871->.859 (logs/beam/evalq_q3.txt). bench.sh running.
- STATUS 15:14 UTC: q2,q3 requant done ($MODELS/new/beam/q2,q3); evalq done for q2/q3 (logs/beam/evalq_q2.txt,q3.txt); bench.sh running under phase3 lock-hold (logs/beam/bench/, ends with BENCHDONE in gate.txt); cache impl committed.
- DOCS written (docs/beam.md Phase 3). Phase3 complete except cached J/token (warmup in window).
- Phase4 (1) cachee.sh queued on gpu lock -> logs/beam/cachee, analyze tools/beam_ce.py
- mixed precision implemented (BEAM_QDIR=q2 BEAM_QHI=q3 BEAM_QHOT=hot file); mixeval.sh queued; hot sets from 6 prompts A, eval on 6 prompts B (NAMES env in beam_evalq.py). B-set baselines: q2 top1 .819 KL .433; q3 .943 KL .031
- queued: cachee.sh, mixeval.sh, actcal.sh (act stats from prompts A), requanta.sh (act-aware q2 -> $MODELS/new/beam/act/q2 ; eval w/ evalq on set B). Lock holder order arbitrary; check logs/beam/{cachee,evalq/mixdone,act/done,rq/act_q2.log}
- STATUS 18:21UTC: cachee.sh in progress (logs/beam/cachee, cfg.txt lists finished configs); mixeval/actcal/requanta queued on gpu lock; patches: mixed-precision (runtime bits), act dump, act-aware requant all committed.
- Phase4 results: cached net J/token (diff N200-N40, warmup cancels): nvfp4+c64 30.8 (reps 38.8/22.7, noisy), q3+c96 13.9 (13.8/14.1), q2+c128 10.0, nvfp4 uncached 45.8 (logs/beam/cachee_summary.txt). Mixed (hot set from prompts A, eval on B 282 pos): q2 top1 .819 KL .433; q3 .943 KL .031; mix64(3136 hot@3b) .918 KL .063 bytes 272GB/39tok vs q3 304; mix32 .911 KL .074. act-aware requant running (requanta.sh).
- STATUS 20:01UTC: cached energy + mixed precision DONE and in docs. act-aware q2 requant (second run, ~layer 16/51) -> $MODELS/new/beam/act/q2; acteval.sh waits for EXIT then forced eval on set B -> evalq/*.actq2.log (analyze: NAMES="science_chat history german sql logic_chat json" tools/beam_evalq.py actq2)
- act-aware q2 B-set: top1 .851 dlogp -.584 KL .237 (evalB_actq2.txt). Phase4 complete. Not done: GPTQ/full Hessian; mixed act-q2+q3 + cache bench.

## 2026-10-30 efficient-AI tracks (pt51u "do it all")
- 14:05 status: FLM v1.0.8 already installed (amdxdna, /dev/accel0). Lemonade not installed (FLM alone is enough to drive the NPU). Started NPU-vs-iGPU driver (tools/npu/driver.sh): qwen3.5:0.8b + llama3.2:1b, FLM NPU vs llama.cpp Vulkan Q4_K_M, zero-shot cascade decisions + decode J/token, logs/npu.
- 14:10 status: NPU driver (logs/npu) and spec-decode energy driver (tools/spec, logs/spec: Glimmer dflash, Qwen3.8 MTP3, gpt-oss EAGLE3) queued on gpu_lock. Edge0 tiering cache-policy sim running offline on prof route traces. New ask: add-before-multiply 3-bit GEMV.
- 14:20 Edge0 tiering sim (docs/efficient-2026-10.md): hit@64 slots/layer perLRU 0.690, globalLRU 0.693, decayLFU 0.702, Belady 0.837 -> policy port not worth it.
- 14:23 UTC CQ probe (tools/cq_probe.py, CPU, partial: layer 5, experts 0/77/191/250, gate+up+down, act-weighted rel err): q3flat 0.030, act-aware q3 0.025, codebook(Lloyd 8-lvl)+act 0.025, Hadamard128+uniform 0.029, Hadamard128+codebook 0.026. Rotation loses the act-aware weighting; CQ gives no gain over our act-aware q3 here. Probe stopped because its CPU load held the quiet gate; rerun more layers when the box is idle. NPU FLM qwen3.5-0.8b accuracy pass: 800 decisions in 406.6 s (0.51 s/decision).
- 14:58 UTC ABM Part A (full q3 model, forced decode, 47 positions x science_chat+sql): bit-plane (ABM1) and code-bin (ABM2) vs MAC: top-1 47/47, top-20 tokens+order identical 47/47, rows bit-identical to 4 d.p. 31-34/47, max |dlogprob| 0.0001 (fp summation order only). Non-disk time (total-disk): MAC 8.8/7.7 s, ABM1 9.5/8.2 s, ABM2 14.5/13.3 s (science/sql); not quiet-gated, Part B is the kernel A/B.
- 15:38 UTC box not quiet (other users at 20-32 cores since ~14:10). Stopped abm.sh Part B after 1 window (its orphaned gate child held gpu_lock for 15-min gate waits); requeued as logs/beam/abm_b.sh with gate max 60 s and per-window busy snapshots. Spec/NPU session gates raised to 3600 s.
- 16:15 UTC ABM Part B rep 1 (NL=8, 224-slot cache, GPU-bound; N=200 minus N=40): GPU time per step MAC 28.75 ms, bit-plane ABM1 28.56 ms (-0.7%, noise), code-bin ABM2 52.88 ms (+84%). Net J/token NOT usable: box had 19-24 busy cores from other users, idle baseline swung 67-137 W, differentials came out negative. Rep 2 running.
- 16:25 UTC ABM Part B done (2 reps): GPU ms/step median MAC 30.0, bit-plane 30.3, code-bin 47.6. Energy unusable (box at 32 busy cores). Documented in docs/beam.md. Spec: Glimmer DFlash r1 net 4.77 J/tok (busy 4 cores), acceptance 0.44 (mean len 2.32). NPU accuracy on 800 items: FLM qwen3.5-0.8b 0.515, Vulkan Q4_K_M 0.510.
- 16:37 UTC box still at 17-28 busy cores; socket nets unreliable (Glimmer base net -1.14 J/tok). GPU-rail (gfx) J/token as a load-insensitive proxy: Glimmer base 1.11, DFlash 1.90 (DFlash 2.0x faster: 0.043 vs 0.088 s/tok). Vulkan qwen3.5-0.8b decision: gfx 0.87 J, 0.066 s/decision. Queue continues: vk gen, flm r1, llama3.2 sessions, q38/oss spec.
- 23:50 UTC quiet rerun ABM Part B (load 0.2 before; gate quiet): GPU ms/step MAC 25.8, bit-plane 25.7, code-bin 33.7; net J/token MAC 1.94 (2.16, 1.71), bit-plane 2.12 (2.81, 1.43; rep2 idle baseline 51 W outlier), code-bin 2.85 (3.05, 2.66). Busy cores in windows 4-5 = beam itself.
- 00:17 UTC quiet spec rep1 net J/token (socket, 320-64 tokens x2): Glimmer base 7.37 / DFlash 5.84 (accept 0.42, mean len 2.25; 0.077 -> 0.044 s/tok); Qwen3.8-27B base 7.36 / MTP n3 4.84 (accept 0.49, len 2.45; idle baseline 37.7 W in one window, so tentative). gpt-oss pending.
- 00:51 UTC spec done (medians of 2 reps; rep2 windows had 5-30 busy cores from other users, idle 14-69 W): net J/tok Glimmer base 8.69 vs DFlash 4.67; Qwen3.8 base 6.82 vs MTP3 4.76; gpt-oss-20b base 0.99 vs EAGLE3 n2 0.78 (oss rep2 very noisy). Clean rep1 only: Glimmer 7.37 -> 5.84 (-21%), Qwen3.8 7.36 -> 4.84 (-34%), oss 0.94 -> 1.15 (+23%). NPU driver running.

## 2026-11-02 NPU vs iGPU energy, quiet reps 1+2 (logs/npu/e, npu_an.py)
Net J/decision (2 reps): l32 vk 2.56 (0.029 s), l32 flm 8.09 (0.42 s, NPU rail 0.72 J),
q35 vk 2.73 (0.040 s), q35 flm 4.19 (0.51 s; idle 30.6 W, busy 4.0 -> noisy, low confidence).
Net J/gen token: l32 vk 0.305, l32 flm 0.293, q35 vk 0.257, q35 flm 0.353.
Accuracy n=800: l32 vk 0.489, l32 flm 0.520, q35 vk 0.510, q35 flm 0.515.
Verdict: iGPU (vk) is ~3x cheaper per decision (prefill-dominated, NPU 15x slower); per generated
token NPU ~= iGPU. Keep decisions on iGPU.

## 2026-11-02 CQ probe, layers 5/20/35/50 x experts 0/250 (logs/cq/probe_full.log)
Mean act-weighted err: q3flat .0316, q3 .0285, rot+q3 .0305, cb .0273, rot+cb .0274. cb wins at L50 (.026 vs .029), rot+cb at L20 (.026 vs .030), tie at L5.
Next if pursued: PPL with per-layer cb.

## 2026-11-02 per-layer codebook quality check (queued)
tools/beam_requant_cb.py: layers 20-29, act-aware q3 (cbA) vs per-layer best-of {q3, Lloyd codebook} (cbB); rest = existing q3.
Engine: BEAM_CBLUT per-layer 8-entry LUT in push constants (spec const CB, MAC path), uncommitted until the build and exactness check pass.
Runner logs/beam/cbq.sh waits for the decision idle loop, then holds gpu_lock: build, requant, forced eval on 6 prompts, beam_evalq.py q3 cbA cbB,
plus a LUT-path exactness check (uniform LUT == cbA on sql). rot+cb was skipped (needs activation rotation in the kernel). Results -> logs/beam/cbq/evalq.txt.
