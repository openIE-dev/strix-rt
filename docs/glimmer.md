# Muse Glimmer 30B (crates/glimmer)

Rust-from-scratch port on rtcore/prefill. Weights: $MODELS/new/glimmer (Q4_K_M text GGUF 15.6 GiB, mmproj Q4_K_M/BF16). Reference: llama.cpp (Vulkan, build e36df6047).

- Text decode: GPU-driven greedy (recorded command buffer, as gemma4). Per layer: RMS (weight+1 folded by the converter), q/k/v + attention-gate GEMVs, per-head q/k RMSNorm,
  interleaved (NORM) RoPE on local layers only (global layers NoPE), GQA 32:2 hd 128, sliding window 2048 on 39 of 52 layers (ring KV), attn x sigmoid(gate), post-attn RMSNorm (eps 1e-8),
  SwiGLU, post-FFN RMSNorm (1e-8); head: x logit_scale 0.196 then tanh softcap 20. Embedding row is RMS-normalised (no weight).
- Prefill: pf coop-matrix GEMMs, chunked (PF_CHUNK, default 256), local ring = win + chunk slots.
- Vision: ViT-G/14 (50 layers, width 1536, sparse 32x32 windows except every 4th + last layer, 2D RoPE, erf-GELU), pixel-shuffle, 3-layer MLP (BF16) to 6656; Lanczos preprocess as mtmd.
- Tokenizer: `src/tok.rs`, GPT-2 byte BPE from the GGUF vocab + merges, llama.cpp GPT4O/llama4 pre-tokenizer, BOS + parse_special, decode. Ids equal llama-tokenize on 24 strings (6 prompts, long prompts, contractions, CJK/Cyrillic, emoji, CRLF/tabs, specials): `logs/glimmer/tok_cmp.py`, 0 mismatches.
- Images: `vision::load_image` (PNG/JPEG -> RGB8). A `<__media__>` marker in the prompt becomes `<|image_start|>`, N placeholder rows, `<|image_end|>` (llama.cpp mtmd layout); ViT embeddings replace the placeholder rows in prefill.
- No chat template: the llama.cpp refs are raw completions (text refs without BOS, the image ref with BOS), so the runner does the same.

## Tools
- `glm_run <gguf> <ids|@file|text:PROMPT|textfile:PATH> [n_new] [ref ids|@file|json:ref.json#idx]`: greedy decode, tok/s, parity vs reference. Env: `FORCED=1` (teacher-forced), `NO_BOS=1`, `IMAGE=<png|jpg>` (+ `MMPROJ`, default mmproj-*.gguf beside the model), `RUNS`, `PF_CHUNK`, `MAX_CTX`; `IMG_EMB`/`IMG_POS` inject a raw f32 embedding dump. `logs/glimmer/sess_text.sh` is the one-lock-session text + image parity run (`text_parity.log`).
- `glm_vis <mmproj> <img.rgb> <out.f32> [ref.f32]`: ViT only, cosine vs a llama.cpp mtmd dump (`logs/glimmer/mtmd/mtmd_dump.cpp`).
- `glm_tok <gguf> <text | @prompts.json>`: print ids.
- Opt-in debug env: `GLM_SKIP=<mask>` (skip kernel groups for power attribution), `GLM_IDOT=1` (failed experiment, below).

## Measured (logs/glimmer/*.log)
### Parity (vs llama.cpp Vulkan greedy refs)
- Text, 6 prompts x 128 tokens, teacher-forced: 765/768 (parity1.log; with prefill parity2_pf.log; text prompts typed straight into glm_run via the Rust tokenizer: text_parity.log, same counts). The 3 misses are near ties (logit gap 0.09-0.22, ref margins 0.16-0.28), consistent with llama.cpp Q8 activation quantisation vs our f32/f16 path. Free-run greedy is identical for 128/128 tokens on 3 of 6 prompts and diverges at token 43/86/73 on the others.
- Long prompts (2880 and 3960 tokens, past the 2048 window, chunk 256 and 512): 48/48 (parity3_long.log).
- Vision prompt (368 tokens: BOS, image_start, 361 image rows, image_end, newline, text): 128/128 forced and free-run with llama.cpp mtmd embeddings; 126/128 forced with our own ViT embeddings (misses at tokens 85 and 119, margins 0.06 and 0.02) (img_parity.log, text_parity.log).
- Vision embeddings vs mtmd (vis2.log): 512x512 (361 tokens) mean per-token cosine 0.9983, overall 0.9979, min 0.817 (one token); 1400x980 (1750 tokens) mean 0.9972, overall 0.9945. ViT GPU time 0.38 s (361 tokens), 3.4-3.8 s (1750 tokens).
  The first ViT attention (one query per workgroup) device-hung at 7000 patches (gfx ring timeout, recovered by ring reset); replaced by coop-matrix flash per window.

### Speed (perf high, bench_ours.log vs bench_llama.log)
- ours pp512 540, pp2048 468 (chunk 512), tg128 13.34 tok/s; llama.cpp Vulkan 451 / 440 / 13.14, ROCm 466 / 461 / 12.77.

### Energy (tools/comp/energy.sh, differential N=320 - N=64; energy_ours.log, energy_llama.log)
- ours 8.44 J/token socket (gfx 5.52); llama.cpp Vulkan 6.98 (2.27); ROCm 8.12 (3.24). Same tok/s, higher power.
- Investigation (eskip.log, sweep1.log, floor.log; `GLM_SKIP` groups: 1 gate/up, 2 down, 4 attention, 8 o-proj, 16 qkv, 32 head): gfx clock 2900 MHz, fclk 2000, uclk 1000 and gfx activity 95-97% are the same as llama.cpp. Weight streaming sits at the DRAM wall (~220-246 GB/s per group); attention is negligible. Whenever a Q4_K/Q6_K float-dequant GEMV group runs, gfx is 67-77 W and socket 100-117 W (floor with nothing running: 14 W gfx). Skipping a group lowers tok/s and power together, so the cost is power per busy cycle in the dequant kernels, not clocks or bandwidth. The lanes-per-row sweep moves J/token by at most ~5%.
- **Failed experiment (opt-in, not default): `GLM_IDOT=1`**, a llama.cpp MMVQ-style GEMV (activations quantised to q8_1 by `glm_xq`, integer dot in `glm_qgv`). Forced parity looked fine and the first 4 free-run ids match, but it is ~3.5x slower (~260 ms/token vs 75 ms; the Q6_K down dispatch is the worst), a 9-token free run did not finish, and so there is no energy number. Not debugged further (suspected code-size/spill and lane mapping); the float path is the default and unchanged.
- **Q4_K item-pair GEMV (default for Glimmer, GEMVQ_Q4P=0 restores the old kernel):** one lane handles two adjacent 32-weight items that share the same 32 qs bytes (low/high nibble), so weight loads and masks halve and byte->float uses cvt_ubyte; GEMV_PROJ default 16,256. Per-item arithmetic is unchanged, only lane accumulation order differs. Differential energy (N=480-64, perf_high, 3 A/B reps, logs/glimmer/final.log): old 8.70/8.80/8.75 (median 8.75) -> new 8.15/8.19/8.34 J/token socket (median 8.19, -6.4%); gfx 5.82 -> 4.9-5.0. Decode speed 13.2 tok/s unchanged. Forced parity vs llama.cpp refs identical to before (same near-tie mismatches on prompts 0, 2, 5; free-run ids identical to the old kernel over 480 tokens).
- Tried, no gain: Q6_K with aligned 32-bit word loads (GEMVQ_Q4P bit 1, opt-in): power 107->103 W but 5% slower, J/token unchanged. Fewer lanes per row (8,128) with pairs: same energy, 3% slower.
- Open: 8.19 vs 6.98 J/token (+17%, was +21%). Remaining: Q6_K and attention path power; gfx channel still about 2x the llama.cpp reading.

### Forced-parity misses (prompts 0, 2, 5) are near-tie noise, not a kernel bug (logs/glimmer/par/)
Teacher-forced on the reference prefix at each mismatch position, the same GGUF was run through llama.cpp Vulkan (-fa on, -fa off), HIP/ROCm and CPU (probe.py, `probe.log`). logprob(ref token) - logprob(our token):

| case | vk fa on | vk fa off | HIP | CPU | ours (FORCED) |
|---|---|---|---|---|---|
| p2 pos 86 | -0.03 | +0.09 | +0.19 | -0.43 | about +0.22 against ref |
| p5 pos 73 | -0.07 | -0.00 | -0.03 | +0.10 | about +0.12 against ref |
| p0 pos 43 | -0.20 | +0.19 | -0.13 | -0.56 | about +0.10 against ref |

Sign flips across llama.cpp backends and flag settings (the preferred token changes between llama.cpp builds themselves) with swings up to 0.6 logprob, larger than our offset. llama.cpp quantises activations to q8_1/q8_K for Q4_K/Q6_K and its kernels differ per backend; ours keeps f32 activations. So the 0.1-0.2 gaps sit inside the cross-backend spread and no layer-level numerics bug is indicated (the earlier GEMV rewrite also leaves forced parity identical). The 0.1 margin gate is too tight for this model; the gate now reads: ours must lie within the spread of llama.cpp backends. No code change.

### Per-group energy attribution after the pair kernel, and sum(x) precompute (logs/glimmer/eskip2.log, xs1.log, xs2.log)
GLM_SKIP, 481 tokens, perf_high (output is wrong when skipping; energy = socket W x time, removed energy per token, socket idle floor included in the 8.3 total):

| group | ms/token | J/token | avg W while running |
|---|---|---|---|
| ffn gate+up (Q4_K) | 34 | 4.2 | 122 |
| ffn down (Q6_K/Q4_K mix) | 23 | 2.6 | 111 |
| qkv+gate proj | 7 | 0.9 | |
| lm_head (Q6_K) | 4 | 0.5 | |
| o-proj | 4 | 0.4 | |
| attention | 0.3 | 0.1 | |

Attention is about 1% of energy, so there is nothing to win there. Q6_K down runs at a lower power per ms (111 W) than Q4_K gate/up (122 W), so Q6_K is not the power hog; the Q6_K aligned-load variant earlier saved power but cost speed. The biggest remaining target is Q4_K gate/up.
- **Change:** the pair kernel still did a redundant sum(x) dot per row per item (about a third of its ALU). New tiny kernel `glm_xs` writes per-32 sums of x once per GEMV input (bound at binding 5, which Glimmer does not use for xw); the GEMV reads them (GEMVQ_Q4P bit 2, default value now 5).
- **Result:** differential energy (3 A/B reps): pairs only 8.63/8.09/8.14 (median 8.14) -> with xs 8.28/7.92/7.91 (median 7.92, -2.6%); wins in 3 of 3 pairs but run-to-run noise is about 0.3. Free-run ids identical (480 tokens), forced parity unchanged (par2.log). Speed 13.14 tok/s.
- **Cumulative:** old kernel median 8.75 -> 7.92 J/token socket (-9.5%); vs llama.cpp Vulkan 6.98 the gap is +13.5%.

### Integer-dot Q4_K pair GEMV with q8 activations (opt-in GEMVQ_Q4P=13, NOT default; logs/glimmer/id*.log, idl.log, par_q13.log)
Form: glm_xs2 quantises x once per GEMV input to int8 per 32 (plus exact sum(x)); the pair kernel does dotPacked4x8 on the nibbles (about 4x fewer ALU ops per weight byte than the float pair kernel, 4 uvec4 x loads per pair). Unlike GLM_IDOT it keeps the pair/lane layout and does not slow down (11.8-12.3 vs 11.9-12.3 tok/s under identical contention).
- Power: the gfx channel drops in every A/B pair (about -10 W in the first sample: 60.6 -> 50.5 W; differential gfx J/token 2.77/2.45/2.56/2.62 -> 2.23/2.28/2.20/2.43 under load).
- Socket J/token could not be measured cleanly: from about 11:30 to 12:40 ET other workers ran rustc/cc1 builds and benchmarks continuously (load average 50-77), decode fell from 13.2 to 10.9-11.9 tok/s and CPU power swamped the socket channel (10-11 J/token for both variants). All 4 A/B pairs were discarded by the 13.0 tok/s clean-box rule, so there is no honest socket number for this variant.
- Parity fails the gate: forced parity is worse than the float kernel, 7 mismatches in 768 positions vs 3, with gaps up to 0.73 (p0 pos 53), 0.51 and 0.31 (p4), outside the spread seen across llama.cpp backends at the old miss positions. Free-run ids diverge from the float kernel at token 43 (the known p0 near-tie, where it now picks the reference token 350), but the extra larger-gap mismatches mean the q8 activation noise here is higher than the llama.cpp backends show. It stays opt-in; the default remains the float pair + sum(x) kernel (GEMVQ_Q4P=5, 7.92 J/token socket measured on a quiet box).
- Next if pursued: quantise x with finer scales or keep the down-proj input in float, re-check parity before any default switch, and re-measure socket on a quiet box.

### Integer-dot retry: min term from the quantised x, finer activation scales (GEMVQ_Q4P bits 4-5 = groups per 32; logs/glimmer/par_q13b.log, par_q29.log, par_q45.log, i45.log)
- Root cause of the first parity failure: the min term used the exact sum(x) while the nibble term used quantised x, so the errors did not cancel. glm_xs2 now stores d*sum(q) (as llama.cpp q8_1 does). Forced parity (6 prompts x 128): per-32 scales (13) 6 mismatches, worst gap 0.46; per-16 scales (29) 4 mismatches, worst gap 0.37 (still above 0.3); per-8 scales (45) 3 mismatches at the same positions as the float kernel (p0 43, p2 86, p5 73) with gaps 0.042, 0.108, 0.047, all smaller than the float kernel (0.092, 0.214, 0.125), so per-8 passes parity and is closer to the references than the float kernel.
- Energy of per-8 vs the float default (5), interleaved A/B pairs, N=480-64, perf_high: the box never became quiet (rustc/cc1 builds from other workers, load 43-56, decode 10.3-11.0 tok/s instead of 13.2 for 40 minutes), so all 6 pairs were below the 13.0 tok/s clean rule and absolute socket values (10-12 J/token) are contaminated. Relative, under the same contention: per-8 lost in 6 of 6 pairs, 3-4% slower decode (10.6-10.7 vs 11.0 tok/s), socket +0.3 J/token (11.4 vs 11.1), gfx +3-5%. The finer scales cost the ALU back that the integer dot saved.
- Decision: not made default (it does not win on socket energy, and the clean-box measurement was not possible). The float pair + sum(x) kernel (Q4P=5, 7.92 J/token measured on a quiet box) stays the default; per-8 int-dot is available as GEMVQ_Q4P=45 (per-32 13, per-16 29 for experiments).
- Remaining ideas: fuse gate+up into one dispatch sharing x loads; per-16 scales with a smarter scale search (the per-16 miss was one 0.37 position); Q6_K down power.

### Fused ffn gate+up+SwiGLU, one dispatch (opt-in GLM_FUSE_FFN=1; logs/glimmer/fuse1.log, fz.log, sess_fuse.sh)
gemvq SWIGLU variant main_q4p_sw: the pair kernel + sum(x) numerics (Q4P=5) for both gate and up rows in one dispatch, x loaded once per pair for both rows, silu(gate)*up in the epilogue, written straight to the down-proj input (removes the f32 gate/up round trip and the separate misc dispatch). Falls back to the two-step path unless both weights are Q4_K.
- Numerics unchanged: free-run ids over 481 tokens identical (md5 e3ff6cb9 both ways) and forced parity identical to the float default (same 3 near-tie mismatches, gaps 0.092/0.214/0.125).
- Energy: the box was never quiet (other workers rustc/cc1 builds, load 37-61, decode 10.8 tok/s instead of 13.2), so no pair met the 13.0 tok/s clean rule and absolute values are contaminated. Interleaved A/B, same contention, 6 pairs (differential N=480-64, perf_high): socket 11.17/11.25/11.20/11.21/11.22/10.81 off vs 11.20/11.14/11.21/11.25/11.13/10.69 on (means 11.18 vs 11.17), gfx 2.51/2.42/2.45/2.40/2.48/2.47 vs 2.55/2.48/2.39/2.45/2.45/2.39 (means 2.46 vs 2.45), decode 10.8 vs 10.8 tok/s. No measurable difference: gate+up energy is weight streaming plus dequant ALU, and the f32 intermediate is negligible next to the weights.
- Decision: stays opt-in, not default (no win on socket J/token). Default remains Q4P=5 (7.92 J/token socket on a quiet box).
- Remaining: Q6_K down power; per-16 int-dot with a smarter scale search; re-measure on a quiet box.

### Q6_K quad kernel for ffn down / lm_head (opt-in GEMVQ_Q4P=71 = 5 + bit 1 + bit 6; logs/glimmer/q6q1.log, q6s.log, q6ab.log)
Form: 4 Q6_K items (128 weights) share the same 64 ql + 32 qh bytes, so one lane handles a quad: aligned 32-bit word loads with funnel shift for 2-mod-4 blocks (26 word loads per 128 weights instead of 72), nibble/2-bit merge for all 4 items from the shared words, byte->float via cvt_ubyte, and the -32 offset taken from per-16 sums of x (glm_xs now also writes per-half sums at xs[8192+...]): sum((v-32)x) = sum(v x) - 32 sum(x).
- Numerics/parity: free-run ids over 481 tokens identical (md5 e3ff6cb9), forced parity same as the float default (3 near-tie mismatches, gaps 0.090/0.213/0.124).
- Energy on a quiet box (load 0.5, nothing else running, perf_high, interleaved A/B, N=480-64, 4 pairs): float default (Q4P=5) 8.095/8.108/8.102/8.097 J/token socket at 12.95 tok/s; Q6 quad 9.398/9.408/9.373/9.333 at 10.98 tok/s. The quad kernel is 16% worse on socket (+1.28 J/token) and 15% slower; the -7 W gfx power does not make up for the time. Lane/workgroup sweep (8,128 / 8,256 / 32,128 / 4,128 / 16,256) never beats default: 10.4-11.0 tok/s (4,128: 8.9).
- Why: scattered 4-byte loads (26 per quad) make it load-issue bound, while the float Q6_K path (ushort loads) already reaches the DRAM wall. Together with the earlier aligned-load variant (-5% speed, equal energy) this says Q6_K dequant ALU is not the lever.
- Decision: opt-in only, default unchanged (Q4P=5). The quiet baseline of the default today is 8.10 J/token (12.95 tok/s) vs 7.92 on an earlier quiet run; run-to-run and day-to-day spread is about 0.15, so the gap to llama.cpp Vulkan (6.98) is +13.5% to +16%.
