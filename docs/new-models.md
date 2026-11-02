# New model families: Gemma 4, Cosmos 3, Nemotron 3 (scope as of 2026-10-15)

Picks: Gemma 4 26B-A4B (MoE) + E4B; Nemotron 3 Nano 30B-A3B (hybrid Mamba-2/MoE) + Nano 4B; Cosmos3-Nano 16B (MoT).

## Models
| Model | Size | Arch | License | Source | Local copy |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B-it | 26B tot / 3.8B active, ~16 GB Q4_K_M | MoE 128 experts, SWA 1024 + global layers, dual RoPE, shared KV across late layers, vision enc | Gemma terms (check card) | google/gemma-4-26B-A4B-it, GGUF unsloth/gemma-4-26B-A4B-it-GGUF | not on drives -> hf download to $MODELS/new/gemma4/26b-a4b |
| Gemma 4 E4B-it | 8B w/ PLE (4.5B eff), 8.9 GB Q8_0+mmproj | dense, SWA 512, Per-Layer Embeddings, vision + USM conformer audio | Gemma | google/gemma-4-E4B-it | local drive: gemma-4-e4b-gguf -> $MODELS/new/gemma4/e4b |
| (Gemma 4 31B dense, 12B Unified) | 31B / 12B | dense; 12B encoder-free multimodal | Gemma | google/* | not picked (31B Q4 fits; 12B Unified is a later stretch) |
| Nemotron 3 Nano 30B-A3B | 24.7 GB Q4_K_M | hybrid Mamba-2 + few attention layers + MoE FFN | NVIDIA Open Model License | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-*, GGUF available | local drive nemotron-3-nano-Q4 -> $MODELS/new/nemotron3 |
| Nemotron 3 Nano 4B | 2.5 GB Q4_0 | 42 layers: 21 Mamba-2, 4 attn, 17 MLP; d=3136 | NVIDIA Open Model License | nvidia/NVIDIA-Nemotron-3-Nano-4B-{BF16,FP8,GGUF} | local drive nemotron-4b-q4_0.gguf -> $MODELS/new/nemotron3 |
| (Nemotron 3 Super 120B-A12B, Ultra 550B) | >50 GB | same hybrid | NVIDIA OML | on drives (Q3 / IQ4_XS / bf16) | out of size budget, not copied |
| Cosmos3-Nano | 16B (8B AR reasoner + 8B diffusion generator), 32 GB bf16 | Mixture-of-Transformers; ViT (understanding), VAE (video/image), sound tokenizer, UniPC flow sched | NVIDIA Open Model License | nvidia/Cosmos3-Nano (Diffusers Cosmos3OmniPipeline); no GGUF | local drive Cosmos3-Nano -> $MODELS/new/cosmos3 |
| (Cosmos3-Super) | 64B | MoT | NVIDIA OML | nvidia/Cosmos3-Super | ~128 GB bf16, not picked |

Also found locally (not used): diffusiongemma-26B-A4B-it (48 GB, local drive), Gemma 4 E2B GGUF/ONNX/LiteRT, gemma4-12b-ltx text encoder (local drive), llama.cpp gemma/nemotron converters in ~/llama-latest.

## What strix-rt has
vkrt shaders: q4k/q6k/q8 GEMV, attn/attn16/attn64 + split/combine (flash-decode), rope variants, rmsnorm, router.comp (MoE, used by lfm2/gptoss), gdn_conv/gdn_step (Gated DeltaNet, qwen38), shortconv (LFM2), embed q4k/q6k. qimage = diffusion DiT + VAE pipeline; mcv = MiniCPM-V vision path + llm prefill; prefill/npu_split for XDNA2.

## Gap list
### Gemma 4 26B-A4B / E4B
- Loader: gemma4 GGUF arch keys (per-layer SWA/global pattern, dual rope theta, shared-KV layer map, softcap, PLE tensors) in crates/gguf + new crates/gemma4.
- Kernels: sliding-window mask in attn*.comp (window 512/1024) + ring-buffer KV for local layers; per-layer rope base switch; GeGLU (gelu_tanh*up) fused act; logit soft-capping; embedding scale sqrt(d); sandwich norms (pre+post rmsnorm, (1+w) form).
- KV sharing: late layers reuse earlier KV — cache indexing change, no new shader.
- MoE (26B): reuse router.comp; 128 experts top-k plus shared dense FFN path — verify router widths.
- PLE (E4B): extra embedding gather per layer + small projection/gate.
- Vision: new SigLIP-style encoder w/ 2D RoPE and variable token budgets; adapt from mcv path. Audio conformer: separate workstream, deferred.
### Nemotron 3 Nano 30B-A3B / 4B
- New Mamba-2 SSD kernels: causal conv1d (generalise gdn_conv/shortconv), selective-scan step for decode (state h = exp(dt*A)h + dt*B x; y=C h + D x), chunked scan for prefill, grouped RMSNorm gating. gdn_step is structurally close but uses delta-rule, so a new ssm_step.comp + chunked prefill shader.
- Loader: nemotron_h layer pattern (M/*/E per layer), ssm tensor names (A_log, D, dt_bias, conv1d, in_proj split z/x/B/C/dt).
- MoE (30B-A3B): router.comp with sigmoid/top-k + shared experts; squared-ReLU expert MLP activation.
- Attention layers: plain GQA, no RoPE (check config) — existing attn works.
### Cosmos3-Nano
- Loader: safetensors (bf16) multi-shard transformer + vision_encoder + vae + sound_tokenizer; Diffusers config parsing (reuse qimage loader).
- AR tower: standard decoder LLM path; MoT routing = per-token modality selects tower weights while sharing attention over the joint sequence — new joint-attention scheduling.
- Diffusion tower: DiT blocks w/ adaLN on timestep, 3D RoPE for video latents, flow-matching UniPC scheduler (qimage has flow scheduler; add UniPC multistep, flow_shift).
- Video VAE (3D causal conv decoder) — new conv3d kernels; biggest new-kernel workstream. Vision ViT encoder for understanding (from mcv). Audio/action heads deferred.
- Memory: bf16 32 GB fits unified memory; Q8 GEMV for DiT is an optional workstream.
- Status (crates/cosmos3, docs/cosmos3.md): t2v works end to end (MoT transformer, UniPC, Wan2.2 video VAE); velocity rel 1.2-2.2e-2, VAE 85 dB vs diffusers; 256x256x9 8 steps 10.9 s / 685 J vs diffusers 18.0 s / 1692 J. Tokenizer, AR text, ViT, audio/action pending.

## Workstreams (ordered)
1. Nemotron 3 Nano 4B: Mamba-2 conv+scan decode, nemotron_h loader (validates SSM path cheaply).
2. Nemotron 30B-A3B: chunked scan prefill + MoE glue.
3. Gemma 4 text (26B-A4B, E4B): SWA/softcap/GeGLU/KV-share/PLE.
4. Gemma 4 vision.
5. Cosmos3-Nano AR text + ViT understanding.
6. Cosmos3-Nano generation: DiT, UniPC, video VAE conv3d.
Validation: logits vs llama.cpp (~/llama-latest) for GGUF models, vs Diffusers on CPU for Cosmos; GPU runs only via logs/wait_quiet.sh + logs/gpu_lock.sh.

## Weights on Framework
$MODELS/new/{gemma4/e4b, gemma4/26b-a4b, nemotron3, cosmos3/Cosmos3-Nano}

## Nemotron 3.5 Super VL 120B-A12B (crates/nemotron35) — 2026-10-16
- Weights: $MODELS/new/nemotron35vl (ggml-org GGUF Q4_0 71.8 GB + mmproj BF16 + mtp Q4_0; sha256 verified). mmproj converted to mmproj_st/ (crates/mcv/ref/mmproj2st.py).
- Reference: llama.cpp worktree ~/llama-n35 @ f3363593f (needs >= cd0a5b6a1 for nemotron_omni); local MTMD_DUMP patch in mtmd.cpp.
- Arch: nemotron_h_moe, 88 blocks, LatentMoE (latent 1024, 512 experts top-22 sigmoid+bias, x5 norm), Q4_0 experts. New crate forks crates/nemotron decode @ aba8338; new kernels n35_q/moe/rlog/router/clamp + vision (C-RADIO ViT-H/16).
- Shared edits: rtcore qtype adds Q4_0=16 (only n35_q.glsl kernels understand it); workspace member added.
- Load: ZEROCOPY=0 required (strixpc max_pinned_mb=65536 -> dmabuf ENOSPC; userptr fails).
- Parity (6 prompts x 48 tok vs llama Vulkan): teacher-forced 285/288, greedy prefix 264/288; all misses at llama margin <0.09 nats.
- Vision parity (rainbow 512^2 vs llama-mtmd-debug): cosine 0.99902, min per-token 0.9734, max|diff| 0.70 (ref rms 0.36).
- Speed: tg 17.65 tok/s (llama 18.66); no batched prefill yet (prompt token-by-token 17.6 tok/s vs llama pp512 451).
- INCIDENT: early n35 runs caused two gfx ring timeouts/resets (11:06, 11:08 UTC); cos saw amdgpu VM list_del warnings ~11:10 UTC. Suspected NaN router id / OOB gather; fixed with NaN-safe router + token clamp, no recurrence.
- Done since: batched prefill, MTP, real-image preprocessing (see section below).

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

## Nemotron 3.5 Super VL: real-image preprocessing + end-to-end image answer (2026-10-17, crates/nemotron35)
- What llama.cpp does for projector `nemotron_omni` (~/llama-n35 @ f3363593f): `mtmd_image_preprocessor_dyn_size`, hparams `PAD_NONE`, bicubic (Pillow-compatible, a=-0.5, 22-bit fixed point, horizontal then vertical pass, each rounded to u8), align = patch 16 x merge 2 = 32, smart_resize with min_pixels 262144 / max_pixels 3407872 (from the mmproj gguf), no tiling (single dynamic-resolution image), x/255 then (x-mean)/std with mean 0.48145467/0.45782751/0.40821072, std 0.26862955/0.26130259/0.27577711 (CLIP values, RGB order), planar -> patch order c,ky,kx. The old `clip.vision.image_size 512` is only the warm-up size; the Super VL does NOT resize to a fixed 512 square (that is the `nemotron_v2_vl` path).
- `nemotron35::imgprep` (src/imgprep.rs): PNG decode (image crate, RGB8, alpha dropped) and JPEG decode (`stbjpeg.rs`, see below), `target_size`, `resize_bicubic` (integer only), `preprocess_rgb/preprocess_file` -> planar f32 [3][h][w]. IEEE f32 + integer, no threads, deterministic.
- Runners: `n35img <mmproj_st> <img> [ref.px] [ref.emb]` (preprocess + ViT, pixel/embedding compare, RUNS=n timing); `n35vl <model.gguf> <mmproj_st> <img> <pre ids> <post ids> [n]` (preprocess -> ViT -> embeddings injected into prefill rows (`Scratch.img`, `n35_icopy.comp`) -> greedy decode); `logs/n35/sess_vl.sh` builds the ids with llama-tokenize (template: `<|im_start|>user\n<img>` ... `</img>PROMPT<|im_end|>\n<|im_start|>assistant\n`, no BOS, same as mtmd-cli); `logs/n35/sess_img.sh` makes the references.
- Reference: ~/llama-n35 llama-mtmd-cli, local patch in mtmd.cpp (MTMD_DUMP_PX = preprocessed f32 tensor, MTMD_DUMP_EMB = ViT output) , greedy (`--temp 0`), 80 tokens.
- Large-image fix: one command buffer for the whole ViT hit the amdgpu job timeout (device lost at 13160 patches / 3290 tokens, ~8 s of attention). `Vision::encode` now submits one stage per command buffer (embed, each of 32 layers, tail) above 2048 patches (`N35V_SPLIT_N`). Output unchanged; 3290 tokens now 8.4 s vs llama.cpp 8.26 s.
- Results, 6 test images (logs/n35/imgparity1.log, imgparity2.log, imgvl1.log):
| image | resized to | tokens | pixel tensor vs mtmd | ViT embedding cosine (min per-token) | greedy answer vs mtmd-cli |
|---|---|---|---|---|---|
| dusk256.png (test frame) | 512x512 | 256 | bit-exact (0 of 786432 differ) | 0.99998 (0.99969) | identical to EOS (44/45 ids + final ".\n\n" vs "." EOS) |
| qi1024.png | 1024x1024 | 1024 | bit-exact | 0.99997 (0.9983) | identical to EOS (43/44, same ending) |
| qi_1500x1001.png | 1504x992 | 1457 | bit-exact | 0.99997 (0.9930) | identical to EOS (39/40, same ending) |
| tiny_100x75.png (upscaled) | 608x448 | 266 | bit-exact | 0.99995 (0.9976) | identical to EOS (54/55, same ending) |
| city_1000x562.jpg | 992x576 | 558 | bit-exact after the stb JPEG port (was 35936/1714176 differing with the image crate decoder) | 0.99996 (0.9965) | identical to EOS (55/56, same ending; was diverging at token 45 before the port) |
| big_3000x2000.jpg (downscaled by the pixel cap) | 2240x1504 | 3290 | bit-exact after the stb JPEG port (was 210556/10106880 differing) | 0.99995 (0.9989) | identical to EOS (43/44, same ending) |
- PNG and JPEG preprocessing are both bit-exact with llama.cpp mtmd (all 6 pixel tensors identical, `logs/n35/imgparity3.log`; all 6 greedy answers identical to EOS, `logs/n35/imgvl2.log`).
- ViT embedding parity cosine ~0.9999 is the same kernel-level difference as the rainbow test (f16 GEMMs vs ggml), not a preprocessing difference.
- Answers (greedy, our runner): dusk256 "...a small wooden cabin at night. The cabin has a lit window and door..."; qi1024 "A red fox sits in the snow in a forest, looking directly at the camera. Next to the fox is a wooden sign with the word STRIX carved into it..." (same as llama.cpp).
- Image-encode time and energy (CPU preprocess + ViT, `n35img` RUNS loop; tools/energy/record.py record 20 Hz, differential between two windows >= 30 s so load/warm-up cancel; socket = SMU average_socket_power, not wall; perf level as found, GPU lock; logs/n35/imgenergy*.{log,py}, sess_imgenergy.sh):
| image | tokens | ViT GPU median | CPU preprocess | socket J/image | gfx J/image | llama.cpp mtmd encode (single run, includes warm path) |
|---|---|---|---|---|---|---|
| dusk256 (-> 512x512) | 256 | 155.7 ms | 2.1 ms | 13.6 (windows 34.9 s / 69.5 s) | 6.3 | 328 ms |
| qi1024 | 1024 | 1073 ms | 3.6 ms | 88.5 (35.9 s / 70.5 s) | 43.2 | 1516 ms |
| qi_1500x1001 (-> 1504x992) | 1457 | 1.58-1.89 s (varied between windows) | 16.5 ms | 159 (35.9 s / 76.8 s; noisy, GPU time differed 20% between the two windows) | 79.0 | 2460 ms |
- Caveats: llama.cpp encode energy was not measured (only its single-run wall times from the mtmd log, one run each, not a controlled comparison); prefill of 1000-3300 image tokens runs at 260-350 tok/s (Super text path), so for a 1024-token image the encode (1.1 s) + prefill (3.3 s) dominates a short answer; JPEG decode is now stb-exact (see below); no multi-image or video input, no tiling (the model has none).

### JPEG: stb_image port (2026-10-17, crates/nemotron35/src/stbjpeg.rs)
- llama.cpp mtmd decodes with `stbi_load_from_memory(.., req_comp=3)` (vendor/stb/stb_image.h). The earlier image-crate (zune-jpeg) decode differed by up to 3 u8 levels in ~2% of samples and changed one answer. `stbjpeg::decode_rgb` is a faithful port of the stb path: baseline and progressive Huffman decode (fast/fast-AC tables, restart intervals, EOB runs, successive approximation, DNL, junk-at-end skipping), dequantise, scalar `stbi__idct_block` (stb documents the SSE2/NEON IDCT as bit-identical), `stbi__resample_row_{1,v_2,h_2,hv_2,generic}` (fancy upsampling; the SIMD hv_2 equals scalar), `stbi__YCbCr_to_RGB_row` fixed point (reduced precision, same in SIMD and scalar; step 3 is the scalar loop anyway), Adobe APP14 / JFIF / RGB-component-id / CMYK / YCCK / grayscale handling. Integer only, no floats at run time (constants are the same `float*4096+0.5` truncations), no threads. `imgprep::decode` routes any file starting FF D8 through it (a panic on corrupt data becomes an error); PNG still uses the image crate (bit-exact on the 4 PNGs tested).
- Verification beyond the two project JPEGs: a C harness (`logs/n35/jpgtest/stbdump.c`, compiled against the same stb_image.h with SSE2 and again with STBI_NO_SIMD) dumps stb output for a 34-file corpus: Go image/testdata (baseline, progressive, 4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 / 4:1:0 / 4:1:1 and odd factors 221122, 222112, 211211, 121121, 221212, restart intervals, separate DC progression, truncated progressive, RGB-component-id, CMYK, grayscale incl. 2x2 sampling) + two sysstat/bcc docs JPEGs + city_1000x562.jpg + big_3000x2000.jpg. stb SSE2 output == stb scalar output on all 34, and `stbjpg_cmp` reports 34/34 byte-exact (`logs/n35/jpgtest/cmp.log`).
- Results after the port (logs/n35/imgparity3.log, imgvl2.log): pixel tensors bit-exact on all 6 test images (incl. city_1000x562.jpg 0/1714176 differ, big_3000x2000.jpg 0/10106880), ViT embedding cosine 0.99995-0.99998 (city 0.99996, min per-token 0.9965, was 0.9631), greedy answers identical to EOS on all 6 (city 55/56 ids + the same final '.'+EOS vs '.\n\n' merge as every other image; before: diverged at token 45).
- Limits: only 8-bit JPEG (as stb); arithmetic-coded or 12-bit files fail like in stb; 16-bit/palette PNG paths of the image crate are not stb-verified (not needed for the tests). CPU decode+preprocess of the 3000x2000 JPEG is ~74 ms (decode inside the one-time file read; per-image encode energy numbers above exclude file decode).

## Beam-1.0-501B-A23B (Reflection) - added 2026-10-18 (disk-streamed)
501B/23.4B-active MoE (3 dense + 49 MoE layers, 256 experts, top-8, gated GQA + SWA, NoPE full layers, scaled residual). NVFP4 checkpoint streamed from NVMe: see docs/beam.md.
Status: runs, correct output; ~0.45-1.2 s/token, ~21 J/token warm. Open: prefill batching, prefetch, numeric parity reference, cold-cache energy.

- Beam-501B phase 3 (disk-streaming, Strix Halo): 3-bit experts 0.89 tok/s (95.7% top-1 vs NVFP4), 2-bit 1.19 tok/s (74%, too lossy); per-layer GPU-visible LRU expert cache with 69-81 GB RAM: NVFP4 2.31 tok/s (hit 77%), 3-bit 2.69, 2-bit 4.78 tok/s, vs 0.72 uncached cold. See docs/beam.md.

- Beam-501B cached energy (net socket, differential, warm-up excluded): NVFP4 uncached 45.8 J/token; +cache 30.8; 3-bit+cache 13.9 (0.25 s/tok); 2-bit+cache 10.0 (0.147 s/tok, quality-limited). Mixed 3/2-bit with 64 hot experts/layer: top-1 91.8% (KL .063) vs 2-bit 81.9% (KL .43), 3-bit 94.3%. See docs/beam.md Phase 4.

- Beam 2-bit with activation-aware scales (calibrated on 6 prompts, tested on 6 other prompts): top-1 85.1% vs 81.9% data-free, KL .237 vs .433 (3-bit: 94.3%, .031). See docs/beam.md.

## Beam-1.0-501B-A23B: quantization follow-up (Nov 2)
Cactus-style codebook 3-bit cuts the act-weighted expert error by 4% on average vs our act-aware q3 (0.0273 vs 0.0285; up to 12% at layers 20-50). Not adopted yet; it needs a perplexity check first. ABM GEMV: exact but no faster, so MAC is kept. Edge0 expert-cache policies: LFU 0.702 vs LRU 0.690 hit rate, vs a Belady optimum of 0.837, so it wasn't ported.
