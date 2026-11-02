# cosmos3: NVIDIA Cosmos3-Nano text-to-video on strix-rt

Model: nvidia/Cosmos3-Nano (Diffusers Cosmos3OmniPipeline, bf16, 33 GB) at $MODELS/new/cosmos3/Cosmos3-Nano.

## Architecture as implemented (crates/cosmos3)
- Cosmos3OmniTransformer is one 36-layer Qwen3-VL-style stack (d 4096, 32 q / 8 kv heads, head 128, SwiGLU 12288) with
  two weight sets per layer (Mixture-of-Transformers): und (text) = to_q/k/v/out, norm_q/k, mlp; gen (diffusion) =
  add_q/k/v_proj, norm_added_q/k, to_add_out, mlp_moe_gen, *_moe_gen layer norms. joint_attn "two_way": text tokens attend
  causally to text only; latent tokens attend (full) to [text K/V, latent K/V]. So the und tower runs once per prompt and its
  per-layer K/V are cached (like qimage's text prefix); each denoise eval runs only the gen tower.
- No adaLN: timestep conditioning is additive, x = proj_in(patchify(latent)) + time_embedder(time_proj(t * 0.001)),
  t = int64(sigma * 1000) (diffusers casts the timesteps to int64). time_embedder runs in f32 on the CPU.
- Interleaved 3D mRoPE (theta 5e6, sections 24/20/20: freq p uses H if p%3==1, W if p%3==2, p < 60, else T), NeoX
  rotate_half pairs. Text positions 0..L; latent positions (frame + L + 15000, row, col) (fps modulation is identity at 24 fps).
- Patchify 2x2 over a 48-channel latent -> 192 features ordered (p, q, c).
- Scheduler: UniPC bh2 order 2 with Karras flow sigmas (sigma_min 0.147, sigma_max 200, rho 7, s/(s+1)); flow_shift is
  ignored on this path (src/sched.rs). CFG 6 with separate cond / uncond prompts (two K/V caches).
- VAE: Wan2.2 AutoencoderKLWan decoder (z 48, base 256, residual up blocks, patch 2). Implemented as a whole-clip pass:
  causal conv3d = new im2col over (kt, ky, kx, c) + prefill coop-matrix GEMM (f16 weights repacked at load); upsample3d
  time_conv runs on frames 1.. only (frame 0 bypasses it, = diffusers' "Rep" cache), DupUp3D shortcut keeps the last
  sub-frame of frame 0 (first_chunk). Mid attention per frame (cos_vattn.comp). Shaders: crates/cosmos3/shaders.
- Kernels/loader reused from crates/qimage unchanged (st.rs zero-copy multi-shard safetensors, qi_norm, qi_qk, qi_small,
  qi_flash (kvdiv = 4 for GQA), qi_ew, qi_vrow); shared shader defaults untouched.

## Reference
~/cosmos_ref: uv venv (python 3.14) reusing ~/venvs/ml (torch 2.12 ROCm 7.13) via a .pth, diffusers git main
(7ecb1e1, 0.41.0.dev0). logs/cos/ref.py captures ids, positions, per-call velocity, layer 0/1/17/35 hidden states; golden at
~/cosmos_ref/g256 (256x256, 9 frames, 8 steps, CFG 6, seed 42, prompt assets/example_t2v_prompt_short.txt, empty negative).
logs/cos/vae_ref.py decodes any latent with diffusers' VAE (CPU fp32) and dumps mid/up0/up1/up2/decoder outputs.

## Parity (256x256x9, 192 latent tokens, cond 71 / uncond 63 text tokens)
- position ids 0/263 mismatches; sigmas and timesteps exact.
- und tower hidden states vs diffusers ROCm bf16: L0 cos 0.999993 (rel 3.7e-3), L1 0.999995, L17 0.999999 (rel 1.4e-2).
- gen tower: input cos 0.999998, L0 0.999994, L1 0.999992, L17 0.999881, L35 0.999899.
- velocity, teacher-forced on the golden inputs, all 16 calls: cos 0.99977-0.99992, rel 1.2e-2 to 2.2e-2 (logs/cos/tf_parity.log).
- UniPC alone on golden CFG velocities: latent rel 2.4e-3 to 3.2e-3 every step (bf16 rounding of the golden inputs).
- free run: latent rel 0.24% -> 22% over 8 steps (CFG 6 amplifies the ~1.3% velocity error; chaotic like qimage).
- VAE on the same latent vs diffusers fp32: video PSNR 85.25 dB, every stage cos 1.000000 (logs/cos/vae_parity.log).
- end to end video vs diffusers pipeline: PSNR 23.64 dB, cos 0.990 (qimage 512: 22.71 dB). Same scene and motion.
Sample: logs/cos/g256_run.mp4 (and logs/cos/bench1.ours2.mp4).

## Speed and energy (Oct 15, logs/cos/bench1.sh under gpu_lock, perf level high, energy recorder 20 Hz socket power)
| 256x256x9, 8 steps, CFG 6 | strix-rt | diffusers ROCm bf16 |
|---|---|---|
| prompt / und prefix | 1.13 s, 41-53 J | (inside denoise) |
| denoise (16 transformer evals) | 7.17 s, 503 J (GPU 6.71 s = 0.42 s/eval) | 9.65 s, 814 J |
| VAE decode | 2.58 s incl. 1.6 s weight repack (GPU 0.97 s), 141 J | CPU fp32 8.31 s, 878 J; ROCm GPU ~210 s (capture run) |
| total | 10.9 s, 685 J | 18.0 s, 1692 J (with the CPU VAE) |
Gen layer profile at 192 tokens (COS_PROF=1): swiglu 3.3 ms (11.7 TF), down 1.9 (10.0), qkv 1.2 (8.0), o 0.56, flash 0.2;
sum 7.3 ms/layer = 262 ms/eval vs 420 ms measured for the full stack: the gap is open work.

Run: target-cos/release/cos --golden ~/cosmos_ref/g256 --stage tower|tf|run|sched|vae|text [--out P] [--lat F --vref DIR]
(env COS_DUMP=0,1,17,35 layer dumps, COS_PROF=1, COS_LPS layers per submit (default 2), COS_VAE_DUMP=1).

## Not done yet
- Own tokenizer + Cosmos prompt templates (ids come from the golden dump today; qimage tok.rs + chat template is the path).
- AR text generation (the und tower + lm_head logits run, stage text, not yet checked vs a reference), ViT understanding
  (Qwen3VLVisionModel, from mcv), sound tokenizer / audio and action heads.
- Bigger resolutions: VAE mid attention kernel is naive (<= 2048 pixels per frame: up to ~720x1280 latent 45x80 = 3600 needs a flash path).

## Oct 15 follow-up (code in, GPU results pending: the GPU lock is wedged, see below)
- Text in, video out: src/prompt.rs (templates, system prompt, chat format, eos + <|vision_start|>; ids match the golden
  71/71 and 63/63) and `cos gen --prompt .. --negative .. --size HxW --frames N --steps N --cfg F --seed N --out P`.
- 720p VAE attention: shaders/cos_vattn_t.comp, a tiled online-softmax mid attention (4 queries per workgroup, 256-key tiles,
  no pixels-per-frame limit). Used automatically above 2048 px/frame, or forced on with COS_VATTN_T=1. Parity script logs/cos/vattn_t.sh,
  720p run logs/cos/gen720.sh (704x1280, 9 frames, 12 steps; runs only if the parity check reaches >= 60 dB).
- Understanding path: src/vit.rs = Qwen3-VL ViT (27 blocks, axial 2D rope, bilinear 48x48 position table, 2x2 merger and DeepStack
  mergers after blocks 8/16/24), with heads zero-padded for qimage flash. fc2 and the patch conv are repacked to f16 because K must be
  a multiple of 64. Model::prefix_emb, set_rope_u and State.ds feed the image embeddings, the Qwen3-VL 3D positions and the DeepStack adds
  (after und layers 0-2) into the und tower. `cos ask --image F | --ref DIR --question Q --new N` decodes greedily; for now each
  token re-runs the und prefix (no KV append yet). vllm-cosmos3 confirms the understanding tower is plain Qwen3-VL with the und weights.
  Reference: logs/cos/vit_ref.py (HF Qwen3-VL on CPU, bf16, with the transformers_cosmos3 key mapping). Its answer for
  assets/example_i2v_input.jpg: "A car is driving on a winding road that curves to the right, ...". Parity script: logs/cos/ask_ref.sh.
- Denoise gap: COS_LTIME=1 prints per-layer GPU times; logs/cos/step4_ab.sh. Earlier A/B: COS_SAMEL=1 (every layer uses layer 0's weights)
  runs at 300-390 ms against 420 ms, so part of the gap comes from streaming fresh weights (memory/TLB), not from the ops.
- Diffusers VAE on ROCm in its own session: logs/cos/vae_gpu_ref.sh (REPS=3, energy recorder, perf level high).

## Oct 16 results (after the reboot)
- Text to video: logs/cos/gen_dog.mp4 (256x448, 17 frames, 20 steps, own tokenizer and noise). Total 66.4 s: denoise 52.7 s GPU, VAE 3.4 s.
- Diffusers VAE on ROCm bf16, own session (logs/cos/vae_gpu_ref.out): 1.04 s on the first call, then 0.64 s / 53-64 J (256x256x9).
  Steady state is faster than ours (GPU 0.97-1.13 s). The old "~210 s" figure was a contended capture run and is withdrawn.
- Tiled VAE attention: forced on at 256 it gives PSNR 85.35 dB vs the CPU fp32 reference (naive kernel 85.25).
  720p (logs/cos/gen720.mp4): 704x1280, 5 frames, 12 steps, 1760 tokens; denoise 42.7 s (1.78 s per eval), VAE 8.0 s.
  9 frames failed in the VAE: each activation buffer is 4.15 GB (> 4 GB), "not enough memory for command submission", device lost.
  The latent is in gen720_9f_fail.log / gen720.lat.f32. Longer 720p clips need temporal-chunked (causal cache) or tiled VAE decode.
- Understanding (logs/cos/ask_ref.log): ViT merger cos 0.9979, DeepStack 0.9994-0.9999, positions 0/591 mismatches,
  first-token logits cos 0.99976, and all 32 greedy tokens match HF. ViT 64 ms. Decode is 0.7 s/token because it re-runs the prefix
  (KV append is the next step). Fixed along the way: lm_head ran on all rows into a buffer sized for 24576 columns; it now runs on the last row only.
- Denoise gap (logs/cos/step4c.log, perf level high): 431 ms per eval. Per layer, layers 0-4 take 7.7 ms (the same as the per-op profile)
  and layers 5-35 take 10-17 ms. With COS_SAMEL=1 (every layer reuses layer 0's weights) it is 275 ms. So the gap comes from the GPU reading
  fresh weights out of GTT system memory (VRAM carve-out is 512 MB), not from the kernels; it changes nothing whether the weights are mmap'd or copied (QI_ZC=0).
  Closing it means changing how weights are allocated or placed (large pages / contiguous GTT, or a bigger UMA carve-out) in vkrt/qimage,
  which are shared crates and were left alone.

## Oct 16 (later): QI_WALIGN default, 720p x 9 frames
- `cos` now sets QI_WALIGN=1 by default (4 KB aligned weight copies; QI_WALIGN=0 = old zero-copy path). Parity identical to the digit
  (logs/cos/walign_parity.log: velocities, free run, video PSNR 23.64 dB, both modes). 256x256x9 denoise GPU 6.73 s -> 4.42 s (16 evals, 276 ms/eval vs 421).
  Cost: load 8-10 s instead of 0.1-4 s (33 GB copy).
- With the zero-copy path 704x1280 now fails in the first submission ("Not enough memory for command submission", device lost) for 5 and 9 frames,
  although the same 5-frame run worked at 10:41; with QI_WALIGN=1 both pass (logs/cos/s720_walign.log, perf level high, 1 step):
  5 frames: 1760 tokens, 1.47 s/eval, VAE 7.21 s GPU; 9 frames: 2640 tokens, 2.3 s/eval, VAE (chunked) 13.07 s GPU.
- VAE temporal chunk: COS_VAE_CHUNK=1 faults the GPU (page fault, also at 256); chunk 2 is exact (85.25 dB). The auto default is now 2.
- Open: KV-cache `cos ask` (WIP, faults), beating diffusers VAE (0.64 s vs our 0.95-1.0 s GPU), full-step 9f 720p quality run.

## Oct 16: KV-cache `cos ask` (exact), BF16 GEMV decode
- The e1dd8df fault is gone with QI_WALIGN=1 (aligned weights); the KV path (und tower for the newest token only, flash over cached rows) matches
  HF: 32/32 greedy tokens identical, first-token logits cos 0.999760 (logs/cos/ask_kv_gemv.log).
- Decode with the coop-matrix GEMM at m=1 ran 277 ms/token (tower 350-450 ms; only 16-48 workgroups per GEMM). New shaders/cos_gemv.comp (BF16 GEMV,
  one 256-lane workgroup per output row, used for qkv/o/gate/up/down/lm_head when m=1) -> **75 ms/token** (re-prefill every token: 320 ms/token). COS_GEMV=0 disables; COS_DEC_TRACE=1 prints tower vs lm_head time.

## Oct 16: 704x1280 failures: kernel ENOMEM in the amdgpu VM update (not memory pressure from other jobs)
- "radv/amdgpu: Not enough memory for command submission" = amdgpu_cs returning -ENOMEM. At 12:03:30 the kernel logged
  `amdgpu_gem_va_update_vm: Could not update BO_VA (-12)` (5x) in the 720p 9-frame full run, right after the denoise finished (latent written) and the
  VAE buffers were allocated; the process then hung in D state in drm_suballoc_new <- amdgpu_ib_get <- amdgpu_vm_update_range <- amdgpu_cs_vm_handling
  (GPU idle, every ring fence signalled, no ring timeout, no other process on /dev/dri). SIGKILL does not remove it (needs a GPU reset or reboot).
- Not caused by other jobs: only the cos process had the render node open. Free memory was 2-9 GB with 50-80 GB of clean page cache in every failing
  run; the 10:41 success had 24 GB free. Page-table / IB allocation in the VM update fails in that state (hypothesis, not yet confirmed by an A/B with
  drop_caches before the run; blocked by the hung process holding the GPU lock).
- Zero-copy (userptr, 4 KB pages over 33 GB of mmap) needs far more VM page-table memory than the aligned-copy path, which is why QI_WALIGN=1 survives
  where zero-copy fails at 720p.

## Oct 16 (after the hard reset): 720p x 9 frames x 12 steps, full run (logs/cos/full720.log, full720.mp4)
- 704x1280, 9 frames, 12 steps, CFG 6, 2640 tokens, perf level high, QI_WALIGN=1 default: load 8.7 s, denoise 55.4 s GPU (57.0 s wall, 2.3 s/eval),
  VAE 12.8 s GPU (13.9 s wall, temporal chunk 2). One lock session, about 80 s.
- Quality (visual, frames 0 and 8): coherent coastal cliff road at sunset, a red vintage car in frame, same scene and consistent motion between frames, no
  VAE tiling or chunk-seam artifacts. There is no pixel reference at this size (our noise differs from diffusers'); 256x256 parity stays 23.64 dB / 85.25 dB VAE.
- Bug found: the 720p VAE lost the device (ring gfx timeout) when it ran in the same process after the 12-step denoise (3 of 3 runs; with 1 denoise step it worked), and
  succeeded every time in a fresh process on the same latent. Not the descriptor cache (clearing it did not help). `cos gen` now writes the latent and exec()s
  `cos vae --lat F --size HxW --frames N --out P` (also usable on its own); COS_VAE_SAME=1 keeps the old in-process decode, COS_NO_VAE=1 stops after the latent.
  Root cause of the in-process failure not found.
- Page-cache drop test: not needed this time; the box had just rebooted (120 GB free, cold cache) and the failure was not ENOMEM. The ENOMEM hypothesis for zero-copy stays untested.
- VAE speed: 256x256x9 GPU 0.96 -> 0.91 s with 128x128 GEMM tiles (PF_MINWG, set by default in the VAE decode only; the same setting costs the gen tower +18%: 276 -> 327 ms, logs/cos/minwg_ab.log). PSNR unchanged 85.25 dB. Still above diffusers 0.64 s: the VAE is bound by the im2col convolution GEMMs (~10.5 TFLOPS); closing the gap needs a faster conv/GEMM kernel (implicit-GEMM conv without the 27x im2col), not started.

## Oct 16: implicit-GEMM VAE conv (shaders/cos_conv.comp)
- Causal KTxKSxKS conv without the im2col buffer: 128x128x32 coop-matrix tiles whose A slab is gathered straight from the channel-last frame stack
  (per-tap slab inside one tap, causal front padding and spatial zero padding handled by the loader, zero-fill outside). Used when cin % 32 == 0,
  cout % 128 == 0 and not 1x1 (COS_IMPL=0 = old im2col path); conv_in (cin 48) and conv_out (cout 12) stay on the old path. Temporal-chunk caches unchanged.
- 256x256x9 VAE GPU time: 0.92-0.96 s -> 0.66-0.67 s (wall 0.74 s), PSNR vs CPU fp32 85.25 dB unchanged (whole clip and COS_VAE_CHUNK=2) (logs/cos/impl_ab.log).
  diffusers ROCm bf16 steady state: 0.64 s per call (own session), so we are at 1.03x of it on GPU time (1.16x on wall incl. readback and unpatchify).
- 704x1280x9 VAE: 12.8 s -> 9.6 s GPU, output bit-identical to the im2col decode of the same latent (cmp).
- In-process 720p VAE after the 12-step denoise now works (COS_VAE_SAME=1: total 79 s, VAE 9.59 s, no ring timeout), so the earlier device loss was tied to the
  old im2col path in that state (3 of 3 failures there, 0 of 1 with implicit conv). Not proven: I did not re-run the old path in-process again to avoid another GPU reset.
  The exec workaround stays as the default.

## Oct 17: measured energy per clip (end to end, fresh process per clip)
Method: `logs/session3.sh` (wait for < 0.8 busy cores system-wide, gpu_lock, perf level high), `tools/energy/record.py record --hz 20 --idle 5` windows (SMU socket power; idle baseline of 5 s taken right before each window; net = window minus idle W x seconds), K clips back to back per window, 3 windows. Prompt: the fixed "red vintage car" prompt ("A red vintage car driving along a coastal road at sunset, with a lighthouse on a cliff in the distance"), seed 42, CFG 6. Each clip is a fresh `cos gen` process: weight load (10-12 s), prefix, denoise, then the exec-fresh-process VAE (the workaround; no device loss in 6 of 6 clips). Raw data: logs/cos/energy3_s256 and energy3_p720 (json, csv, logs); `ana.py` and `ph.py` summarise them.
| clip | windows (K) | wall per clip | socket J per clip, net (windows) | gross J per clip | phases only (prefix+denoise+VAE), net | J / frame | J / Mpx-frame |
|---|---|---|---|---|---|---|---|
| 256x256x9, 8 steps | 2 clean x K=3 | 18.2-18.3 s | 626, 633 | 789-791 | about 412 J (denoise 298 J / 4.9 s, VAE 106 J / 2.4 s) | 70 | 1070 |
| 704x1280x9, 12 steps | 3 x K=1 | 77.9-78.5 s | 6320, 6367, 6435 (median 6367) | 7009-7129 | about 6130-6240 J (denoise 5190-5280 J / 54.5 s, VAE 930-950 J / 12.4 s) | 707 | 785 |
Caveats (read before quoting):
- The 256 figure of 685 J from Oct 15 was a different prompt (kitchen), gross-ish and excluded process start; the comparable number for this prompt is ~412 J for the phases and ~630 J per clip including the 10-11 s weight load of a fresh process. A server that keeps the weights resident would pay roughly the phase figure.
- Phase split is aligned from the CSV by the end time of the last clip (+/- 0.3 s), so it is approximate (about 5%); window totals come from tools/energy/record.py directly.
- Discarded windows (not committed): one 256 window at 5.1 busy cores from other users gave 1039 J/clip net, two earlier sessions with neighbours building (idle baselines 28-104 W instead of ~9 W) gave 464-1843 J (256) and 2.9-6.3 kJ (720p) and are not trustworthy. The three valid 720p windows and two valid 256 windows ran at 0.2-0.7 average busy cores including our own process.
- Socket power only (SMU), not wall power. The idle baseline at perf level high was 8.5-9 W on the quiet box.
- Quality is unrelated to these joules: no pixel reference at 720p (see above); 256 parity unchanged.
