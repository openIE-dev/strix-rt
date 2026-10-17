# qimage: Qwen-Image-2.1 text-to-image on strix-rt

Model: Qwen/Qwen-Image-2.1 (HF, BF16, 33 GB) at $MODELS/qwen-image-2.1. No GGUF needed.

Pipeline (crates/qimage):
- tokenizer (tok.rs): byte-level BPE from processor/tokenizer.json; ids identical to HF (50/50, drop_idx 14).
- text encoder (Qwen3-VL-8B text part, 36 layers): prefill coop-matrix GEMMs on zero-copy BF16 weights; last hidden state pre final norm.
- DiT (32 single-stream blocks, d 4096): coop-matrix GEMMs, two-pass coop-matrix flash attention (qi_flash), text prefix K/V computed once (t=0 modulation) and cached.
- scheduler: flow-match Euler, exponential shift (mu from seq len), shift_terminal 0.02, on the CPU (sigmas match diffusers to 6e-8).
- VAE decoder (vae.rs): channel-last f32, im2col + GEMM in pixel chunks, mid attention via GEMMs. F32 weights repacked to f16 once.
- Weights: st.rs mmaps safetensors and imports page-cache folios as dma-bufs (memory tier, zero copy, chunks < 3.5 GiB).

Parity at 512x512, 40 steps, seed 42, vs diffusers bf16 on ROCm (~/qi_ref/golden512):
- text encoder vs HF CPU per layer: rel 3e-3 to 6e-3 (layers 1-4). Final embeds vs golden: rel 4.3e-2, mostly from the first kept token (massive-activation row, 26%); other rows 1-4%.
- DiT, teacher-forced per step (golden input latents): velocity rel 0.8e-2 to 1.8e-2 over all 40 steps.
- VAE alone on golden final latents: 50.65 dB PSNR, SSIM 0.9988.
- Free-running end to end: 22.71 dB PSNR, SSIM 0.932. The image is visually the same scene; the main difference is the sign text, a chaotic detail.

Timing (512, Oct 14, GPU shared/unstable, treat as provisional): denoise 57.3 s GPU for 40 steps (1.43 s/step), VAE 0.4-0.64 s;
diffusers ROCm 65.5 s/image. 1024 numbers pending: GPU ring resets from ~18:04 make later timings suspect.

Build note: the uncommitted pf_deq.glsl edit in the main tree reads BF16 as uvec4 (16-byte alignment), but 60% of
these safetensors tensors are only 8-byte aligned, which gives garbage GEMMs. qimage was validated in the worktree
~/qiwt (HEAD prefill), target dir target-qi.

Run: target-qi/release/qi [--golden DIR] [--size 1024] [--steps 40] [--out x.png] [--stage te|vae|all]

## 1024x1024 (Oct 14, 23:13 UTC, under logs/gpu_lock.sh, perf high, no ring timeouts during the run)
Submissions are split per 2 blocks (QI_BPS), flash attention dispatches at most 16 query tiles (QI_FLASH_QT), and weight
chunks are block-aligned so fused SwiGLU is on.
- strix-rt: 293.3 / 299.3 s/image wall (denoise 287-290 s GPU = 7.2 s/step, VAE 1.84 s, TE 0.2 s, prefix 0.27 s).
- diffusers bf16 ROCm: 429.2 s/image (rep 1; rep 0 927 s ran during the reset period).
- Free run vs golden1024: PSNR 37.91 dB, SSIM 0.9917. VAE alone on golden latents: 51.27 dB.
- 512 after the changes: denoise 54.1 s (was 57.3), PSNR unchanged at 22.71 dB.
Sample: logs/qi1024.png

## Optimization round (Oct 15, 00:4x UTC, gpu_lock, perf high -> auto)
Per-op profile at 1024 (one image block, `qi --stage prof`, logs/qi_prof.sh):
before: qkv 29.6 ms (13.9 TFLOPS), flash 86.0 ms (3.2), o 10.0, swiglu 67.5 (12.2), down 44.3 (9.3); elementwise < 4 ms total.
- flash attention: single pass with lazy rescaling (O rescaled via smem only when a row max grows by > 2^8), 8 waves x 16
  queries per workgroup, S/P share one smem region: 86 -> 36 ms/block (7.6 TFLOPS). Step 7.17 -> 5.44 s.
- prefill GEMM knobs (PF_KS, PF_LS4) and F16 instead of BF16 weights (QI_F16W): no gain; GEMMs sit at ~13 TFLOPS.
- own 128x128 coop-matrix BF16 GEMM (qi_gemm, QI_GEMM=1): correct (512 PSNR 22.66) but slower (step 10.5 s); opt-in only.
Result at 1024: 222.3 / 228.2 s/image (denoise 217.5 s = 5.44 s/step), PSNR 38.04 dB, SSIM 0.9918 vs golden1024
(diffusers 429 s/image). Remaining time: GEMMs ~75%, flash ~19%.

## Activation stride padding (Oct 15)

Power-of-2 activation row strides (h: FFD 12288, a: D 4096 halves) camp DRAM channels on Strix Halo.
Padding both by 64 halves (QI_HPAD, QI_APAD, default 64) at 1024, per block:

| op | before | after |
|---|---|---|
| qkv | 29.7 ms 13.9 TF | 15.8 ms 26.2 TF |
| o | 10.5 ms 13.1 TF | 5.2 ms 26.3 TF |
| swiglu (fused, A via LDS on BF16) | 47.2 ms 12.2 TF | 43.9 ms 18.8 TF |
| down | 48.8 ms 8.4 TF | 15.6 ms 26.5 TF |
| flash | 36.7 ms 7.6 TF | unchanged |

Step 5.57 -> 4.04 s. qimage 1024: denoise 170 s, wall 179-188 s/image (was 222 s), PSNR 38.04 dB, SSIM 0.9918 vs golden1024.
512 parity unchanged (22.71 dB). LLM prefill (Qwen3.8-27B Q4_K): 16/16 greedy at 512 and 2048, pp512 538.6 tok/s;
pp2048 432.8 tok/s in that run was a cold first run, not a regression: A/B (logs/q38_npu_ab.log) default 437.0 then 486.5 tok/s, NPU_SPLIT=0 NPU_TB=0 487.8 / 483.7, 16/16 greedy in all four. pf_q38 (Q4_K) never touches the NPU: NPU_SPLIT/NPU_TB default 0 and only apply to Bonsai ternary projections, so 4751a63 cannot affect it. Read the 2nd+ run.
