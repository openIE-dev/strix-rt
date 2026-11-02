# Efficient-AI tracks, 2026-10-30 (approval pt51u, "do it all")

All energy numbers: tools/energy/record.py SMU socket channel, differential windows (long minus short), quiet gate, gpu_lock. Times in ET.

## (3) Edge0 storage tiering vs Beam expert cache
Edge0 (github.com/Edge0-AI/edge0, docs/streaming.md, read 2026-10-30): 4-bit weights on SSD, mmap on demand; a **cross-layer shared LRU**
(`cache_slots=64` total, default for both tiers), **hot residency from decayed counts** (`hot_decay 0.75` every 4 steps, `pin_bonus 2`),
**trained pre-router heads** that predict layer N+1 routing at layer N for prefetch, an optional **lossy "staged" decode** (fixed slots;
experts not ready map to a zero slot and are dropped; `staged_replace` swaps the model's routing for the predicted set), madvise WILLNEED readahead,
whole-layer prefill loads when the file is page-cached, and LoRA adapters distilled to repair 4-bit loss. Their device (M4 Pro, 24 GB) is ~1.5 GB/s-class
flash per token budget; ours is NVMe-bound at 6-7 GB/s.

Offline cache simulation on Beam's own routing traces (tools/beam_cachesim, logs/beam/prof: 12 prompts, ~753 tokens x 49 MoE layers, first half warms, hit rate on the second half):

| slots/layer (equal total) | per-layer LRU (ours) | global shared LRU (Edge0) | decayed-LFU (Edge0 hot counts) | LRFU l=0.9 | Belady optimum |
|---|---|---|---|---|---|
| 32 | 0.540 | 0.535 | 0.565 | 0.561 | 0.715 |
| 64 | 0.690 | 0.693 | 0.702 | 0.700 | 0.837 |
| 96 | 0.787 | 0.794 | 0.789 | 0.789 | 0.898 |
| 128 | 0.851 | 0.856 | 0.852 | 0.852 | 0.937 |

Conclusion: Edge0's replacement policies are worth at most +0.3-1.2 pts of hit rate over our per-layer LRU (+2.5 pts at 32 slots); not worth
porting. The remaining headroom (LRU to Belady, ~15 pts at 64 slots) needs knowledge of *future tokens'* routing, which a 1-layer-ahead
pre-router does not give; our own next-layer-router prefetch (87% top-8 recall) already showed no speedup because the disk is saturated.
Edge0's lossy "drop experts not ready" is the one idea that changes bytes/token; it is a quality trade (like top-k reduction) and is not tested here.

## NPU vs iGPU for decisions (quiet reps 1+2, Oct 31; logs/npu/e)
| config | net J/decision | s/decision | net J/gen tok | acc (n=800) |
|---|---|---|---|---|
| Llama3.2-1B Q4_K_M, iGPU vk | 2.56 | 0.029 | 0.305 | 0.489 |
| Llama3.2-1B, NPU (FLM) | 8.09 | 0.42 | 0.293 | 0.520 |
| Qwen3.5-0.8B Q4_K_M, iGPU vk | 2.73 | 0.040 | 0.257 | 0.510 |
| Qwen3.5-0.8B, NPU (FLM) | 4.19 (noisy: idle 30.6 W) | 0.51 | 0.353 | 0.515 |
NPU rail itself is 0.72-0.99 J/decision, but whole-system time is 12-15x longer, so the iGPU wins per decision by about 3x. Per generated token they're about even.

## Speculative decoding J/token (quiet rep 1; logs/spec/quiet_summary.txt)
Glimmer 30B 7.37 -> 5.84 (DFlash, acc 0.42); Qwen3.8 27B 7.36 -> 4.84 (MTP3, acc 0.49, tentative);
gpt-oss 20B 0.94 -> 1.15 (EAGLE3 n2, acc 0.48: a loss for small active models).

## Add-before-multiply GEMV (Beam; docs/beam.md)
Exact (top-1 47/47). Quiet: MAC 25.8 ms / 1.94 J/tok, bit-plane 25.7 / 2.12, code-bin 33.7 / 2.85. MAC stays default.

## Cactus-style rotation+codebook 3-bit (Beam experts; logs/cq/probe_full.log, Nov 2)
Act-weighted relative error, layers 5/20/35/50 x experts 0/250:
| layer | q3flat | q3 (ours) | rot+q3 | cb | rot+cb |
|---|---|---|---|---|---|
| 5 | 0.030 | 0.025 | 0.029 | 0.025 | 0.026 |
| 20 | 0.031 | 0.030 | 0.029 | 0.029 | 0.026 |
| 35 | 0.031 | 0.029 | 0.030 | 0.028 | 0.027 |
| 50 | 0.035 | 0.029 | 0.034 | 0.026 | 0.030 |
| mean | 0.0316 | 0.0285 | 0.0305 | 0.0273 | 0.0274 |
Codebook is about 4% lower error on average, with up to 12% lower in the mid and deep layers (rot+cb at 20, cb at 50) and no gain at layer 5. That's a small win, and it's only worth pursuing if a per-layer choice of {q3, cb, rot+cb} costs no decode time. The next step would be a perplexity check, not a kernel.
