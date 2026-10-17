# Ternary (1.58-bit) kernels: `crates/ternary`

GPU decode GEMVs, an online Walsh-Hadamard transform, and a batched int8 GEMM for ternary {-1,0,+1}
weights with a scale per group of 128. Measured on the Framework Desktop (Radeon 8060S gfx1151, RADV,
~256 GB/s), 2026-10-14. Raw logs are in `logs/tern_*`.

Code layout (separate from vkrt so it doesn't touch other workers' files):
- `crates/ternary/src/lib.rs`: format definitions, CPU reference dequant, `fwht_ref`
- `src/gg.rs`: a GGUF reader that accepts the fork-private types. `crates/gguf` divides by the block
  size and panics on unknown types.
- `src/st.rs`: a minimal safetensors reader
- `shaders/tern_gemv.comp`: one source built per format as `-DPQ2|PTQ1|TQ21|TH5`, with lanes-per-row
  `T` and rows-per-cluster `NR` set as defines. `build.rs` builds 48 variants.
- `shaders/tern_fwht.comp`, `tern_quantx.comp`, `tern_gemm.comp`
- bins: `tern_inspect`, `tern_validate`, `tern_bench`, `tern_gemm`
- `tools/fork_dequant.c`: dequantizes GGUF rows with a fork's own `ggml` (`to_float` trait). It's used
  as the reference.

Models are in `$MODELS/{bonsai2,kitten2,parakeet-redux}`. The Prism fork (`prism` branch @ e3de715)
and `kitten-tts-2-cpp` are cloned to `~/src/{prism-llama,kitten-cpp}`, and their `ggml-base` is built
in `build-cpu/`.

## Byte layouts (from source)

### PrismML PQ2_0 (ggml type 142): `ggml-common.h` / `ggml-quants.c` in PrismML-Eng/llama.cpp
```
block_pq2_0 { ggml_half d; uint8_t qs[32]; }      // 34 B per 128 weights = 2.125 bpw
w[j] = d * (((qs[j/4] >> 2*(j%4)) & 3) - 1)        // 00=-1, 01=0, 10=+1 (11=+2 unused for ternary)
```
Blocks are only 2-byte aligned, so the kernel reads them through a `uint16_t` SSBO view.

### PrismML PTQ1_0 (type 143): base-3 packing like TQ1_0, at group 128
```
block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; }   // 28 B per 128 = 1.75 bpw
trit n of byte b:  q = (uint8)(b * 3^n);  t = (q*3) >> 8;  w = d*(t-1)    ("fixed-point" trits, MSD first)
elements   0.. 79 : n in 0..5, m in 0..16 -> trit n of qs[m]        (elem = n*16 + m)
elements  80..119 : n in 0..5, m in 0..8  -> trit n of qs[16+m]     (elem = 80 + n*8 + m)
elements 120..127 : n in 0..4, h in 0..2  -> trit n of qh[h]        (elem = 120 + n*2 + h)
```
(The fork uses `ptq1_0_stages = {32,16,8}` on a 24-byte `qs`, so the 32-stage never runs.) Blocks are
4-byte aligned, 7 words each, and the scale is the top 16 bits of word 6. On the GPU the trits come out
with an iterated `q*=3; t=q>>8; q&=255` on `uvec4`.

### Bonsai 2 Hadamard basis (GGUF KV, read by `llama-model.cpp`)
- `prism.hadamard.version=1`, `block_size=1024`, `transform="normalized-sylvester-walsh-hadamard"`,
  `axis="input-last-dimension"`, `sign_mode="explicit"`
- Sign vectors are stored in the GGUF, not derived from a seed. `prism.hadamard.sign_widths = [5120, 6144, 17408]`
  and `prism.hadamard.sign_values` is an I32 array of 28672 ±1 values: the vectors for each width,
  concatenated. Each folded weight uses the vector that matches its input width (`ne[0]`).
- Forward, for every weight in `prism.hadamard.weight_names` (401 entries: `output`, and per layer
  `attn_qkv`, `attn_gate`, `ssm_out`, `attn_q/k/v/output`, `ffn_gate/up/down`):
  `x' = H_1024 (s ⊙ x)` per 1024-block, then `W x'`. Here `H[r][c] = (-1)^popcount(r&c) / 32`.
- `prism.hadamard.gdn_v_grouped=true`: before the signs, the input of each `ssm_out` gets a permutation
  from tiled `[hd, nk, rep]` to grouped `[hd, rep, nk]`, with hd = 6144/48 = 128, nk = ssm.group_count = 16
  and rep = 48/16 = 3.
- `prism.hadamard.inverse_weight_names = ["token_embd.weight"]`: embedding rows are stored rotated, so
  after the lookup apply `h = s ⊙ (H z)` (signs after H).
- The transform is shared through a memo: `attn_qkv` and `attn_gate`, and likewise `ffn_gate` and
  `ffn_up`, consume the same transformed activation.

### KittenML TQ2_1 (type 43, kitten-tts-2-cpp): the `cpp/model-tq2_1.gguf` packing
```
block_tq2_1 { uint8_t qs[64]; ggml_half d[2]; }    // 68 B per 256 = 2.125 bpw, one scale per 128
w[g*128 + l*32 + m] = d[g] * (((qs[g*32+m] >> 2l) & 3) - 1)
```
Note: the GGUF that the Kitten fork runs is 2 bits/weight. The 5-trits-per-byte packing (1.6 bpw) only
exists in `lm/model-ternary.safetensors`.

### KittenML `kittenml-ternary-1.58` safetensors (`kittenml/kittentts2/ternary.py`, pip `kittenml` 0.9.4)
- `<name>::codes` is U8 `[ceil(rows*K/5)]`: one flat stream for the whole tensor, not padded per row.
  Byte = Σ_{i<5} digit_i·3^i (LSD first), digit = sign+1. Row r element i is global trit `r*K+i`, so
  with K=2048 rows start mid-byte.
- `<name>::scale` is BF16 `[rows, K/128]` (group amax).

### moondream parakeet-redux `thrush-ternary-v2` (`ternary.json` + `model.safetensors`)
- `<module>.qweight` is U8 `[out, ceil(in/5)]`: row-major with each row padded to a whole byte (1024 → 205 B).
  Element i is base-3 digit i%5 of byte i/5, LSD first, `code-1`.
- `<module>.scales` is F16 `[out, ceil(in/128)]`, and `w = scales[row, col/128] * (code-1)`.
- 264 quantized modules: FF linear1/linear2, q/k/v/o/relative_k projections, and conv pointwise1/2
  (`as_conv1d`). The depthwise conv, subsampling and the VAD head stay F16, and norms stay F32.
- Photon itself is not open; this layout comes from `ternary.json` and the tensors.

## CPU reference validation (`tern_validate`, logs/tern_validate.txt): ALL OK
| check | result |
|---|---|
| PQ2_0 vs Prism fork `to_float`: 7 tensors × 32 rows (attn_qkv, ffn_down, attn_q, ffn_up, ssm_out, output, token_embd) | 0 bit mismatches |
| PTQ1_0 vs Prism fork: same 7 tensors | 0 bit mismatches |
| TQ2_1 vs Kitten fork: 4 tensors × 32 rows | 0 bit mismatches |
| PQ2_0 vs PTQ1_0 decode of the same Bonsai tensors | identical (same checkpoint, lossless packings) |
| Kitten flat 5-trit safetensors vs TQ2_1 GGUF (3 tensors) | 0 code mismatches; scales equal at the rows checked |
| parakeet: all 120.9M packed bytes < 243; per-module zero fraction vs `ternary.json` (264 modules) | max diff 0 |
| FWHT butterfly vs explicit matrix (n=1024) | 1.2e-7 |

## GPU GEMV (decode) results

Method: weight copies cycled so each dispatch reads a cold copy, with ≥256 MiB in total (≥1 copy for
lm_head), 100 iterations, GPU timestamps. GB/s = (W + scales + x + y) / t. Gweights/s = N·K / t.
Correctness is checked against an f64 CPU GEMV on the dequantized weights. The best of 12 (T, NR)
variants is shown, with a 64-wide workgroup and required subgroup 64. Run: `SWEEP=1 tern_bench all`
(logs/tern_bench_all.txt). No other GPU process was running during the run (checked before and after).

| tensor (Bonsai 2 27B) | N x K | PQ2_0 GB/s | % of 256 | Gw/s | PTQ1_0 GB/s | % of 256 | Gw/s | max_abs/max\|ref\| |
|---|---|---|---|---|---|---|---|---|
| attn_qkv (GDN in) | 10240x5120 | 214.8 | 83.9% | 805 | 198.2 | 77.4% | 901 | ≤1.7e-7 |
| attn_gate (GDN z) | 6144x5120 | 172.5 | 67.4% | 646 | 142.2 | 55.5% | 646 | ≤2.2e-7 |
| ssm_out (+FWHT perm) | 5120x6144 | 212.5 | 83.0% | 796 | 184.6 | 72.1% | 838 | ≤2.2e-7 |
| attn_q (+gate) | 12288x5120 | 213.8 | 83.5% | 802 | 161.2 | 63.0% | 733 | ≤2.0e-7 |
| attn_k / attn_v | 1024x5120 | 166.6 | 65.1% | 616 | 110.1 | 43.0% | 493 | ≤2.1e-7 |
| attn_output | 5120x6144 | 212.7 | 83.1% | 797 | 154.9 | 60.5% | 704 | ≤3.9e-7 |
| ffn_gate / ffn_up | 17408x5120 | 216.0 | 84.4% | 810 | 198.6 | 77.6% | 904 | ≤2.0e-7 |
| ffn_down | 5120x17408 | 224.0 | 87.5% | 840 | 186.7 | 72.9% | 850 | ≤1.3e-6 |
| lm_head (output) | 248320x5120 | 220.8 | 86.3% | 829 | 223.4 | 87.3% | 1018 | ≤2.0e-7 |

Q4_K baseline, same session (`target/release/bench`, q7.gguf, logs/tern_q4k_baseline.txt):
ffn_gate 18944x3584 216.2 GB/s (383 Gw/s), ffn_down 200.1 GB/s (355 Gw/s), token_embd 152064x3584
205.7 GB/s (365 Gw/s). That matches PROGRESS.md's 200-230.

Takeaways:
- PQ2_0 reaches the same fraction of roofline as Q4_K, 213-224 GB/s on the big matrices. At 2.125 bpw
  that is 800-840 Gweights/s, about 2.2x Q4_K's weights/s.
- PTQ1_0 moves 18% fewer bytes. The trit decode costs ALU, so it lands at 185-223 GB/s on the large
  shapes and 142-161 GB/s on some mid-size ones. In weights/s it wins on ffn_gate/up, attn_qkv and
  lm_head (up to 1018 Gw/s) and loses on attn_q and attn_output. This matches PrismML's own
  "dense trits win where bandwidth binds" note. A per-tensor choice (PTQ1 for ffn and lm_head, PQ2 for
  attention) is possible because both packings decode to identical weights.
- Small shapes (attn_k/v at 1.3 MB) are launch- and tail-bound at 110-167 GB/s. In a real decode they
  should be fused with each other: k and v share the same input.
- Run-to-run variance is about ±10% on the mid-size shapes. An earlier run (logs/tern_bench_pq2_sweep.txt)
  had PQ2 attn_gate at 194.7 GB/s and was taken while another worker's `lfm` process was running.
- Decode estimate, not measured end to end: weight bytes per token ≈ file size − token_embd ≈ 6.87 GB
  for PQ2_0 and 5.67 GB for PTQ1_0. At about 215 and 190 GB/s aggregate that is ~31-33 tok/s as the
  matmul-only bound.

Other formats (same method):

| tensor | fmt | N x K | GB/s | Gw/s | max_abs/max\|ref\| |
|---|---|---|---|---|---|
| kitten ffn_gate/up | TQ2_1 | 6144x2048 | 188.8 | 704 | 1.9e-7 |
| kitten ffn_down | TQ2_1 | 2048x6144 | 153.8 | 573 | 2.0e-7 |
| kitten attn_q/o | TQ2_1 | 2048x2048 | 156.0 | 579 | 2.7e-7 |
| kitten attn_k/v | TQ2_1 | 1024x2048 | 118.6 | 437 | 2.4e-7 |
| kitten ffn_up (flat 5-trit safetensors) | thrush5 flat | 6144x2048 | 65.5 | 280 | 1.2e-7 |
| kitten ffn_down (flat 5-trit) | thrush5 flat | 2048x6144 | 81.0 | 346 | 2.1e-7 |
| parakeet ff linear1 | thrush5 row | 4096x1024 | 72.2 | 305 | 1.2e-7 |
| parakeet ff linear2 | thrush5 row | 1024x4096 | 65.4 | 277 | 1.6e-7 |
| parakeet q_proj | thrush5 row | 1024x1024 | 61.7 | 258 | 1.8e-7 |

The thrush5 kernel is correct but slow. Rows aren't aligned, it needs byte loads and a shared LUT
(byte → 5×2-bit codes), and scales are f32 (they count in the GB/s figure). The LUT already made it
1.6-1.8x faster than per-element `/3`. **Recommendation:** at load time, repack thrush/kittenml
safetensors into PTQ1_0-style blocks. It's lossless: parakeet scales are already F16, and kitten's
BF16 scales would need a range check. Then reuse the 185-223 GB/s PTQ1 kernel. For Kitten, just load
the TQ2_1 GGUF.

## FWHT kernel (`tern_fwht.comp`)
There is one 256-lane workgroup per 1024-block (4 elements per lane, 10 shared-memory butterfly stages).
It supports the GDN `ssm_out` permutation (flags bit0: signs before H) and the inverse with signs after
H (bit1, for the `token_embd` lookup), and runs batched with workgroup = token·(K/1024) + block.

| K | time | max_abs vs CPU ref | inverse round trip |
|---|---|---|---|
| 5120 | 2.3-5.3 us | 0 | 2.4e-7 |
| 6144 + GDN perm | 2.8 us | 0 | (perm is not self-inverse; n/a) |
| 17408 | 2.7-6.6 us | 0 | 2.7e-7 |

This is a few µs per call against 40-110 µs GEMVs. Per token, Bonsai needs ~4 distinct transforms per
layer, thanks to the memo (qkv+gate, ssm_out or attn_output, gate+up, down), so about 64×4×3 µs ≈ 0.8 ms.
That is worth fusing later: put the FWHT into the producer (RMSNorm) or into the GEMV prologue.

## Batched GEMM / prompt processing (stretch): `tern_quantx.comp` + `tern_gemm.comp`
Activations are quantized per token per 128-group to symmetric int8. The 64×128 ternary W tile is
decoded to int8 in shared memory. The math uses `KHR_cooperative_matrix` s8×s8→s32 16×16×16, with A
loaded straight from the int8 activation buffer. Each 128-group's s32 accumulators are spilled to shared
memory and scaled by `xs[m,g]·ws[n,g]` in fp32. The tile is 64×64 per WG (4×wave32). The
`VK_KHR_shader_integer_dot_product` route was not used: `vkrt::Ctx::new` doesn't enable
`shaderIntegerDotProduct`, and I didn't edit vkrt. Cooperative matrix is already enabled there.

M = 512 tokens (`tern_gemm 512`, logs/tern_gemm_m512.txt):

| fmt | tensor | N x K | gemm µs | TOPS (2MNK/t) | quantx µs | max_rel vs int8-x ref | max_rel vs fp32-x ref |
|---|---|---|---|---|---|---|---|
| PQ2_0 | ffn_gate | 17408x5120 | 5024 | 18.2 | 71 | 1.9e-7 | 3.1e-3 |
| PQ2_0 | ffn_down | 5120x17408 | 4794 | 19.0 | 276 | 2.4e-7 | 8.0e-3 |
| PQ2_0 | attn_qkv | 10240x5120 | 2818 | 19.1 | 71 | 1.9e-7 | 4.3e-3 |
| PQ2_0 | attn_output | 5120x6144 | 1660 | 19.4 | 91 | 2.0e-7 | 4.8e-3 |
| PTQ1_0 | ffn_gate | 17408x5120 | 5492 | 16.6 | 72 | 1.9e-7 | 3.1e-3 |
| PTQ1_0 | ffn_down | 5120x17408 | 5481 | 16.7 | 298 | 2.4e-7 | 8.0e-3 |
| PTQ1_0 | attn_qkv | 10240x5120 | 3218 | 16.7 | 73 | 1.9e-7 | 4.3e-3 |
| PTQ1_0 | attn_output | 5120x6144 | 1856 | 17.4 | 76 | 2.0e-7 | 4.8e-3 |

The kernel matches the int8-activation reference exactly (to fp32 rounding). The int8 activation
quantization error against fp32 x is 3-8e-3 relative to max |y|. At ~18 TOPS, 2×24.35 G MAC-ops per
token works out to roughly 370 tok/s as a matmul-only prefill bound. That is an estimate, not a
measured prefill rate.

Next steps:
- Bigger N tiles, so the W decode is amortized over more coop ops.
- Packed u32 writes into shared memory, instead of byte stores.
- Double buffering.
- A quantx kernel that does a whole token per workgroup. Right now there are M·K/128 tiny workgroups;
  K=17408 takes 276 µs.

## What Bonsai 2's full forward pass still needs (arch `qwen35`, 64 layers)
Hyperparameters from the GGUF: d=5120, ffn=17408, vocab=248320, tied output (`output.weight` is present
here). `full_attention_interval=4`, so there are 16 gated full-attention layers (blk 3, 7, …) and 48
Gated DeltaNet layers.

1. **Gated DeltaNet layers (48)**
   - `attn_qkv` [5120→10240] produces q, k (16 groups × 128 each) and v (48 heads × 128).
   - `attn_gate` [5120→6144] is z.
   - `ssm_conv1d` is a causal depthwise conv, kernel 4, over 10240 channels, with SiLU. It needs a
     conv-state cache.
   - `ssm_alpha` and `ssm_beta` (BF16 [5120→48]) feed the decay `g = -exp(ssm_a)·softplus(alpha + dt_bias)`
     and `β = sigmoid(beta)`.
   - The gated delta rule: per-head recurrent state 128×128 (48 heads), with L2-normalized q/k. Prefill
     needs a chunked form.
   - `ssm_norm` is a gated RMSNorm (128) times SiLU(z).
   - `ssm_out` [6144→5120] needs the GDN-permuted FWHT, which is implemented above.
   - New kernels: conv1d+state, the delta-rule recurrence (decode) and chunked scan (prefill), and the
     gated norm. None of these exist in strix-rt yet. The `shortconv.comp` that another worker is
     writing for LFM2 is related but not the same op.
2. **Gated attention layers (16)**
   - `attn_q` [5120→12288] = 24 heads × 256 × 2 (query + output gate).
   - `attn_k`/`attn_v` [5120→1024] = 4 KV heads × 256 (GQA 6:1).
   - `attn_q_norm`/`attn_k_norm` RMSNorm over 256.
   - Partial M-RoPE: 64 of 256 dims, sections [11, 11, 10], freq_base 1e7.
   - Output gated by `sigmoid(gate)` before `attn_output` [6144→5120].
   - The existing attn kernels assume head dims of 64/128 (`attn64.comp`, `attn.comp`), so head_dim 256
     plus the gate needs a variant.
3. **Hadamard plumbing:** one FWHT per distinct (activation, width), signs read from the GGUF, the
   inverse rotation after the `token_embd` lookup, and `output.weight` (folded) taking a forward transform
   of the final normed hidden state.
4. **MTP:** this GGUF has no `nextn`/MTP tensors and no `qwen35.nextn_predict_layers` key. The fork's
   `qwen35.cpp` supports a NextN block (a dense decoder block appended after the main stack, used for
   speculative drafting), but nothing ships here. An MTP draft would need its weights from elsewhere.
5. **Vision** (mmproj, 27 blocks) is out of scope.
6. Others: norms are F32 [5120] (`attn_norm`, `post_attention_norm`, `output_norm`). Tokenizer: gpt2 BPE,
   `pre=qwen35`, 248320 tokens.
