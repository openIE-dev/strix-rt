# NPU (AMD XDNA2, RyzenAI-npu5, 8 columns) from Rust: `crates/npu`

Measured 2026-10-14 on the Framework Desktop (Strix Halo, kernel 7.0.0-38, amdxdna in-tree, XRT 2.21.75, RADV Mesa 26.0.8).
No numbers here are estimates unless they are marked as estimates.

## What is in the crate
- `csrc/xrt_shim.cpp`: an `extern "C"` wrapper over the XRT C++ API (`libxrt_coreutil`), compiled by `build.rs` with the system g++ (no `cc` crate).
  Calls: open (xclbin, hw_context, kernel, instruction BO), BO alloc / userptr / dma-buf import / dma-buf export / map / sync, run.
  Every call catches C++ exceptions and returns the message through `npu_last_error`.
- `src/lib.rs`
  - `Npu`, `Bo`: safe wrappers.
  - `Gemm`: an int8 GEMM on the whole-array dc2 design (mlir-aie `wa_dc`, builds under `~/npu/kernel/builds/`). It packs row-major A and B into the pre-tiled layouts the design reads contiguously: A as `[M/m][K/k][m][k]` and B as `[K/k][N/n][k][n]`. C comes back row-major.
  - `quantize_rows` (per token), `quantize_cols` (per output channel), and `linear_f32` (a batched prompt-processing linear layer: f32 X, int8 W, int32 accumulation on the NPU, dequantization to f32).
- Binaries (build with `CARGO_TARGET_DIR=~/npu/target cargo build --release -p npu`; a separate target dir keeps the decode worker's target untouched):
  - `npu_gemm <build> M K N m k n out_bytes [iters]`: benchmark plus sampled verification.
  - `npu_share <build> M K N m k n`: the zero-copy tests (T1–T6).
  - `npu_linear <i32 build> M K N m k n [iters]`: the quantized linear layer.

## (1) Rust → XRT → NPU
- **Build:** dc2 144x72x144 int8→int8, 8064³ (`wa_dc_dc2_r1_st800_l24_at_bt_d221_8c_8064x8064x8064_144x72x144`).
- **Run-only time** (one run plus wait per iteration, 100 iterations): **avg 25,231 µs = 41.57 TOPS** (best iteration 42.12, worst 41.17).
  The C++ harness baseline is 40.90 TOPS, so the Rust path adds no overhead.
- **Other timings:** open plus allocation 154 ms. Packing plus sync of A and B (65 MB each) 41 ms.
- **Verification:** 20,000 random C entries match the exact int32 dot product truncated to int8 (wraparound). The int8-output design returns the low byte; it does not saturate. That makes the int8-output build correct but not useful for inference. Use the int32 build (3).

## (2) Zero-copy NPU ↔ GPU (Vulkan) ↔ page cache (`npu_share`)
All tests run on 62 MB buffers (8064² int8) and verify a full GEMM through the shared pages, with no host copy between devices.

| test | path | result |
|---|---|---|
| T1 | XRT BO `export_buffer()` → dma-buf → `VK_EXT_external_memory_dma_buf` import; the GPU reads the NPU's C | **PASS**. Export 0.021 ms, import 0.066 ms. The imported memory type is 5 (HOST_VISIBLE, COHERENT, CACHED). The GPU copies 62 MB in 0.53 ms (122.6 GB/s); the bytes are identical; a fresh NPU result is visible to the GPU with no host sync |
| T2 | same import; the GPU writes the NPU's A, then the NPU computes | **PASS**. GPU write 0.60 ms (108 GB/s); the NPU result is correct with no host sync |
| T3 | Vulkan allocation exported as a dma-buf (type 2, HOST_VISIBLE and COHERENT, GTT) → XRT `bo(device, fd)`; the NPU writes C into GPU memory | **PASS**. NPU import 1.83 ms. GEMM 25,652 µs = **40.88 TOPS**, the same as native BOs (40.89 TOPS in the same run) |
| T4 | strixpc dma-buf (pinned anonymous pages) → XRT import | **blocked**: the import itself goes through, but XRT always mmaps an imported BO, and `strixpc`'s `.mmap` returns `-EINVAL` (`mmap(... ) failed (err=-22)`) |
| T5 | XRT userptr BO over an anonymous mmap | **PASS**. Create 2.8 ms; GEMM 40.70–40.87 TOPS |
| T6 | userptr BO over a **file-backed MAP_SHARED (page-cache) mapping** | PROT_READ\|PROT_WRITE: **PASS**. Create 4.9 ms; GEMM 40.76 TOPS. PROT_READ only: **fails** (`CREATE_BO err=-14 EFAULT`; amdxdna pins user pages for write) |

- **Sync cost on native host-only BOs:** C `from_device` 0.55 ms, A `to_device` 0.51 ms (62 MB each).
- **Overhead summary:** sharing costs nothing at run time; every shared path ran within about 0.5% of native TOPS. The setup costs are export 0.02 ms, Vulkan import 0.07 ms, XRT import of a Vulkan dma-buf 1.8 ms, and userptr 2.8–4.9 ms for 62 MB.
- **For strixpc → NPU there are two routes:**
  - Implement `spc_mmap` in `kmod/strixpc`, for example remapping the pinned pages with `vm_insert_pages`. That is owned by the decode worker, so I did not touch it.
  - Skip strixpc for the NPU: a userptr BO on a writable MAP_SHARED mapping of the weights file already works (T6). It needs the file opened read-write; it does not need to be written.

## (3) Batched prompt-processing int8 GEMM, int32 output (`npu_linear`)
Design: dc2 with int32 output. The 144x72x144 tile cannot fit: C alone would be 83 KB at int32 against 64 KB of L1. Measured int32-output builds (C++ harness, PASS):

| problem | tile | avg TOPS (C++ harness) |
|---|---|---|
| 2048x4096x4096 | 64x128x64 | 16.66 |
| 2048x4096x4096 | 128x64x64 | 18.78 |
| 4096x4096x14336 | 64x128x64 | 18.59 |
| 4096x4096x14336 | 128x64x64 | 20.38 |
| 8064³ | 96x64x96, 112x72x72 | build fail (B tiling constraint; n must be a multiple of 16) |

Through Rust (`linear_f32`: per-token quantization of X, W pre-quantized per channel and uploaded once):

| problem (tokens x K x N) | NPU run only | end-to-end linear_f32 | int32 exact check | f32 rel. RMS error |
|---|---|---|---|---|
| 4096 x 4096 x 14336 | 24.4–25.8 ms = **18.7–19.7 TOPS** | 63.1 ms (7.6 TOPS-equiv.) | 0/3000 mismatches, PASS | 0.0070 |
| 2048 x 4096 x 4096 | 3.50–3.77 ms = **18.2–19.7 TOPS** | 11.9 ms (5.8 TOPS-equiv.) | 0/3000 mismatches, PASS | 0.0069 |

The two ranges come from two runs of the same binary.

- **Against the 40.9 TOPS int8→int8 baseline:** the useful int32-output op runs at about 19–20 TOPS. The loss comes from the smaller tile that int32 C forces in L1 (m·n/(m+n) drops from 72 to 43) and from 4x the C write traffic.
- **End-to-end overhead:** the host side dominates, with threaded quantization straight into the tiled BO and threaded dequantization of a 235 MB f32 output.
- **Next steps:**
  - Fuse dequantization into a consumer (the GPU via T1-style sharing, or keep int32).
  - An int16-output or shift-requantized int8 variant would get the large tile back.
  - Overlap host work with the next NPU run.

## (4) Ternary (1.58-bit) GEMM feasibility — assessment, not built
- **Formats:**
  - PrismML Ternary Bonsai (1.7B/4B/8B): {-1, 0, +1}, g128 FP16 scales, GGUF Q2_0.
  - Ternary Bonsai 2 27B: g128 FP16 scales plus a blockwise Hadamard rotation, PQ2_0 (2.16 bpw) or PTQ1_0 (1.76 bpw).
  Source: prismml.com and the Hugging Face model cards.
- **The AIE has a better path than unpacking to int8.** aie2p has a native **int8 x int4** `mmul<4,16,16>` (aie_api `detail/aie2p/mmul_8_4.hpp`): 1024 MACs per instruction against 512 for int8 x int8. Ternary values are exact in int4.
  - Store the weights as int4 in DDR (2x the bytes of 2-bit, half of int8): no in-core unpacking, and in principle 2x the compute ceiling of int8 x int8 (an estimate, not measured).
  - Storing 2-bit in DDR and unpacking to int4 in-core with vector shift and mask halves the weight bytes again. But it spends VPU slots in a loop that is already compute-bound for prompt processing (kernel-only 43.8 TOPS ≈ 75% of the int8 peak), so it only pays off where DMA-bound (decode/GEMV).
  - The DMA engines cannot bit-unpack, so the memtiles cannot do it.
- **Group scales (g128):** the int32 partial sum has to be scaled every 128 K. Two options:
  - Make the K tile 128 and keep an f32 output tile in L1 (the same L1-pressure problem as int32 C).
  - Accumulate per K group into int32 and do a fused multiply-add into f32 per group.
  Either is feasible; the overhead is estimated, not measured.
- **Bonsai 2's Hadamard rotation** acts on the activations: apply it in the per-token quantization step on the host or GPU before the NPU GEMM.
- **Decode** (GEMV, batch 1) is DMA-bound. The NPU's measured DDR copy ceiling is about 73–76 GB/s combined (about 37 GB/s each way), well below the GPU path, so ternary decode belongs on the GPU. Prompt processing on the NPU with int8 x int4 is the promising split.

## Round 6 (2026-10-14): on-device pipeline, epilogue, int8 x int4

### (2) GPU quant/Hadamard -> NPU -> GPU dequant, all through shared dma-bufs (`npu_e2e`)
The shaders are `shaders/quant_tiled.comp` (per-token amax, optional blockwise-1024 FWHT, int8 A written pre-tiled into the NPU BO) and `shaders/dequant.comp` (`y = C * sx[r] * sw[c]`, reading the NPU's int32 C BO). There is no host f32 round-trip. Build: i32-out dc2, 128x64x64.

| shape | Hadamard | e2e wall | GPU quant | NPU GEMM | GPU dequant | verify |
|---|---|---|---|---|---|---|
| 4096x4096x14336 | off | 25.24 ms (19.06 TOPS-eq) | 0.718 ms | 21.48 ms (22.39 TOPS) | 2.217 ms | PASS, rel RMS vs f32 0.0072 |
| 4096x4096x14336 | on | 27.10–27.15 ms (17.7 TOPS-eq) | 1.79 ms | 21.9 ms (21.9 TOPS) | 2.24 ms | PASS, rel RMS 0.0097 |
| 512x5120x17408 | on | 5.50 ms (16.59 TOPS-eq) | 0.738 ms | 4.10 ms (22.25 TOPS) | 0.369 ms | PASS, rel RMS 0.0100 |

- Verify checks Y exactly against an integer reference computed on the GPU-quantized A. GPU and host int8 A differ only on rounding ties: 10–14 of 16.7M values, none by more than 1.
- The round-5 host path took 63.1 ms, so this is about 2.3x faster.
- One Hadamard run took 130.9 ms; I'm treating it as a GPU-contention outlier. Two reruns gave 27.1 ms.
- GPU-only comparison: there's no GPU number for 4096x4096x14336 int8. For 512x5120x17408, the other worker's ternary coop-matrix GEMM takes 5024 µs (18.2 TOPS), compared with NPU e2e at 5.50 ms.

### (1) Requantize epilogue so 144x72x144 fits: not feasible in this dataflow
- The K loop accumulates partials in the L1 C tile across K/k calls. A 144x144 int32 C tile is 83 KB, more than the 64 KB L1.
- Requantizing per call would need a separate int32 accumulator plus a narrow output buffer (5–6·m·n bytes). That doesn't fit either.
- Accumulating in int16/bf16 between calls loses exactness or overflows.
- The best int32-partial tile is still 128x64x64, which gives about 22 TOPS NPU-only. Getting back to about 40 TOPS needs K fully inside one call (e.g. k=K with an L2 A stream) or a different dataflow. Not attempted.

### (3) int8 x int4 GEMM with g128 scales (`mm_q4.cc`, `wa_q4`, `npu_q4`, `GemmQ4`)
- Uses `aie::mmul<4,16,16,int8,int4>`. B is packed host-side as 16x16 nibble blocks with a per-tile scale trailer.
- Integer epilogue per group: `C = round_even(((C<<15) + acc*q) >> 15)`, where q is an int16 fixed-point group scale and the per-column float S·32768/32767 is applied by the consumer.
- Kernel stack must be 0x1000; at 0x800 it corrupts results.

| tile mxkxn | epilogue | 512x5120x17408 | check |
|---|---|---|---|
| 64x128x64 | float (accfloat) | 7.17 TOPS | float verify OK |
| 64x128x64 | int16-q, conv_even | 9.73 TOPS (9380 µs) | integer exact PASS 0/2000; float error ≤1.1e-3 of typical abs(y) (C kept as integers per group) |
| 64x64x128 | int16-q, applied per k-half | 5.34 TOPS | exact PASS |

The GPU coop-matrix ternary GEMM for the same shape runs at 18.2 TOPS, so the NPU int4 path is about 2x slower today. The int8 dc2 path on the NPU (4.10 ms for this shape) beats both. Bottleneck: the int4 mmul does only 8 MACs per group per 2x2 block, and the unpack-load and epilogue dominate. Larger m and n don't fit in L1 with int32 C.

## Round 7 (2026-10-14): NPU + GPU co-execution, full-K dataflow model

### (1) Co-execution: N split between NPU and GPU (`npu_coexec`)
- NPU: dc2 int8 x int8 -> int32 GEMM, 128x64x64 tile, one xclbin per N split. It runs on its own thread that owns the XRT objects.
- GPU: crates/ternary's coop-matrix GEMM (`ternary::shaders::GEMM_PQ2`, int8 x PQ2_0 ternary with per-128-group scales -> f32), dispatched through vkrt. crates/npu now depends on `ternary` read-only; nothing in crates/ternary was changed.
- Each iteration starts both engines and waits for both. Reported numbers are medians; GPU spikes from other processes reach 30–70 ms. Inputs are synthetic; correctness is covered per engine by npu_e2e and tern_gemm.
- Timing covers the GEMM only. It excludes GPU quant (0.7 ms at 512 rows) and dequant of the NPU half (about 0.2 ms at 512 rows, 1.2 ms at 4096 rows). Raw logs: `~/npu/kernel/r7co.log`, `r7co2.log`.

| shape | split NPU/GPU cols | together wall (median) | combined TOPS | NPU alone, full N | GPU alone, full N | speedup vs best single |
|---|---|---|---|---|---|---|
| 512x5120x17408 | 9216 / 8192 | **2.301–2.305 ms** | **39.6–39.7** | 4.69 ms (19.5 TOPS) | 5.16 ms (17.7 TOPS) | 2.0x |
| 512x5120x17408 | 9728 / 7680 | 2.361–2.368 ms | 38.5–38.7 | | | |
| 512x5120x17408 | 8704 / 8704 | 2.48–2.52 ms | 36.2–36.8 | | | |
| 512x5120x17408 | 10240 / 7168 | 2.47 ms | 36.9 | | | |
| 4096x4096x14336 | 7680 / 6656 | **12.20 ms** | **39.4** | 22.10 ms (21.8 TOPS) | 26.02 ms (18.5 TOPS) | 1.81x |
| 4096x4096x14336 | 8192 / 6144 | 12.50–12.62 ms | 38.1–38.5 | | | |
| 4096x4096x14336 | 7168 / 7168 | 13.0–13.5 ms | 35.6–37.0 | | | |
| 4096x4096x14336 | 6656 / 7680 | 14.32 ms | 33.6 | | | |

DDR contention: each engine's own time while overlapped stays within about ±3% of its time alone at every split, so contention costs nothing here. Both GEMMs are compute-bound. The NPU part reads about 20 GB/s and the GPU's ternary weights are about 5 GB/s at 2.3 ms, far below DDR bandwidth. Best split is about 53–54% of columns on the NPU, which matches the throughput ratio 21.5 : 18.

Caveat: the GPU half uses ternary PQ2_0 weights with per-128-group scales, while the NPU half uses int8 weights with per-column scales. For a real Bonsai layer, the NPU half has two options. It could fold group scales into int8 weights (t * s_g / S_col quantized to int8, about 0.4% weight error), which is untested for accuracy. Or it could use the q4/g128 path, which runs at 9.7 TOPS and would cut the NPU share to about 35%.

### (2) Full-K-in-one-call with requantize at the end: memory model
L1 per core is 64 KB, including the stack (0x800 to 0x1000 bytes). The per-core budget is acc int32 (4mn) + output (c·mn int8) + A (2mk) + B (2kn).
- **144x144 tile: impossible.** The int32 accumulator alone is 82.9 KB > 64 KB, whatever the output width. aie2p accumulator registers (~1 KB) hold only one 8x16 sub-block, and keeping the accumulator in registers across all of K would need the full-K A/B panels in L1 (128x4096 = 512 KB). So narrow output does not bring the 144-class tile back.
- The largest feasible tiles with an int32 accumulator: 96x32x96 with requant output, single-buffered C (36.9 + 9.2 + 12.3 = 58.4 KB + stack), and 128x32x64 (60 KB). 104 or 112 squares don't fit once there's an output buffer.
- I measured the int32-out variants of these smaller-k, larger-mn tiles (all PASS, 4096-class shapes padded to tile multiples). All are slower than 128x64x64 (20.4 TOPS in the C++ harness):
  - 96x32x96: 13.72 TOPS
  - 128x32x80: 14.38 TOPS
  - 128x32x64: 12.81 TOPS
- Per-call cost is about 1.3–1.4 µs at k=32 versus 1.65 µs at k=64. Fixed per-call overhead (C tile load/store, locks, kernel prologue) dominates, so shrinking k to make room for a bigger mn loses.
- Requant-at-end would only shrink the C drain and output DDR traffic, about 12% of tile time at C_DEPTH=1, onto tiles that are already slower. I did not implement it.
- Conclusion: about 22 TOPS is the ceiling for exact int32 accumulation in this IRON whole_array dataflow. The 40-TOPS-class result for exact int8 prefill comes from co-execution, measured above at 39.4–39.7 TOPS combined.

## Round 8 (2026-10-14): NPU+GPU split linear on real Bonsai 2 PQ2_0 weights

### Interface for crates/prefill: `npu::split` (`crates/npu/src/split.rs`)
```rust
let sl = npu::split::SplitLinear::new(&ctx, m, k, n, nn, &npu::split::build_dir(m, k, nn), Some(signs_k))?; // one per shape
let w  = sl.load(&ctx, pq2_bytes)?;            // per layer: NPU int8 B (folded) + scales, GPU PQ2_0 rows nn..n
let t  = sl.forward(&ctx, &w, &x, ldx, &y, ldy, y_col)?;   // x f32 [m][ldx] -> y f32 [m][ldy] cols y_col..y_col+n
```
- Constraints:
  - m % 128 == 0, k % 1024 == 0, nn % 512 == 0, n % 64 == 0.
  - An NPU xclbin is needed per (m, k, nn) (`~/npu/kernel/bench.sh` with the dc2 int32 env, see run_r8.sh).
  - `nn = 0` gives the GPU-only path through the same code.
- One NPU hardware context per shape. Weights are per-layer BOs, so a model needs about 3 contexts (gate/up share), not one per layer.
- `forward` blocks: three queue submits, with the NPU running on a scoped thread while the GPU GEMM runs. It returns `SplitTimes` (pre, gemm, npu, gpu, post, total).
- Input is f32 today. prefill's pf_gemm takes f16 `a`; an f16 input variant of `rot_quant` is a small change.
- Pipeline:
  1. `rot_quant` phase 0: one WG per (token, 1024-block), `xr = H_1024(s ⊙ x)` with the GGUF signs, plus per-block amax.
  2. Phase 1: per-token int8 A written into the NPU's dma-buf.
  3. crates/ternary `tern_quantx` on xr, for the GPU half.
  4. Concurrently: the NPU dc2 GEMM and `tern_gemm_split`. The latter is a copy of crates/ternary's coop-matrix PQ2_0 GEMM with ldy/y_col added; the original is untouched.
  5. `dequant_split` writes the NPU's int32 C into the same output.
- Weight format, fold: `int8 = round(t · round(d_g / S_n))` with `S_n = max_g |d_g| / 127` per output column.

### Accuracy
- Setup: layer 10. Activations are real token embeddings (inverse-rotated per the GGUF), RMS-normed with the layer's post_attention_norm. The down input is silu(gate)·up from the GPU-only outputs.
- Metric: relative RMS vs an f32 reference (f64 Hadamard, exact dequantized ternary weights), over 24 tokens × 160 columns.

| op | fused, NPU cols | fused, GPU cols | GPU-only path | exact-g128 NPU emulation | fused vs GPU-only |
|---|---|---|---|---|---|
| ffn_gate | 0.0046 | 0.0032 | 0.0032 | 0.0038 | 0.0039 |
| ffn_up | 0.0044 | 0.0032 | 0.0032 | 0.0037 | 0.0039 |
| ffn_down | 0.0090 | 0.0057 | 0.0059 | 0.0087 | 0.0075 |

- "Exact-g128 emulation" is the exact option: ternary int8 {-1,0,1} with the group scale applied per K-group in fp32. It differs from fold only in weight handling: 0.0038 vs 0.0046 on gate/up, 0.0087 vs 0.0090 on down.
- Most of the NPU-vs-GPU gap is per-token activation quant (NPU) vs per-128-group activation quant (GPU), not the weight fold.
- The exact per-group epilogue needs k=128 calls with an fp32 epilogue (the q4 experience: 7–10 TOPS), so **fold** is the chosen format. The same values hold at M=2048.

### End-to-end, real weights
- e2e = rotate+quant + split GEMM + dequant, median of 20, measured while the GPU was quiet (the runner waits for no other GPU jobs).
- GPU-only = same API with nn=0: rotate + tern_quantx + full-N tern GEMM.

| M | op | NxK | split NPU/GPU | fused e2e | TOPS | pre | split GEMM (NPU / GPU) | dequant | GPU-only e2e | TOPS | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | ffn_gate | 17408x5120 | 9216/8192 | 2.900 ms | 31.48 | 0.219 | 2.418 (2.310 / 2.407) | 0.263 | 5.492 ms | 16.62 | 1.89x |
| 512 | ffn_up | 17408x5120 | 9216/8192 | 2.903 ms | 31.44 | 0.218 | 2.418 (2.226 / 2.406) | 0.266 | 5.542 ms | 16.47 | 1.91x |
| 512 | ffn_down | 5120x17408 | 2560/2560 | 3.422 ms | 26.67 | 0.867 | 2.465 (2.070 / 2.454) | 0.087 | 5.803 ms | 15.73 | 1.70x |
| 2048 | ffn_gate | 17408x5120 | 9728/7680 | 10.998 ms | 33.19 | 1.052 | 9.062 (9.029 / 8.689) | 0.888 | 21.701 ms | 16.82 | 1.97x |
| 2048 | ffn_up | 17408x5120 | 9728/7680 | 11.000 ms | 33.19 | 1.045 | 9.068 (9.035 / 8.703) | 0.891 | 21.557 ms | 16.93 | 1.96x |
| 2048 | ffn_down | 5120x17408 | 3072/2048 | 13.043 ms | 27.99 | 3.178 | 9.547 (9.516 / 7.781) | 0.301 | 23.038 ms | 15.85 | 1.77x |
| 2048 | ffn_down | 5120x17408 | 2560/2560 | 13.394 ms | 27.26 | 3.372 | 9.738 (8.874 / 9.717) | 0.267 | (one contended run at 192.6 ms; earlier run 22.489 ms) | | |
| 2048 | ffn_gate (9216 split) | | 9216/8192 | 11.572 ms | 31.55 | | | | 21.058 ms | 17.34 | 1.82x |

- Per FFN block (gate+up+down): 512 tokens 9.23 ms fused vs 16.84 ms GPU-only (1.82x); 2048 tokens 35.04 ms vs 66.30 ms (1.89x).
- The remaining overhead on ffn_down is the prologue (0.87 ms at 512, 3.2 ms at 2048). It's memory-bound: 17408-wide rows read and written several times (x, xr, two quant passes). Fusing tern_quantx into phase 1 and keeping xr in f16 would roughly halve it; not done.
- Logs: `~/npu/kernel/r8run.log` (final runs), `r8_512a.log`/`r8_512b.log` (earlier, before the output buffer moved to device-local memory; readback memory slowed GPU writes about 3x).

## Round 10: NPU_SPLIT=1 in crates/prefill (Bonsai 2 PQ2_0)

- Hook (`crates/prefill/src/npu_split.rs`, flag off by default): attn_qkv, attn_gate, attn_q, attn_output, ffn_gate/up/down split by N.
  No extra rotate/quant: `npu_prep.comp` builds the NPU A (per-token int8, tiled) from prefill's own rotated f32 `xr`;
  GPU rows [nn, N) go through `Pf::tgemm(row0=nn)` on prefill's xq/xs; dequant of the NPU columns is recorded at the start of the next
  command buffer. Stream per split: [.. prep] (fence) | [GPU half] | NPU run on host while the GPU half runs | [dequant ..].
- Correctness (L=512): logits cosine 0.999988 vs token-by-token, top10 10/10, greedy argmax equal.
- Context switches: each SplitLinear had its own xclbin/hw_context; alternating shapes cost ~2.2 ms per NPU run
  (npu_alt m=512: 3584 cols 1.17 -> 3.33 ms, 9216 cols 2.48 -> 4.63 ms). The shim now shares one hw_context per xclbin and all
  shapes with the same M,K use one xclbin (dc2 core code depends on K, not N; K=6144 insts on a K=5120 xclbin fail, state 8).
- Whole model, mixed-load sessions (perf level auto, not the final numbers):
  | config | pp512 (PF_M 512) | pp2048 (PF_M 512) | pp2048 (PF_M 2048) |
  |---|---|---|---|
  | split off | 270.0 | 313.5 | 265.3 |
  | split, per-shape contexts | 291–294 | 290.7 | 380.7 |
  | split, shared per-K contexts | 350.6 | 337.8 | 415.5 |

## Round 11 (2026-10-14): whole-model NPU_SPLIT prefill, final numbers + split tuning

Bonsai 2 27B PQ2_0, `pf_q38` (random prompt, seed 12345), perf level high, `logs/gpu_idle_run.sh`, median of 14 reps
(2 sessions x 7), PF_M = L (one chunk). Logs: `logs/r11/`.

| config | pp512 median (best) | pp2048 median (best) |
|---|---|---|
| GPU only (NPU_SPLIT off) | 1556 ms = 329.0 tok/s (329.5) | 6338 ms = 323.1 tok/s (323.9) |
| NPU_SPLIT=1, round-10 ratios | ~1433 ms = 357 tok/s (368) | ~4970 ms = 412 tok/s (425) |
| NPU_SPLIT=1, round-11 ratios (default now) | 1431 ms = 357.8 tok/s (363) | 4786 ms = 427.9 tok/s (430.7) |
| speedup vs GPU only | 1.09x | 1.32x |

- Correctness (split on, tuned): greedy 16/16 vs token-by-token at L=512 and L=2048, logits cosine 0.999990 / 0.999987
  (split off: 0.999992 / 0.999990), top10 10/10.
- Split sweep (NPU columns of ffn_gate/up 5120x17408 / ffn_down 17408x5120; only existing xclbin builds), medians:
  M=512: 8704/2560 1412 ms, 9728/2560 1421, 9216/2560 (old) ~1430-1479, 8704/3072 1511, 9728/3072 1466.
  M=2048: 9216/2560 4749 ms, 9728/2560 4836, 9216/3072 4861, 9728/3072 (old) 4972.
  New defaults in `nn_for`: gate/up 8704 (M<2048) / 9216 (M>=2048), down 2560 at both. `NPU_NN=KxN:nn,...` overrides
  per shape (nn=0 keeps it on the GPU); a matching `{M}x{K}x{nn}` build must exist.
- Dropping attn_qkv/attn_q from the split (NPU_SPLIT_OPS) is slower (pp512 ~1470 ms, pp2048 ~5190 ms); FFN-only is
  slower still (1524 / 5402 ms). Keep all seven projections.
- Bottleneck: the NPU path is serial. Per prefill, NPU run time + host wait for the GPU prep segment ~= wall time
  (M=512: 960 + 443 ms of 1431; M=2048: 2713 + 1962 ms of 4786). The GPU half of each split is fully hidden; the NPU
  sits idle while the GPU runs the non-split work (GDN, attention, ssm_out, small projections, prep). Next: overlap
  NPU split k with the GPU non-split work that follows (dequant behind a fence instead of stream order), then
  re-tune ratios. Also suspect: at M=512 the 5120x10240 (nn 5632, 3.85 ms) and 5120x12288 (nn 6656, 4.1 ms) NPU runs
  are ~2.4x slower per column than 5120x17408 (nn 8704, 2.5 ms); at M=2048 they are on par. Check the shared K=5120
  xclbin / instruction streams for those two shapes.
- Runner fix: `logs/gpu_idle_run.sh` now sources the argv0-anchored `logs/pf_busy.sh`; the unanchored pattern
  matched the waiting runner itself once its args named a binary (self-deadlock).

## Round 12 (2026-10-14): NPU_SPLIT vs the f16 GPU baseline (PF_TF16=1, ca69fe1)

Under NPU_SPLIT the GPU half and all unsplit ternary GEMMs now also use the f16 coop-matrix path (8569980; before,
NPU_SPLIT forced the int8 `pf_tgemm` everywhere). Bonsai 2 27B PQ2_0, `pf_q38`, perf level high, idle runner, PF_M = L,
7 reps per session, medians. Logs: `logs/r12/`.

| config | pp512 | pp2048 |
|---|---|---|
| GPU only, PF_TF16=1 (new baseline) | 975 ms = 525 tok/s | ~4560 ms = ~449 tok/s |
| split, round-11 shares, GPU half f16 | 1376 ms = 372 | ~4740 ms = 432 |
| split, round-11 shares, GPU half int8 (PF_TF16=0) | 1416 ms = 362 | ~4740 ms = 432 |
| split all 7 ops, gate/up 6144 / down 1536 (best all-ops) | 1248 ms = 410 | (6144/2048) 4368 ms = 469 |
| split FFN only, gate/up 4096 / down 1536 (best at 512) | 1142 ms = 448 | |
| split FFN only, gate/up 6144 / down 2048 | 1238 ms | 4161 ms = 492 tok/s |

- Result: against the f16 baseline the NPU split now **loses at pp512** in every configuration (best 1142 ms vs 975 ms)
  and **wins about 1.10x at pp2048**, and only when the NPU takes the FFN alone (gate/up 6144 of 17408 columns,
  down 2048 of 5120). Defaults in `nn_for` now: with PF_TF16=1, no split for M < 2048, FFN-only 6144/2048 for
  M >= 2048 (NPU_NN still overrides). Correctness: greedy 16/16 at L=512 and L=2048 with the f16 GPU half and round-11 shares (cosine 0.99999); the new FFN-only default is not yet re-checked (GPU device-lost at 22:27 UTC, logs/r12/chk_default_2048.log).
- Bottleneck 1, serial NPU path: NPU time + host wait for the prep segment ~= wall time (e.g. FFN-only 2048:
  1639 + 2430 ms of 4150 per prefill). Each NPU run waits for all GPU work queued before it, and the GPU waits for the
  NPU result before the next layer. Only the GPU half of each split overlaps. The f16 GPU is ~1.6x faster than the int8
  path at 512, so this fixed serialization now outweighs the extra throughput.
- Bottleneck 2, the NPU slows down while the GPU runs: at M=512, `ffn_gate` with nn=4096 takes 2.43 ms in the split but
  ~1.3 ms standalone (r12build perf, 16-20 TOPS), and nn=4096 vs 8704 take the same 2.4-2.5 ms. That points to shared
  DDR/fabric or power-budget contention with the concurrent f16 GEMM, not NPU compute. At M=2048 the slowdown is ~30%
  (6144 cols: 8.0 ms in split vs ~6.2 ms standalone).
- To win at pp512: pipeline across layers (NPU FFN of layer l overlapping GPU attention/GDN of layer l+1 is not
  possible due to the residual dependency, so the realistic option is splitting by M instead: the NPU does a token
  block of the FFN while the GPU does the rest), and investigate the contention slowdown first.
- Caveat: from ~22:04 UTC the GPU had ring timeouts/resets (`ring gfx_0.0.0 timeout`, other workers' jobs, including
  ROCm python that `pf_busy.sh` does not detect). Sessions after that (s2*_a/b/c/d) are unreliable: the GPU-only
  baseline itself drifted to 4.7-6.2 s. Repeat the pp2048 FFN-only sweep (gate/up 4096|6144 x down 1536|2048, builds now
  exist in ~/npu/kernel/builds) on a quiet machine.

## Round 13 (2026-10-14): pp2048 FFN-only re-check under the GPU lock

Every run held `logs/gpu_lock.sh` (docs/benchmarking.md). Perf level high, PF_M=2048, 7 reps. Logs: `logs/r13/`.
- Greedy check of the new default (NPU_SPLIT=1, FFN-only 6144/2048 at M=2048): 16/16 vs token-by-token, logits
  cosine 0.999991.
- Session a (all runs flat, no drift), median ms (tok/s):

  | config (gate/up nn / down nn) | pp2048 |
  |---|---|
  | GPU only (f16) | 4586 (446.6) |
  | FFN-only 4096/1536 | 4224 (484.8) |
  | FFN-only 4096/2048 | 4216 (485.8) |
  | FFN-only 6144/1536 | 4200 (487.6) |
  | FFN-only 6144/2048 (default) | 4283 (478.2) |

- Session b drifted upward within every run (first rep ~4130-4250, last 4600-6300, GPU-only too). The GPU sat at
  90-92 C / 107 W after long qimage runs, so this is most likely thermal/power throttling; only the first reps are
  usable there: GPU only 4596; FFN-only 4224-4236; 6144/2048 4125-4149.
- Conclusion: all FFN-only shares from 4096-6144 / 1536-2048 are within ~2% of each other (noise level); NPU_SPLIT gives
  **1.07-1.09x at pp2048** over the f16 baseline. Default kept at 6144/2048. The limit is still the serial NPU path
  (per prefill: NPU 1556 ms + host wait for prep 2539 ms ~= 4190 ms wall).

## Round 14 (2026-10-15): NPU_FUSE (ffn_gate + ffn_up share one sync point)

`NPU_FUSE=1` (default, d124f72): ffn_up gets its own SplitLinear, so both NPU A preps are recorded in one segment.
Both GPU halves go in one submit, and the NPU runs gate then up back to back: one host wait per layer fewer.
Greedy 16/16 at 2048 (cosine 0.999991) and at 512 (0.999993). Median ms, two sessions, under the GPU lock:

| config | pp512 | pp2048 |
|---|---|---|
| GPU only (f16) | 1002 / 985 | 4519 / 4635 |
| FFN split, NPU_FUSE=0 | 1165 / 1209 | 4185 / 4192 |
| FFN split, NPU_FUSE=1 | 1103 / 1114 | 4111 / 4109 |

pp2048: **1.11x** over GPU only (was 1.07-1.09x). pp512: still ~10% slower than GPU only.

**NPU contention, measured directly:** with fusion the gate NPU run overlaps both GPU halves, but up runs after they
finish. Same shape, same A:
- M=512: gate 3.9 ms vs up 1.4 ms (2.8x).
- M=2048: gate 9.3 ms vs up 6.8 ms.

The NPU is ~3x slower while the f16 GPU GEMM runs. Overlap at M=512 therefore buys almost nothing: the NPU half
finishes no sooner than the GPU could have done it. Round 15 checks fabric clock / GPU clock.

## Round 15 (2026-10-15): what slows the NPU runs (pp512)

FFN-only split 4096/1536, NPU_FUSE=1, PF_M=512, one lock hold (`logs/r15/`). NPU ms per call:

| clocks | gate (K=5120) | up (K=5120) | down (K=17408) | split pp512 | GPU-only pp512 |
|---|---|---|---|---|---|
| perf high | 3.78 | 1.25 | 3.85 | 1077 ms | 945 ms |
| manual, fclk 2000 MHz forced | 3.78 | 1.32 | 3.97 | 1112 ms | 947 ms |
| manual, sclk 1100 requested (did not take, stayed 2900) | 3.77 | 1.29 | 3.90 | 1066 ms | 948 ms |

- Forcing the fabric clock to max changes nothing, so the slow gate/down runs are not DDR/fabric contention.
- The extra time is a constant ~2.5 ms: M=512: 3.8 vs 1.3 ms; M=2048: 9.3 vs 6.8 ms (round 14). Gate and down are the
  calls that **switch xclbin / hw_context**. The K=5120 and K=17408 kernels are different xclbins (dc2 core code depends
  on K), and they alternate every layer: gate follows down of the previous layer, down follows up. Up runs right after
  gate on the same xclbin and is fast. That matches the ~2.2 ms context-switch cost measured in round 10.
- So the real bottleneck is 2 context switches per layer: 64 layers x 2 x ~2.5 ms ~= 320 ms per prefill. That is
  more than the whole pp512 deficit (1077 vs 945 ms). Removing it would put the split at roughly 760-800 ms at pp512
  (~1.2x over the f16 GPU) and ~3.8 s at pp2048. This is an estimate, not a measurement.
- Next: one xclbin for both K. Either a K-generic dc2 core (K loop count from the instruction stream / runtime
  parameter), or ffn_down as K=5120-sized chunks with accumulation (17408 = 3 x 5120 + 2048, so it needs a K=2048
  remainder or K padding to 20480 with zero rows). Alternatively put both kernels in one xclbin on disjoint columns.
  The earlier "NPU slows while the GPU runs" reading (round 14) was wrong.

## Round 16 (2026-10-15): one xclbin for both K (NPU_KCHUNK) + confirmation

- Confirmation (`logs/r15/gu_only.log`): splitting only ffn_gate/up (all K=5120, no xclbin switch) brings gate from
  3.78 to 1.54 ms. That confirms the round-15 diagnosis.
- `NPU_KCHUNK=1` (default, 32abf7b): ffn_down (K=17408) runs as 4 zero-padded K=5120 chunks on the same xclbin as
  the K=5120 shapes. `npu_prep` takes a K window (x0, kv) and gives each chunk its own per-token scale. Each chunk
  has its own folded weight and scales, and the chunk dequants accumulate into y. New insts builds: 512x5120x1536,
  512x5120x2048, 2048x5120x2048.
- Greedy 16/16 vs token-by-token at 512 and 2048 (cosine 0.999993 / 0.999991).
- NPU ms/call with K-chunking:
  - M=512: gate 1.14, up 1.21, down chunks 0.49-0.62 (4 chunks ~2.1 ms, vs 3.85 ms with the switch).
  - M=2048: gate/up 6.8, down 4 x ~2.45.

Session a (session b drifted upward within runs, thermal), median ms:

| config | pp512 | pp2048 |
|---|---|---|
| GPU only (f16) | 954 | 4599 |
| FFN split with K switch (NPU_KCHUNK=0) | 1148 | 4126 |
| FFN split, K-chunked down | 923 (555 tok/s) | 4117 (497 tok/s) |
| ffn_gate/up only (4096) | 908 (564 tok/s) | |

- pp512 is now a win, but a small one: gate/up-only 1.05x, K-chunked FFN 1.03x. Before this fix the split was 1.20x
  slower than GPU only. pp2048: 1.12x; at M=2048 the switch cost was already small relative to the GEMMs.
- New defaults (4751a63): M<2048 splits ffn_gate/up (4096 NPU cols); M>=2048 FFN-only 6144/2048 with K-chunked down.
- What still limits it: the serial prep -> NPU -> dequant chain per split (host wait for the GPU prep segment). With
  the switch cost gone, the next gains are more NPU columns at M=512 (now cheap) and adding the K=5120 attention
  projections (attn_qkv/gate/q share the xclbin now).

## Round 17 (2026-10-15): pp512 column sweep, attention projections, where the ceiling is

Default pp512 split (gate/up 4096, `logs/r16/chk512_default.log`): greedy 16/16, cosine 0.999994. `logs/r17/`,
two sessions, median ms:

| pp512 config | session a | session b |
|---|---|---|
| GPU only | 967 | 968 |
| gate/up 3584 | 913 | 914 |
| gate/up 4096 (default) | 918 | 920 |
| gate/up 5632 | 936 | 931 |
| gate/up 6144 | 944 | 946 |
| gate/up 8704 | 1018 | 1021 |
| + attn qkv 5632 / gate 3584 / q 6656 | 968 | 962 |
| + attn qkv 3584 / gate 2048 / q 4096 | 941 | 919 |

- Best: **1.05-1.06x** at pp512 (3584-4096 NPU columns of 17408). The attention projections add nothing (greedy
  16/16 with them, cosine 0.999994, but no speedup). Default stays gate/up 4096.
- Why the ceiling is low at M=512:
  - Per prefill the NPU is busy only ~170 ms of ~915 ms. The f16 coop-matrix GPU does gate+up at ~3x the NPU's
    per-column rate (NPU 4096 cols ~1.15 ms vs GPU 13312 cols in about the same time), so the NPU can take only
    ~20-25% of a GEMM's columns before it becomes the critical path (5632+ gets slower).
  - Each extra split adds a fence wait and two submits (~0.1-0.2 ms), which is why the many small attention splits
    break even.
  - Upper bound with perfect overlap of every ternary GEMM at a 20% share: ~1.2x. Gate/up are about a third of the
    GEMM time, so ~1.06x is close to what this layout can give.
- Pipelining "prep with the previous layer" is not possible as such. The NPU A for layer l is the rotated output of
  layer l's own norm, which depends on layer l's attention/GDN output (residual chain). The NPU is idle during
  non-GEMM GPU work (GDN scan, attention, norms) because nothing independent exists for it within one token block.
  The only way to give it work there is token-block pipelining: split the chunk into two halves and run the
  NPU FFN of half A while the GPU does attention of half B (K/V of half A are ready), then swap. That needs the FFN of
  half A fully on the NPU (gate/up/down plus SwiGLU and requant in between) or ping-ponging small GPU kernels. Bound:
  ~1.2-1.3x at pp512. This is a sizeable refactor of `qwen35::record` (two interleaved half-chunk streams with shared
  recurrent state order), not done in this round.

## Round 18: token-block pipeline (`NPU_TB=1`, commit 6a1660d)

Each prefill chunk is split into two token blocks, A and B, with separate `Q35` scratch states. The whole FFN of A runs on the
NPU: gate and up are fused, and down runs as 4 K=5120 chunks. SwiGLU, Hadamard and requant run as small GPU ops between them. B runs fully on the
GPU. `tb_chunk` drives it layer by layer: A mixer (GPU) -> B mixer (GPU) || NPU A gate/up -> A SwiGLU + down prep
(GPU) -> B FFN (GPU) || NPU A down -> A residual add. The KV cache and GDN state stay ordered because the A mixer is always queued before the B mixer.

| Bonsai 2 27B PQ2_0, pp2048 (PF_M=2048, A=B=1024) | best ms (run a / b) | tok/s | vs GPU |
|---|---|---|---|
| GPU only (PF_TF16 baseline) | 4527 / 4580 | 452 | 1.00x |
| NPU_SPLIT FFN column split, KCHUNK (round 16) | 4083 / 4096 | 501 | 1.11x |
| **NPU_TB=1** | **3810 / 3821** | **537** | **1.19x** |

Greedy: 16/16 continuation tokens match, argmax matches, logits cosine 0.999993, rep-to-rep diff 0
(logs/r18/tbchk2048.log). NPU time per layer at M=1024: gate 11 ms, up 9.1 ms, down 4 x 3.1 ms, about 33 ms in total. That is
roughly the GPU time for the B mixer plus FFN, so the two sides are about balanced. The host waits about 1.1 s per run on the
A-mixer fence before the A FFN can start, and the per-layer sync adds a gap. Overlapping the A mixer of layer l+1 with the NPU down of
layer l is the next step.

pp512 is not possible in this form. The XDNA2 GEMM requires M to be a multiple of 512 at the 128-row tile, so a 256-token half needs the
64x64x64 tile, and that tile only reaches 4.3 TOPS (~/npu/kernel/r18b256.log). That would make the A FFN about 30 ms per layer, slower than
the GPU. For pp512, the best remains the round-16 gate/up column split (900-918 ms vs 950-983 ms, about 1.05x).

## Rounds 19-20: no per-layer host sync, pp4096, energy per token

**Cross-layer overlap.** The A attention for layer l+1 cannot run during the NPU down for layer l: it reads A's residual stream, and that
stream depends on down(l). The B attention for layer l+1 cannot fill the gap either, because B's tokens attend to A's layer-(l+1) KV,
and GDN state runs in A-then-B token order. So within this design, the only things that overlap are the ones that already do:
B attention with A gate/up, and B FFN with A down. Commit 35204b9 drops the per-layer host sync instead (`NPU_TB_SYNC`, default 16 layers).
All GPU work is now ordered by queue order plus the leading barrier of each command buffer. Layer l+1 is recorded while the GPU drains
layer l. The gain is small: pp2048 went from 3832-3842 ms (sync every layer) to 3784-3805 ms. The remaining critical path per layer is
A attention, then NPU FFN (about 33 ms), then the A residual add. To go further would need a third token block, or the NPU taking only
part of the A FFN.

| Bonsai 2 27B PQ2_0 | GPU only | NPU_TB (PF_M=2048, 1024+1024) | NPU_TB (PF_M=4096, 2048+2048) |
|---|---|---|---|
| pp2048 best ms | 4591-4601 | 3784-3805 (1.21x) | - |
| pp4096 best ms | 9526 (PF_M=2048); 12220 (PF_M=4096) | **8075-8101 (1.18x)** | 8476-8499 |

**Greedy check at pp4096.** NPU_TB reaches 6/16 greedy continuation tokens, with first-token argmax and logits cosine 0.999992
matching the GPU-only run. The GPU f16 path gets 16/16. This is a numerics effect, not a pipeline bug: the GPU-only int8-activation
path (`PF_TF16=0`) produces the identical token sequence and the same 6/16 (logs/r20/i8chk4096.log). The NPU uses int8 activations,
and on this repetitive synthetic prompt a near-tie at token 7 flips. At pp2048, NPU_TB is 16/16.

**Energy** (energy recorder, 1 kHz). Each figure is a REPS=6 run minus a REPS=1 run, which leaves 5 pp2048 prefills; model load and weight fold cancel out. Units are J/token.

| pp2048 | gfx | npu | cpu cores | soc_other | socket total | vs GPU |
|---|---|---|---|---|---|---|
| GPU only | 0.129 | 0.000 | 0.004 | 0.212 | **0.345** | 1.00 |
| NPU_SPLIT=1 (FFN columns) | 0.107 | 0.004 | 0.008 | 0.175 | **0.294** | 0.85 |
| NPU_TB=1 | 0.101 | 0.003 | 0.039 | 0.134 | **0.277** | 0.80 |

NPU_TB uses about 20% less energy per token. Most of the saving is in soc_other (memory, data fabric and IO). That is consistent with
the shorter runtime at a similar socket power (about 140 W). The SMU's `average_ipu_power` for the NPU reads near zero, which is
implausible for an NPU at about 15 TOPS, so treat the npu column as unreliable. The NPU energy probably lands in soc_other. The cpu-core
cost of NPU_TB (0.039 J/token) is the host thread spinning on NPU completion. The other worker's NPU-default regression report on
Qwen3.8-27B (4751a63) was traced to a cold first run (7012463), so this round needed no change for it.

The column split (NPU_SPLIT=1, PF_M=2048) gets 16/16 at pp4096 in 8521 ms (one timed run). Only part of each FFN runs with int8 activations there, so it perturbs less than NPU_TB (logs/r20/splitchk4096.log). If greedy identity at long context matters more than the 5% speed difference, use NPU_SPLIT for long prompts.

## Rounds 21-24: NPU wait, three-block pipeline, NPU_AUTO default

**NPU wait (round 21).** The host does not spin. XRT `run.wait()` and `vkWaitForFences` both block in the kernel. NPU_TB REPS=1 vs
REPS=4 shows 0.5 s of user time for 3 extra pp2048 prefills, which is under 2% of one core (logs/r21/cpu.log). The 0.039 J/token
"cpu cores" figure in round 20 came from other CPU-heavy jobs on the box. Re-measured behind `logs/wait_quiet.sh` (round 23), the NPU
modes use 0.008-0.009 J/token on the cores, the same as GPU only (0.010). There was nothing to replace.

**Block size / partial FFN (round 21).** These all lose to 1024+1024 at pp2048 (3786-3806 ms):
- NPU_TB=1 with A=512: 3965-4058 ms
- A keeps half its FFN columns on the GPU (`NPU_NN=5120x17408:8704,17408x5120:2560`): 3945-3962 ms
- A=512 with PF_M=1536: 4090-4241 ms

**Three blocks, NPU_TB=3 (commits 089018d, 51a4230).** Blocks A and B have 512 tokens each and run their whole FFN on the NPU. Each
has its own NPU A/C buffers (a second `SplitLinear` set over the same weight BOs). Block C has the rest of the tokens and runs entirely
on the GPU. The NPU order per layer is gate/up A, gate/up B, down A, down B. The GPU runs the B and C mixers, A's SwiGLU, C's FFN, and
then layer l+1's A mixer while the NPU runs B's down for layer l. This is the cross-layer overlap that two blocks cannot provide.
The host syncs only at the end of a chunk. A sync in the middle of a chunk destroyed the fences of the already-submitted next-layer
segments and caused VK_ERROR_UNKNOWN.

| Bonsai 2 27B PQ2_0 (wait_quiet + GPU lock) | pp2048 best ms | pp4096 best ms | greedy 2048 / 4096 |
|---|---|---|---|
| GPU only | 4498-4545 | 9567-9611 | 16 / 16 |
| NPU_TB=1 (1024+1024) | 3732-3782 (1.20x) | 8075-8137 (1.18x) | 16 / 6 |
| **NPU_TB=3 (512+512+1024)** | **3528-3556 (1.28x)** | **7537-7580 (1.27x)** | 16 / 6 |
| NPU_TB=3, A=B=1024 | 4147-4195 (PF_M=3072) | 7816-7824 (PF_M=4096) | - |

The pp4096 miss is the same int8 near-tie as round 20. It produces the same 6/16 token sequence as the GPU int8 path.

**Energy per token, pp2048** (energy recorder; REPS=6 minus REPS=1; logs/r23/e_*.json), in J/token:

| | gfx | npu | cores | soc_other | socket |
|---|---|---|---|---|---|
| GPU only | 0.132 | 0.000 | 0.010 | 0.178 | **0.320** |
| NPU_TB=1 | 0.100 | 0.003 | 0.009 | 0.136 | **0.248** (-23%) |
| NPU_TB=3 | 0.092 | 0.003 | 0.008 | 0.139 | **0.242** (-24%) |

**NPU_AUTO (commits 7ada862, 49e4368).** This is the new default for ternary models at PF_M >= 2048 when neither NPU_TB nor NPU_SPLIT
is set. Set NPU_AUTO=0 for GPU only. `Q35::new` builds the column-split state plus a 512+512+1024 three-block set. Loading takes
about 6 s more for the extra weight fold, and the NPU weights take about 24 GB. `prefill` picks the mode by prompt length L:

| L | mode | measured (PF_M=2048, logs/r24) | greedy |
|---|---|---|---|
| <= 1024 | GPU only | 2054 vs 2045 ms (GPU) | 16/16 |
| 1025-2048 | NPU_TB=3 | 3538 vs 4530 ms (1.28x) | 16/16 |
| > 2048 | NPU column split | 8535 vs 9563 ms (1.12x) | 16/16 |

At PF_M < 2048 nothing changes. The per-model NPU_SPLIT default work by the other worker applies there.
