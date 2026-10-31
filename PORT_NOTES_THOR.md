# Thor port notes (lab notebook, 2026-10-30)

Porting strix-rt to an NVIDIA Jetson AGX Thor devkit (Tegra T264: Blackwell iGPU, 20 SMs, CC 11.0, warp 32;
14-core Neoverse-class CPU; 122 GiB unified LPDDR5X at ~273 GB/s; JetPack 7.2 / L4T r39.2.1, Ubuntu 24.04,
NVIDIA driver 595.78, Vulkan 1.4.329). Everything below is from the logs on the Thor itself.

## What it took to build (three changes)

1. **`npu` crate hard-links AMD XRT.** `build.rs` now probes `/usr/include/xrt/xrt_bo.h`; when absent it skips the
   shim compile/link, emits `cargo:rustc-cfg=npu_stub`, and `lib.rs` substitutes no-op stubs for the 13 extern fns
   (open returns null -> every NPU path errors cleanly at runtime). Zero change on hosts that have XRT.
2. **Ubuntu 24.04's packaged glslc predates `GL_EXT_integer_dot_product`** (gemvq.comp fails). Fixed with
   shaderc 2026.4 from conda-forge (`pixi global install -c conda-forge shaderc`); `~/.pixi/bin` first in PATH.
3. **`kitten`'s `dot4` / `sdpa_rows` were `#[cfg(target_arch = "x86_64")]` AVX2-only** with ungated call sites.
   Added portable fallbacks (multi-accumulator scalar; LLVM auto-vectorizes to NEON). A tuned NEON/SVE2 pass is
   an open slot.

With those, the whole workspace builds (cargo build --release, all 60+ bins) and runs.

## Correctness smoke (not a full parity run)

Nemotron 3 Nano 4B Q4_K_M via `nh`: "The capital of France is" -> top-1 token 6993 (" Paris") at logit 18.0,
4 logits clear of #2. Tokenizer read from the GGUF by `glm_tok` matches round-trips. Weights import: 2693 MiB in
0.42 s (3 arenas on the 4.7 GiB Qwen3 file earlier: 4788 MiB in 0.76 s). Teacher-forced agreement vs a reference
run (FORCE mode) still to do.

## First numbers (decode, differential long-short: n=256 vs n=16, 240-token delta)

Energy from the board's INA3221 rails (VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0) and the INA238 module-input rail,
sampled at 20 Hz by `crates/jenergy` (`jrecord`, a Rust sibling of tools/energy/record.py with the same CLI/JSON).
Board rails, not wall power. Idle floor ~30 W module input.

| Nemotron 3 Nano 4B Q4_K_M, decode | strix-rt (as ported, untuned) | llama.cpp CUDA (conda-forge 0.8.0-dev 7e2fea0) |
|---|---|---|
| tok/s | 21.9 | 66.6 (tg128), 67.0 (tg256) |
| J/token, module input (INA238) | 2.247 | 1.163 |
| J/token, VDD_GPU | 0.915 | 0.536 |
| J/token, VDD_CPU_SOC_MSS | 0.463 | 0.254 |
| prompt | 6-token prompt only (NOT a pp512 number) | pp512 2251 tok/s |

Read it the honest way: the port is 3.05x slower and ~1.9x more joules per token than the CUDA bar. The implied
GPU-rail draw during decode is ~20 W (strix-rt) vs ~36 W (llama.cpp): our kernels starve the GPU rather than
strain it, so closing the speed gap should close most of the energy gap with it.

## Leads (in rough order of expected payoff)

- **NV_coopmat2**: this Vulkan driver exposes cooperative-matrix-2 (tensor cores) and `int dot: 1`
  (llama.cpp's vulkan device line confirms both). strix-rt's integer-dot GEMV failed on RADV; Blackwell's DP4A
  path may redeem it, and coopmat2 is a bigger lever than any subgroup tuning.
- Occupancy/tile sweep for 20 SMs x warp 32 (all shader tile sizes were swept for gfx1151's 40 CUs).
- Prefill batching through the same pp512 harness used for the Strix numbers, before comparing prefill at all.
- NEON/SVE2 versions of kitten's CPU kernels.
- Teacher-forced parity runs (FORCE) per model as they come up on this box.
- Same-model cross-silicon table once Nemotron 30B-A3B numbers land here (Strix Halo side is already published).

## Environment pins

rustc 1.99.1; shaderc 2026.4 (conda-forge); llama.cpp conda-forge 0.8.0-dev build 7e2fea0 (CUDA+Vulkan);
models under ~/models per MODELS.md; this clone: ~/work/strix-rt on the Thor (`cbio@thor.local`).

## Prefill on NVIDIA: device-lost hunt (2026-10-30, evening)

PF=1 batched prefill dies with VK_ERROR_DEVICE_LOST. Validation layers found real spec violations first:
vkrt did not enable `maintenance4` (modern glslc emits LocalSizeId) or `vulkanMemoryModel`(+DeviceScope)
(GL_KHR_memory_scope_semantics) — both now enabled in vkrt (correct on AMD too; RADV just tolerated it).
Remaining violations on this driver: int8 coopmat shaders use 16x16x16 but NVIDIA only lists K=32 for s8/u8
(ternary/Bonsai path, needs a KSize-32 variant), and one shader wants 62592 B of shared vs 49152 max.

Supported coopmat (NVIDIA 595.78, subgroup 32, max shared 49152): f16xf16->f16/f32 at 16x16x16, 16x8x16, 16x8x8;
s8/u8 at 16x16x32, 16x8x32; bf16 16x16x16->f32. `devinfo` (new bench bin) prints this table.

The crash itself is narrower than any of that: PF_PROBE bisect on pf_gemm (Q4_K, f16 16x16x16 — a listed config):
PROBE=3 (skeleton) OK, PROBE=2 (dequant+staging, no MMA) OK, PROBE=1 (MMA on, loads off) device lost.
pf_peak (register-only coopmats) runs fine. So the kill is coopMatLoad-fed MulAdd — A loaded straight from the
activation buffer (ADIR) and/or B from shared with LS4 stride. Next: standalone repro shader, then sweep
layout/stride/BK variants. pf_peak rates for reference: f16 29.5 TFLOPS, s8 "1778 TOPS" (out-of-spec shape on NV,
treat as invalid until re-measured at K=32).
