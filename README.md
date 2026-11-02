# strix-rt

**A from-scratch Rust + Vulkan LLM and vision runtime for one AMD Strix Halo box, built to measure joules per token, with every number reproducible, including the experiments that failed.**

We think the energy a computer spends to do a piece of thinking (joules, not tokens per second) is the honest way to measure machine intelligence. This repo is our lab bench for that idea. We publish it, notes and dead ends included, so anyone can copy what worked, skip what didn't, and save some energy. There is no moat here and we don't want one.

## Measured so far (one machine, numbers straight from the logs)

| Workload (Ryzen AI Max+ 395, Radeon 8060S gfx1151, 128 GB) | strix-rt | llama.cpp (our reference bar) | Where it comes from |
|---|---|---|---|
| Gemma 4 26B-A4B UD-Q4_K_M, prompt processing pp512 | 1637 tok/s | 1719 tok/s (Vulkan) | [docs/prefill.md](docs/prefill.md), `logs/g4/pp_sweep3.log`, `logs/g4/rebench_llama.log` |
| Nemotron 3 Nano 30B-A3B Q4_K_M, prompt processing pp511 | 1469 tok/s | 1687 tok/s (Vulkan, pp512) | [docs/prefill.md](docs/prefill.md), `logs/nh/high.log`, `logs/nh/bench30b_v3/l_pp.txt` |
| Muse Glimmer 30B Q4_K_M, decode energy (SMU socket) | 7.9-8.1 J/token (7.92 on one quiet run, 8.10 on a later quiet run; 12.95-13.3 tok/s) | 6.98 J/token Vulkan, 8.12 ROCm (13.1 / 12.8 tok/s) | [docs/glimmer.md](docs/glimmer.md), `logs/glimmer/xs2.log`, `logs/glimmer/q6s.log`, `logs/glimmer/energy_llama.log` |
| Gemma 4 26B-A4B, decode energy (GPU "gfx" channel, same session) | 0.685 J/token | 0.569 J/token (Vulkan) | [docs/comparisons.md](docs/comparisons.md), `logs/g4/bench_26b3_energy.log` |
| Nemotron 3.5 Super VL 120B-A12B Q4_0, image input vs llama.cpp mtmd | pixel tensors bit-exact on 6/6 test images (PNG and JPEG); greedy answers identical to EOS on 6/6 | (reference) | [docs/new-models.md](docs/new-models.md), `logs/n35/imgparity3.log`, `logs/n35/imgvl2.log` |

Read it the honest way: on these rows llama.cpp is still faster at prefill and still spends fewer joules per decoded token. We match its outputs, and we are close enough to learn something from every gap. The full lab notebook, with more models and a few rows where strix-rt comes out ahead, is in [docs/comparisons.md](docs/comparisons.md).

## What this is, and what it isn't

**It is**
- A from-scratch runtime: GGUF parsing, quantized GEMV/GEMM kernels in GLSL compute shaders, attention, MoE routing, Mamba-2 / Gated DeltaNet state, vision towers and tokenizers, written in Rust on [ash](https://github.com/ash-rs/ash) (Vulkan, Mesa RADV).
- An energy bench: every serious number comes from a differential energy measurement with the box's own power sensors, read from standard Linux sysfs files (see [How we measure](#how-we-measure)).
- A record of failures. Integer-dot GEMV, a Q6_K quad kernel, hybrid tiles, a persistent work queue and fused gate+up all failed here, and the write-ups are below and in `docs/`.

**It isn't**
- A general runtime. It is tuned for one chip (gfx1151) on one machine with one OS, kernel and Mesa build. On other hardware it may not run at all.
- Faster than llama.cpp in general. llama.cpp is our bar for quality and energy, and on most rows above it is still ahead.
- Wall-power accurate. Energy is the SoC's **socket** power as the SMU reports it (plus a GPU "gfx" channel), not a meter at the wall. Fans, VRMs, SSD and PSU losses are not counted.
- Quiet. Several jobs often shared the box, so some runs are noisy. The docs say which runs were taken on a busy box and which were thrown out.
- Polished. Model support is uneven, most binaries are research harnesses driven by environment variables, and the logs are raw.

## Quick start

Needs the hardware and software listed under [Hardware and software](#hardware-and-software). No GPU work happens at build time.

```sh
git clone https://github.com/openIE-dev/strix-rt && cd strix-rt
cargo build --release                      # ~45 s on the 16-core Strix Halo; compiles all GLSL shaders with glslc
export MODELS=~/models                     # where you keep GGUF files (see MODELS.md); weights are NOT in this repo

# Greedy decode with Muse Glimmer 30B, prompt typed as text, 64 new tokens (prints tok/s)
./target/release/glm_run $MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf "text:The Strix Halo APU is" 64
```

The exact commands behind each number above, plus the energy harness, are in [REPRODUCE.md](REPRODUCE.md).

## Hardware and software

What it was built and measured on (the only setup we have tested):

- AMD Ryzen AI Max+ 395 "Strix Halo" in a Framework Desktop: Radeon 8060S iGPU (gfx1151, RDNA 3.5, 40 CUs), XDNA2 NPU, 128 GB unified memory (about 256 GB/s peak; decode tops out near 220-230 GB/s in practice)
- Ubuntu 26.04, Linux 7.0, Mesa RADV 26.0.8 (Vulkan 1.4), `glslc` (shaderc)
- Rust 1.99 (stable)
- `libxrt-dev` (AMD XRT headers and `libxrt_coreutil`): the `npu` crate links it and `prefill` depends on `npu`, so the workspace does not build without it, even if you never touch the NPU
- Optional: `kmod/strixpc` (a small kernel module for zero-copy GGUF imports, GPL-2.0, see [kmod/strixpc/README.md](kmod/strixpc/README.md)), `sudo` to pin the GPU performance level, and Python 3 for the energy recorder (`tools/energy/record.py`, standard library only)
- The reference numbers used llama.cpp `e36df6047` (Vulkan and ROCm 7.13 builds) and `f3363593f` for Nemotron 3.5 Super; details in [docs/comparisons.md](docs/comparisons.md)

## Layout

```
crates/
  gguf/        GGUF v3 parser, mmap, CPU dequant reference
  vkrt/        thin Vulkan layer (ash): device, buffers, pipelines, timestamps + shared GEMV/attention shaders
  rtcore/      shared model plumbing: weight arenas, quant-type tables, GPU-driven decode helpers
  prefill/     batched prompt processing: coop-matrix GEMMs (dense + grouped MoE), flash attention
  npu/         XDNA2 NPU experiments (XRT shim)          ternary/   1.58-bit kernels
  qwen2/ lfm2/ qwen38/ gptoss/ gemma4/ glimmer/ nemotron/ nemotron35/ mcv/    text and vision-language models
  qimage/ k6/ cosmos3/ yue2/ kitten/ parakeet/    image, video, music, TTS and speech-to-text ports
  kev/ laya/   "System One" decision models          serve/     OpenAI-style servers (experimental)
  bench/       micro-benchmarks and older model runners (strix, lfm, q38, nh, ...)
docs/          lab notebook, one file per topic (start at docs/README.md)
logs/          raw logs and session scripts behind the numbers (large media and multi-100 KB dumps removed)
tools/comp/    benchmark and energy harness scripts
tools/energy/  record.py: stand-alone socket-energy recorder (reads amdgpu gpu_metrics, hwmon, RAPL from sysfs)
kmod/strixpc/  dma-buf kernel module for zero-copy weight import (GPL-2.0)
PROGRESS.md    the original day-by-day progress log
```

## How we measure

- **Energy = differential windows.** Run the same command twice, once with a short and once with a long token count (for example N=64 and N=320 or 480 decoded tokens), and integrate power over each whole run. J/token = (J(long) - J(short)) / (long - short). Model load, prefill and warm-up are in both runs, so they cancel. Long windows (at least 30 s) keep the sensor's averaging lag small. Script: `tools/comp/energy.sh`; prefill and image-encode energy use the same idea (for example 20 x pp2048 minus 20 x pp512, or REPS=6 minus REPS=1).
- **What is measured: socket power, not wall power.** The main channel is the SMU's `average_socket_power` from the amdgpu `gpu_metrics` sysfs file: the whole APU (CPU cores, GPU, memory controller, fabric). A second channel, `average_gfx_power`, is the GPU core alone. The hwmon `PPT` reading and the RAPL package counter are cross-checks. DRAM chips, VRM losses, fans, SSD and PSU are not in any of them.
- **Pinned clocks.** `tools/comp/perf_high.sh` forces the amdgpu performance level to `high` for the run and restores `auto` afterwards, the same condition we use for the llama.cpp numbers.
- **One GPU job at a time.** `logs/gpu_lock.sh` holds an exclusive `flock`, and `logs/wait_quiet.sh` waits until no process is using more than 100% CPU, because CPU jobs steal memory bandwidth and power budget from the GPU ([docs/benchmarking.md](docs/benchmarking.md)).
- **Repeats and A/B interleaving.** Typically 3 to 6 reps, A/B pairs interleaved in one locked session, medians reported. Run-to-run noise on socket J/token is about ±0.3 J for Glimmer and up to about 12% between sessions, so we treat differences under about 10% across sessions as not significant.
- **Clean-box rule.** If decode speed shows the box was busy (for Glimmer, below 13.0 tok/s instead of 13.2), the pair is thrown out. Several experiments below have no clean socket number for exactly this reason, and we say so instead of publishing a contaminated one.
- **Parity before speed.** Kernel changes are checked against llama.cpp greedy references: free-run ids, and teacher-forced agreement with top-2 margins. A miss only counts as noise if it is a near-tie inside the spread between llama.cpp's own backends.

**Replicate it with standard sysfs readings.** Nothing here needs special tools. `tools/energy/record.py` (Python 3, standard library, about 150 lines) samples these files at 20 Hz while your command runs and integrates them with the trapezoid rule:

| Channel | File | Unit |
|---|---|---|
| socket | `/sys/class/drm/card*/device/gpu_metrics`, `struct gpu_metrics_v3_0`, `average_socket_power` (byte offset 112) | mW |
| gfx | same file, `average_gfx_power` (offset 124) | mW |
| PPT | `/sys/class/drm/card*/device/hwmon/hwmon*/power1_average` (label `PPT`) | µW |
| CPU package | `/sys/class/powercap/intel-rapl:0/energy_uj` (usually root-only; an energy counter, so no integration needed) | µJ |

Check the offsets against `kgd_pp_interface.h` for your kernel; the header's format byte must read 3. Then run `tools/energy/record.py record --json short.json -- <cmd with N=64>` and the same with N=320, subtract the socket joules and divide by 256. Pin clocks first (`echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level`), make sure nothing else is running, and repeat at least 3 times. Some numbers in `docs/` and `logs/` were recorded with an earlier in-house recorder that read the same `gpu_metrics` socket and gfx averages; the scripts in this repo now call `tools/energy/record.py`.

## Results summary

- **Prefill.** Gemma 4 26B-A4B is at 1637 vs 1719 tok/s (-4.8%) after tuning the packed expert GEMM's wave layout. Nemotron 3 Nano 30B is at 1469 vs 1687 (-13%). Probes show the expert GEMMs are bound by coop-matrix issue and LDS fragment loads, not by dequant or memory ([docs/prefill.md](docs/prefill.md)).
- **Decode energy.** Glimmer went from 8.75 to 7.92 J/token (socket, -9.5%) through two kernel changes: an item-pair Q4_K GEMV, and a per-32 sum(x) precompute. A later quiet-box run of the same default measured 8.10, so we quote 7.9-8.1 J/token: +13.5% to +16% above llama.cpp Vulkan (6.98), and at or just below ROCm (8.12). Clocks, activity and bandwidth match llama.cpp; the gap is power per busy cycle in the float-dequant GEMVs ([docs/glimmer.md](docs/glimmer.md)). On Gemma 4 26B the GPU channel says we spend more (0.685 vs 0.569 J/token). The socket channel came out lower in one session (1.375 vs 1.777) but varied from 1.38 to 2.11 across three sessions, so we don't claim a socket win there.
- **Decode is at the DRAM wall.** Every weight-streaming group on Gemma and Glimmer already runs at about 220-230 GB/s, so wider loads can't make decode faster. Only watts per byte can still move.
- **Vision parity.** Nemotron 3.5 Super VL image preprocessing is bit-exact with llama.cpp mtmd for PNG and JPEG, the latter through a faithful Rust port of stb_image's JPEG decoder (byte-exact on a 34-file corpus). Greedy answers match llama.cpp token for token to EOS on all 6 test images.
- More models (Qwen3.8, LFM2.5, MiniCPM-V, gpt-oss, Bonsai ternary, Kandinsky 6 video, YuE2 music, Kitten TTS, Parakeet STT, Cosmos3), including rows where strix-rt draws less energy than llama.cpp and many where it doesn't: [docs/comparisons.md](docs/comparisons.md), [PROGRESS.md](PROGRESS.md).

## Lessons from failed experiments

Each of these looked like a sure win on paper. All stay in the code as opt-in switches so you can rerun them.

1. **Integer-dot GEMV (llama.cpp MMVQ style) for Glimmer's dense decode.** First try (`GLM_IDOT=1`) was about 3.5x slower (about 260 vs 75 ms/token). A second design that kept our lane layout didn't slow down and cut GPU power (about -10 W), but with per-32 activation scales it failed parity (7 near-tie misses instead of 3, gaps up to 0.73). Per-8 scales (`GEMVQ_Q4P=45`) passed parity, then lost on socket energy in 6 of 6 A/B pairs (+0.3 J/token, 3-4% slower): the finer scales cost back the ALU work the integer dot saved. *Lesson:* fewer ALU ops per weight byte only pays off if the activation quantization stays cheap, and none of these runs got a clean-box socket number. (The same idea on Gemma 26B's Q4_K expert gate/up did help: GPU power 41.2 to 38.2 W at the same 54 tok/s. Context matters.)
2. **Hybrid 32/16-row expert tiles** (`PFG_HYB`, Nemotron Nano prefill). +0.3%, which is noise. Padding only changes for the last 1-16 rows of an expert, and the extra dispatch plus duplicated dequant cancel the MMA work saved. *Lesson:* measure how much padding actually costs before you engineer it away.
3. **Persistent work queue** (`PF_QH_PERS`, Gemma prefill). Equal or 0.5% slower at every size. The grouped GEMMs already launch more than 1000 uniform workgroups on 40 CUs, so there is no idle tail to recover. *Lesson:* persistent kernels fix load imbalance, and this kernel didn't have any.
4. **Fused gate+up+SwiGLU in one dispatch** (`GLM_FUSE_FFN=1`, Glimmer decode). Numerics identical, energy identical (11.18 vs 11.17 J/token, measured under the same contention). The f32 intermediate it removes is tiny next to the weight stream. *Lesson:* at the DRAM wall, removing intermediate traffic only helps when that traffic is a real share of the bytes.
5. **Q6_K quad kernel for ffn down and lm_head** (`GEMVQ_Q4P=71`, Glimmer decode). Four Q6_K items share the same 96 bytes of packed weights, so one lane decodes all four from aligned 32-bit words: 26 word loads per 128 weights instead of 72, and GPU power dropped by about 7 W. Parity was unchanged. On a quiet box, 4 interleaved pairs: 9.33-9.41 J/token vs 8.10 for the default (+16%), at 10.98 vs 12.95 tok/s (15% slower), and no lane/workgroup setting in the sweep beat the default. The scattered word loads made it load-issue bound, while the plain float Q6_K path already sits at the DRAM wall. *Lesson:* fewer instructions per byte is worthless if it costs memory-level parallelism; watts saved per second don't help if the run takes longer.

There are smaller ones throughout `docs/`: f16-typed shared arrays that made `coopMatLoad` 6x slower, a split-K GEMM that gave nothing, 1-ulp attention differences that flipped a near-tie, and a GPU hang caused by one command buffer running longer than the amdgpu lockup timeout.

## What we are building, and next steps

We work in public, so this list will change as the logs come in.

- **Close the Glimmer energy gap** (7.9-8.1 vs 6.98 J/token): Q6_K dequant ALU is now ruled out as the lever (see failure 5); next are int-dot with per-16 scales and a smarter scale search, and re-measuring the opt-in variants on a quiet box.
- **Prefill structure:** the remaining prefill gap needs a different MMA/LDS structure for the 24-32-row expert tiles; tile sizes, dequant and scheduling are ruled out.
- **Measure what we haven't:** llama.cpp prefill energy and image-encode energy, so the energy table also covers prompt processing.
- **Better measurement:** check the gpu_metrics gfx-activity and clock fields in `tools/energy/record.py` under load, add a wall-power meter next to the SMU numbers, and run more repeats on a dedicated quiet box.
- **Not yet released** (needed to rerun a few checks bit for bit; we plan to publish them):
  - the llama.cpp `mtmd` tensor-dump patch used for the Nemotron 3.5 Super VL pixel-parity check
  - the 6 vision test images behind that check (and the other image/video test inputs; media files were left out of this repo)
  - the Gemma 4 26B chat-prompt parity files (`logs/g4/26bc_prompt_ids.txt`, `26bc_vk_ids.txt`) used for the teacher-forced checks
  - the Laya decision-model golden outputs (`crates/laya/ref/golden.*`, too large for this snapshot)
- **Energy receipts per decision:** `crates/receipt` (new) hashes each agent slice (domain-separated SHA-256), chains it to the previous receipt, signs it with ed25519, records compute and wait joules with their provenance, and checks incoming event envelopes against replay. `kev_serve` now returns a signed receipt with every `/v1/decisions` response and appends it to a local ledger (`KEV_RECEIPT_KEY`, `KEV_RECEIPT_LEDGER`). Next: publish measured J/decision for the small decision models next to the hosted ones. See [docs/infinity-suspend-resume.md](docs/infinity-suspend-resume.md).
- **Make it easier to copy:** fewer environment switches, one bench script per headline number, and a portable path for other RDNA parts if someone wants to try.

## License

Dual-licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option (`SPDX-License-Identifier: MIT OR Apache-2.0`). Copyright 2026 Open Interface Engineering, Inc.

Exception: `kmod/strixpc/` is a Linux kernel module and is licensed GPL-2.0 (see its SPDX headers). Third-party code and data we derive from or reference are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). **Model weights are not included** and come with their own licenses; see [MODELS.md](MODELS.md).

## Contributions: read-only for now

This repository is published to read, run and copy. **We don't accept pull requests for now**, and PRs will be closed without review. That isn't because we don't value your work; we just can't review and maintain outside changes yet. Issues are open for questions, reproduction reports and feedback, and we read them, but please don't expect a reply or a fix. Forks are very welcome: take whatever helps you save energy.

A small GitHub Actions workflow (`.github/workflows/close-prs.yml`) closes new pull requests automatically with a short, friendly note, so nobody waits on a review that won't come.

Security: see [SECURITY.md](SECURITY.md). In short, this is research code that runs on your own machine; please report anything security-relevant privately to contact@openie.dev.

## About OpenIE

strix-rt is a project of **Open Interface Engineering, Inc. (OpenIE)**, a Delaware public benefit corporation (B-Corp). Our goal is to teach, for free, how to make computers spend less energy for the same intelligence, so that people copy it and the planet stays livable for humans. We measure what we build in joules.

- Website: [openie.dev](https://openie.dev)
- GitHub: [github.com/openIE-dev](https://github.com/openIE-dev)
