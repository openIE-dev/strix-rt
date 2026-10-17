# Reproducing the numbers

Everything here was run on one machine (see [README: Hardware and software](README.md#hardware-and-software)). These are the
commands that produced the headline table, copied from the session scripts in `logs/`. Expect your numbers to differ:
it is a different box, a different thermal setup, and maybe a different Mesa build. If you get something very different, please
open an issue with your log. We read them.

## 0. Build

```sh
sudo apt install glslc libxrt-dev          # shader compiler + XRT (needed by the npu crate, which prefill depends on)
cargo build --release                      # all binaries land in target/release/
```

Verified from a clean checkout of this staging tree: `cargo build --release --workspace` finishes in about 45 s
(Rust 1.99.0, shaderc 2026.1, `-j 16`). The build doesn't touch the GPU.

## 1. Conventions used by the scripts

- **Paths.** The session scripts in `logs/**/*.sh` and `tools/comp/*.sh` assume the repo is checked out at `~/strix-rt`,
  model files under `$MODELS` (we used `export MODELS=~/models`, layout as in [MODELS.md](MODELS.md)), and llama.cpp builds under
  `~/llama-latest/build-vulkan` and `~/llama-latest/build-rocm71x`. Adjust these or symlink to match.
- **Target directories.** Several jobs shared the machine, and each built into its own target dir (`target-glm`, `target-g4p`,
  `target-nhp`, `target-n35`, ...). With a single build you can symlink them: `for d in glm g4p nhp n35 comp; do ln -s target target-$d; done`.
- **One GPU job at a time.** Wrap every measurement as the scripts do:
  `logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/perf_high.sh <cmd>`.
  `perf_high.sh` writes to `/sys/class/drm/card1/device/power_dpm_force_performance_level` with `sudo` (check your card index),
  and restores `auto` on exit. `tools/comp/run.sh` wraps all three.
- **Energy.** `tools/comp/energy.sh TAG "CMD with @N@"` runs the command at N=64 and N=320 decoded tokens
  (`energy480.sh`: N=64 and N=480) under `tools/energy/record.py record --hz 20 --json ...` and prints
  `(J_long - J_short) / (N_long - N_short)` for the socket and gfx channels. The recorder is in this repo (Python 3, standard
  library only) and reads nothing but sysfs; see the next section. Some logged numbers were taken with an earlier in-house
  recorder that read the same `gpu_metrics` socket and gfx averages.

## 1a. Measuring energy yourself with standard sysfs readings

You don't need our scripts to check a number. On a Strix Halo (or other recent AMD APU) running Linux:

1. **Pick the sensors.** `/sys/class/drm/card1/device/gpu_metrics` is a binary `struct gpu_metrics_v3_0` (bytes 2-3 of the header
   must read format 3, content 0). `average_socket_power` (u32 at byte offset 112, mW) is the whole APU socket; `average_gfx_power`
   (offset 124, mW) is the GPU core. `/sys/class/drm/card1/device/hwmon/hwmon*/power1_average` (label `PPT`, µW) is a second view
   of package power and tracks socket power closely (about 8 W each at idle on our box). `/sys/class/powercap/intel-rapl:0/energy_uj` (µJ
   counter, often root-only) covers the CPU package. Check the offsets in `drivers/gpu/drm/amd/include/kgd_pp_interface.h` for
   your kernel; the card index may differ.
2. **Record.** `tools/energy/record.py record --hz 20 --json out.json -- <cmd>` samples those files while `<cmd>` runs and
   integrates power over time (trapezoid rule). `tools/energy/record.py snapshot` prints one reading, including GPU clocks and
   activity. A short shell loop that reads `power1_average` every 50 ms and sums it works too.
3. **Take differential windows.** Run the workload at a short and a long length (N=64 and N=320 tokens, or 1 vs 6 repetitions),
   each window at least 30 s, and compute `(J_long - J_short) / (N_long - N_short)`. Load, warm-up and idle power cancel.
4. **Control the box.** Pin the GPU clock level (`echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level`,
   and write `auto` afterwards), stop other jobs, interleave A/B runs, repeat at least 3 times and report medians and the spread.
5. **Say what it is.** These are SoC socket readings from the SMU firmware, averaged over a short window. They are not wall power:
   DRAM devices, VRMs, fans, storage and PSU losses are not included. Compare runs measured the same way on the same box.

## 2. Headline rows

Set `M` to the model file for each block.

### Gemma 4 26B-A4B UD-Q4_K_M prefill (strix-rt 1637 tok/s, llama.cpp Vulkan 1719)

```sh
M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
# strix-rt, pp512, 5 reps (logs/g4/pp_sweep3.sh; the "mi1" rows are now the default, PF_QMM_MI=1 is kept explicit here)
for r in 1 2 3 4 5; do PF_QMM_MI=1 G4_QMMH=1 G4_PF=1 MAX_CTX=4096 ./target/release/g4_run $M rand:512 2 | grep prompt; done
# parity: g4_run <model> @prompt_ids 64 @llama_ref_ids with FORCED=1 prints teacher-forced agreement. The "26bc" chat prompt/reference
# id files used in pp_mi1.sh are not released yet, so make your own reference with llama.cpp (logs/g4/refchat.sh shows how).
# llama.cpp Vulkan (logs/g4/bench.sh <tag> $M llama)
~/llama-latest/build-vulkan/bin/llama-bench -m $M -ngl 999 -fa on -b 2048 -ub 512 -p 512,2048 -n 128 -r 3 -o json
```
Our logs: `logs/g4/pp_sweep3.log` (mi1 rows 1636.8-1639.3 after a cold first rep), `logs/g4/pp_mi1.log`, `logs/g4/rebench_llama.log` (1718.98 ± 6.22).

### Nemotron 3 Nano 30B-A3B Q4_K_M prefill (strix-rt pp511 1469 tok/s, llama.cpp Vulkan pp512 1687)

```sh
M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
# strix-rt (logs/nh/high_one.sh): 511-token prompt + 2 decode steps, 6 reps
PF=1 RUNS=6 ./target/release/nh $M "$(cat logs/nh/p512.ids)" 2 | grep '^prefill'
# llama.cpp Vulkan (flags as recorded in logs/nh/bench30b_v3/l_pp.txt: ngl 99, fa 1; the exact script is not in the repo)
~/llama-latest/build-vulkan/bin/llama-bench -m $M -ngl 99 -fa 1 -p 512 -n 0
```
Our logs: `logs/nh/high.log` (1468.3-1470.5), `logs/nh/bench30b_v3/l_pp.txt` (1687.08 ± 3.69, build e36df6047).

### Muse Glimmer 30B Q4_K_M decode energy (strix-rt 7.9-8.1 J/token socket; llama.cpp Vulkan 6.98, ROCm 8.12)

```sh
M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
# strix-rt, differential N=480-64 (logs/glimmer/sess_xs2.sh; GEMVQ_Q4P=5 is the default kernel, 1 is the variant without sum(x))
for r in 1 2 3; do GEMVQ_Q4P=5 tools/comp/energy480.sh xs5_r$r "./target/release/glm_run $M @logs/glimmer/p/0.ids @N@" | tail -1; done
# llama.cpp, differential N=320-64 (logs/glimmer/sess_energy.sh vulkan|rocm)
tools/comp/energy.sh glimmer_llama_vulkan "~/llama-latest/build-vulkan/bin/llama-bench -m $M -ngl 999 -fa 1 -p 0 -n @N@ -r 1"
# speed (logs/glimmer/sess_bench_ours.sh)
RUNS=3 ./target/release/glm_run $M @logs/glimmer/p/0.ids 129 | grep '^prompt'
```
Our logs: `logs/glimmer/xs2.log` (Q4P=5: 8.275 / 7.922 / 7.914, median 7.92), `logs/glimmer/energy_llama.log` (6.977 / 8.115),
`logs/glimmer/bench_ours.log` (13.34 tok/s). A later quiet-box session of the same default, interleaved with the Q6_K quad
kernel (`GEMVQ_Q4P=71`), measured 8.095 / 8.108 / 8.102 / 8.097 J/token at 12.95 tok/s (docs/glimmer.md, Q6_K quad section;
speed and power samples in `logs/glimmer/q6s.log`, `q6ab.log`; the per-run energy JSONs from that session are not in this
snapshot). So the default is 7.9-8.1 J/token, +13.5% to +16% above llama.cpp Vulkan. Note that our differential window is 416 tokens and llama.cpp's is 256.
Both cancel load and warm-up, but they are not the same window.

### Gemma 4 26B-A4B decode energy (gfx 0.685 vs 0.569 J/token)

```sh
M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
bash logs/g4/bench.sh 26b3 $M energy      # runs tools/comp/energy.sh for g4_run and llama-bench in one session (logs/g4/job_en3.sh)
```
Our logs: `logs/g4/bench_26b3_energy.log`; earlier sessions `bench_26b_energy.log`, `bench_26b2_energy.log` show how much the socket
channel moves between sessions.

### Nemotron 3.5 Super VL image preprocessing parity (bit-exact on 6/6; answers equal to EOS on 6/6)

```sh
D=$MODELS/new/nemotron35vl          # GGUF + mmproj; mmproj_st/ is made by crates/mcv/ref/mmproj2st.py
# preprocess + ViT, compared with llama.cpp mtmd dumps of the same image
./target/release/n35img $D/mmproj_st <image.png|jpg> imgref/<name>.px imgref/<name>.emb
# end to end: preprocess -> ViT -> prefill injection -> greedy answer (logs/n35/sess_vl.sh builds the ids with llama-tokenize)
bash logs/n35/sess_vl.sh <image.png|jpg>
# JPEG decoder vs stb_image, on a corpus dumped by logs/n35/jpgtest/stbdump.c
./target/release/stbjpg_cmp <corpus dir> <stb dump dir>
```
The references come from `llama-mtmd-cli` with a small **local patch** that dumps the preprocessed pixels and the ViT output
(`MTMD_DUMP_PX`, `MTMD_DUMP_EMB`; formats in the header of `crates/nemotron35/src/bin/n35img.rs`). That patch is not in this repo
yet. The six test images are not included either (one was generated by our own `qi` text-to-image port, and the others are a
video frame and resized copies; see the table in [docs/new-models.md](docs/new-models.md)). Our logs: `logs/n35/imgparity3.log`, `logs/n35/imgvl2.log`.

## 3. Failed experiments (rerun them)

All of these are opt-in switches on the default binaries:

| Experiment | Switch | Doc |
|---|---|---|
| MMVQ-style integer-dot GEMV, Glimmer | `GLM_IDOT=1`; pair int-dot `GEMVQ_Q4P=13` (per-32), `29` (per-16), `45` (per-8) | docs/glimmer.md |
| Hybrid 32/16-row expert tiles, Nemotron Nano | `PFG_HYB=16` or `8` (`logs/nh/hyb_one.sh`) | docs/prefill.md |
| Persistent work queue, Gemma prefill | `PF_QH_PERS=80/160/240/480` (`logs/g4/pers_one.sh`) | docs/prefill.md |
| Fused gate+up+SwiGLU, Glimmer decode | `GLM_FUSE_FFN=1` (`logs/glimmer/sess_fuse.sh`) | docs/glimmer.md |
| Q6_K quad kernel (ffn down, lm_head), Glimmer decode | `GEMVQ_Q4P=71` (`logs/glimmer/sess_q6q.sh`, `sess_q6s.sh`) | docs/glimmer.md |

## 4. Everything else

Each doc in `docs/` names the log and usually the session script behind every number. `PROGRESS.md` is the day-by-day record.
A few scripts reference tools or reference setups that are not in this repo (PyTorch/diffusers reference environments, the llama.cpp
mtmd dump patch, vLLM containers); the docs say so where it matters.
