#!/bin/bash
# ROCm 7.13 llama-bench rerun, short session (pp512, pp2048, tg128; -r 3), perf level high
source ~/strix-rt/tools/comp/rocm713.env; cd ~/strix-rt
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
for m in LFM2.5-24B-A2B-Q4_K_M.gguf Qwen3.8-27B-UD-Q4_K_M.gguf; do echo "== $m"
timeout 140 ~/llama-latest/build-rocm713/bin/llama-bench -m $MODELS/$m -ngl 999 -fa on -b 2048 -ub 512 -p 512,2048 -n 128 -r 3 2>&1 | grep -E "pp|tg"; done
