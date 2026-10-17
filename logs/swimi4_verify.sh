#!/bin/bash
cd ~/strix-rt; D=/sys/class/drm/card1/device; echo high | sudo tee $D/power_dpm_force_performance_level
for L in 512 2048 2048; do echo "== pf_q38 $L"; target/release/pf_q38 $MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf $L 2>&1 | grep -E "greedy|tok/s"; done
echo auto | sudo tee $D/power_dpm_force_performance_level
logs/qi1024_run.sh
