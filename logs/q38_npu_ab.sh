#!/bin/bash
cd ~/strix-rt
D=/sys/class/drm/card1/device; echo high | sudo tee $D/power_dpm_force_performance_level
trap "echo auto | sudo tee $D/power_dpm_force_performance_level" EXIT
for e in "" "NPU_SPLIT=0 NPU_TB=0" "" "NPU_SPLIT=0 NPU_TB=0"; do echo "== [$e] pf_q38 2048"; env $e target/release/pf_q38 $MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf 2048 2>&1 | grep -E "pp2048|greedy|match|/16"; done
