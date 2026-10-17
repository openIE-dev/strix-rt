#!/bin/sh
# LM decode gemv tunables sweep (KT_REPS=4, warm reps), GPU-idle gated, perf level high
cd ~/strix-rt; D=/sys/class/drm/card1/device; G=$MODELS/kitten2/hf/cpp/model-tq2_1.gguf
busy() { pgrep -f "logs/r11/pfq|release/(pf_|npu_|tern_|q38|lfm)|npu/target/release|llama-bench|llama-cli|build-.*/bin/llama" >/dev/null; }
w=0; while busy; do sleep 5; w=$((w+5)); done; echo "[waited ${w}s; start $(date +%T)]"
echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
for r in 1 2; do
for cfg in "X=1" "GEMV_TERN=4,64" "GEMV_TERN=16,64" "GEMV_TERN=8,128" "GEMV_TERN=8,32" "GEMVQ_NR=2" "GEMVQ_NR=4" "ATTN_NC=8" "ATTN_NC=32"; do
  echo "$cfg: $(env $cfg KT_REPS=4 ./target/release/kt $G logs/kt/bruno logs/kt/prompt.json 2>&1 >/tmp/sw.json | grep ^rep | awk '{print $5}' | sort | sed -n 2p) same=$(cmp -s /tmp/sw.json /tmp/kt_b1.json && echo y || echo n)"
done; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null
echo "restored $(cat $D/power_dpm_force_performance_level); busy at end: $(busy && echo YES || echo no) $(date +%T)"
