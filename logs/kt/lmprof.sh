#!/bin/sh
# wait for idle GPU (bounded), then KT_LMPROF segment profile of one LM step
cd ~/strix-rt; G=$MODELS/kitten2/hf/cpp/model-tq2_1.gguf
busy() { pgrep -f "logs/r11/pfq|release/(pf_|npu_|tern_|q38|lfm)|npu/target/release|llama-bench|llama-cli|build-.*/bin/llama" >/dev/null; }
w=0; while busy && [ $w -lt ${MAXWAIT:-900} ]; do sleep 2; w=$((w+2)); done; echo "[waited ${w}s busy=$(busy && echo YES || echo no)]"
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null
env "$@" KT_REPS=3 KT_LMPROF=1 ./target/release/kt $G logs/kt/bruno logs/kt/prompt.json 2>&1 >/tmp/lmprof.json | grep -v "^weights"
echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null
echo "[busy end=$(busy && echo YES || echo no)]"
