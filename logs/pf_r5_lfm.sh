#!/bin/sh
# Round 5 lfm2: fused select+sort (PF_MSEL) correctness + tile/fusion sweep at perf level high
cd ~/strix-rt; D=/sys/class/drm/card1/device; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
busy() { pgrep -f "release/(q38|qtest|lfm|npu_|tern_|bench|pf_)|npu/target[a-z_]*/release|llama-bench|llama-cli|build-.*/bin/" >/dev/null; }
w=0; while busy; do sleep 5; w=$((w+5)); done; echo "[waited ${w}s; start $(date +%T)]"
echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
for ms in 0 1; do echo "== check PF_MSEL=$ms"; PF_MSEL=$ms REPS=2 ./target/release/pf_lfm $L 512 16 2>&1 | grep -E "top5|argmax|  pf"; done
run() { echo "$* : $(env "$@" REPS=6 NOREF=1 ./target/release/pf_lfm $L 512 1 2>&1 | grep ^prefill)"; }
for r in 1 2; do
run PF_MSEL=0; run PF_MSEL=1; run PF_MSEL=1 PF_MSWI=1; run PF_MSEL=1 PF_GBM=64 PF_GBM2=64; run PF_MSEL=1 PF_GBM=64; run PF_MSEL=1 PF_GBM2=64; run PF_MSEL=1 PF_GBM=64 PF_MSWI=1; run PF_MSEL=1 PF_LPS=24
done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level); other GPU job at end: $(busy && echo YES || echo no)"
