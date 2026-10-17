#!/bin/sh
cd ~/strix-rt; D=/sys/class/drm/card1/device; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
. logs/pf_busy.sh
w=0; while busy; do sleep 5; w=$((w+5)); done; echo "[waited ${w}s; start $(date +%T)]"
echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
run() { waitidle; echo "$* : $(env "$@" REPS=5 NOREF=1 ./target/release/pf_lfm $L 512 1 2>&1 | grep ^prefill)"; }
for s in 0 1 4 8 16 29; do run PF_GBM2=64 PF_SKIP=$s; done
PF_GBM2=64 PF_LPS=1 PF_PROF=1 REPS=1 NOREF=1 ./target/release/pf_lfm $L 512 1 2>&1 | grep layers
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level); busy at end: $(busy && echo YES || echo no)"
