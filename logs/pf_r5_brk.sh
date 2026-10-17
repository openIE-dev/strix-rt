#!/bin/sh
# Round 5 lfm2 breakdown: gather (SKIP 32) / swiglu (SKIP 64) cost, MSWI with GBM2=64, perf level high
cd ~/strix-rt; D=/sys/class/drm/card1/device; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; B=./target-r5/release/pf_lfm
. logs/pf_busy.sh
waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
run() { waitidle; echo "$* : $(env "$@" REPS=6 NOREF=1 $B $L 512 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; }
X="PF_GBM2=64 PF_LPS=24"
for r in 1 2; do run $X; run $X PF_SKIP=32; run $X PF_SKIP=64; run $X PF_SKIP=16; run $X PF_MSWI=1; run $X PF_SKIP=8; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level)"
