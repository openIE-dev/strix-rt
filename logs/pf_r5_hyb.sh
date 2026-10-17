#!/bin/sh
# Round 5 lfm2: SKIP-profile + hybrid 64/32 expert tiles (PF_HYB) A/B, perf level high; binary from target-r5
trap "echo SIGNAL TERM $(date +%T)" TERM; trap "echo SIGNAL HUP" HUP; trap "echo SIGNAL INT" INT
cd ~/strix-rt; D=/sys/class/drm/card1/device; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; B=./target-r5/release/pf_lfm
. logs/pf_busy.sh
waitidle; echo "[start $(date +%T)]"
echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
for h in 0 1; do echo "== check PF_HYB=$h"; waitidle; PF_HYB=$h PF_GBM2=64 REPS=2 $B $L 512 16 2>&1 | grep -E "top5 ref|  pf|rror"; echo "exit $?"; done
run() { waitidle; echo "$* : $(env "$@" REPS=6 NOREF=1 $B $L 512 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; }
for s in 1 16; do run PF_GBM2=64 PF_SKIP=$s; done
for r in 1 2; do run PF_GBM2=64; run PF_HYB=1; run PF_HYB=1 PF_MSWI=1; run PF_HYB=1 PF_LPS=24; run PF_GBM2=64 PF_LPS=24; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level)"
