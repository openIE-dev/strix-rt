#!/bin/sh
# Round 5 lfm2 tile knobs on top of GBM2=64 LPS=24 MSWI=1 (perf level high). Correctness check first.
cd ~/strix-rt; D=/sys/class/drm/card1/device; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; B=./target-r5/release/pf_lfm
. logs/pf_busy.sh
waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
X="PF_GBM2=64 PF_LPS=24 PF_MSWI=1"
for v in "PF_MSWI=0" "$X" "$X PF_G32N=128"; do waitidle; echo "== check $v"; env $v REPS=2 $B $L 512 16 2>&1 | grep -E "top5 ref|  pf"; done
run() { waitidle; echo "$* : $(env "$@" REPS=6 NOREF=1 $B $L 512 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; }
for r in 1 2; do run $X; run $X PF_G32N=128; run $X PF_MINWG=64; run $X PF_MINWG=128; run $X PF_G32N=128 PF_MINWG=128; run $X PF_LPS=40; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level)"
