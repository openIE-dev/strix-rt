#!/bin/sh
cd ~/strix-rt; D=/sys/class/drm/card1/device; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
. logs/pf_busy.sh; waitidle; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
run() { l=$1; shift; waitidle; echo "pp$l $* : $(env "$@" PF_M=$l REPS=5 NOREF=1 ./target/release/pf_lfm $L $l 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; }
for r in 1 2; do run 2048; run 512; run 2048 PF_WIDE_RPE=100000 PF_G32N=256 PF_GBM2=32; done
waitidle; echo "check 2048: $(PF_M=2048 REPS=1 ./target/release/pf_lfm $L 2048 8 2>&1 | grep "^  pf")"
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level)"
