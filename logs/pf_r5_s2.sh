#!/bin/sh
# Round 5 rerun: q38 before/after (session run was contended) + lfm2 pp2048 knob sweep, perf high
cd ~/strix-rt; D=/sys/class/drm/card1/device; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
. logs/pf_busy.sh; waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
run() { b=$1; m=$2; l=$3; shift 3; waitidle; echo "$b pp$l $* : $(env "$@" PF_M=$l REPS=5 NOREF=1 ./target/release/$b $m $l 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; }
for r in 1 2; do run pf_q38 $Q 512 PF_KS=1; run pf_q38 $Q 512; run pf_q38 $Q 2048 PF_KS=1; run pf_q38 $Q 2048; done
for v in "" PF_G32N=256 PF_GBM2=32 PF_MSWI=0 PF_LPS=8 PF_KS=1 "PF_G32N=256 PF_GBM2=32"; do run pf_lfm $L 2048 $v; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level) [end $(date +%T)]"
