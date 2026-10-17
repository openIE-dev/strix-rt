#!/bin/sh
# Round 5 final (one session, perf high): q38 + lfm2 llama.cpp vs strix-rt before/after, pp512/pp2048; lfm2 greedy check
cd ~/strix-rt; D=/sys/class/drm/card1/device; LB=~/llama.cpp/build-vulkan/bin/llama-bench; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
BEFORE="PF_KS=1 PF_MSEL=0 PF_GBM2=32 PF_MSWI=0 PF_LPS=8 PF_G32N=256 PF_WIDE_RPE=100000"
. logs/pf_busy.sh; waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
run() { b=$1; m=$2; l=$3; shift 3; waitidle; echo "$b pp$l $* : $(env "$@" PF_M=$l REPS=6 NOREF=1 ./target/release/$b $m $l 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; }
for m in $Q $L; do waitidle; $LB -m $m -ngl 99 -p 512,2048 -n 0 -r 5 2>&1 | grep -E "pp(512|2048)"; echo "[busy after: $(busy && echo YES || echo no)]"; done
for l in 512 2048; do run pf_q38 $Q $l $BEFORE; run pf_q38 $Q $l; run pf_lfm $L $l $BEFORE; run pf_lfm $L $l; done
for l in 512 2048; do for v in before after; do waitidle; e=""; [ $v = before ] && e="$BEFORE"
  echo "lfm2 $v L=$l: $(env $e PF_M=$l REPS=1 ./target/release/pf_lfm $L $l 8 2>&1 | grep -E "^  pf" )"; done; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level) [end $(date +%T)]"
