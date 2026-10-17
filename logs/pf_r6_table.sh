#!/bin/sh
# Round 6 final table, one session at perf level high: llama.cpp (Bonsai: prism fork) vs strix-rt defaults, pp512 / pp2048
cd ~/strix-rt; D=/sys/class/drm/card1/device; LB=~/llama.cpp/build-vulkan/bin/llama-bench; FB=~/src/prism-llama/build-vulkan/bin/llama-bench
Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; P2=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; P1=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
. logs/pf_busy.sh; waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
for mb in "q38 pf_q38 $Q $LB" "lfm2 pf_lfm $L $LB" "pq2 pf_q38 $P2 $FB" "ptq1 pf_q38 $P1 $FB"; do set -- $mb
  waitidle; echo "== $1 llama:"; $4 -m $3 -ngl 99 -p 512,2048 -n 0 -r 5 2>&1 | grep -E "pp(512|2048)"; echo "[busy after: $(busy && echo YES || echo no)]"
  for l in 512 2048; do waitidle; echo "$1 strix pp$l: $(PF_M=$l REPS=6 NOREF=1 ./target/release/$2 $3 $l 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; done; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level) [end $(date +%T)]"
