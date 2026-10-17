#!/bin/sh
# Round 5 clean single session (perf level high, restored to auto): pp512 / pp2048 for all models, llama.cpp vs strix-rt
# "before" (round-4 settings via env: no K-slab, separate rsel, 32-row tiles, unfused SwiGLU, 8 layers/submit) and "after" (defaults).
# Every run waits for GPU idle (logs/pf_busy.sh); strix-rt reports median of the reps after dropping the first.
cd ~/strix-rt; D=/sys/class/drm/card1/device; LB=~/llama.cpp/build-vulkan/bin/llama-bench; FB=~/src/prism-llama/build-vulkan/bin/llama-bench
Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; P2=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; P1=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
. logs/pf_busy.sh
BEFORE="PF_KS=1 PF_MSEL=0 PF_GBM2=32 PF_MSWI=0 PF_LPS=8 PF_G32N=256"
waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "perf level $(cat $D/power_dpm_force_performance_level)"
med() { tr -d "[]," | awk "{n=0; for(i=2;i<=NF;i++){a[n++]=\$i}; asort(a); m=(n%2)?a[int(n/2)+1]:(a[n/2]+a[n/2+1])/2; printf \"median %.1f ms -> %.2f tok/s (reps %s)\", m, $1*1000/m, \$0}"; }
st() { tag=$1; bin=$2; m=$3; l=$4; shift 4; waitidle
  line=$(env "$@" PF_M=$l REPS=6 NOREF=1 ./target/release/$bin $m $l 1 2>&1 | grep ^prefill)
  echo "strix $tag pp$l: $(echo "$line" | sed "s/.*all \[\([^]]*\)\].*/\1/" | cut -d, -f2- | med $l) | $line [busy after: $(busy && echo YES || echo no)]"; }
ll() { waitidle; echo "llama $1:"; $2 -m $3 -ngl 99 -p 512,2048 -n 0 -r 5 2>&1 | grep -E "pp(512|2048)"; echo "[busy after: $(busy && echo YES || echo no)]"; }
for name in q38 lfm2 pq2 ptq1; do
  case $name in q38) m=$Q; b=pf_q38; lb=$LB;; lfm2) m=$L; b=pf_lfm; lb=$LB;; pq2) m=$P2; b=pf_q38; lb=$FB;; ptq1) m=$P1; b=pf_q38; lb=$FB;; esac
  echo "=== $name"; ll $name $lb $m
  for l in 512 2048; do st before $b $m $l $BEFORE; st after $b $m $l; done
done
echo "=== lfm2 greedy check (pf before vs after, L=512 and 2048)"
for l in 512 2048; do for v in before after; do waitidle; e=""; [ $v = before ] && e="$BEFORE"
  echo "$v L=$l: $(env $e PF_M=$l REPS=1 ./target/release/pf_lfm $L $l 8 2>&1 | grep -E "^  pf|argmax" | tr "\n" " ")"; done; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored perf level: $(cat $D/power_dpm_force_performance_level) [end $(date +%T)]"
