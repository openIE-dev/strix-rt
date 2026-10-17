#!/bin/bash
# ab_loop.sh PREFIX VA VB NPAIRS : wait for a quiet box (outside the GPU lock), run an A/B pair, keep it only if both runs decode >= 13.0 tok/s (others were not contending); until NPAIRS kept
cd ~/strix-rt; P=$1; kept=0; try=0
while [ $kept -lt $4 ] && [ $try -lt 40 ]; do try=$((try+1))
  logs/wait_quiet.sh true > /dev/null
  out=$(GPU_LOCK_TAG="glimmer A/B $P" logs/gpu_lock.sh tools/comp/perf_high.sh logs/glimmer/pair_ab.sh ${P}_r$try $2 $3 2>&1)
  ta=$(echo "$out" | grep "TOKS q$2" | grep -o "[0-9.]* tok" | cut -d" " -f1); tb=$(echo "$out" | grep "TOKS q$3" | grep -o "[0-9.]* tok" | cut -d" " -f1)
  if awk "BEGIN{exit !($ta>=13.0 && $tb>=13.0)}"; then kept=$((kept+1)); echo "KEEP try$try"; echo "$out" | grep ENERGY; else echo "DISCARD try$try toks $ta $tb"; fi
done
