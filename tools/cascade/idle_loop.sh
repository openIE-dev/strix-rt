#!/bin/bash
# Retry idle windows until 3 clean of each (or 4 h). Takes the GPU lock per attempt only.
cd ~/strix-rt; D=logs/cascade/floor; end=$(( $(date +%s) + 14400 ))
while [ $(date +%s) -lt $end ]; do
  for t in none srv; do n=$(ls $D/idle_${t}_c*.json 2>/dev/null | wc -l); [ $n -lt 3 ] && GPU_LOCK_TAG="idle $t" logs/gpu_lock.sh tools/cascade/idle_win.sh $t $((n+1)); done
  [ $(ls $D/idle_*_c*.json 2>/dev/null | wc -l) -ge 6 ] && { echo IDLE CLEAN DONE; exit 0; }
done; echo IDLE TIMEOUT
