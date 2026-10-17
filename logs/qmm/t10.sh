#!/bin/bash
cd ~/strix-rt
for rep in 1 2 3 4 5; do for c in "2 8" "2 2" "1 4"; do set -- $c
  echo "## NJ=$1 KD=$2 rep=$rep"
  GPU_LOCK_WAIT=1800 logs/wait_quiet.sh logs/gpu_lock.sh bash logs/qmm/t10_one.sh $1 $2 9>&-
done; done; echo DONE
