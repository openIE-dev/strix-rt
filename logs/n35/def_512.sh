#!/bin/bash
cd ~/strix-rt
for f in p30 p200; do echo "== $f 512"; GPU_LOCK_WAIT=900 logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/perf_high.sh bash logs/n35/sess_def.sh $f 512 9>&- 2>&1 | grep -v "quiet\|perf level"; done; echo DONE
