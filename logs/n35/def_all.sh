#!/bin/bash
cd ~/strix-rt
g() { GPU_LOCK_WAIT=900 logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/perf_high.sh bash logs/n35/sess_def.sh $1 $2 9>&- 2>&1 | grep -v "quiet\|perf level"; }
for f in p30 p90 div30 p200 long_prompt p512; do echo "== $f 128"; g $f 128; done
echo "== p30 512"; g p30 512
echo "== p200 512"; g p200 512
echo DONE
