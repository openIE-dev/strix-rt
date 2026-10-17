#!/bin/bash
# final numbers: bench ours for E4B and 26B, 26B energy; one lock session each
cd ~/strix-rt; E=$MODELS/new/gemma4/e4b/gemma-4-E4B-it-Q8_0.gguf; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for job in "e4b2 $E ours" "26b2 $M ours" "26b2 $M energy"; do set -- $job
  logs/wait_quiet.sh; GPU_LOCK_TAG=g4-final-$1-$3 logs/gpu_lock.sh tools/comp/perf_high.sh bash logs/g4/bench.sh $1 $2 $3 > logs/g4/bench_$1_$3.log 2>&1; done
echo alldone
