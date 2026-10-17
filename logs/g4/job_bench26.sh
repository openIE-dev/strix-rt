#!/bin/bash
# 26B pp512/pp2048/tg128 ours + llama.cpp Vulkan + energy; one lock session per mode
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for mode in ours llama energy; do
  logs/wait_quiet.sh
  GPU_LOCK_TAG=g4-bench26-$mode logs/gpu_lock.sh tools/comp/perf_high.sh bash logs/g4/bench.sh 26b $M $mode > logs/g4/bench_26b_$mode.log 2>&1
done
echo alldone
