#!/bin/bash
# energy re-measure after G4_SPEC/G4_IDOT: 26B and E4B, ours vs llama.cpp Vulkan (one lock session each)
cd ~/strix-rt; E=$MODELS/new/gemma4/e4b/gemma-4-E4B-it-Q8_0.gguf; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for job in "26b3 $M" "e4b3 $E"; do set -- $job
  logs/wait_quiet.sh; GPU_LOCK_TAG=g4-en-$1 logs/gpu_lock.sh tools/comp/perf_high.sh bash logs/g4/bench.sh $1 $2 energy > logs/g4/bench_$1_energy.log 2>&1; done
echo alldone
