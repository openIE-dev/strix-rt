#!/bin/bash
# GPU power breakdown during 26B decode: ours (full / no experts / no dense / no lm_head) vs llama.cpp Vulkan tg
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; B=~/llama-latest/build-vulkan/bin; S=logs/g4/esample.sh
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-epw logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for s in 0 1 2 8; do G4_SKIP=\$s bash $S ours_s\$s ./target/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 513; done
bash $S llama $B/llama-bench -m $M -ngl 999 -fa on -p 0 -n 512 -r 1"
