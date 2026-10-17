#!/bin/bash
# 26B parity after router fix: token-by-token greedy+forced, then 512-chunk prefill forced; layer-0 routes
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-26bpar2 logs/gpu_lock.sh bash -c "
DUMPX=1 DBG_LAYERS=1 DBG_PART=3 timeout 40 ./target/release/g4_run $M @logs/g4/26b_prompt_ids.txt 1 | grep ROUTE
RC=vk bash logs/g4/par.sh 26bc $M 64
echo PF; G4_PF=1 FORCED=1 RC=vk timeout 60 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt"
