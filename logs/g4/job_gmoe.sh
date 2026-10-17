#!/bin/bash
# 26B grouped-MoE prefill: forced parity after 512-chunk prefill, then pp512/pp2048
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-gmoe logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
G4_PF=1 FORCED=1 timeout 60 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt | grep -v ^weights
for L in 512 2048; do G4_PF=1 MAX_CTX=4096 timeout 60 ./target/release/g4_run $M rand:\$L 2 | grep prompt | sed \"s/^/pp\$L /\"; done"
