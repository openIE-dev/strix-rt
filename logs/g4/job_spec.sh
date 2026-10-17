#!/bin/bash
# (a) 26B decode power: MoE GEMV format-specialized + shared x (G4_SPEC=1) vs generic (0); (b) G4_QMMH=1 packed-LDS prefill parity + pp
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; S=logs/g4/esample.sh
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-spec logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for s in 0 1; do G4_SPEC=\$s bash $S spec\$s ./target/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 513; done
G4_PF=1 FORCED=1 timeout 60 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt | grep FORCED | cut -c1-160"
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-qmmh logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
G4_QMMH=1 G4_PF=1 FORCED=1 timeout 60 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt | grep FORCED | cut -c1-160
for L in 512 2048; do G4_QMMH=1 G4_PF=1 MAX_CTX=4096 timeout 60 ./target/release/g4_run $M rand:\$L 2 | grep prompt | sed \"s/^/qmmh pp\$L /\"; done"
echo alldone
