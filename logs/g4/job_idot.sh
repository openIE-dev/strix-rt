#!/bin/bash
# 26B decode: Q4_K gate_up through int8 dot (G4_IDOT=1, q8 x per workgroup) vs float q_any (0): power, speed, greedy + forced parity
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; S=logs/g4/esample.sh
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-idot logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for s in 0 1; do G4_IDOT=\$s bash $S idot\$s ./target/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 513; done
G4_PF=1 timeout 60 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt | grep parity
G4_PF=1 FORCED=1 timeout 60 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt | grep FORCED | cut -c1-160"
echo alldone
