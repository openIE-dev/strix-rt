#!/bin/bash
# multi-workgroup router logits: 26B tg128, greedy + forced parity (token and prefill)
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-rlog logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for r in 0 1; do echo -n \"G4_RLOG=\$r: \"; G4_RLOG=\$r RUNS=2 timeout 30 ./target/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 129 | grep -o 'decode 128.*tok/s |' | tail -n 1; done
RC=vk bash logs/g4/par.sh 26bc $M 64 | grep -E 'parity|FORCED' | cut -c1-200
G4_PF=1 FORCED=1 timeout 60 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt | grep FORCED | cut -c1-200
echo -n 'pp2048: '; G4_PF=1 MAX_CTX=4096 timeout 40 ./target/release/g4_run $M rand:2048 2 | grep -o 'prompt.*s)'"
