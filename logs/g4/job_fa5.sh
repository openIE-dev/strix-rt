#!/bin/bash
# g4_attnfa512 (global layers) prefill: greedy continuation after a 2000-token real-text prompt, FA on vs off (E4B, 26B); pp2048 timings; short forced parity
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; E=$MODELS/new/gemma4/e4b/gemma-4-E4B-it-Q8_0.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-fa5 logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for mm in $E $M; do for fa in 0 1; do echo \"== \$(basename \$mm) FA5=\$fa\"; G4_PF=1 G4_PF_FA5=\$fa MAX_CTX=4096 timeout 40 ./target/release/g4_run \$mm @logs/g4/long_ids.txt 24 | grep -E '^ids|prompt' | cut -c1-260; done; done
for mm in $E $M; do echo -n \"pp2048 \$(basename \$mm): \"; G4_PF=1 MAX_CTX=4096 timeout 40 ./target/release/g4_run \$mm rand:2048 2 | grep -o 'prompt.*s)'; done
G4_PF=1 FORCED=1 timeout 60 ./target/release/g4_run $E @logs/g4/e4b_prompt_ids.txt 64 @logs/g4/e4b_cont_ids.txt | grep FORCED | cut -c1-200"
