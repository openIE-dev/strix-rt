#!/bin/bash
# decode skip-profile (26B, E4B) and prefill attention share at pp2048
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; E=$MODELS/new/gemma4/e4b/gemma-4-E4B-it-Q8_0.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-prof logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for s in 0 1 2 4 8 16 31; do echo -n \"26b skip \$s: \"; G4_SKIP=\$s timeout 30 ./target/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 129 | grep -o 'decode 128.*' ; done
for s in 0 4; do echo -n \"26b pp2048 pfskip \$s: \"; G4_PF=1 G4_PF_SKIP=\$s MAX_CTX=4096 timeout 40 ./target/release/g4_run $M rand:2048 2 | grep -o 'prompt.*s)'; done
for s in 0 4; do echo -n \"e4b pp2048 pfskip \$s: \"; G4_PF=1 G4_PF_SKIP=\$s MAX_CTX=4096 timeout 40 ./target/release/g4_run $E rand:2048 2 | grep -o 'prompt.*s)'; done"
