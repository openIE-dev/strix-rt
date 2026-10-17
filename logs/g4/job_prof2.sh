#!/bin/bash
# 26B decode: marginal cost of small kernels on top of skip 31 (all GEMV/attn skipped)
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-prof2 logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for s in 31 63 95 159 287 543 1023 8; do echo -n \"skip \$s: \"; G4_SKIP=\$s timeout 30 ./target/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 129 | grep -o 'decode 128.*tok/s |' ; done"
