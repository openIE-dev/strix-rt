#!/bin/bash
# decode: norm workgroup size (G4_NORM_WG) effect, 26B and E4B
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; E=$MODELS/new/gemma4/e4b/gemma-4-E4B-it-Q8_0.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-norm logs/gpu_lock.sh tools/comp/perf_high.sh bash -c "
for mm in $M $E; do for wg in 1024 256 64; do for s in 0 31; do echo -n \"\$(basename \$mm) wg \$wg skip \$s: \"; G4_NORM_WG=\$wg G4_SKIP=\$s timeout 30 ./target/release/g4_run \$mm @logs/g4/e4b_prompt_ids.txt 129 | grep -o 'decode 128.*tok/s |' ; done; done; done
G4_NORM_WG=256 timeout 30 ./target/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt | grep parity"
