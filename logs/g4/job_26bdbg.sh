#!/bin/bash
# 26B layer-0 intermediates (attn_out, ffn_mlp, mlp+moe, routes) at the last prompt position vs logs/g4/26b_evalcb_cpu.txt
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-26bdbg logs/gpu_lock.sh bash -c "
for p in 1 2 3; do echo PART \$p; DUMPX=1 DBG_LAYERS=1 DBG_PART=\$p timeout 40 ./target/release/g4_run $M @logs/g4/26b_prompt_ids.txt 1; done
echo L1; DUMPX=1 DBG_LAYERS=1 timeout 40 ./target/release/g4_run $M @logs/g4/26b_prompt_ids.txt 1"
