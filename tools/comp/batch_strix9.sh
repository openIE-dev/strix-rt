#!/bin/bash
# round 8 after the fan curve (f862ce7): strix-serve r6b settings (NB 8, PF_PASSES 1) eq + bench, then llama.cpp Vulkan rerun
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
GPU_LOCK_TAG="strix-serve eq8" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 48 > logs/serve/eq_q38_r8.log 2>&1
$R serve $S strix_q38_r8 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
$R serve tools/comp/serve_bench.sh llamavk_q38_fan llama-vk $MD/Qwen3.8-27B-UD-Q4_K_M.gguf -np 8 -c 16384
echo BATCH_STRIX9_DONE
