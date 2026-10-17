#!/bin/bash
# strix-serve round 2: balanced gemv groups; eq recheck, step times, q38 concurrency with PF_M=1024 (one prefill chunk) and no prefix-cache save
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
GPU_LOCK_TAG="strix-serve eq2" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 48 > logs/serve/eq_q38_r2.log 2>&1
GPU_LOCK_TAG=serve_steps2 logs/gpu_lock.sh env ZEROCOPY=dmabuf timeout 120 target-serve/release/serve_steps $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 600 > logs/serve/steps_q38_r2.log 2>&1
PF_M=1024 $R serve $S strix_q38_r2 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048 --cache 0
echo BATCH_STRIX2_DONE
