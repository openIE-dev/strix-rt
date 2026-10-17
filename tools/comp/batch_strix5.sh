#!/bin/bash
# strix-serve round 5: packed prefill, 512-row passes; eq recheck, then q38 + Bonsai concurrency
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
GPU_LOCK_TAG="strix-serve eq5" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 48 > logs/serve/eq_q38_r5.log 2>&1
$R serve $S strix_q38_r5 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
echo BATCH_STRIX5_DONE
