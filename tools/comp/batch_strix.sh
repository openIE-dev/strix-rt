#!/bin/bash
# strix-serve: Bonsai eq check, then Soothill concurrency C=1,2,4,8 for Qwen3.8 and Bonsai 2 PQ2_0
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
GPU_LOCK_TAG="strix-serve eq bonsai" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf 8 48 > logs/serve/eq_bonsai.log 2>&1
$R serve $S strix_q38 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
$R serve $S strix_bonsai strix $MD/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf --slots 8 --ctx 2048
echo BATCH_STRIX_DONE
