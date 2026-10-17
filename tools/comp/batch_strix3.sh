#!/bin/bash
# strix-serve round 3: chunked prefill (SERVE_CHUNK=256) interleaved with decode; eq recheck, then q38 + Bonsai concurrency
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
GPU_LOCK_TAG="strix-serve eq3" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 48 > logs/serve/eq_q38_r3.log 2>&1
$R serve $S strix_q38_r3 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
$R serve $S strix_bonsai_r3 strix $MD/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf --slots 8 --ctx 2048
echo BATCH_STRIX3_DONE
