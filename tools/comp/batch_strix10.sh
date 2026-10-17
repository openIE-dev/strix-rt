#!/bin/bash
# round 9: r8 settings + GEMV lanes/2 with 2x workgroup rows (decode power sweep: lower J/token at m=1..4)
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
export GEMV_PROJ=8,128 GEMV_OUT=16,256 GEMV_SWIGLU=16,256 GEMV_DOWN=8,128
GPU_LOCK_TAG="strix-serve eq9" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 48 > logs/serve/eq_q38_r9.log 2>&1
$R serve $S strix_q38_r9 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
echo BATCH_STRIX10_DONE
