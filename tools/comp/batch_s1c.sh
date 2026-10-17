#!/bin/bash
# strix-serve1 r2: deterministic LFM norms + greedy steps of all slots in one submission
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh
GPU_LOCK_TAG="s1c eq lfm" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve1_eq.sh $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 8 48 --slots 8 --ctx 4096 > logs/serve/eq1_lfm_r2.log 2>&1
GPU_LOCK_TAG="s1c eq mcv" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve1_eq.sh $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf 8 48 --slots 8 --ctx 4096 > logs/serve/eq1_mcv_r2.log 2>&1
$R serve $S strix1_lfm_r2 strix1 $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf --slots 8 --ctx 4096
$R serve $S strix1_mcv_r2 strix1 $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf --slots 8 --ctx 4096
echo BATCH_S1C_DONE
