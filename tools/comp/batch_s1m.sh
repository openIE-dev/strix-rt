#!/bin/bash
# strix-serve1 batched decode (record_multi): lfm steps, eq lfm, then C=1/2/4/8 per-level gpu_lock sessions for both models
cd ~/strix-rt; S=tools/comp/serve_bench_strix.sh
GPU_LOCK_TAG=s1steps-lfm logs/gpu_lock.sh bash -c "ZEROCOPY=dmabuf LFM_DET=1 tools/comp/perf_high.sh timeout 150 target-serve/release/s1_steps $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 8" > logs/serve/s1_steps_lfm.log 2>&1
GPU_LOCK_TAG="s1m eq lfm" logs/gpu_lock.sh tools/comp/serve1_eq.sh $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 8 48 --slots 8 --ctx 4096 > logs/serve/eq1_lfm_m1.log 2>&1
tools/comp/serve_split.sh $S strix1_mcv_m1 strix1 $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf --slots 8 --ctx 4096
tools/comp/serve_split.sh $S strix1_lfm_m1 strix1 $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf --slots 8 --ctx 4096
echo BATCH_S1M_DONE
