#!/bin/bash
# strix-serve1 LFM2.5 batched prefill: eq, then C=1/2/4/8 (one lock session per level)
cd ~/strix-rt
GPU_LOCK_TAG="pfb eq lfm" logs/gpu_lock.sh tools/comp/serve1_eq.sh $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 8 48 --slots 8 --ctx 4096 > logs/serve/eq1_lfm_pfb.log 2>&1
grep -q "8/8" logs/serve/eq1_lfm_pfb.log && tools/comp/serve_split.sh tools/comp/serve_bench_strix.sh strix1_lfm_pfb strix1 $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf --slots 8 --ctx 4096
echo BATCH_PFB_DONE
