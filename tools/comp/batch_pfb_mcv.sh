#!/bin/bash
cd ~/strix-rt
GPU_LOCK_TAG="pfb eq mcv" logs/gpu_lock.sh tools/comp/serve1_eq.sh $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf 8 48 --slots 8 --ctx 4096 > logs/serve/eq1_mcv_pfb.log 2>&1
GPU_LOCK_TAG="pfb img mcv" logs/gpu_lock.sh tools/comp/serve1_test.sh $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf > logs/serve/s1test_mcv_pfb.log 2>&1
grep -q "8/8" logs/serve/eq1_mcv_pfb.log && tools/comp/serve_split.sh tools/comp/serve_bench_strix.sh strix1_mcv_pfb strix1 $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf --slots 8 --ctx 4096
echo BATCH_PFB_MCV_DONE
