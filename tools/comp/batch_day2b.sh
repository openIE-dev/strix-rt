#!/bin/bash
# strix reruns after strixpc reload (the first day2 pass died on missing /dev/strixpc); waits for batch_day2
cd ~/strix-rt; M=$MODELS; L=$M/LFM2.5-24B-A2B-Q4_K_M.gguf; V=$M/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
while pgrep -f "batch_day[2].sh" >/dev/null; do sleep 20; done
CS="2 4 8" tools/comp/serve_split.sh tools/comp/serve_bench_strix.sh strix1_lfm_pfb2 strix1 $L --slots 8 --ctx 4096
GPU_LOCK_TAG="pfb eq mcv" logs/gpu_lock.sh timeout -k 10 290 tools/comp/serve1_eq.sh $V 8 48 --slots 8 --ctx 4096 > logs/serve/eq1_mcv_pfb.log 2>&1
GPU_LOCK_TAG="pfb img mcv" logs/gpu_lock.sh timeout -k 10 290 tools/comp/serve1_test.sh $V > logs/serve/s1test_mcv_pfb.log 2>&1
grep -q "8/8" logs/serve/eq1_mcv_pfb.log && tools/comp/serve_split.sh tools/comp/serve_bench_strix.sh strix1_mcv_pfb strix1 $V --slots 8 --ctx 4096
echo BATCH_DAY2B_DONE
