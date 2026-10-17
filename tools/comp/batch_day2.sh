#!/bin/bash
# reruns after the GPU-hang reboot; every GPU session is its own gpu_lock hold, each under 5 min
cd ~/strix-rt; M=$MODELS; L=$M/LFM2.5-24B-A2B-Q4_K_M.gguf; V=$M/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
CS="2 4 8" tools/comp/serve_split.sh tools/comp/serve_bench_strix.sh strix1_lfm_pfb2 strix1 $L --slots 8 --ctx 4096
MODELS="$L $V" NAMES=lfm,mcv NEED_GB=60 GPU_LOCK_TAG=merge2 logs/gpu_lock.sh timeout -k 10 290 tools/comp/merge_test.sh > logs/serve/merge_test2.log 2>&1
GPU_LOCK_TAG="pfb eq mcv" logs/gpu_lock.sh timeout -k 10 290 tools/comp/serve1_eq.sh $V 8 48 --slots 8 --ctx 4096 > logs/serve/eq1_mcv_pfb.log 2>&1
GPU_LOCK_TAG="pfb img mcv" logs/gpu_lock.sh timeout -k 10 290 tools/comp/serve1_test.sh $V > logs/serve/s1test_mcv_pfb.log 2>&1
grep -q "8/8" logs/serve/eq1_mcv_pfb.log && tools/comp/serve_split.sh tools/comp/serve_bench_strix.sh strix1_mcv_pfb strix1 $V --slots 8 --ctx 4096
CS=8 tools/comp/serve_split.sh tools/comp/serve_bench.sh llamavk_mcv_fan llama-vk $V -np 8 -c 16384
tools/comp/serve_split.sh tools/comp/serve_bench.sh llamahip_lfm_fan llama-hip $L -np 8 -c 16384
tools/comp/serve_split.sh tools/comp/serve_bench.sh llamahip_mcv_fan llama-hip $V -np 8 -c 16384
echo BATCH_DAY2_DONE
