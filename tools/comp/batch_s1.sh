#!/bin/bash
cd ~/strix-rt
logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve1_test.sh $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf > logs/serve/s1_lfm.log 2>&1
logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve1_test.sh $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ~/mcv_ref/fox.png > logs/serve/s1_mcv.log 2>&1
