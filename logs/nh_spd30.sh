#!/bin/bash
# 30B decode speed A/B (MOE_LPR) + parity ids on the short prompt
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for l in 8 32; do echo "MOE_LPR=$l"; MOE_LPR=$l timeout 60 target-nh/release/nh $M "$(cat logs/nh/p30.ids)" 128 2>&1 | grep -v ^weights | cut -c1-200; done
