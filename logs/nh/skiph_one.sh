#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for k in 0 32 64 1 2 8 4 16 256 128 0; do echo "NH_SKIP=$k $(NH_SKIP=$k PF=1 RUNS=4 timeout 80 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "^prefill" | cut -c15-33 | tr "\n" " ")"; done
