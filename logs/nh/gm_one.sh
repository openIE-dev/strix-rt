#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for g in 32 16 64 128 32800 32; do echo "## PFG_M=$g"
PFG_M=$g PFPROF=4 timeout 60 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "prefill chunk|_exps|rror" | cut -c1-100
PFG_M=$g PF=1 RUNS=3 timeout 60 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "^prefill|rror" | cut -c1-60
done
