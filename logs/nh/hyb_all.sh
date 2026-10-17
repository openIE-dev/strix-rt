#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for f in p512 long p30; do for h in 0 16 8; do echo "ids $f HYB=$h"; PFG_HYB=$h PF=1 RUNS=1 target-nhp/release/nh $M "$(cat logs/nh/$f.ids)" 48 2>&1 | tail -4 | cut -c1-300; done; done
