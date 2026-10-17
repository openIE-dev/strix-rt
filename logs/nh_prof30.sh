#!/bin/bash
# per-segment profile of one 30B decode step (PROF=n repeats, last reported) + parity/speed check on p30
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
PROF=4 timeout 60 target-nh/release/nh $M "$(cat logs/nh/p30.ids)" 1 2>&1 | grep -v ^weights
timeout 60 target-nh/release/nh $M "$(cat logs/nh/p30.ids)" 128 2>&1 | grep -v ^weights | cut -c1-400
