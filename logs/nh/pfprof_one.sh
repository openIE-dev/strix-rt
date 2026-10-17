#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
PFPROF=4 timeout 90 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -v ^weights | cut -c1-120
