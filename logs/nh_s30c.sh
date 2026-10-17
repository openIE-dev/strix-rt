#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
MAX_CTX=2048 timeout 60 target-nh/release/nh $M "$(tail -c +5 logs/nh/long_prompt.ids | sed s/^/[/)" 24 2>&1 | grep -v ^weights
