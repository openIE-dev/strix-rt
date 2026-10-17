#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
TOPK=6 MAX_CTX=2048 timeout 60 target-nh/release/nh $M "$(cat logs/nh/div30.ids)" 1 2>&1 | grep top
