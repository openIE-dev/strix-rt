#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
logs/nh_ref.sh $M 48 "The history of the Roman Empire begins with" logs/nh/ref30b
timeout 120 target-nh/release/nh $M "$(cat logs/nh/p30.ids)" 48 2>&1 | grep -v ^weights
