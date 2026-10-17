#!/bin/bash
# 30B-A3B parity: llama.cpp ref (short + long prompt), strix-rt token-by-token and batched prefill
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
[ -f logs/nh/ref30b.ids ] || logs/nh_ref.sh $M 48 "The history of the Roman Empire begins with" logs/nh/ref30b
[ -f logs/nh/ref30b_long.ids ] || logs/nh_ref.sh $M 24 "$(cat logs/nh/long_prompt.txt)" logs/nh/ref30b_long
timeout 100 target-nh/release/nh $M "$(cat logs/nh/p30.ids)" 48 2>&1 | grep -v ^weights
PF=1 timeout 60 target-nh/release/nh $M "$(cat logs/nh/p30.ids)" 48 2>&1 | grep -v ^weights
PF=1 MAX_CTX=2048 timeout 60 target-nh/release/nh $M "$(tail -c +5 logs/nh/long_prompt.ids | sed s/^/[/)" 24 2>&1 | grep -v ^weights
