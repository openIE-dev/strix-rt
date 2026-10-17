#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for f in logs/nh/p90.ids logs/nh/p200.ids logs/nh/p512.ids; do for k in 0 32; do echo "$f NH_SKIP=$k $(NH_SKIP=$k PF=1 RUNS=4 timeout 80 target-nhp/release/nh $M "$(cat $f)" 2 2>&1 | grep -E "^prefill" | tail -2 | cut -c1-45 | tr "\n" " ")"; done; done
