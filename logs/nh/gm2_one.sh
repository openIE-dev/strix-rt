#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for r in 1 2; do for g in 32 32800 64; do echo "## PFG_M=$g round $r"
PFG_M=$g PF=1 RUNS=5 timeout 80 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "^prefill|rror" | cut -c1-40 | tr "\n" " "; echo; done; done
for f in logs/nh/p512.ids logs/nh/long_prompt.ids logs/nh/p30.ids logs/nh/div30.ids logs/nh/p200.ids logs/nh/p90.ids; do
 for g in 32 64 32800 128; do echo "$f PFG_M=$g $(PFG_M=$g PF=1 timeout 60 target-nhp/release/nh $M "$(cat $f)" 48 2>&1 | grep ^ids | md5sum | cut -c1-8)"; done
done
