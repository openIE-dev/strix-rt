#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for r in 1 2; do for w in 0 64 100 128 200 400; do echo "PF_MINWG=$w $(PF_MINWG=$w PF=1 RUNS=5 timeout 80 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "^prefill" | cut -c15-40 | tr "\n" " ")"; done; done
for f in logs/nh/p512.ids logs/nh/long_prompt.ids logs/nh/p30.ids logs/nh/div30.ids logs/nh/p200.ids logs/nh/p90.ids; do
 for w in 0 128 400; do echo "$f MINWG=$w $(PF_MINWG=$w PF=1 timeout 60 target-nhp/release/nh $M "$(cat $f)" 48 2>&1 | grep ^ids | md5sum | cut -c1-8)"; done
done
