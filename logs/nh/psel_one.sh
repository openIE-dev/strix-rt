#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for tb in 1 2 4 8; do echo "## NH_PSEL_TB=$tb"
NH_PSEL_TB=$tb PFPROF=4 timeout 60 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "prefill chunk|psel"
NH_PSEL_TB=$tb PF=1 RUNS=3 timeout 60 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "^prefill" | cut -c1-60
done
for f in logs/nh/p512.ids logs/nh/long_prompt.ids logs/nh/p30.ids logs/nh/div30.ids /tmp/nh_p200.ids /tmp/nh_p90.ids; do
 for tb in 1 4 8; do echo "$f TB=$tb $(NH_PSEL_TB=$tb PF=1 timeout 60 target-nhp/release/nh $M "$(cat $f)" 48 2>&1 | grep ^ids | md5sum | cut -c1-8)"; done
done
