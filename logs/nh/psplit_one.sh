#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for ps in 1 2 4 8; do echo "## NH_PSPLIT=$ps"
NH_PSPLIT=$ps PFPROF=4 timeout 60 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "prefill chunk|m\.scan"
NH_PSPLIT=$ps PF=1 RUNS=3 timeout 60 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 32 2>&1 | grep -E "^ids|^prefill" | cut -c1-300
done
