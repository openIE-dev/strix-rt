#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
echo "level $(cat /sys/class/drm/card1/device/power_dpm_force_performance_level)"
for r in 1 2 3; do for h in 0 16 8 0 16; do echo -n "HYB=$h "; PFG_HYB=$h PF=1 RUNS=5 timeout 80 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "^prefill|rror" | tail -1 | cut -c1-70; done; done
