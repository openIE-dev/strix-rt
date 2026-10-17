#!/bin/bash
# one locked session: nemotron 30B prefill pp511 with NH_SKIP=$1 (profiling mask), 4 runs
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
echo "## NH_SKIP=$1"; NH_SKIP=$1 PF=1 RUNS=4 timeout 90 target-nhp/release/nh $M "$(cat logs/nh/p512.ids)" 2 2>&1 | grep -E "^prefill" | cut -c1-120
