#!/bin/bash
# 30B MoE prefill tile sweep: PFG_M (grouped GEMM rows) x PF_M (chunk), pp511 + 16-token continuation ids for parity
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf; P="$(cat logs/nh/p512.ids)"
for pm in 512 256; do for g in 16 32 64; do echo "PF_M=$pm PFG_M=$g"; PF=1 PF_M=$pm PFG_M=$g RUNS=4 timeout 40 target-nh/release/nh $M "$P" 16 2>&1 | grep -v ^weights | cut -c1-160; done; done
