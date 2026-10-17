#!/bin/bash
cd ~/strix-rt; N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for r in 1 2 3 4 5; do for cfg in "2 8 8" "4 4 4" "2 4 4" "4 8 8" "4 4 8" "2 8 4" "4 2 2" "1 8 8" "2 16 8"; do set -- $cfg
 for f in blk.1.ffn_up_exps blk.1.ffn_down_exps; do echo "MI=$1 NJ=$2 WR=$3 $f $(PF_MI=$1 PF_NJ=$2 PF_WR=$3 QMMH=1 ITERS=30 timeout 100 target-qh/release/pf_gemm $N 512 $f 2>&1 | grep -E "^  QMMH|f16 " | sed -E "s/.*(f16 [0-9]+ us [0-9.]+ TF).*/\1/" | head -1)"; done; done; done
