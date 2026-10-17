#!/bin/bash
cd ~/strix-rt; N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for f in blk.1.ffn_up_exps blk.1.ffn_down_exps blk.1.ffn_; do for c in "2 8" "1 4"; do set -- $c; echo "== $f NJ=$1 KD=$2"
PF_QH_NJ=$1 PF_QH_KD=$2 QMMH=1 ITERS=5 timeout 120 target-qh/release/pf_gemm $N 512 $f 2>&1 | grep -E "Q5_0|qgemmh vs|QMMH|NONDET|panick" | cut -c1-200; done; done
