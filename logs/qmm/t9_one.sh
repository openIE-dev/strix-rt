#!/bin/bash
# usage: t9_one.sh NJ KD  (runs inside the GPU lock; one short session)
N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for f in blk.1.ffn_up_exps blk.1.ffn_down_exps; do
PF_QH_NJ=$1 PF_QH_KD=$2 QMMH=1 ITERS=20 timeout 100 ~/strix-rt/target-fa3/release/pf_gemm $N 512 $f 2>&1 | grep -E "^  QMMH" | cut -c1-150
done
