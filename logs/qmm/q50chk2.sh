#!/bin/bash
cd ~/strix-rt; N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for f in blk.1.ffn_up_exps blk.1.ffn_down_exps blk.1.ffn_up_shexp blk.1.ffn_up_shexp; do echo "== $f"
QMMH=1 ITERS=5 timeout 120 target-qh/release/pf_gemm $N 512 $f 2>&1 | grep -E "Q5_0|qgemmh vs|QMMH|NONDET|panick|^\| blk" | cut -c1-200; done
