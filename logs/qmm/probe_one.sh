#!/bin/bash
cd ~/strix-rt; N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for r in 1 2 3 4 5; do for p in 2 0 1; do for f in blk.1.ffn_up_exps blk.1.ffn_down_exps; do echo "PROBE=$p MS=512 $f $(PF_PROBE=$p QMMH=1 ITERS=30 timeout 100 target-qh/release/pf_gemm $N 512 $f 2>&1 | grep -E "^  QMMH" | sed -E "s/.*(f16 [0-9]+ us [0-9.]+ TF).*/\1/")"; done; done; done
