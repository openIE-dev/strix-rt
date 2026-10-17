#!/bin/bash
# pf_qgemmh tile/k-step variants on Nemotron 30B blk.1 experts (Q5_0 up K=2688, Q8_0 down K=1856), M=512; (NJ,KD) sweep
cd ~/strix-rt; N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for c in "2 8" "2 4" "2 2" "1 8" "1 4"; do set -- $c; echo "== NJ=$1 KD=$2"; PF_QH_NJ=$1 PF_QH_KD=$2 QMMH=1 ITERS=10 timeout 50 target-fa3/release/pf_gemm $N 512 blk.1.ffn_ 2>&1 | grep -E "QMMH.*exps|rror|panic" | cut -c1-200; done; echo DONE
