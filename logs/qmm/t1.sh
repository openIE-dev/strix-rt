#!/bin/bash
# pf_qgemm (packed-LDS int8 Q4_K) vs f16 pf_gemm on LFM dense Q4_K tensors
cd ~/strix-rt; for M in 512 2048; do QMM=1 ITERS=5 timeout 120 target-fa3/release/pf_gemm $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf $M 2>&1 | grep -E "QMM|Q4K|rror|panic|aggregate"; done; echo DONE
