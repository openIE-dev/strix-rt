#!/bin/bash
# pf_qgemmh first run (isolated, after the Oct 15 hang): LFM blk.0 Q4_K M=512 MI=2, bitwise vs pf_gemm
cd ~/strix-rt; PF_QMM_MI=2 QMMH=1 ITERS=10 timeout 60 target-fa3/release/pf_gemm $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 512 blk.0. 2>&1 | grep -E "QMMH|rror|panic"; echo "DONE rc=$?"
