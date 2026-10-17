#!/bin/bash
# pf_qgemmh dequant rounding vs pf_gemm: coopmat mul-sub (0), per-element fma (4), per-element precise mul,sub (8); LFM blk.0.ffn_gate M=512
cd ~/strix-rt; for p in 0 4 8; do echo "== PROBE=$p $(PF_QMM_PROBE=$p QMMH=1 ITERS=10 timeout 60 target-fa3/release/pf_gemm $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 512 blk.0.ffn_gate 2>&1 | grep -E "QMMH|rror|panic")"; done; echo DONE
