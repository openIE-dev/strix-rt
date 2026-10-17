#!/bin/bash
# pf_qgemm probes on LFM blk.0.ffn_gate (M=512, 2048): epilogue cost (PROBE=1), MMA cost (PROBE=2), MI=2
cd ~/strix-rt
for M in 512 2048; do for c in X=0 PF_QMM_PROBE=1 PF_QMM_PROBE=2 PF_QMM_PROBE=3 PF_QMM_MI=2; do echo "== M$M $c $(env $c QMM=1 ITERS=10 timeout 60 target-fa3/release/pf_gemm $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf $M blk.0.ffn_gate 2>&1 | grep -oE "f16 [0-9]+ us [0-9.]+ TF|qgemm [0-9]+ us [0-9.]+ TF|cos [0-9.]+|rror.*|panic.*" | tr "\n" " ")"; sleep 5; done; done; echo DONE
