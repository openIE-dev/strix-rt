#!/bin/bash
# pf_qgemmh (packed Q4_K LDS, f16 register unpack, f16 activations) vs pf_gemm: bitwise diff + speed; LFM PF_QMM=2 smoke
cd ~/strix-rt; B=target-fa3/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
for M in 512 2048; do for mi in 2 1; do echo "== M$M MI=$mi"; PF_QMM_MI=$mi QMMH=1 ITERS=10 timeout 60 $B/pf_gemm $L $M blk.0. 2>&1 | grep -E "QMMH|rror|panic"; sleep 5; done; done
for q in 2 0; do echo "== QMM=$q L512"; PF_QMM=$q REPS=3 timeout 60 $B/pf_lfm $L 512 4 2>&1 | grep -E "logits max|pp512|rror|panic"; sleep 10; done; echo DONE
