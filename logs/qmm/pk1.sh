#!/bin/bash
# packed-LDS worker: pf_qgemmh WR 2 vs 1 after the b5de625 fix, M=512; LFM Q4_K ffn_gate and Nemotron blk.1 experts
cd ~/strix-rt; B=target-fa3/release; N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for wr in 2 1 2 1; do echo "== LFM WR=$wr $(PF_QH_WR=$wr QMMH=1 ITERS=10 timeout 50 $B/pf_gemm $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 512 blk.0.ffn_gate 2>&1 | grep -E "QMMH|rror|panic")"; sleep 3; done
for c in "2 8" "1 8" "1 2"; do set -- $c; echo "== NH WR=$1 KD=$2"; PF_QH_WR=$1 PF_QH_KD=$2 QMMH=1 ITERS=10 timeout 50 $B/pf_gemm $N 512 blk.1.ffn_ 2>&1 | grep -E "QMMH.*exps|rror|panic" | cut -c1-200; done; echo DONE
