#!/bin/bash
# pf_qgemmh with Q8_0/Q5_0 support: Nemotron 30B expert slices (blk.1, blk.3) + LFM blk.0, M=512, vs pf_gemm
cd ~/strix-rt; B=target-fa3/release
for f in blk.1. blk.3.; do echo "== nemotron $f"; QMMH=1 ITERS=10 timeout 100 $B/pf_gemm $MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf 512 $f 2>&1 | grep -E "QMMH|exps|rror|panic" | cut -c1-200; done
echo "== lfm"; QMMH=1 ITERS=10 timeout 60 $B/pf_gemm $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf 512 blk.0.ffn_gate 2>&1 | grep -E "QMMH|rror|panic"; echo DONE
