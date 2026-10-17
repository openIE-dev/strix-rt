#!/bin/bash
cd ~/strix-rt; N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
N=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for r in 1 2 3 4 5; do for cfg in "PF_LS4=9" "PF_LS4=10" "PF_LS4=11" "PF_LS4=13" "PF_ADIR=0 PF_LS4=5" "PF_KS=1 PF_LS4=5" "PF_SG=64" "PF_PROBE=1 PF_LS4=9" "PF_PROBE=1 PF_LS4=16"; do
 for f in blk.1.ffn_up_exps blk.1.ffn_down_exps; do echo "$cfg $f $(env $cfg QMMH=1 ITERS=30 timeout 100 target-qh/release/pf_gemm $N 512 $f 2>&1 | grep -E "f16 " | sed -E "s/.*(f16 [0-9]+ us [0-9.]+ TF).*/\1/" | head -1)"; done; done; done
