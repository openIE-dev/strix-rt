#!/bin/bash
# 30B prefill: f16 grouped GEMM (PF_QMM=0) vs packed-LDS pf_qgemmh (PF_QMM=1): pp511 speed + continuation ids (p512, long 830)
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for q in 0 1; do echo "PF_QMM=$q p512"; PF=1 PF_QMM=$q RUNS=4 timeout 40 target-nh/release/nh $M "$(cat logs/nh/p512.ids)" 32 2>&1 | grep -v ^weights | cut -c1-400
echo "PF_QMM=$q long"; PF=1 PF_QMM=$q RUNS=1 timeout 40 target-nh/release/nh $M "$(cat logs/nh/long_prompt.ids)" 48 2>&1 | grep -v ^weights | cut -c1-500; done
