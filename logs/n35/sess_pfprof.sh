#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; P=$(cat logs/nh/p512.ids)
ZEROCOPY=0 MAX_CTX=1024 PF=1 PFPROF=3 timeout 250 target-n35/release/n35 $M "$P" 2>&1 | grep -E "prefill chunk|ms " 
