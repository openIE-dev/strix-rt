#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf
ZEROCOPY=0 PROF=5 MAX_CTX=64 timeout 120 target/release/n35 $M 1784,6330 2>&1 | grep -v "^weights"
