#!/bin/bash
# one lock session: single decode steps with the first N blocks only; stops at the first failure (never retries a hang)
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; free -g | head -2
for N in ${LAYERS:-1 2 8 88}; do
  echo "== N35_LAYERS=$N"; N35_LAYERS=$N ZEROCOPY=0 DEBUG1=1 MAX_CTX=64 timeout 90 target/release/n35 $M 1784,6330 2>&1 | grep -v "^weights"; rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && { echo "stop: rc $rc"; break; }
done
