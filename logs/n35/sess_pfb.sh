#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf
for f in p512 p200; do P=$(cat logs/nh/$f.ids)
echo "== $f PF=1"; ZEROCOPY=0 STEP_CHUNK=16 MAX_CTX=1024 PF=1 RUNS=3 timeout 250 target-n35/release/n35 $M "$P" 32 2>&1 | grep -E "^ids|prompt" | cut -c1-260
done
