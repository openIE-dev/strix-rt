#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; P=$(cat logs/n35/p512_9.ids)
for pf in 0 1; do echo "PF=$pf"; ZEROCOPY=0 STEP_CHUNK=16 MAX_CTX=1024 PF=$pf TOPK=5 timeout 250 target-n35/release/n35 $M "$P" 2>&1 | grep "^top"; done
