#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; P=$(cat logs/nh/${1:-p30}.ids)
ZEROCOPY=0 STEP_CHUNK=16 MAX_CTX=1024 MTP_CMP=${3:-2} H_MODE=${HM:-1} timeout 280 target-n35/release/n35 $M "$P" ${2:-64} 2>&1 | grep -E "plain|MTP|decode GPU|mtp:|Error|error|panick" | cut -c1-300
