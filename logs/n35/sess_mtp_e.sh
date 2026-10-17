#!/bin/bash
# logs/n35/sess_mtp_e.sh PROMPT N ONLY(1 plain,2 mtp only): sampled watts over active samples
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; P=$(cat logs/nh/${1:-p30}.ids)
bash logs/g4/esample.sh n35e_$3 env ZEROCOPY=0 STEP_CHUNK=16 MAX_CTX=1024 MTP_CMP=1 ONLY=$3 timeout 280 target-n35/release/n35 $M "$P" ${2:-256}
grep -E "ONLYPLAIN|MTP run|decode GPU" /tmp/esample_n35e_$3.out | cut -c1-250
