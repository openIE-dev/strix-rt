#!/bin/bash
# one lock session: strix-rt n35 greedy + teacher-forced parity vs llama.cpp refs (ref_llama.txt)
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; free -g | head -2
P=$(python3 -c 'import json;print(",".join(map(str,json.load(open("logs/n35/ref_llama.json"))[0]["ids"])))')
ZEROCOPY=${ZC:-0} STEP_CHUNK=${STEP_CHUNK:-16} MAX_CTX=256 TOPK=5 timeout 200 target/release/n35 $M "$P" 2>&1 | tail -3
ZEROCOPY=${ZC:-0} STEP_CHUNK=${STEP_CHUNK:-16} MAX_CTX=256 timeout 200 target/release/n35 $M PARITY logs/n35/ref_llama.txt 2>&1
