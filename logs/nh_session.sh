#!/bin/bash
# nh_session.sh: one GPU session: llama.cpp long-prompt reference + strix-rt decode-prefill and batched-prefill parity
cd ~/strix-rt; M=${M:-$MODELS/new/nemotron3/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf}; T=${T:-4b}
P="$(cat logs/nh/long_prompt.ids)"; S="[1, 1784, 6330, 1307, 1278, 9718, 18344, 19570, 1454]"
[ -f logs/nh/ref${T}_long.ids ] || logs/nh_ref.sh $M 48 "$(cat logs/nh/long_prompt.txt)" logs/nh/ref${T}_long
PF=1 timeout 60 target-nh/release/nh $M "$S" 32 2>&1 | grep -v ^weights
MAX_CTX=2048 timeout 60 target-nh/release/nh $M "$P" 24 2>&1 | grep -v ^weights
PF=1 MAX_CTX=2048 timeout 60 target-nh/release/nh $M "$P" 24 2>&1 | grep -v ^weights
