#!/bin/bash
# one lock session per image: llama.cpp mtmd-cli (greedy, dumps preprocessed pixels + embeddings + answer), then strix-rt n35img on the same file
# usage: sess_img.sh <img> [prompt]
cd ~/strix-rt/logs/n35; D=$MODELS/new/nemotron35vl; I=$1; N=$(basename $I | sed "s/\..*//"); P=${2:-"Describe this image in one or two sentences."}
mkdir -p imgref
MTMD_DUMP_PX=imgref/$N.px MTMD_DUMP_EMB=imgref/$N.emb timeout 600 ~/llama-n35/build-vulkan/bin/llama-mtmd-cli -m $D/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf \
  --mmproj $D/mmproj-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16.gguf --image $I -p "$P" -n 80 --temp 0 --top-k 1 -ngl 99 -c 4096 --no-warmup > imgref/$N.ans 2> imgref/$N.err; echo "mtmd-cli rc $?"
ls -la imgref/$N.px imgref/$N.emb
