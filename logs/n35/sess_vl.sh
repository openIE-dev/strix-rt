#!/bin/bash
# one lock session: strix-rt n35vl (real-image preprocess + ViT + prefill injection + greedy decode) on <img>; ids of llama.cpp answer for comparison
# usage: sess_vl.sh <img> [variant: think|nothink] [n_new]
cd ~/strix-rt/logs/n35; D=$MODELS/new/nemotron35vl; I=$1; V=${2:-think}; NN=${3:-60}; B=$(basename $I | sed "s/\..*//")
M=$D/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; T=~/llama-n35/build-vulkan/bin/llama-tokenize
P=${PROMPT:-"Describe this image in one or two sentences."}
tk() { $T -m $M --no-bos --ids --log-disable -p "$1"; }
tk "<|im_start|>user
<img>" > /tmp/vl_pre.ids
if [ $V = think ]; then TAIL="<think></think>"; else TAIL=""; fi
tk "</img>$P<|im_end|>
<|im_start|>assistant
$TAIL" > /tmp/vl_post.ids
MAX_CTX=${MAX_CTX:-4096} ~/strix-rt/target/release/n35vl $M $D/mmproj_st $I /tmp/vl_pre.ids /tmp/vl_post.ids $NN 2>&1 | grep -v "^weights"
