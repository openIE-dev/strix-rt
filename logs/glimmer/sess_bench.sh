#!/bin/bash
# one lock session: llama-bench pp512/pp2048/tg128. usage: sess_bench.sh vulkan|rocm
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
case $1 in vulkan) B=~/llama-latest/build-vulkan/bin/llama-bench;; rocm) B=~/llama-latest/build-rocm714/bin/llama-bench;; esac
free -g | head -2
timeout 280 $B -m $M -ngl 999 -fa 1 -p 512,2048 -n 128 -r 3 -o md 2>&1 | grep -v "^ggml_vulkan\|^load_backend"
