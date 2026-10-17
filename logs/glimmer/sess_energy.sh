#!/bin/bash
# one lock session (perf high): differential decode energy (N=320 minus N=64) for llama-bench vulkan|rocm
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
case $1 in vulkan) B=~/llama-latest/build-vulkan/bin/llama-bench;; rocm) B=~/llama-latest/build-rocm714/bin/llama-bench;; esac
tools/comp/energy.sh glimmer_llama_$1 "timeout 200 $B -m $M -ngl 999 -fa 1 -p 0 -n @N@ -r 1"
