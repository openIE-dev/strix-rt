#!/bin/bash
# one lock session: llama.cpp mtmd-debug rainbow-512 encode dump, then strix-rt n35v on the same image
cd ~/strix-rt/logs/n35; D=$MODELS/new/nemotron35vl; free -g | head -2
MTMD_DUMP=$PWD/ref_rainbow512.f32 timeout 120 ~/llama-n35/build-vulkan/bin/llama-mtmd-debug -m ${TEXT_M:-$D/mtp-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf} \
  --mmproj $D/mmproj-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16.gguf -p encode --image rainbow -n 512 -ngl 0 > mtmd_debug.log 2>&1; echo "mtmd-debug rc $?"
grep -a -i "dumped\|error\|failed" mtmd_debug.log | head -5
timeout 120 ~/strix-rt/target/release/n35v $D/mmproj_st 512 ref_rainbow512.f32 2>&1 | grep -v "^weights"
