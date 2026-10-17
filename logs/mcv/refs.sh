#!/bin/bash
# reference dumps (llama.cpp Vulkan mtmd greedy) for extra images
M=$MODELS/minicpmv47
cd ~/mcv_ref
for im in gold chess; do mkdir -p ref_$im
  timeout 300 ./mcvdump $M/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf $M/mmproj-model-f16.gguf $im.png prompt_img.txt ref_$im 128 > ref_$im/log.txt 2>&1; echo "$im rc=$?"
done
echo REFSDONE
