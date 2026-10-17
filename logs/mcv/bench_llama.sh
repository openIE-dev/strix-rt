#!/bin/bash
# llama.cpp (Vulkan) baseline: mtmd image encode + llama-bench pp/tg
M=$MODELS/minicpmv47; B=~/llama.cpp/build-vulkan/bin; cd ~/mcv_ref
for im in fox gold chess; do
  timeout 300 $B/llama-mtmd-cli -m $M/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf --mmproj $M/mmproj-model-f16.gguf --image $im.png -p "Describe this image in one paragraph." -n 64 --temp 0 -ngl 99 2>&1 | grep -E "encoded in|decoded in|eval time|load time" | sed "s/^/$im: /"
done
timeout 600 $B/llama-bench -m $M/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf -ngl 99 -fa 1 -p 91,512 -n 128 -r 3
echo BLDONE
