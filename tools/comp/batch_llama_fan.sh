#!/bin/bash
# llama.cpp Vulkan + ROCm reruns for LFM2.5 and MiniCPM-V under the fan curve, one gpu_lock session per concurrency level
cd ~/strix-rt; B=tools/comp/serve_bench.sh
for k in vk:llama-vk hip:llama-hip; do
  tools/comp/serve_split.sh $B llama${k%%:*}_lfm_fan ${k#*:} $MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf -np 8 -c 16384
  tools/comp/serve_split.sh $B llama${k%%:*}_mcv_fan ${k#*:} $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf -np 8 -c 16384
done
echo BATCH_LLAMA_FAN_DONE
