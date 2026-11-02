#!/bin/bash
# tools/npu/driver.sh: NPU (FLM) vs iGPU (llama.cpp Vulkan) for the same small models; 2 reps each, each config under gpu_lock.
cd ~/strix-rt; M=$MODELS/npu_cmp
for rep in 1 2; do for c in "flm qwen3.5:0.8b" "vk $M/q35-08b-q4km.gguf" "flm llama3.2:1b" "vk $M/l32-1b-q4km.gguf"; do
  GPU_LOCK_TAG="npu $c $rep" logs/gpu_lock.sh bash tools/npu/npu_session.sh $c $rep
done; done; echo DRIVER_DONE
