#!/bin/bash
# batch 3: llama-server (Vulkan + HIP) and vLLM concurrency C=1,2,4,8
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench.sh; MD=$MODELS
for t in gptoss:bench/gpt-oss-20b-MXFP4 q36:bench/Qwen3.6-35B-A3B-Q8_0 lfm:LFM2.5-24B-A2B-Q4_K_M q38:Qwen3.8-27B-UD-Q4_K_M mcv:minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M; do
  for k in vk hip; do $R serve $S llama${k}_${t%%:*} llama-$k $MD/${t#*:}.gguf -np 8 -c 16384; done; done
for m in Qwen3.6-35B-A3B-FP8 LFM2.5-24B-A2B-FP8 Qwen3.8-27B-FP8; do $R serve $S vllm_$m vllm $m --max-model-len 4096 --gpu-memory-utilization 0.6; done
