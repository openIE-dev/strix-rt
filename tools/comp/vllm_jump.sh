#!/bin/bash
# Investigate vLLM gpt-oss C=4 -> C=8 jump: (a) order 8,1,4,6,8 (warm-up vs batch-size effect), (b) enforce-eager C=1,8
cd ~/strix-rt; S=tools/comp/serve_bench.sh; A="--max-model-len 4096 --gpu-memory-utilization 0.6"
CONC="8 1 4 6 8" $S vllmj_order vllm gpt-oss-20b $A
CONC="1 8" $S vllmj_eager vllm gpt-oss-20b $A --enforce-eager
