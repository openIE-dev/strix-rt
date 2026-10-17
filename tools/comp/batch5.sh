#!/bin/bash
# vLLM AWQ-INT4 concurrency (gfx1151 has no FP8 support in vLLM: RocmPlatform.supports_fp8 = CDNA/RDNA4 only)
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench.sh
for m in Qwen3.8-27B-AWQ-INT4 Qwen3.6-35B-A3B-AWQ LFM2.5-24B-A2B-AWQ-INT4; do $R serve $S vllm_$m vllm $m --max-model-len 4096 --gpu-memory-utilization 0.6; done
