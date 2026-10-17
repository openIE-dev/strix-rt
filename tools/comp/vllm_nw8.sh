#!/bin/bash
# Root-cause test for the vLLM gpt-oss 4->5 client step: force num_warps=8 in triton_kernels opt_flags (stock: 2 warps when m=tokens*topk<=16)
cd ~/strix-rt; P=/opt/venv/lib/python3.12/site-packages/vllm/third_party/triton_kernels/matmul_details/opt_flags.py
VLLM_DOCKER_EXTRA="-v $HOME/strix-rt/tools/comp/vpatch/opt_flags_nw8.py:$P:ro" CONC="1 2 4 5 8" tools/comp/serve_bench.sh vllm_nw8 vllm gpt-oss-20b --max-model-len 4096 --gpu-memory-utilization 0.6
