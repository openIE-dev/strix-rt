#!/bin/bash
# one lock session (perf high): llama.cpp Vulkan llama-bench pp512/tg128, then strix-rt n35 decode (RUNS=3, 12-token prompt, 128 tokens)
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; free -g | head -2
if [ "$1" != ours ]; then timeout 200 ~/llama-n35/build-vulkan/bin/llama-bench -m $M -ngl 999 -fa on -p 512 -n 128 -r 3 -o md 2>&1 | grep -v "^ggml_vulkan\|^load_backend"; fi
if [ "$1" != llama ]; then P=$(python3 -c 'import json;print(",".join(map(str,json.load(open("logs/n35/ref_llama.json"))[0]["ids"])))')
  ZEROCOPY=${ZC:-0} STEP_CHUNK=32 MAX_CTX=512 RUNS=3 timeout 200 target/release/n35 $M "$P" 128 2>&1 | grep -v "^warm"; fi
