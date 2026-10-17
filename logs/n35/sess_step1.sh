#!/bin/bash
# one lock session: one step per submission (STEP_CHUNK=1), 8-token prompt + 16 tokens, 2 runs
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf
P=$(python3 -c "import json;print(\",\".join(map(str,json.load(open(\"logs/n35/ref_llama.json\"))[0][\"ids\"])))")
ZEROCOPY=0 STEP_CHUNK=1 MAX_CTX=64 RUNS=3 timeout 120 target/release/n35 $M "$P" 16 2>&1 | grep -v "^weights"
