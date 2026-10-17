#!/bin/bash
# one lock session: llama-server greedy refs for 2 long prompts (>2048 window), 48 tokens
cd ~/strix-rt/logs/glimmer; D=$MODELS/new/glimmer
~/llama-latest/build-vulkan/bin/llama-server -m $D/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf -ngl 999 -c 8192 -fa on --port 18237 -np 1 > srv_ref_long.log 2>&1 &
SP=$!
trap "kill $SP 2>/dev/null; wait $SP 2>/dev/null" EXIT
for i in $(seq 1 200); do kill -0 $SP 2>/dev/null || { echo server died; tail -8 srv_ref_long.log; exit 1; }; curl -s --max-time 2 http://127.0.0.1:18237/health | grep -q ok && break; sleep 1; done
echo ready $i
timeout 200 python3 ref_llama.py 18237 48 ref_long.json prompts_long.json
