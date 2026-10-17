#!/bin/bash
# one lock session: llama-server (Vulkan, mmproj) greedy refs: 6 text prompts + 1 image prompt, 128 tokens
cd ~/strix-rt/logs/glimmer; D=$MODELS/new/glimmer
~/llama-latest/build-vulkan/bin/llama-server -m $D/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --mmproj $D/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf -ngl 999 -c 4096 -fa on --port 18236 -np 1 > srv_ref.log 2>&1 & SP=$!
trap "kill $SP 2>/dev/null; wait $SP 2>/dev/null" EXIT
for i in $(seq 1 200); do kill -0 $SP 2>/dev/null || { echo server died; tail -8 srv_ref.log; exit 1; }; curl -s --max-time 2 http://127.0.0.1:18236/health | grep -q ok && break; sleep 1; done
echo "ready after $i s"
timeout 100 python3 ref_img.py 18236 128 ref_llama_img.json $D/test_image.png
